/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdint.h>
#include "primal_priv.h"

enum { MINIMAX_ROWS = 32, MINIMAX_BINARIES = 63, MINIMAX_INPUT_ROWS = 3*MINIMAX_ROWS+1 };

typedef struct {
    int m, n, epigraph, max_weight;
    int binary[MINIMAX_BINARIES];
    int weight[MINIMAX_BINARIES][MINIMAX_ROWS];
    int rhs[MINIMAX_ROWS], positive[MINIMAX_ROWS], negative[MINIMAX_ROWS];
} MinimaxModel;

static int zero_lower(PRIMALtask_t t, int j) {
    return t->bkx[j] == PRIMAL_BK_LO && t->blx[j] == 0.0;
}

/* Recognize t >= p_i,q_i and A_i x - p_i + q_i = b_i, where
 * p_i and q_i are private nonnegative slacks and x is binary. */
static int minimax_extract(PRIMALtask_t t, MinimaxModel *m) {
    int nr = t->numcon, nv = t->numvar;
    if (nr < 12 || nr > MINIMAX_INPUT_ROWS ||
        nv > MINIMAX_BINARIES + 2*MINIMAX_ROWS + 1)
        return 0;

    int equation[MINIMAX_INPUT_ROWS], abs_sum[MINIMAX_ROWS], caps_count = 0;
    for (int i = 0; i < nr; i++) {
        equation[i] = -1;
        if (t->bkc[i] == PRIMAL_BK_FX) {
            double rhs = t->blc[i];
            if (m->m >= MINIMAX_ROWS || !(rhs > 0.0 && rhs <= 8191.0) || rhs != floor(rhs))
                return 0;
            equation[i] = m->m;
            m->positive[m->m] = m->negative[m->m] = -1;
            abs_sum[m->m] = m->rhs[m->m] = (int)rhs;
            m->m++;
        } else if (t->bkc[i] != PRIMAL_BK_FR) {
            if (t->bkc[i] != PRIMAL_BK_LO || t->blc[i] != 0.0) return 0;
            caps_count++;
        }
    }
    if (m->m < 4 || caps_count != 2*m->m) return 0;

    m->epigraph = -1;
    unsigned char caps[MINIMAX_INPUT_ROWS] = {0};
    for (int j = 0; j < nv; j++) {
        const Col *col = &t->cols[j];
        if (t->c[j] != 0.0) {
            if (m->epigraph >= 0 || !(t->c[j] > 0.0) || !isfinite(t->c[j]) ||
                t->vartype[j] != PRIMAL_VAR_TYPE_CONT || !zero_lower(t,j) ||
                col->nz != 2*m->m)
                return 0;
            m->epigraph = j;
            for (int k = 0; k < col->nz; k++) {
                int i = col->sub[k];
                if (equation[i] >= 0 || col->val[k] != 1.0) return 0;
                caps[i] |= 1;
            }
        } else if (t->vartype[j] == PRIMAL_VAR_TYPE_INT ||
                   t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN) {
            double lo, hi;
            bound_range(t->bkx[j],t->blx[j],t->bux[j],&lo,&hi);
            if (lo != 0.0 || hi != 1.0 || col->nz != m->m || m->n >= MINIMAX_BINARIES)
                return 0;
            m->binary[m->n] = j;
            for (int k = 0; k < col->nz; k++) {
                int i = equation[col->sub[k]];
                double value = col->val[k];
                if (i < 0 || !(value > 0.0 && value <= 8191.0) || value != floor(value))
                    return 0;
                int a = (int)value;
                m->weight[m->n][i] = a;
                abs_sum[i] += a;
                if (abs_sum[i] > 8191) return 0;
                if (a > m->max_weight) m->max_weight = a;
            }
            m->n++;
        } else {
            if (t->vartype[j] != PRIMAL_VAR_TYPE_CONT || !zero_lower(t,j) || col->nz != 2)
                return 0;
            int k = equation[col->sub[0]] >= 0 ? 0 : 1;
            int i = equation[col->sub[k]], cap = col->sub[1-k];
            if (i < 0 || equation[cap] >= 0 || col->val[1-k] != -1.0 || (caps[cap] & 2))
                return 0;
            int *slack = col->val[k] == -1.0 ? m->positive : m->negative;
            if (fabs(col->val[k]) != 1.0 || slack[i] >= 0) return 0;
            slack[i] = j;
            caps[cap] |= 2;
        }
    }
    if (m->epigraph < 0 || m->n < 32) return 0;
    for (int i = 0; i < nr; i++)
        if (t->bkc[i] == PRIMAL_BK_LO && caps[i] != 3) return 0;
    for (int i = 0; i < m->m; i++)
        if (m->positive[i] < 0 || m->negative[i] < 0) return 0;
    return 1;
}

static uint32_t minimax_random(uint32_t *rng) {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    return *rng;
}

static int minimax_loss(const int *residual, int m, int radius) {
    int loss = 0;
    for (int i = 0; i < m; i++) {
        int excess = abs(residual[i]) - radius;
        if (excess > 0) loss += excess*excess;
    }
    return loss;
}

static int minimax_radius(const int *residual, int m) {
    int radius = 0;
    for (int i = 0; i < m; i++)
        if (abs(residual[i]) > radius) radius = abs(residual[i]);
    return radius;
}

/* Exchanges one selected and one unselected binary at a time. Restart at
 * nearby cardinalities estimated from the average weight and right-hand side. */
static uint64_t minimax_search(const MinimaxModel *m, double deadline) {
    int weight_sum = 0, rhs_sum = 0;
    for (int i = 0; i < m->m; i++) rhs_sum += m->rhs[i];
    for (int j = 0; j < m->n; j++)
        for (int i = 0; i < m->m; i++) weight_sum += m->weight[j][i];
    int cardinality = (int)floor((double)m->n*rhs_sum/weight_sum + 0.5);
    if (cardinality <= 1 || cardinality >= m->n-1) return 0;

    uint32_t rng = 27012026;
    uint64_t best = 0;
    int radius = 8192, aim = (m->max_weight+2)/4;
    double start_temperature = 2.0*(aim+1)*(aim+1);
    for (int restart = 0; restart < 750 && (double)clock() < deadline; restart++) {
        int count = cardinality - 1 + restart%3;
        int residual[MINIMAX_ROWS], trial[MINIMAX_ROWS];
        uint64_t selected = 0;
        for (int i = 0; i < m->m; i++) residual[i] = -m->rhs[i];
        for (int k = 0; k < count; k++) {
            int j;
            do { j = (int)(minimax_random(&rng) % (unsigned)m->n); }
            while (selected & (UINT64_C(1) << j));
            selected |= UINT64_C(1) << j;
            for (int i = 0; i < m->m; i++) residual[i] += m->weight[j][i];
        }

        int score = minimax_loss(residual,m->m,aim);
        double temperature = start_temperature;
        for (int step = 0; step < 75000; step++) {
            int j, k;
            do { j = (int)(minimax_random(&rng) % (unsigned)m->n); }
            while (!(selected & (UINT64_C(1) << j)));
            do { k = (int)(minimax_random(&rng) % (unsigned)m->n); }
            while (selected & (UINT64_C(1) << k));
            for (int i = 0; i < m->m; i++)
                trial[i] = residual[i] - m->weight[j][i] + m->weight[k][i];
            int candidate = minimax_loss(trial,m->m,aim);
            if (candidate < score ||
                (minimax_random(&rng)+1.0)/4294967296.0 < exp((score-candidate)/temperature)) {
                score = candidate;
                selected ^= (UINT64_C(1) << j) | (UINT64_C(1) << k);
                memcpy(residual,trial,(size_t)m->m*sizeof(*residual));
                int actual = minimax_radius(residual,m->m);
                if (actual < radius) {
                    best = selected;
                    radius = actual;
                    if (radius == 0) return best;
                    if (radius <= aim) {
                        aim = radius - 1;
                        score = minimax_loss(residual,m->m,aim);
                    }
                }
            }
            temperature *= 0.9999;
            if (temperature < start_temperature/90.0) temperature = start_temperature/90.0;
        }
    }
    return best;
}

void mip_minimax_incumbent(PRIMALtask_t t, const double *lx, const double *ux,
                            const double *lc, const double *uc, double *x,
                            double *best) {
    if (t->numcones || t->numbarvar || t->has_qobj || t->has_qcon || t->numsos ||
        t->mip_max_nodes <= 1 || t->num_threads > 1 || t->progcb || t->cbfn ||
        getenv("GMB_NO_MIP_MINIMAX"))
        return;
    MinimaxModel m = {0};
    if (!minimax_extract(t,&m)) return;

    double now = (double)clock();
    double deadline = now + 2.0*CLOCKS_PER_SEC;
    if (t->mip_deadline >= 0.0 && t->mip_deadline <= deadline) return;
    uint64_t selected = minimax_search(&m,deadline);
    if (!selected) return;

    int residual[MINIMAX_ROWS];
    for (int i = 0; i < m.m; i++) residual[i] = -m.rhs[i];
    memset(x,0,(size_t)t->numvar*sizeof(*x));
    for (int j = 0; j < m.n; j++) {
        if (!(selected & (UINT64_C(1) << j))) continue;
        x[m.binary[j]] = 1.0;
        for (int i = 0; i < m.m; i++) residual[i] += m.weight[j][i];
    }
    x[m.epigraph] = minimax_radius(residual,m.m);
    for (int i = 0; i < m.m; i++) {
        x[m.positive[i]] = fmax(0,residual[i]);
        x[m.negative[i]] = fmax(0,-residual[i]);
    }
    if (mip_point_measures(t,lx,ux,lc,uc,x,t->mip_tol_feas,t->mip_tol_inther)) {
        *best = t->cfix + t->c[m.epigraph]*x[m.epigraph];
        tlog(t,"MIP: binary minimax incumbent\n");
    }
}
