/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"

enum { POWER_LINK = 1, POWER_LOSS = 2, POWER_SCALAR = 1, POWER_MEMBER = 2 };

typedef struct {
    int count, first, last;
    double first_value, last_value;
} PowerRow;

typedef struct {
    int head_row, allocation_row, tail_row, loss_row;
    double gamma;
} PowerLoss;

typedef struct {
    int balance_row;
    double cost, alpha;
    PowerLoss *loss;
    unsigned char *column_role, *row_role;
} PowerRisk;

/* Read a cone coordinate equal to a positive multiple of a scalar. */
static int power_copy(PRIMALtask_t t, const PowerRow *a, int member,
                       int *row, int *scalar, double *coefficient) {
    const Col *c = &t->cols[member];
    if (c->nz != 1 || c->val[0] != -1.0) return 0;
    int r = c->sub[0];
    if (a[r].count != 2 || t->bkc[r] != PRIMAL_BK_FX || t->blc[r] != 0.0)
        return 0;
    *row = r;
    if (a[r].first == member) {
        *scalar = a[r].last;
        *coefficient = a[r].last_value;
    } else {
        *scalar = a[r].first;
        *coefficient = a[r].first_value;
    }
    return *scalar != member && *coefficient > 0.0;
}

/* All admissible loss columns and allocations have one further incidence. */
static int power_other(const Col *c, int row, double coefficient) {
    if (c->nz != 2) return -1;
    int k = c->sub[0] == row ? 0 : 1;
    if (c->sub[k] != row || c->val[1-k] != coefficient) return -1;
    return c->sub[1-k];
}

/* Find lambda >= ||gamma*r||_p with lambda=sum allocations and
 * r_i >= b_i - a_i'x - eta. The latter rows share a free costly offset. */
static int power_risk_pattern(PRIMALtask_t t, PowerRisk *risk) {
    int n = t->numcones;
    for (int i = 0; i < n; i++) {
        if (t->cone_type[i] != PRIMAL_CT_PPOW || t->cone_param[i] != risk->alpha ||
            t->cone_nmem[i] != 3) return 0;
        for (int k = 0; k < 3; k++) {
            int j = t->cone_mem[i][k];
            if (risk->column_role[j] || t->bkx[j] != PRIMAL_BK_FR || t->c[j] != 0.0) return 0;
            risk->column_role[j] = POWER_MEMBER;
        }
    }
    PowerRow *a = calloc((size_t)t->numcon, sizeof(*a));
    if (!a) return 0;
    int ok = 0, epigraph = -1;
    double min_gamma = INFINITY;
    for (int j = 0; j < t->numvar; j++) {
        if (!isfinite(t->c[j])) goto done;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            if (!isfinite(c->val[k]) || c->val[k] == 0.0) goto done;
            PowerRow *r = &a[c->sub[k]];
            if (r->count++ == 0) {
                r->first = j;
                r->first_value = c->val[k];
            }
            r->last = j;
            r->last_value = c->val[k];
        }
    }
    risk->balance_row = -1;
    for (int i = 0; i < n; i++) {
        PowerLoss *loss = &risk->loss[i];
        const int *mem = t->cone_mem[i];
        int l, u, r;
        double lv, uv;
        if (!power_copy(t, a, mem[0], &loss->head_row, &l, &lv) || lv != 1.0 ||
            !power_copy(t, a, mem[1], &loss->allocation_row, &u, &uv) || uv != 1.0 ||
            !power_copy(t, a, mem[2], &loss->tail_row, &r, &loss->gamma))
            goto done;
        if (i == 0) {
            if (risk->column_role[l] || t->bkx[l] != PRIMAL_BK_FR ||
                !(t->c[l] > 0.0) || t->cols[l].nz != n+1) goto done;
            epigraph = l;
            risk->cost = t->c[l];
            risk->column_role[l] = POWER_SCALAR;
        } else if (l != epigraph) goto done;

        if (u == r || risk->column_role[u] || risk->column_role[r] || t->c[u] != 0.0 ||
            t->c[r] != 0.0 || t->bkx[u] != PRIMAL_BK_FR || t->bkx[r] != PRIMAL_BK_LO ||
            t->blx[r] != 0.0) goto done;
        int balance = power_other(&t->cols[u], loss->allocation_row, -1.0);
        if (balance < 0 || t->bkc[balance] != PRIMAL_BK_FX || t->blc[balance] != 0.0 ||
            (risk->balance_row >= 0 && balance != risk->balance_row)) goto done;
        risk->balance_row = balance;
        int loss_row = power_other(&t->cols[r], loss->tail_row, 1.0);
        if (loss_row < 0 || t->bkc[loss_row] != PRIMAL_BK_LO ||
            risk->row_role[loss_row]) goto done;
        loss->loss_row = loss_row;
        risk->column_role[u] = risk->column_role[r] = POWER_SCALAR;
        risk->row_role[loss->head_row] = risk->row_role[loss->allocation_row] = POWER_LINK;
        risk->row_role[loss->tail_row] = POWER_LINK;
        risk->row_role[loss_row] = POWER_LOSS;
        min_gamma = fmin(min_gamma, loss->gamma);
    }
    if (a[risk->balance_row].count != n+1) goto done;
    const Col *ep = &t->cols[epigraph];
    int positive = 0;
    for (int i = 0; i < ep->nz; i++)
        if (ep->sub[i] == risk->balance_row && ep->val[i] == 1.0) positive = 1;
    if (!positive) goto done;
    risk->row_role[risk->balance_row] = POWER_LINK;

    /* Increasing this coordinate changes only the loss rows. */
    for (int j = 0; j < t->numvar && !ok; j++) {
        const Col *c = &t->cols[j];
        if (risk->column_role[j] || t->bkx[j] != PRIMAL_BK_FR || c->nz != n ||
            !(t->c[j] > 0.0 && t->c[j] / risk->cost <= min_gamma))
            continue;
        ok = 1;
        for (int k = 0; k < c->nz; k++)
            if (risk->row_role[c->sub[k]] != POWER_LOSS || c->val[k] != 1.0) ok = 0;
    }
done:
    free(a);
    return ok;
}

/* Solve the dual of the LP at zero loss. This puts one row per remaining
 * scalar into the linear solver even when there are many loss rows. */
static int power_zero_lp(PRIMALtask_t t, const PowerRisk *risk, double *x, double *y) {
    int nr = t->numcon, nv = t->numvar, m = 0, n = 0, ok = 0;
    for (int j = 0; j < nv; j++) if (!risk->column_role[j]) n++;
    int *row = malloc((size_t)nr * sizeof(*row));
    int *index = malloc((size_t)nr * sizeof(*index));
    double *value = malloc((size_t)nr * sizeof(*value));
    PRIMALtask_t lp = NULL;
    if (!row || !index || !value) goto done;
    for (int i = 0; i < nr; i++) row[i] = risk->row_role[i] == POWER_LINK ? -1 : m++;
    if (PRIMAL_maketask(t->env, n, m, &lp) ||
        PRIMAL_appendcons(lp, n) || PRIMAL_appendvars(lp, m) ||
        PRIMAL_putobjsense(lp, PRIMAL_OPTIMIZE_MAXIMIZE)) goto done;
    param_copy(lp, t);
    lp->tol_pfeas = t->tol_co_pfeas;
    lp->tol_dfeas = t->tol_co_dfeas;
    lp->tol_gap = t->tol_co_gap;
    for (int i = 0; i < nr; i++) {
        if (row[i] < 0) continue;
        PRIMALboundkeye bound;
        switch (t->bkc[i]) {
        case PRIMAL_BK_FX: bound = PRIMAL_BK_FR; break;
        case PRIMAL_BK_LO: bound = PRIMAL_BK_UP; break;
        case PRIMAL_BK_UP: bound = PRIMAL_BK_LO; break;
        default: goto done;
        }
        double b = t->bkc[i] == PRIMAL_BK_UP ? t->buc[i] : t->blc[i];
        if (!isfinite(b) || PRIMAL_putvarbound(lp, row[i], bound, 0.0, 0.0) ||
            PRIMAL_putcj(lp, row[i], -b)) goto done;
    }
    for (int j = 0, k = 0; j < nv; j++) {
        if (risk->column_role[j]) continue;
        if (t->bkx[j] != PRIMAL_BK_FR && (t->bkx[j] != PRIMAL_BK_LO || t->blx[j] != 0.0))
            goto done;
        const Col *c = &t->cols[j];
        if (c->nz > nr) goto done;
        for (int e = 0; e < c->nz; e++) {
            index[e] = row[c->sub[e]];
            value[e] = c->val[e];
            if (index[e] < 0) goto done;
        }
        PRIMALboundkeye bound = t->bkx[j] == PRIMAL_BK_FR ? PRIMAL_BK_FX : PRIMAL_BK_LO;
        if (PRIMAL_putconbound(lp, k, bound, -t->c[j], -t->c[j]) ||
            PRIMAL_putarow(lp, k, c->nz, index, value))
            goto done;
        k++;
    }
    if (PRIMAL_optimize(lp) != PRIMAL_RES_OK || !lp->has_sol ||
        lp->solsta != PRIMAL_SOL_STA_OPTIMAL) goto done;
    for (int j = 0, k = 0; j < nv; j++)
        if (!risk->column_role[j]) x[j] = lp->y[k++];
    for (int i = 0; i < nr; i++)
        if (row[i] >= 0) y[i] = lp->x[row[i]];
    ok = 1;
done:
    PRIMAL_deletetask(&lp);
    free(row); free(index); free(value);
    return ok;
}

/* Pair the loss prices with supporting dual power cones at the origin.
 * The unused leading-coordinate mass completes epigraph stationarity. */
static int power_risk_duals(PRIMALtask_t t, const PowerRisk *risk, double *y) {
    double alpha = risk->alpha, cost = risk->cost;
    double remaining = alpha * cost, allocation_price = (1.0-alpha) * cost;
    y[risk->balance_row] = -allocation_price;
    for (int i = 0; i < t->numcones; i++) {
        const PowerLoss *p = &risk->loss[i];
        double q = -y[p->loss_row] / p->gamma;
        if (!(q >= 0.0) || !isfinite(q)) return 0;
        double leading = alpha * cost * pow(q / cost, 1.0 / alpha);
        remaining -= leading;
        y[p->head_row] = -leading;
        y[p->allocation_row] = -allocation_price;
        y[p->tail_row] = q;
    }
    if (!(remaining >= -0.1 * t->tol_co_dfeas)) return 0;
    y[risk->loss[0].head_row] -= fmax(0.0, remaining);
    return 1;
}

static int power_risk_publish(PRIMALtask_t t, const PowerRisk *risk,
                               const double *x, const double *y,
                               double *gradient, double *activity) {
    int nv = t->numvar, nr = t->numcon;
    double primal = t->cfix, dual = t->cfix;
    for (int j = 0; j < nv; j++) {
        if (!isfinite(x[j])) return 0;
        primal += t->c[j] * x[j];
        double g = t->c[j];
        const Col *c = &t->cols[j];
        for (int e = 0; e < c->nz; e++) {
            int i = c->sub[e];
            g += c->val[e] * y[i];
            activity[i] += c->val[e] * x[j];
        }
        if (!isfinite(g)) return 0;
        gradient[j] = g;
        if (risk->column_role[j] == POWER_MEMBER) continue;
        if (t->bkx[j] == PRIMAL_BK_FR) {
            if (fabs(g) > t->tol_co_dfeas) return 0;
        } else if (x[j] < -t->tol_co_pfeas || g < -t->tol_co_dfeas ||
                   fabs(x[j] * g) > t->tol_co_gap) return 0;
    }
    for (int i = 0; i < nr; i++) {
        if (!isfinite(y[i]) || !isfinite(activity[i])) return 0;
        double b = t->bkc[i] == PRIMAL_BK_UP ? t->buc[i] : t->blc[i];
        double r = activity[i] - b;
        switch (t->bkc[i]) {
        case PRIMAL_BK_LO: if (y[i] > 0.0) return 0; r = -r; break;
        case PRIMAL_BK_UP: if (y[i] < 0.0) return 0; break;
        default: r = fabs(r); break;
        }
        if (!(r <= t->tol_co_pfeas)) return 0;
        dual -= b * y[i];
    }
    for (int i = 0; i < t->numcones; i++) {
        const int *m = t->cone_mem[i];
        double s[3] = {gradient[m[0]], gradient[m[1]], gradient[m[2]]};
        if (s[0] < 0.0 || !(s[1] > 0.0) || (s[0] == 0.0 && s[2] != 0.0) ||
            !(cone_dual_signed_slack(PRIMAL_CT_PPOW, risk->alpha, s, 3) >= -t->tol_co_dfeas))
            return 0;
    }
    if (!isfinite(primal) || !isfinite(dual)) return 0;
    double scale = fmax(1.0, fmax(fabs(primal), fabs(dual)));
    double p = primal/scale, d = dual/scale;
    if (!(fabs(p-d) <= t->tol_co_gap * (1.0/scale+fabs(p)+fabs(d)))) return 0;
    memcpy(t->x, x, (size_t)nv * sizeof(*x));
    for (int j = 0; j < nv; j++)
        if (t->bkx[j] == PRIMAL_BK_LO) t->slx[j] = -fmax(0.0, gradient[j]);
    for (int i = 0; i < nr; i++) {
        t->y[i] = y[i];
        t->slc[i] = fmin(0.0, y[i]);
        t->suc[i] = fmax(0.0, y[i]);
    }
    t->pobj = primal;
    t->dobj = dual;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_OPTIMAL;
    return 1;
}

int power_zero_risk_solve(PRIMALtask_t t) {
    int n = t->numcones, nv = t->numvar, nr = t->numcon;
    if (n <= 0 || (size_t)nv < 5 * (size_t)n + 2 || (size_t)nr < 4 * (size_t)n + 1 ||
        t->numbarvar || t->has_warm || t->cbfn || t->progcb ||
        t->optimizer_max_time >= 0.0 || t->lower_obj_cut > -0.5 * DBL_MAX ||
        t->upper_obj_cut < 0.5 * DBL_MAX || iter_cap(t->max_iter_intpnt) <= 1 ||
        t->cone_type[0] != PRIMAL_CT_PPOW || getenv("GMB_NO_EXP_REDUCED") ||
        getenv("GMB_NO_EXP_ABSORB") || getenv("GMB_EXP_DA") ||
        getenv("GMB_SDP_DENSE") || getenv("GMB_SDP_SPARSE")) return 0;
    double alpha = t->cone_param[0];
    if (!(alpha > 0.0 && alpha < 1.0)) return 0;
    size_t vars = (size_t)nv, rows = (size_t)nr;
    PowerLoss *loss = malloc((size_t)n * sizeof(*loss));
    unsigned char *role = calloc(vars + rows, sizeof(*role));
    double *x = calloc(2 * (vars + rows), sizeof(*x));
    int solved = 0;
    if (loss && role && x) {
        double *y = x + vars, *gradient = y + rows, *activity = gradient + vars;
        PowerRisk risk = {-1, 0.0, alpha, loss, role, role + vars};
        solved = power_risk_pattern(t, &risk) && power_zero_lp(t, &risk, x, y) &&
                 power_risk_duals(t, &risk, y) && power_risk_publish(t, &risk, x, y, gradient, activity);
    }
    free(loss); free(role); free(x);
    if (solved) tlog(t, "optimal solution found (zero-loss power reduction)\n");
    return solved;
}
