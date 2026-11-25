/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"
#include "linalg.h"
#include <float.h>

enum {
    DESIGN_DIM = 6, DESIGN_INTS = 12, DESIGN_ROWS = 384, DESIGN_VARS = 192,
    DESIGN_COLS = DESIGN_VARS + DESIGN_DIM*(DESIGN_DIM+1)/2 + DESIGN_DIM*(2*DESIGN_DIM+1),
    DESIGN_SMALL_NODES = 180000, DESIGN_LARGE_NODES = 50000000
};

typedef struct {
    int d, cols, count, budget, objective, bar[2], offset[2];
    int root[DESIGN_COLS], integer[DESIGN_COLS], var[DESIGN_INTS], upper[DESIGN_INTS];
    int diag[DESIGN_DIM], logdiag[DESIGN_DIM];
    double fixed[DESIGN_COLS], lo[DESIGN_ROWS], up[DESIGN_ROWS];
    double matrix[DESIGN_INTS+1][DESIGN_DIM*DESIGN_DIM];
    double *raw, *rows;
} Design;

typedef struct {
    PRIMALtask_t task;
    const Design *model;
    int nodes, max_nodes, saturated, tail[DESIGN_INTS+1];
    int row_of[1 << DESIGN_DIM], col_of[1 << DESIGN_DIM];
    int x[DESIGN_INTS], best_x[DESIGN_INTS];
    double shift, best_det;
} DesignSearch;

static int design_alive(PRIMALtask_t t) {
    return t->mip_deadline < 0.0 || (double)clock() < t->mip_deadline;
}

static int design_root(Design *p, int j) {
    while (p->root[j] != j) {
        p->root[j] = p->root[p->root[j]];
        j = p->root[j];
    }
    return j;
}

static int design_bar_col(const Design *p, int which, int i, int j) {
    /* Local packed order is independent of the public packed bar layout. */
    int major = i > j ? i : j, minor = i > j ? j : i;
    return p->offset[which] + major*(major+1)/2 + minor;
}

static int design_assign(double *value, int j, double x) {
    if (!isfinite(x) || (!isnan(value[j]) && value[j] != x)) return 0;
    value[j] = x;
    return 1;
}

static int design_data(PRIMALtask_t t, Design *p) {
    int nv = t->numvar;
    for (int j = 0; j < p->cols; j++) {
        p->root[j] = j;
        p->fixed[j] = NAN;
        p->integer[j] = -1;
    }
    for (int j = 0; j < nv; j++) {
        if (t->vartype[j] != PRIMAL_VAR_TYPE_CONT) {
            if ((t->vartype[j] != PRIMAL_VAR_TYPE_INT && t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) ||
                p->count == DESIGN_INTS) return 0;
            p->integer[j] = p->count;
            p->var[p->count++] = j;
        }
        for (int e = 0; e < t->cols[j].nz; e++) {
            double *entry = &p->raw[(size_t)t->cols[j].sub[e]*p->cols + j];
            double value = t->cols[j].val[e];
            if (*entry != 0.0 || !isfinite(value)) return 0;
            *entry = value;
        }
    }
    for (int k = 0; k < t->nbarA; k++) {
        int sym = t->barA_sym[k], which = t->barA_bar[k] == p->bar[0] ? 0 : 1;
        double *row = p->raw + (size_t)t->barA_con[k]*p->cols;
        double weight = t->barA_coef[k];
        if (fabs(weight) != 1.0 && fabs(weight) != 0.5) return 0;
        for (int e = 0; e < t->sym_nnz[sym]; e++) {
            int i = t->sym_subi[sym][e], j = t->sym_subj[sym][e];
            if (i >= (which+1)*p->d || j >= (which+1)*p->d) return 0;
            double a = (i == j ? weight : 2.0*weight)*t->sym_val[sym][e];
            int col = design_bar_col(p, which, i, j);
            if (row[col] != 0.0 || fabs(a) != 1.0) return 0;
            row[col] = a;
        }
    }
    return p->count > 0;
}

/* Contract exact zero-RHS equalities x_i = x_j. Keep the original rows
 * for validation and discard ambiguous cancellations from the pattern scan. */
static int design_contract(PRIMALtask_t t, Design *p) {
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->raw + (size_t)r*p->cols;
        bound_range(t->bkc[r], t->blc[r], t->buc[r], &p->lo[r], &p->up[r]);
        if (!(p->lo[r] <= p->up[r])) return 0;
        if (p->lo[r] != 0.0 || p->up[r] != 0.0) continue;
        int first = -1, last = -1, j = 0;
        for (; j < p->cols; j++) if (row[j] != 0.0) {
            if (last >= 0 || p->integer[j] >= 0 || fabs(row[j]) != 1.0) break;
            if (first < 0) first = j;
            else last = j;
        }
        if (j == p->cols && last >= 0 && row[first] == -row[last])
            p->root[design_root(p,last)] = design_root(p,first);
    }
    for (int j = 0; j < p->cols; j++) design_root(p,j);
    for (int j = 0; j < t->numvar; j++)
        if (t->bkx[j] == PRIMAL_BK_FX && p->integer[j] < 0 &&
            !design_assign(p->fixed,p->root[j],t->blx[j])) return 0;
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->raw + (size_t)r*p->cols;
        double *merged = p->rows + (size_t)r*p->cols;
        int j = 0;
        for (; j < p->cols; j++) if (row[j] != 0.0) {
            int g = p->root[j];
            if (merged[g] != 0.0 && merged[g] != -row[j]) break;
            merged[g] += row[j];
        }
        if (j < p->cols) {
            memset(merged,0,(size_t)p->cols*sizeof(*merged));
            continue;
        }
        if (p->lo[r] != p->up[r] || !isfinite(p->lo[r])) continue;
        int only = -1;
        for (j = 0; j < p->cols; j++) if (merged[j] != 0.0) {
            if (only >= 0) break;
            only = j;
        }
        if (j == p->cols && only >= 0 && p->integer[only] < 0 && fabs(merged[only]) == 1.0 &&
            !design_assign(p->fixed,only,p->lo[r]*merged[only])) return 0;
    }
    return 1;
}

static int design_box(PRIMALtask_t t, Design *p) {
    for (int j = 0; j < p->count; j++) {
        int col = p->var[j];
        double lb, ub;
        bound_range(t->bkx[col],t->blx[col],t->bux[col],&lb,&ub);
        if (t->vartype[col] == PRIMAL_VAR_TYPE_INT_BIN) {
            lb = fmax(0.0,lb);
            ub = fmin(1.0,ub);
        }
        for (int r = 0; r < t->numcon; r++) {
            const double *row = p->rows + (size_t)r*p->cols;
            double sign = row[col];
            if (fabs(sign) != 1.0) continue;
            int k = 0;
            for (; k < p->cols; k++) if (k != col && row[k] != 0.0) break;
            if (k < p->cols) continue;
            lb = fmax(lb,sign > 0 ? p->lo[r] : -p->up[r]);
            ub = fmin(ub,sign > 0 ? p->up[r] : -p->lo[r]);
        }
        if (!(ceil(lb) == 0.0 && floor(ub) >= 0.0 && floor(ub) <= 16.0)) return 0;
        p->upper[j] = (int)floor(ub);
        p->budget += p->upper[j];
    }
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->rows + (size_t)r*p->cols;
        double sign = row[p->var[0]];
        if (fabs(sign) != 1.0) continue;
        int k = 0;
        for (; k < p->cols; k++)
            if (row[k] != (p->integer[k] >= 0 ? sign : 0.0)) break;
        if (k == p->cols) {
            double bound = floor(sign > 0 ? p->up[r] : -p->lo[r]);
            if (bound < 0) return 0;
            if (bound < p->budget) p->budget = (int)bound;
        }
    }
    return 1;
}

static int design_objective(PRIMALtask_t t, Design *p) {
    int col = -1;
    for (int j = 0; j < t->numvar; j++) if (t->c[j] != 0.0) {
        if (col >= 0 || t->c[j] != -1.0 || p->integer[j] >= 0) return 0;
        col = p->root[j];
    }
    if (col < 0) return 0;
    int wanted[DESIGN_COLS] = {0};
    wanted[col] = 1;
    for (int i = 0; i < p->d; i++) {
        int j = p->logdiag[i];
        if (wanted[j] != 0 || j == p->diag[i]) return 0;
        wanted[j] = -1;
    }
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->rows + (size_t)r*p->cols;
        if (p->lo[r] != 0.0 || p->up[r] != 0.0 || fabs(row[col]) != 1.0) continue;
        int j = 0;
        for (; j < p->cols; j++) if (row[j] != row[col]*wanted[j]) break;
        if (j == p->cols) { p->objective = col; return 1; }
    }
    return 0;
}

static int design_affine(PRIMALtask_t t, Design *p, int target, int i, int j) {
    int d = p->d;
    if (p->integer[target] >= 0) return 0;
    if (!isnan(p->fixed[target])) {
        double value = p->fixed[target], magnitude = fabs(value);
        if (magnitude > 1e30 || (value != 0.0 && magnitude < 1e-30)) return 0;
        p->matrix[0][i*d+j] = p->matrix[0][j*d+i] = value;
        return 1;
    }
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->rows + (size_t)r*p->cols;
        double rhs = p->lo[r], magnitude = fabs(rhs);
        if (rhs != p->up[r] || magnitude > 1e30 ||
            (rhs != 0.0 && magnitude < 1e-30) || fabs(row[target]) != 1.0)
            continue;
        int k = 0;
        for (; k < p->cols; k++) {
            if (k != target && row[k] != 0.0 &&
                (p->integer[k] < 0 || fabs(row[k]) < 1e-30 || fabs(row[k]) > 1e30)) break;
        }
        if (k < p->cols) continue;
        p->matrix[0][i*d+j] = p->matrix[0][j*d+i] = rhs*row[target];
        for (k = 0; k < p->count; k++)
            p->matrix[k+1][i*d+j] = p->matrix[k+1][j*d+i] = -row[p->var[k]]*row[target];
        return 1;
    }
    return 0;
}

static int design_pattern(PRIMALtask_t t, Design *p) {
    int d = p->d;
    for (int i = 0; i < d; i++) {
        p->diag[i] = p->root[design_bar_col(p,1,i,i)];
        if (p->root[design_bar_col(p,1,i,d+i)] != p->diag[i]) return 0;
        int found = 0;
        for (int k = 0; k < d; k++) {
            const int *member = t->cone_mem[k];
            if (p->root[member[0]] != p->diag[i]) continue;
            if (found || p->fixed[p->root[member[1]]] != 1.0) return 0;
            p->logdiag[i] = p->root[member[2]];
            found = 1;
        }
        if (!found) return 0;
        for (int j = 0; j < i; j++) {
            int off = p->root[design_bar_col(p,1,i,j)];
            int cross = p->root[design_bar_col(p,1,i,d+j)];
            if (p->fixed[off] != 0.0 || p->fixed[cross] != 0.0) return 0;
        }
        for (int j = 0; j <= i; j++) {
            int target = p->root[design_bar_col(p,1,d+i,d+j)];
            if (p->root[design_bar_col(p,0,i,j)] != target ||
                !design_affine(t,p,target,i,j)) return 0;
        }
    }
    return design_objective(t,p);
}

/* Each coefficient A is within eta I of a positive semidefinite vv'.
 * The row sums of A-vv' bound its spectral norm. fma evaluates each
 * entry in one rounding, and the inflation covers the norm reduction. */
static double design_psd_shift(const Design *p) {
    int d = p->d;
    double eta = 0.0, scale = 0.0;
    for (int k = 1; k <= p->count; k++) {
        const double *a = p->matrix[k];
        int pivot = 0;
        for (int i = 1; i < d; i++) if (a[i*d+i] > a[pivot*d+pivot]) pivot = i;
        double v[DESIGN_DIM] = {0};
        if (a[pivot*d+pivot] > 0.0) {
            double head = sqrt(a[pivot*d+pivot]);
            for (int i = 0; i < d; i++) v[i] = a[i*d+pivot]/head;
        }
        for (int i = 0; i < d; i++) {
            double norm = 0.0;
            for (int j = 0; j < d; j++) {
                scale = fmax(scale,fabs(a[i*d+j]));
                norm += fabs(fma(-v[i],v[j],a[i*d+j]));
            }
            eta = fmax(eta,norm);
        }
    }
    double shift = nextafter(p->budget*(eta*(1.0+64.0*DBL_EPSILON)+DBL_MIN),INFINITY);
    return shift <= 1e-11*fmax(1.0,scale) ? shift : -1.0;
}

/* The absolute Leibniz expansion bounds the sum of magnitudes even
 * when the signed sum cancels. */
static double design_det_small(const double *a, int d, int i, unsigned todo, int absolute) {
    if (!todo) return 1.0;
    double value = 0.0;
    int sign = 1;
    for (int j = 0; j < d; j++) if (todo & (1u << j)) {
        double term = a[i*d+j] * design_det_small(a,d,i+1,todo ^ (1u << j),absolute);
        value += absolute ? term : sign*term;
        sign = -sign;
    }
    return value;
}

static double design_det(const DesignSearch *s, const double *a, int absolute) {
    int d = s->model->d;
    unsigned all = (1u << d)-1u;
    if (d <= 4) return design_det_small(a,d,0,all,absolute);
    /* The minor for a subset of columns uses the same number of final
     * rows. Subsets share the minors on the layer immediately below. */
    double minor[1 << DESIGN_DIM];
    minor[0] = 1.0;
    for (unsigned cols = 1; cols <= all; cols++) {
        double sum = 0.0;
        int sign = 1;
        const double *row = a + s->row_of[cols]*d;
        for (unsigned remaining = cols; remaining;) {
            unsigned bit = remaining & (0u-remaining);
            sum += sign*row[s->col_of[bit]]*minor[cols ^ bit];
            if (!absolute) sign = -sign;
            remaining ^= bit;
        }
        minor[cols] = sum;
    }
    return minor[all];
}

static int design_enumerate(DesignSearch *s, int k, int slack, const double *matrix) {
    const Design *p = s->model;
    if (++s->nodes == s->max_nodes || ((s->nodes & 255) == 0 && !design_alive(s->task))) return 0;
    int size = p->d*p->d;
    if (k == p->count) {
        double det = design_det(s,matrix,0);
        if (det > s->best_det) {
            s->best_det = det;
            memcpy(s->best_x,s->x,(size_t)p->count*sizeof(*s->x));
        }
        return 1;
    }
    int max = slack < p->upper[k] ? slack : p->upper[k];
    int min = s->saturated && slack > s->tail[k+1] ? slack-s->tail[k+1] : 0;
    for (int w = min; w <= max; w++) {
        double next[DESIGN_DIM*DESIGN_DIM];
        for (int i = 0; i < size; i++) next[i] = matrix[i] + w*p->matrix[k+1][i];
        s->x[k] = w;
        if (!design_enumerate(s,k+1,slack-w,next)) return 0;
    }
    return 1;
}

static double design_error(const DesignSearch *s) {
    const Design *p = s->model;
    double magnitude[DESIGN_DIM*DESIGN_DIM];
    for (int i = 0; i < p->d*p->d; i++) {
        magnitude[i] = fabs(p->matrix[0][i]);
        int order[DESIGN_INTS];
        for (int j = 0; j < p->count; j++) {
            int k = j;
            while (k > 0 && fabs(p->matrix[order[k-1]+1][i]) < fabs(p->matrix[j+1][i])) {
                order[k] = order[k-1];
                k--;
            }
            order[k] = j;
        }
        /* Spend the whole budget on the largest coefficients first to
         * bound the absolute sum in this entry. */
        int remaining = p->budget;
        for (int j = 0; j < p->count && remaining > 0; j++) {
            int k = order[j];
            int take = remaining < p->upper[k] ? remaining : p->upper[k];
            magnitude[i] += take*fabs(p->matrix[k+1][i]);
            remaining -= take;
        }
    }
    for (int i = 0; i < p->d; i++) magnitude[i*p->d+i] += s->shift;
    /* A monomial traverses at most six entry sums of 14 operations,
     * six products and 1+...+6 additions. The nonnegative envelope
     * has the same bound. 512 eps covers both the forward error and
     * rounding in the envelope, including the final inflation. */
    double total = design_det(s,magnitude,1);
    return 512.0*DBL_EPSILON*total + 2048.0*DBL_MIN;
}

static int design_lift(const Design *p, const DesignSearch *s, double *x) {
    int d = p->d;
    double matrix[DESIGN_DIM*DESIGN_DIM], lower[DESIGN_DIM*DESIGN_DIM];
    double *row[DESIGN_DIM], diag[DESIGN_DIM];
    for (int i = 0; i < d*d; i++) {
        matrix[i] = p->matrix[0][i];
        for (int k = 0; k < p->count; k++) matrix[i] += s->best_x[k]*p->matrix[k+1][i];
        lower[i] = matrix[i];
    }
    for (int i = 0; i < d; i++) row[i] = lower + i*d;
    DMat factor = {d,d,row};
    if (dmat_cholesky(&factor,d)) return 0;
    double values[DESIGN_COLS];
    memcpy(values,p->fixed,(size_t)p->cols*sizeof(*values));
    double logdet = 0.0;
    for (int i = 0; i < d; i++) {
        diag[i] = row[i][i]*row[i][i];
        double logdiag = log(diag[i]);
        if (!design_assign(values,p->diag[i],diag[i]) ||
            !design_assign(values,p->logdiag[i],logdiag)) return 0;
        logdet += logdiag;
    }
    if (!design_assign(values,p->objective,logdet)) return 0;
    for (int i = 0; i < d; i++) {
        for (int j = i+1; j < d; j++) {
            int target = p->root[design_bar_col(p,1,i,d+j)];
            if (!design_assign(values,target,row[i][i]*row[j][i])) return 0;
        }
        for (int j = 0; j <= i; j++) {
            int target = p->root[design_bar_col(p,1,d+i,d+j)];
            if (!design_assign(values,target,matrix[i*d+j])) return 0;
        }
    }
    for (int j = 0; j < p->count; j++) values[p->var[j]] = s->best_x[j];
    for (int j = 0; j < p->cols; j++) {
        double v = values[p->root[j]];
        x[j] = isnan(v) ? 0.0 : v;
    }
    return 1;
}

static int design_measures(PRIMALtask_t t, const Design *p, const double *x,
                           double *bars) {
    double ftol = fmin(1e-8,fmin(t->mip_tol_feas,t->tol_co_pfeas));
    for (int j = 0; j < t->numvar; j++) {
        double lo, up;
        bound_range(t->bkx[j],t->blx[j],t->bux[j],&lo,&up);
        if (!(x[j] >= lo-ftol && x[j] <= up+ftol)) return 0;
    }
    for (int r = 0; r < t->numcon; r++) {
        const double *row = p->raw + (size_t)r*p->cols;
        double activity = 0.0;
        for (int j = 0; j < p->cols; j++) activity += row[j]*x[j];
        if (!isfinite(activity) || !(activity >= p->lo[r]-ftol && activity <= p->up[r]+ftol))
            return 0;
    }
    for (int k = 0; k < p->d; k++) {
        const int *mem = t->cone_mem[k];
        double v[] = {x[mem[0]],x[mem[1]],x[mem[2]]};
        if (!(cone_signed_slack(PRIMAL_CT_PEXP,0.0,v,3) >= -ftol)) return 0;
    }
    for (int k = 0; k < 2; k++) {
        int d = p->d*(k+1);
        double *matrix = bars + (k ? p->d*p->d : 0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) matrix[i*d+j] = x[design_bar_col(p,k,i,j)];
        if (!(bar_min_eig(d,matrix,NULL) >= -ftol)) return 0;
    }
    return 1;
}

/* A verified lift matching the determinant upper bound proves optimality.
 * Other dimensions, missing model identities, or an incomplete enumeration
 * use the ordinary MIP solver. */
int mip_logdet_solve(PRIMALtask_t t) {
    int d = t->numcones;
    if (d < 2 || d > DESIGN_DIM || t->numbarvar != 2 || t->nbarC ||
        t->numvar > DESIGN_VARS || t->numcon > DESIGN_ROWS ||
        t->has_qobj || t->has_qcon || t->numsos || t->cbfn || t->progcb ||
        t->has_warm || !design_alive(t)) return 0;
    Design p = {0};
    p.d = d;
    p.bar[0] = t->barDim[0] == d ? 0 : 1;
    p.bar[1] = 1-p.bar[0];
    if (t->barDim[p.bar[0]] != d || t->barDim[p.bar[1]] != 2*d) return 0;
    for (int k = 0; k < d; k++)
        if (t->cone_type[k] != PRIMAL_CT_PEXP || t->cone_nmem[k] != 3) return 0;
    p.offset[0] = t->numvar;
    p.offset[1] = p.offset[0] + d*(d+1)/2;
    p.cols = p.offset[1] + d*(2*d+1);
    size_t entries = (size_t)t->numcon*p.cols;
    p.raw = calloc(2*entries,sizeof(*p.raw));
    if (!p.raw) return 0;
    p.rows = p.raw + entries;
    int solved = 0;
    if (!design_data(t,&p) || !design_contract(t,&p) ||
        !design_box(t,&p) || !design_pattern(t,&p)) goto done;

    DesignSearch s = {0};
    s.task = t;
    s.model = &p;
    s.max_nodes = d <= 4 ? DESIGN_SMALL_NODES : DESIGN_LARGE_NODES;
    s.shift = design_psd_shift(&p);
    s.saturated = s.shift >= 0.0;
    if (!s.saturated) s.shift = 0.0;
    for (int k = p.count-1; k >= 0; k--) s.tail[k] = s.tail[k+1] + p.upper[k];
    for (int i = 0; i < d; i++) s.col_of[1u << i] = i;
    s.row_of[0] = d;
    for (unsigned mask = 1; mask < (1u << d); mask++)
        s.row_of[mask] = s.row_of[mask & (mask-1u)]-1;
    /* If A_k+eta I is PSD, every feasible M(w) is dominated by an
     * M(u)+eta*budget I with u>=w and sum(u)=budget. Only that enlarged
     * matrix enters the bound. The lift keeps the original coefficients. */
    double start[DESIGN_DIM*DESIGN_DIM];
    memcpy(start,p.matrix[0],(size_t)d*d*sizeof(*start));
    for (int i = 0; i < d; i++) start[i*d+i] += s.shift;
    if (!design_enumerate(&s,0,p.budget,start) || s.best_det <= 0.0) goto done;
    double x[DESIGN_COLS], bars[5*DESIGN_DIM*DESIGN_DIM];
    if (!design_lift(&p,&s,x) || !design_measures(t,&p,x,bars)) goto done;
    double upper = nextafter(s.best_det + design_error(&s),INFINITY);
    double log_gap = log(upper) - x[p.objective];
    double tolerance = fmin(1e-8,t->tol_co_gap);
    if (!(fabs(log_gap) <= tolerance) || !design_alive(t) ||
        !isfinite(t->cfix - x[p.objective])) goto done;
    memcpy(t->x,x,(size_t)t->numvar*sizeof(*t->x));
    memcpy(t->barx[p.bar[0]],bars,(size_t)d*d*sizeof(*bars));
    memcpy(t->barx[p.bar[1]],bars+d*d,(size_t)4*d*d*sizeof(*bars));
    t->pobj = t->dobj = t->cfix - x[p.objective];
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    tlog(t,"MIP: determinant bound certified for integer design\n");
    solved = 1;
done:
    free(p.raw);
    return solved;
}
