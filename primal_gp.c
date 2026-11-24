/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"

enum {
    GP_DIM = 16, GP_TERMS = 48, GP_GROUPS = 16,
    GP_VARS = 4 * GP_TERMS + GP_DIM, GP_ROWS = 3 * GP_TERMS + GP_GROUPS,
    GP_STEPS = 20
};

typedef struct {
    int row;
    double upper;
} GpGroup;

typedef struct {
    int head, scale, arg, alias, group, link_row, scale_row, arg_row;
    double link, scale_coef, arg_coef, u, weight, bias, log_weight;
    double a[GP_DIM];
} GpTerm;

typedef struct {
    int dim, terms, groups;
    int free_var[GP_DIM];
    double c[GP_DIM];
    GpTerm term[GP_TERMS];
    GpGroup group[GP_GROUPS];
} GpModel;

typedef struct {
    double f[GP_GROUPS], jac[GP_GROUPS][GP_DIM], z[GP_TERMS];
    double pri[GP_GROUPS], dual[GP_DIM], gap, merit;
} GpEval;

/* Recognize positive sums of disjoint exponential epigraphs, each with
 * its own fixed scale and affine argument. A free alias may copy a head. */
static int gp_extract(PRIMALtask_t t, GpModel *g) {
    int nv = t->numvar, nr = t->numcon;
    int used[GP_VARS] = {0}, row_type[GP_ROWS] = {0};
    int row_count[GP_ROWS] = {0}, row_first[GP_ROWS], row_last[GP_ROWS];
    for (int i = 0; i < g->terms; i++) {
        if (t->cone_type[i] != PRIMAL_CT_PEXP || t->cone_nmem[i] != 3) return 0;
        const int *m = t->cone_mem[i];
        for (int k = 0; k < 3; k++) {
            if (used[m[k]] || t->c[m[k]] != 0.0) return 0;
            used[m[k]] = 1;
        }
    }
    for (int j = 0; j < nv; j++) {
        if (t->bkx[j] != PRIMAL_BK_FR || !isfinite(t->c[j])) return 0;
        const Col *col = &t->cols[j];
        for (int k = 0; k < col->nz; k++) {
            int r = col->sub[k];
            if (!isfinite(col->val[k]) || col->val[k] == 0.0) return 0;
            if (row_count[r]++ == 0) row_first[r] = j;
            row_last[r] = j;
        }
    }
    for (int i = 0; i < g->terms; i++) {
        GpTerm *p = &g->term[i];
        const int *m = t->cone_mem[i];
        p->head = m[0]; p->scale = m[1]; p->arg = m[2];
        const Col *h = &t->cols[p->head], *u = &t->cols[p->scale], *v = &t->cols[p->arg];
        if (h->nz != 1 || u->nz != 1 || v->nz != 1) return 0;
        int r = h->sub[0], ur = u->sub[0], vr = v->sub[0];
        if (ur == vr || ur == r || vr == r || row_type[ur] || row_type[vr] ||
            row_count[ur] != 1 || t->bkc[ur] != PRIMAL_BK_FX || t->bkc[vr] != PRIMAL_BK_FX)
            return 0;
        p->scale_row = ur; p->arg_row = vr;
        p->scale_coef = u->val[0]; p->arg_coef = v->val[0];
        p->u = t->blc[ur] / p->scale_coef;
        p->bias = (t->blc[vr] / p->arg_coef) / p->u;
        if (!(p->u > 0.0) || !isfinite(p->u) || !isfinite(p->bias)) return 0;
        row_type[ur] = -1;
        row_type[vr] = -i - 2;

        p->alias = p->link_row = -1;
        double weight = h->val[0];
        if (t->bkc[r] == PRIMAL_BK_FX) {
            if (row_type[r] || row_count[r] != 2 || t->blc[r] != 0.0) return 0;
            int j = row_first[r] == p->head ? row_last[r] : row_first[r];
            const Col *a = &t->cols[j];
            if (used[j] || t->c[j] != 0.0 || a->nz != 2) return 0;
            int k = a->sub[0] == r ? 0 : 1;
            if (a->sub[k] != r || a->val[k] != -h->val[0]) return 0;
            used[j] = 1;
            row_type[r] = -1;
            p->alias = j; p->link_row = r; p->link = h->val[0];
            r = a->sub[1-k];
            weight = a->val[1-k];
        }
        if (t->bkc[r] != PRIMAL_BK_UP || !(weight > 0.0) ||
            !isfinite(t->buc[r]) || !(t->buc[r] > 0.0) || row_type[r] < 0) return 0;
        if (row_type[r] == 0) {
            if (g->groups == GP_GROUPS) return 0;
            g->group[g->groups++] = (GpGroup){r, t->buc[r]};
            row_type[r] = g->groups;
        }
        p->group = row_type[r] - 1;
        p->weight = weight;
        p->log_weight = log(weight) + log(p->u) - log(t->buc[r]);
    }
    for (int j = 0; j < nv; j++) {
        if (used[j]) continue;
        const Col *col = &t->cols[j];
        if (g->dim == GP_DIM || col->nz == 0) return 0;
        int d = g->dim++;
        g->free_var[d] = j;
        g->c[d] = t->c[j];
        for (int k = 0; k < col->nz; k++) {
            int r = col->sub[k], role = row_type[r];
            if (role > -2) return 0;
            GpTerm *p = &g->term[-role-2];
            p->a[d] -= (col->val[k] / p->arg_coef) / p->u;
            if (!isfinite(p->a[d])) return 0;
        }
    }
    for (int r = 0; r < nr; r++)
        if (!row_type[r]) return 0;
    return g->dim > 0 && g->groups > 0;
}

/* The Jacobian rows are weighted means of the affine slopes. Their
 * Hessians are covariances. Only the weighted sum is needed in Newton. */
static int gp_eval(const GpModel *g, const double *x, const double *slack,
                   const double *lambda, GpEval *e, DMat *matrix) {
    double peak[GP_GROUPS], sum[GP_GROUPS] = {0}, weight[GP_TERMS];
    for (int k = 0; k < g->groups; k++) peak[k] = -INFINITY;
    for (int i = 0; i < g->terms; i++) {
        const GpTerm *p = &g->term[i];
        double z = p->bias;
        for (int j = 0; j < g->dim; j++) z += p->a[j] * x[j];
        e->z[i] = z;
        weight[i] = z + p->log_weight;
        if (!isfinite(weight[i])) return 0;
        peak[p->group] = fmax(peak[p->group], weight[i]);
    }
    memset(e->jac, 0, sizeof(e->jac));
    for (int i = 0; i < g->terms; i++) {
        const GpTerm *p = &g->term[i];
        int k = p->group;
        weight[i] = exp(weight[i] - peak[k]);
        sum[k] += weight[i];
        for (int j = 0; j < g->dim; j++) e->jac[k][j] += weight[i] * p->a[j];
    }
    double obj = 0.0, pn = 0.0, dn = 0.0;
    e->gap = 0.0;
    for (int j = 0; j < g->dim; j++) {
        e->dual[j] = g->c[j];
        obj += g->c[j] * x[j];
    }
    for (int k = 0; k < g->groups; k++) {
        double q = 1.0 / sum[k];
        e->f[k] = peak[k] + log(sum[k]);
        e->pri[k] = e->f[k] + slack[k];
        pn = fmax(pn, fabs(e->pri[k]));
        e->gap += slack[k] * lambda[k];
        for (int j = 0; j < g->dim; j++) {
            e->jac[k][j] *= q;
            e->dual[j] += lambda[k] * e->jac[k][j];
        }
    }
    for (int j = 0; j < g->dim; j++) {
        if (!isfinite(e->dual[j])) return 0;
        dn = fmax(dn, fabs(e->dual[j]));
    }
    e->merit = pn + dn + (0.5 * e->gap) / (0.5 + fabs(obj));
    if (!isfinite(obj) || !isfinite(e->gap) || !isfinite(e->merit)) return 0;
    if (!matrix) return 1;
    for (int j = 0; j < g->dim; j++)
        memset(matrix->v[j], 0, (size_t)(j+1) * sizeof(double));
    for (int i = 0; i < g->terms; i++) {
        const GpTerm *p = &g->term[i];
        int k = p->group;
        double q = lambda[k] * weight[i] / sum[k];
        const double *a = e->jac[k];
        for (int j = 0; j < g->dim; j++)
            for (int l = 0; l <= j; l++)
                matrix->v[j][l] += q * (p->a[j] - a[j]) * (p->a[l] - a[l]);
    }
    for (int k = 0; k < g->groups; k++) {
        const double *a = e->jac[k];
        double q = lambda[k] / slack[k];
        for (int j = 0; j < g->dim; j++)
            for (int l = 0; l <= j; l++) matrix->v[j][l] += q * a[j] * a[l];
    }
    return 1;
}

static void gp_newton(const GpModel *g, const GpEval *e, const DMat *factor,
                      const double *slack, const double *lambda, const double *comp,
                      double *dx, double *ds, double *dl) {
    for (int j = 0; j < g->dim; j++) dx[j] = -e->dual[j];
    for (int k = 0; k < g->groups; k++) {
        double q = (comp[k] + lambda[k] * e->pri[k]) / slack[k];
        for (int j = 0; j < g->dim; j++) dx[j] -= q * e->jac[k][j];
    }
    dmat_chol_solve(factor, dx, g->dim);
    for (int k = 0; k < g->groups; k++) {
        double step = -e->pri[k];
        for (int j = 0; j < g->dim; j++) step -= e->jac[k][j] * dx[j];
        ds[k] = step;
        dl[k] = (comp[k] - lambda[k] * step) / slack[k];
    }
}

static double gp_step(int n, const double *x, const double *dx) {
    double a = 1.0;
    for (int k = 0; k < n; k++)
        if (dx[k] < 0.0) a = fmin(a, -x[k] / dx[k]);
    return a;
}

/* Recover dual components of a supporting plane at each lifted head. */
static int gp_publish(PRIMALtask_t t, const GpModel *g,
                      const double *point, const double *lambda, const GpEval *e) {
    double x[GP_VARS] = {0}, y[GP_ROWS] = {0}, activity[GP_ROWS] = {0};
    for (int j = 0; j < g->dim; j++) x[g->free_var[j]] = point[j];
    for (int k = 0; k < g->groups; k++)
        y[g->group[k].row] = (lambda[k] / g->group[k].upper) * exp(-e->f[k]);
    for (int i = 0; i < g->terms; i++) {
        const GpTerm *p = &g->term[i];
        double z = e->z[i], ez = exp(z);
        x[p->scale] = p->u;
        x[p->arg] = p->u * z;
        x[p->head] = p->u * ez;
        if (!(x[p->head] > 0.0)) return 0;
        double yg = y[g->group[p->group].row];
        double d0 = yg * p->weight, d2 = -d0 * ez;
        double d1 = -d2 * (z - 1.0);
        d1 += 4.0 * DBL_EPSILON * (fabs(d1) + fabs(d2));
        if (p->alias >= 0) {
            x[p->alias] = x[p->head];
            y[p->link_row] = yg * p->weight / p->link;
        }
        y[p->scale_row] = d1 / p->scale_coef;
        y[p->arg_row] = d2 / p->arg_coef;
    }
    double primal = t->cfix, dual = t->cfix, gradient[GP_VARS];
    for (int j = 0; j < t->numvar; j++) {
        if (!isfinite(x[j])) return 0;
        primal += t->c[j] * x[j];
        gradient[j] = t->c[j];
        const Col *col = &t->cols[j];
        for (int k = 0; k < col->nz; k++) {
            int r = col->sub[k];
            activity[r] += col->val[k] * x[j];
            gradient[j] += col->val[k] * y[r];
        }
        if (!isfinite(gradient[j])) return 0;
    }
    for (int j = 0; j < g->dim; j++)
        if (!(fabs(gradient[g->free_var[j]]) <= t->tol_co_dfeas)) return 0;
    for (int i = 0; i < g->terms; i++) {
        const GpTerm *p = &g->term[i];
        if (p->alias >= 0 && !(fabs(gradient[p->alias]) <= t->tol_co_dfeas)) return 0;
        double u[3] = {x[p->head], x[p->scale], x[p->arg]};
        double d[3] = {gradient[p->head], gradient[p->scale], gradient[p->arg]};
        if (!(d[0] > 0.0 && d[2] < 0.0) ||
            !(cone_signed_slack(PRIMAL_CT_PEXP, 0.0, u, 3) >= -t->tol_co_pfeas) ||
            !(cone_dual_signed_slack(PRIMAL_CT_PEXP, 0.0, d, 3) >= -t->tol_co_dfeas))
            return 0;
    }
    for (int r = 0; r < t->numcon; r++) {
        if (!isfinite(activity[r]) || !isfinite(y[r])) return 0;
        double b = t->bkc[r] == PRIMAL_BK_UP ? t->buc[r] : t->blc[r];
        double violation = activity[r] - b;
        if (t->bkc[r] == PRIMAL_BK_FX) violation = fabs(violation);
        else if (y[r] < 0.0) return 0;
        if (!(violation <= t->tol_co_pfeas)) return 0;
        dual -= b * y[r];
    }
    if (!isfinite(primal) || !isfinite(dual)) return 0;
    double scale = fmax(1.0, fmax(fabs(primal), fabs(dual)));
    double p = primal / scale, d = dual / scale;
    if (!(fabs(p - d) <= t->tol_co_gap * (1.0 / scale + fabs(p) + fabs(d)))) return 0;

    memcpy(t->x, x, (size_t)t->numvar * sizeof(*x));
    for (int i = 0; i < t->numcon; i++) {
        t->y[i] = y[i];
        t->slc[i] = fmin(0.0, y[i]);
        t->suc[i] = fmax(0.0, y[i]);
    }
    t->pobj = primal;
    t->dobj = dual;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_OPTIMAL;
    tlog(t, "optimal solution found (reduced log-sum-exp constraints)\n");
    return 1;
}

int exp_sum_constraints_solve(PRIMALtask_t t) {
    int n = t->numcones, limit = iter_cap(t->max_iter_intpnt);
    if (n <= 0 || n > GP_TERMS || t->numvar > GP_VARS || t->numcon > GP_ROWS ||
        t->numcon <= 2 * n || t->numvar <= 3 * n || t->numbarvar ||
        t->has_warm || t->cbfn || t->progcb || limit <= 1 ||
        t->optimizer_max_time >= 0.0 || t->lower_obj_cut > -0.5 * DBL_MAX ||
        t->upper_obj_cut < 0.5 * DBL_MAX || getenv("GMB_NO_EXP_REDUCED") ||
        getenv("GMB_NO_EXP_ABSORB") || getenv("GMB_EXP_DA") || getenv("GMB_NO_EXP_HIGHER") ||
        getenv("GMB_SDP_DENSE") || getenv("GMB_SDP_SPARSE"))
        return 0;
    GpModel g = {0};
    g.terms = n;
    if (!gp_extract(t, &g)) return 0;

    double storage[GP_DIM][GP_DIM], *rows[GP_DIM];
    for (int j = 0; j < g.dim; j++) rows[j] = storage[j];
    DMat matrix = {g.dim, g.dim, rows};
    double x[GP_DIM] = {0}, slack[GP_GROUPS] = {0}, lambda[GP_GROUPS];
    for (int k = 0; k < g.groups; k++) lambda[k] = 1.0;
    GpEval e, trial;
    if (!gp_eval(&g, x, slack, lambda, &e, NULL)) return 0;
    for (int k = 0; k < g.groups; k++) slack[k] = fmax(1.0, -e.f[k]);
    double threshold = 0.25 * fmin(t->tol_co_gap, fmin(t->tol_co_pfeas, t->tol_co_dfeas));
    if (limit > GP_STEPS) limit = GP_STEPS;
    for (int it = 0; it < limit; it++) {
        if (!gp_eval(&g, x, slack, lambda, &e, &matrix)) return 0;
        if (e.merit <= threshold && gp_publish(t, &g, x, lambda, &e)) return 1;
        if (!(e.gap > 0.0) || dmat_cholesky(&matrix, g.dim)) return 0;

        double dx[GP_DIM], ds[GP_GROUPS], dl[GP_GROUPS], comp[GP_GROUPS];
        for (int k = 0; k < g.groups; k++) comp[k] = -slack[k] * lambda[k];
        gp_newton(&g, &e, &matrix, slack, lambda, comp, dx, ds, dl);
        double ap = gp_step(g.groups, slack, ds), ad = gp_step(g.groups, lambda, dl);
        double affine_gap = 0.0;
        for (int k = 0; k < g.groups; k++)
            affine_gap += (slack[k] + ap * ds[k]) * (lambda[k] + ad * dl[k]);
        double ratio = affine_gap / e.gap;
        if (!isfinite(ratio)) return 0;
        ratio = fmin(1.0, fmax(0.0, ratio));
        double center = (ratio * ratio * ratio) * (e.gap / g.groups);
        for (int k = 0; k < g.groups; k++) comp[k] += center - ds[k] * dl[k];
        gp_newton(&g, &e, &matrix, slack, lambda, comp, dx, ds, dl);

        double step = 0.995 * fmin(gp_step(g.groups, slack, ds), gp_step(g.groups, lambda, dl));
        double tx[GP_DIM], ts[GP_GROUPS], tl[GP_GROUPS];
        int accepted = 0;
        for (int bt = 0; bt < 20; bt++, step *= 0.5) {
            int positive = 1;
            for (int j = 0; j < g.dim; j++) tx[j] = x[j] + step * dx[j];
            for (int k = 0; k < g.groups; k++) {
                ts[k] = slack[k] + step * ds[k];
                tl[k] = lambda[k] + step * dl[k];
                if (!(ts[k] > 0.0 && tl[k] > 0.0)) positive = 0;
            }
            if (positive && gp_eval(&g, tx, ts, tl, &trial, NULL) &&
                trial.merit <= e.merit * (1.0 - 0.0001 * step)) {
                accepted = 1;
                break;
            }
        }
        if (!accepted) return 0;
        memcpy(x, tx, (size_t)g.dim * sizeof(*x));
        memcpy(slack, ts, (size_t)g.groups * sizeof(*slack));
        memcpy(lambda, tl, (size_t)g.groups * sizeof(*lambda));
    }
    return 0;
}
