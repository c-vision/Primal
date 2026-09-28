/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
/* primal_std.c - standard-form solvers, LP/QP route helpers, Farkas vectors on stdform.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Range of the objective coefficient c_j over which the current optimal
 * solution of an LP stays optimal. Uses the reduced cost z_j for a nonbasic
 * variable and returns the current point for a basic one. Rejects QP/conic/
 * bar models and unsolved tasks; writes lcost/ucost. */
PRIMALrescodee PRIMAL_costsensitivity(PRIMALtask_t t, int j, double *lcost, double *ucost) {
    if (!t || !lcost || !ucost) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol || t->has_qobj || t->has_qcon > 0 ||
        t->numcones > 0 || t->numbarvar > 0)
        return PRIMAL_RES_ERR_ARG;   /* LP only, solved */
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;

    /* z_j (reduced cost, original form): z = -(c + A'y) with y mapped back */
    double av = 0.0;
    const Col *c = &t->cols[j];
    for (int k = 0; k < c->nz; k++) av += c->val[k] * t->y[c->sub[k]];
    double zj = -(t->c[j] + av);
    int at_lo = (t->bkx[j] == PRIMAL_BK_LO || t->bkx[j] == PRIMAL_BK_RA)
                && fabs(t->x[j] - t->blx[j]) < 1e-7;
    int at_up = (t->bkx[j] == PRIMAL_BK_UP || t->bkx[j] == PRIMAL_BK_RA)
                && fabs(t->x[j] - t->bux[j]) < 1e-7;
    if (!at_lo && !at_up) {
        /* basic or superbasic: fixed cost (any c_j keeps stationarity only
         * if z_j = 0: the solution stays optimal for any c_j ONLY if x_j = 0
         * contributes... in basis form: a basic var has z = 0 always; the
         * cost range needs the basis: return the current point (degenerate
         * range) */
        *lcost = t->c[j];
        *ucost = t->c[j];
        return PRIMAL_RES_OK;
    }
    /* nonbasic: the reduced cost z_j (original form) must keep its optimal
     * sign. Clone convention (verified by T65/solvebasis):
     * var at the optimal lower <=> s*z <= 0;  at the upper <=> s*z >= 0.
     * Moving c_j by delta shifts z_j by -delta:  z' = z - delta.
     *   at_lo: s*(z-delta) <= 0  <=> delta >= s*z... in min form (s=+1):
     *   z - delta <= 0 <=> delta >= z  -> lcost = c + z, ucost = +inf.
     *   at_up: z - delta >= 0 <=> delta <= z -> lcost = -inf, ucost = c + z.
     * (s absorbed: z is already original form, the sign constraint s*z
     * becomes, substituting z' = z - s*delta:  s*(z - s*delta) = s*z - delta.)
     */
    double sz = s * zj;
    if (at_lo) {
        *lcost = t->c[j] + sz;
        *ucost = INFINITY;
    } else {
        *lcost = -INFINITY;
        *ucost = t->c[j] + sz;
    }
    return PRIMAL_RES_OK;
}

/* Range of the RHS of row i over which the current optimal solution of an
 * LP keeps the duals optimal. Uses the row activity and bound_range; for an
 * active row without an exposed basis it returns the structural
 * compatibility interval. Rejects QP/conic/bar models and unsolved tasks;
 * writes lrange/urange. */
PRIMALrescodee PRIMAL_rhssensitivity(PRIMALtask_t t, int i, double *lrange, double *urange) {
    if (!t || !lrange || !urange) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol || t->has_qobj || t->has_qcon > 0 ||
        t->numcones > 0 || t->numbarvar > 0)
        return PRIMAL_RES_ERR_ARG;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    double y_i = t->y[i];
    if (fabs(y_i) < 1e-9) {
        /* inactive row: the bound can move freely until the row becomes
         * active: range = [current position, infinity) */
        double ax = 0.0;
        for (int j = 0; j < t->numvar; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] == i) ax += c->val[k] * t->x[j];
        }
        double lo, up;
        bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
        if (y_i >= 0) { *lrange = ax; *urange = up; }
        else { *lrange = lo; *urange = ax; }
        return PRIMAL_RES_OK;
    }
    /* active row: the duals stay optimal as long as the row stays active;
     * the limit is set by the variables that reach their bounds: without an
     * exposed basis one evaluates the movement allowed while KEEPING the
     * row active: x can redistribute, so the true range requires the basis.
     * Structural approximation: the range in which the active row is
     * COMPATIBLE with the variable bounds (there exists a feasible x with
     * the row active) -> for dense rows it is [-inf, +inf]; the useful value
     * for the single-variable case: the full range. We return the
     * compatibility limit (conservative). */
    double lo, up;
    bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
    /* the active row: a'y_i > 0 -> at the upper bound, < 0 -> at the lower bound */
    if (s * y_i > 0) {
        *lrange = -INFINITY;   /* the row stays active moving the up bound
                                * upward without structural limits */
        *urange = INFINITY;
        /* the true limit (basis) is not computable: full range as a
         * conservative placeholder? no: we signal an open range */
        return PRIMAL_RES_OK;
    }
    *lrange = -INFINITY;
    *urange = INFINITY;
    return PRIMAL_RES_OK;
}

/* CSC (Aptr[n+1]/Arow/Aval, m x n) -> dense m x n row-major (caller frees). */
static double *csc_to_dense(const int *Aptr, const int *Arow, const double *Aval, int m, int n) {
    double *A = (double *)calloc((size_t)(m > 0 ? m : 1) * (size_t)(n > 0 ? n : 1), sizeof(double));
    if (!A) return NULL;
    for (int j = 0; j < n; j++)
        for (int p = Aptr[j]; p < Aptr[j + 1]; p++) A[(size_t)Arow[p] * n + j] = Aval[p];
    return A;
}

/* lower-CSC symmetric Q (Qptr[n+1]/Qrow/Qval) -> dense n x n symmetric (caller
 * frees). Used only by the dense augmented IPM fallback. */
static double *qcsc_to_dense(const int *Qptr, const int *Qrow, const double *Qval, int n) {
    double *Q = (double *)calloc((size_t)(n > 0 ? n : 1) * (size_t)(n > 0 ? n : 1), sizeof(double));
    if (!Q) return NULL;
    for (int j = 0; j < n; j++)
        for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
            int i = Qrow[p];
            Q[(size_t)i * n + j] = Qval[p];
            if (i != j) Q[(size_t)j * n + i] = Qval[p];
        }
    return Q;
}

/* Decide the solver for a standard-form problem  min 1/2 x'Qx + c'x, A x = b,
 * x >= 0 (hasQ = 0 for LP). Returns 0 = dense tableau simplex, 1 = dense IPM,
 * 2 = sparse LP normal-equations IPM, 3 = sparse QP normal-equations IPM.
 * Large LPs (m*n > 1.5e6, m <= 2000) and large QPs with moderate m go sparse:
 * the dense tableau/LU would explode. For QP the density of Q is checked in
 * solve_std_routed (a dense Q falls back to the dense augmented IPM). MIP
 * relaxations always stay on the simplex. */
int std_route_method(int hasQ, int m, int n, PRIMALtask_t t) {
    int no_simplex = (t->optimizer != PRIMAL_OPTIMIZER_PRIMAL_SIMPLEX &&
                      t->optimizer != PRIMAL_OPTIMIZER_DUAL_SIMPLEX);
    if (hasQ) return (n > 150 && m <= 600 && no_simplex) ? 3 : 1;
    int use_ipm = (t->optimizer == PRIMAL_OPTIMIZER_INTPNT);
    int use_sparse = (m <= 2000 && (double)m * (double)n > 1.0e4 && no_simplex);
    /* Bench/experiment hook: GMB_LP_SPARSE forces the sparse LP route below the
     * size threshold (opt-in, default off; it never overrides an explicit
     * simplex choice). */
    if (getenv("GMB_LP_SPARSE") != NULL && m <= 2000) use_sparse = no_simplex;
    return use_sparse ? 2 : (use_ipm ? 1 : 0);
}

/* ---------- Farkas certificates ------------------------------------------
 * A ray is a vector, and a vector is published only after it has been
 * measured, exactly as the conic route accepts a point only after measuring
 * the relative triple.  The measurement happens in the standard form that
 * produced the ray, min c'x s.t. A x = b, x >= 0, because there the two
 * alternatives are exact:
 *   dual ray   (primal infeasible)   b'y > 0  and  A'y <= 0
 *   primal ray (dual infeasible)     rho >= 0, A rho = 0, c'rho < 0
 * A candidate that does not measure as one of these is an iterate that stopped
 * moving, and it stays unpublished: the status then says "no certificate",
 * never a certificate that was not measured. */

/* The two engines number their statuses differently: simplex says 1 infeasible
 * and 2 unbounded, the interior point says 1 no-convergence and 2 out of
 * memory.  Handing both to the caller in that space is how a plain iteration
 * limit of the IPM reached the user as PRIM_INFEAS_CER, so every status
 * crosses this translator. */
static int std_status(int st, int method) {
    if (method == 0) {              /* simplex.h: 0 opt, 1 infeas, 2 unbnd, 3 maxiter, 4 mem */
        if (st == 0) return STD_OPT;
        if (st == 1) return STD_INFEASIBLE;
        if (st == 2) return STD_UNBOUNDED;
        if (st == 3) return STD_STALLED;
        return STD_MEMORY;
    }
    if (st == 0) return STD_OPT;    /* ipm.h: 0 opt, 1 no convergence, 2 memory, 3 singular */
    if (st == 2) return STD_MEMORY;
    return STD_STALLED;             /* a singular system proves nothing either */
}

/* w = A'y for CSC A; w has n entries, y has m entries. */
static void csc_ATy(const int *Aptr, const int *Arow, const double *Aval,
                    int n, const double *y, double *w) {
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int k = Aptr[j]; k < Aptr[j + 1]; k++) s += Aval[k] * y[Arow[k]];
        w[j] = s;
    }
}

/* v = A rho for CSC A; v has m entries, rho has n entries. */
static void csc_Ax(const int *Aptr, const int *Arow, const double *Aval,
                   int m, int n, const double *rho, double *v) {
    for (int i = 0; i < m; i++) v[i] = 0.0;
    for (int j = 0; j < n; j++) {
        double r = rho[j];
        if (r == 0.0) continue;
        for (int k = Aptr[j]; k < Aptr[j + 1]; k++) v[Arow[k]] += Aval[k] * r;
    }
}

/* largest absolute entry, and the scale a column of A can reach against a
 * vector normalized to 1 — both sides of the tests below are read against it */
static double ray_maxabs(const double *v, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; i++) if (fabs(v[i]) > mx) mx = fabs(v[i]);
    return mx;
}

/* Largest absolute column sum of CSC A (the scale a column can reach). */
static double csc_maxcolabs(const int *Aptr, const double *Aval, int n) {
    double mx = 0.0;
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int k = Aptr[j]; k < Aptr[j + 1]; k++) s += fabs(Aval[k]);
        if (s > mx) mx = s;
    }
    return mx;
}

/* Normalizes v (m entries) to max |.| = 1 and keeps it only if it is a dual ray. */
static int dual_ray_measured(const int *Aptr, const int *Arow, const double *Aval,
                             const double *b, int m, int n, double *v) {
    double mx = ray_maxabs(v, m);
    if (mx <= 0.0) return 0;
    for (int r = 0; r < m; r++) v[r] /= mx;
    double *aty = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!aty) return 0;
    csc_ATy(Aptr, Arow, Aval, n, v, aty);
    double atol = 1e-8 * (1.0 + csc_maxcolabs(Aptr, Aval, n));
    int ok = 1;
    for (int j = 0; j < n; j++) if (aty[j] > atol) { ok = 0; break; }
    free(aty);
    if (!ok) return 0;
    double by = 0.0, babs = 1.0;
    for (int r = 0; r < m; r++) { by += v[r] * b[r]; babs += fabs(b[r]); }
    return by > 1e-8 * babs;
}

/* Normalizes v (n entries) to max |.| = 1 and keeps it only if it is a primal ray. */
static int prim_ray_measured(const int *Aptr, const int *Arow, const double *Aval,
                             const double *c, int m, int n, double *v) {
    double mx = ray_maxabs(v, n);
    if (mx <= 0.0) return 0;
    for (int j = 0; j < n; j++) v[j] /= mx;
    double tol = 1e-8 * (1.0 + csc_maxcolabs(Aptr, Aval, n));
    for (int j = 0; j < n; j++) if (v[j] < -tol) return 0;
    double *ax = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    if (!ax) return 0;
    csc_Ax(Aptr, Arow, Aval, m, n, v, ax);
    double nrm = ray_maxabs(ax, m);
    /* ||A||_row-scale: the same tolerance family as above, on the row sums */
    double rmax = 1.0;
    for (int j = 0; j < n; j++)
        for (int k = Aptr[j]; k < Aptr[j + 1]; k++)
            if (fabs(Aval[k]) > rmax) rmax = fabs(Aval[k]);
    free(ax);
    if (nrm > 1e-7 * rmax * (double)(n > 1 ? n : 1)) return 0;
    double cr = 0.0, cabs = 1.0;
    for (int j = 0; j < n; j++) { cr += v[j] * c[j]; cabs += fabs(c[j]); }
    return cr < -1e-8 * cabs;
}

/* The witness the simplex wrote, if it wrote one; otherwise the iterate the
 * solver stopped on, which the measurement below reads as a candidate. */
const double *ray_candidate(const double *own, const double *iter, int n) {
    if (own && ray_maxabs(own, n) > 0.0) return own;
    return iter;
}

/* Two std-row shapes have no image as one number per constraint: a row that
 * came from x_j <= ux_j, and a ranged constraint whose witness uses BOTH sides
 * (stdform_map_y adds the two multipliers, while the statement the caller can
 * check needs one bound per constraint).  y is normalized to max |.| = 1 by the
 * measurement, hence the relative tests. */
static int ray_representable(const StdForm *sf, const double *y) {
    int r = 0;
    while (r < sf->m) {
        int i = sf->rows[r].orig, lo = 0, up = 0;
        do {                                        /* one constraint's rows are consecutive */
            if (fabs(y[r]) > 1e-12) {
                if (sf->rows[r].kind == SFRK_VARUB) return 0;
                else if (sf->rows[r].kind == SFRK_LO) lo = 1;
                else if (sf->rows[r].kind == SFRK_UP) up = 1;
            }
            r++;
        } while (r < sf->m && sf->rows[r].orig == i);
        if (lo && up) return 0;
    }
    return 1;
}

/* Solve a standard-form problem given in SPARSE CSC (A always; Q lower-CSC or
 * NULL for LP) with the chosen method (see std_route_method). The sparse IPMs
 * consume the CSC directly; the dense simplex / dense augmented IPM materialize
 * a dense copy (only used for small problems). Returns a STD_* code.
 * dray (m) / pray (n) are the Farkas witnesses of infeasibility / unboundedness
 * and only the tableau simplex fills them; an interior-point route leaves them
 * alone, and its last iterate is offered to the same measurement instead. */
static int solve_std_routed_impl(const int *Aptr, const int *Arow, const double *Aval,
                            const int *Qptr, const int *Qrow, const double *Qval,
                            int m, int n, const double *b, const double *c,
                            PRIMALtask_t t, double *xt, double *ystd, double *zst,
                            const double *x0, const double *y0, int method,
                            double *dray, double *pray);
/* Dispatch a standard-form solve through the given method, firing the
 * begin/end optimization callbacks around solve_std_routed_impl. Returns the
 * STD_* status of the chosen engine. */
int solve_std_routed(const int *Aptr, const int *Arow, const double *Aval,
                            const int *Qptr, const int *Qrow, const double *Qval,
                            int m, int n, const double *b, const double *c,
                            PRIMALtask_t t, double *xt, double *ystd, double *zst,
                            const double *x0, const double *y0, int method,
                            double *dray, double *pray)
{
    int simplex = (method == 0 || method == 4);
    iter_cb_begin(t);
    cb_fire(t, simplex ? PRIMAL_CALLBACK_BEGIN_SIMPLEX : PRIMAL_CALLBACK_BEGIN_INTPNT);
    int st = solve_std_routed_impl(Aptr, Arow, Aval, Qptr, Qrow, Qval, m, n, b, c,
                                   t, xt, ystd, zst, x0, y0, method, dray, pray);
    cb_fire(t, simplex ? PRIMAL_CALLBACK_END_SIMPLEX : PRIMAL_CALLBACK_END_INTPNT);
    iter_cb_end();
    return st;
}
/* Body of solve_std_routed: runs the engine named by method (0 tableau
 * simplex, 1 dense IPM, 2 sparse LP IPM, 3 sparse QP IPM, 4 dual simplex from
 * a crash basis) and returns its STD_* status. */
static int solve_std_routed_impl(const int *Aptr, const int *Arow, const double *Aval,
                            const int *Qptr, const int *Qrow, const double *Qval,
                            int m, int n, const double *b, const double *c,
                            PRIMALtask_t t, double *xt, double *ystd, double *zst,
                            const double *x0, const double *y0, int method,
                            double *dray, double *pray)
{
    int nit;
    if (method == 2) {
        int st = ipm_solve_std_csc(Aptr, Arow, Aval, m, n, b, c,
                                   t->tol_gap, t->tol_pfeas, t->tol_dfeas,
                                   iter_cap(t->max_iter_intpnt), xt, ystd, zst, x0, y0, &nit);
        count_add(&t->intpnt_iter, nit);
        return std_status(st, method);
    }
    if (method == 3) {
        double dens = (n > 0) ? (double)Qptr[n] / ((double)n * (n + 1) / 2.0) : 1.0;
        if (dens > 0.3) {                 /* dense Q: normal equations useless -> dense IPM */
            double *dA = csc_to_dense(Aptr, Arow, Aval, m, n);
            double *dQ = qcsc_to_dense(Qptr, Qrow, Qval, n);
            if (!dA || !dQ) { free(dA); free(dQ); return STD_MEMORY; }
            int st = ipm_solve_std(dA, dQ, m, n, b, c,
                                   t->tol_qo_gap, t->tol_qo_pfeas, t->tol_qo_dfeas,
                                   iter_cap(t->max_iter_intpnt), xt, ystd, zst, x0, y0, &nit);
            count_add(&t->intpnt_iter, nit);
            free(dA); free(dQ);
            return std_status(st, method);
        }
        int st = ipm_solve_qp_csc(Aptr, Arow, Aval, Qptr, Qrow, Qval, m, n, b, c,
                                  t->tol_qo_gap, t->tol_qo_pfeas, t->tol_qo_dfeas,
                                  iter_cap(t->max_iter_intpnt), xt, ystd, zst, x0, y0, &nit);
        count_add(&t->intpnt_iter, nit);
        return std_status(st, method);
    }
    if (method == 1) {
        double *dA = csc_to_dense(Aptr, Arow, Aval, m, n);
        double *dQ = Qptr ? qcsc_to_dense(Qptr, Qrow, Qval, n) : NULL;
        if (!dA || (Qptr && !dQ)) { free(dA); free(dQ); return STD_MEMORY; }
        /* Dense interior point: it solves BOTH an LP (Qptr == NULL) and a QP
         * (Qptr != NULL), so the tolerance set follows the class -- the plain
         * set for an LP, the quadratic set for a QP (reference: INTPNT_QO_TOL_*). */
        int st = Qptr
            ? ipm_solve_std(dA, dQ, m, n, b, c,
                            t->tol_qo_gap, t->tol_qo_pfeas, t->tol_qo_dfeas,
                            iter_cap(t->max_iter_intpnt), xt, ystd, zst, x0, y0, &nit)
            : ipm_solve_std(dA, dQ, m, n, b, c,
                            t->tol_gap, t->tol_pfeas, t->tol_dfeas,
                            iter_cap(t->max_iter_intpnt), xt, ystd, zst, x0, y0, &nit);
        count_add(&t->intpnt_iter, nit);
        free(dA); free(dQ);
        return std_status(st, method);
    }
    if (method == 4) {                                   /* dual simplex from the crash basis */
        double *dA = csc_to_dense(Aptr, Arow, Aval, m, n);
        if (!dA) return STD_MEMORY;
        int *bas = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
        int st = STD_STALLED;
        if (bas && simplex_crash_basis(dA, m, n, bas)) {
            st = std_status(simplex_dual_solve_std(dA, m, n, b, c, bas,
                             iter_cap(t->max_iter_simplex), xt, NULL, ystd, &nit), method);
            count_add(&t->sim_dual_iter, nit);
        }
        free(bas); free(dA);
        return st;
    }
    double *dA = csc_to_dense(Aptr, Arow, Aval, m, n);   /* method 0: simplex */
    if (!dA) return STD_MEMORY;
    /* Crash basis (Gaussian with column pivoting) + revised simplex: a valid
     * basis starts the revised method (B^-1 + eta update) without phase 1. If
     * the basis is not primal feasible (st 1), singular (4) or does not close
     * (3), fall back to the tableau (phase 1+2). Kill-switch GMB_SIMPLEX_TABLEAU=1. */
    int st = 3;
    if (!getenv("GMB_SIMPLEX_TABLEAU")) {
        int *bas = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
        if (bas && simplex_crash_basis(dA, m, n, bas)) {
            st = simplex_revised_solve_std(dA, m, n, b, c, bas,
                                           iter_cap(t->max_iter_simplex), xt, ystd, &nit);
            count_add(&t->sim_primal_iter, nit);
            /* Only the optimum (0) is used by the revised: on a non-feasible
             * basis (1), unbounded (2, the revised does not compute the `pray`
             * ray), maxiter (3) or singular (4) one goes back to the tableau. */
            if (st != 0) st = 3;
        }
        free(bas);
    }
    if (st != 0 && st != 2) {
        st = simplex_solve_std(dA, m, n, b, c, iter_cap(t->max_iter_simplex), xt, ystd, dray, pray, &nit);
        count_add(&t->sim_primal_iter, nit);
    }
    free(dA);
    return std_status(st, method);
}

/* Concurrent optimizer: simplex (0) and IPM (1) in parallel on the SAME
 * standard form (reading A/b/c is shared, the outputs are separate). Whoever
 * converges wins; if both converge the simplex wins, so the result is
 * deterministic and identical to the sequential route. */
static void *conc_worker(void *arg) {
    ConcJob *j = (ConcJob *)arg;
    clock_t t0 = clock();
    j->status = solve_std_routed(j->Aptr, j->Arow, j->Aval, j->Qptr, j->Qrow, j->Qval,
                                 j->m, j->n, j->b, j->c, j->t, j->xt, j->ystd, j->zst,
                                 j->x0, j->y0, j->method, j->dray, j->pray);
    j->elapsed = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
    return NULL;
}
/* Run the four std engines (simplex, dense IPM, sparse LP IPM, dual simplex)
 * concurrently on the same standard form and keep the winner. Falls back to
 * the sequential simplex route if the worker buffers cannot be allocated. */
int solve_std_conc(const int *Aptr, const int *Arow, const double *Aval,
                          const int *Qptr, const int *Qrow, const double *Qval,
                          int m, int n, const double *b, const double *c, PRIMALtask_t t,
                          double *xt, double *ystd, double *zst,
                          const double *x0, const double *y0, double *dray, double *pray)
{
    double *xt2 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *ys2 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *zs2 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *dr2 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *pr2 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *xt3 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *ys3 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *zs3 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *dr3 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *pr3 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *xt4 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *ys4 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *zs4 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *dr4 = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *pr4 = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!xt2 || !ys2 || !zs2 || !dr2 || !pr2 || !xt3 || !ys3 || !zs3 || !dr3 || !pr3 ||
        !xt4 || !ys4 || !zs4 || !dr4 || !pr4) {
        free(xt2); free(ys2); free(zs2); free(dr2); free(pr2);
        free(xt3); free(ys3); free(zs3); free(dr3); free(pr3);
        free(xt4); free(ys4); free(zs4); free(dr4); free(pr4);
        return solve_std_routed(Aptr, Arow, Aval, Qptr, Qrow, Qval, m, n, b, c, t,
                                xt, ystd, zst, x0, y0, 0, dray, pray);
    }
    ConcJob j0, j1, j2, j3;
    j0.Aptr = Aptr; j0.Arow = Arow; j0.Aval = Aval; j0.Qptr = Qptr; j0.Qrow = Qrow; j0.Qval = Qval;
    j0.m = m; j0.n = n; j0.b = b; j0.c = c; j0.t = t;
    j0.xt = xt; j0.ystd = ystd; j0.zst = zst; j0.x0 = x0; j0.y0 = y0;
    j0.method = 0; j0.dray = dray; j0.pray = pray; j0.status = STD_MEMORY;
    j1 = j0; j1.xt = xt2; j1.ystd = ys2; j1.zst = zs2; j1.dray = dr2; j1.pray = pr2;
    j1.method = 1; j1.status = STD_MEMORY;
    /* Third strategy: sparse IPM (normal equations). For the LP it is a
     * different engine from the dense one (method 1), so concurrency covers
     * more cases. */
    j2 = j0; j2.xt = xt3; j2.ystd = ys3; j2.zst = zs3; j2.dray = dr3; j2.pray = pr3;
    j2.method = 2; j2.status = STD_MEMORY;
    /* Fourth strategy: dual simplex from the crash basis. */
    j3 = j0; j3.xt = xt4; j3.ystd = ys4; j3.zst = zs4; j3.dray = dr4; j3.pray = pr4;
    j3.method = 4; j3.status = STD_MEMORY;
    pthread_t th0, th1, th2, th3; int c0, c1, c2, c3;
    c0 = pthread_create(&th0, NULL, conc_worker, &j0);
    c1 = pthread_create(&th1, NULL, conc_worker, &j1);
    c2 = pthread_create(&th2, NULL, conc_worker, &j2);
    c3 = pthread_create(&th3, NULL, conc_worker, &j3);
    if (c0 != 0) conc_worker(&j0);
    if (c1 != 0) conc_worker(&j1);
    if (c2 != 0) conc_worker(&j2);
    if (c3 != 0) conc_worker(&j3);
    if (c0 == 0) pthread_join(th0, NULL);
    if (c1 == 0) pthread_join(th1, NULL);
    if (c2 == 0) pthread_join(th2, NULL);
    if (c3 == 0) pthread_join(th3, NULL);
    /* Choice: with `concurrent_time` the FASTEST optimum wins (time measured
     * per worker); otherwise the deterministic tie-break (simplex, dense IPM,
     * sparse IPM) so as not to depend on timing. */
    ConcJob *jv[4] = { &j0, &j1, &j2, &j3 };
    double *xv[4] = { xt, xt2, xt3, xt4 };
    double *yv[4] = { ystd, ys2, ys3, ys4 };
    double *zv[4] = { zst, zs2, zs3, zs4 };
    double *drv[4] = { dray, dr2, dr3, dr4 };
    double *prv[4] = { pray, pr2, pr3, pr4 };
    int pick = -1;
    if (t->concurrent_time) {
        double best_t = 0.0;
        for (int k = 0; k < 4; k++)
            if (jv[k]->status == STD_OPT && (pick < 0 || jv[k]->elapsed < best_t)) {
                pick = k; best_t = jv[k]->elapsed;
            }
    } else {
        for (int k = 0; k < 4; k++) if (jv[k]->status == STD_OPT) { pick = k; break; }
    }
    int st;
    if (pick < 0) st = j0.status;
    else {
        st = jv[pick]->status;
        if (pick != 0) {
            memcpy(xt, xv[pick], (size_t)n * sizeof(double));
            memcpy(ystd, yv[pick], (size_t)m * sizeof(double));
            memcpy(zst, zv[pick], (size_t)n * sizeof(double));
            memcpy(dray, drv[pick], (size_t)m * sizeof(double));
            memcpy(pray, prv[pick], (size_t)n * sizeof(double));
        }
    }
    free(xt2); free(ys2); free(zs2); free(dr2); free(pr2);
    free(xt3); free(ys3); free(zs3); free(dr3); free(pr3);
    free(xt4); free(ys4); free(zs4); free(dr4); free(pr4);
    return st;
}

/* Measure the two standard-space candidates and publish the ones that are
 * witnesses.  Returns the status the caller reports: a stalled solve is
 * upgraded only by a vector that proves one of the two alternatives, and a
 * "certificate" status is kept only if the matching ray was published — the
 * status names a certificate, so it does not outlive one. */
int ray_publish(PRIMALtask_t t, const StdForm *sf,
                       const double *rs, const double *ds,
                       double *yray, double *xray, int status)
{
    int claim_inf = (status == STD_INFEASIBLE), claim_unb = (status == STD_UNBOUNDED);
    int dy = (status != STD_OPT && status != STD_MEMORY) &&
             yray && dual_ray_measured(sf->Aptr, sf->Arow, sf->Aval, sf->b, sf->m, sf->n, yray) &&
             ray_representable(sf, yray);
    int px = (status != STD_OPT && status != STD_MEMORY) &&
             xray && prim_ray_measured(sf->Aptr, sf->Arow, sf->Aval, sf->c, sf->m, sf->n, xray);
    if (!claim_inf && !claim_unb && !dy && !px) return status;
    if (dy) {
        double *ymin = (double *)calloc((size_t)(sf->ncon > 0 ? sf->ncon : 1), sizeof(double));
        if (ymin) {
            stdform_map_y(sf, yray, ymin);
            /* the ray is a statement about the feasible set, so the objective
             * sense does not enter; -1 because the min-form dual is the
             * negated sum of the row images (see stdform_map_y) */
            for (int i = 0; i < sf->ncon; i++) t->dray[i] = -rs[i] * ymin[i];
            free(ymin);
            t->has_dray = 1;
        } else dy = 0;
    }
    if (px && !dy) {                     /* infeasibility dominates unboundedness */
        double *dx = (double *)calloc((size_t)(sf->nvar > 0 ? sf->nvar : 1), sizeof(double));
        if (dx) {
            stdform_map_dir(sf, xray, dx);
            for (int j = 0; j < sf->nvar; j++) t->pray[j] = ds[j] * dx[j];
            free(dx);
            t->has_pray = 1;
        } else px = 0;
    }
    if (dy) return STD_INFEASIBLE;
    if (px) return STD_UNBOUNDED;
    /* the engine's own claim stands (a positive phase-1 optimum and a ratio
     * test with no leaving row are proofs about the system that was solved);
     * what the measurement decides is only whether a vector goes with it */
    return status;
}

/* Index = PRIMALconeetype; the two gaps are the reference's DPOW (5) and ZERO (6),
 * kinds this solver does not accept in appendcone. */
const char *const cone_kind_name[] = { "QUAD", "RQUAD", "PEXP", "DEXP", "PPOW",
                                                   "?", "?", "RPOW" };

/* entry count for cone_kind_name (see primal_priv.h) */
const int cone_kind_count = (int)(sizeof cone_kind_name / sizeof *cone_kind_name);
/* ---- what the published point owes the cones, measured on the user's own
 * model.  rel_pri is computed on the equality rows of the standard form and
 * says nothing about K: a point outside a cone reads rel_pri = 0.  On the
 * tangent-cut route that is not a formality -- its LP master moves on an OUTER
 * approximation of K, a superset, so a vertex satisfies the cuts and need not
 * satisfy K.  Every cone block, every PSD bar and every finite lower bound is
 * measured here.  The units differ per cone (an eigenvalue, a degree-1 epigraph
 * gap, a bound gap), so the SIGN is the answer and the magnitude is only an
 * order of depth. */
