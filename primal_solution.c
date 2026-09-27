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
/* primal_solution.c - solution getters, violations, slices, getsolution.
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

static double quadratic_row_gradient(PRIMALtask_t t, int j) {
    double g = 0.0;
    for (int i = 0; i < t->numcon; i++) if (t->qcon && t->qcon[i]) {
        double a = 0.0;
        for (int k = 0; k < t->numvar; k++)
            a += t->qcon[i][j*t->numvar+k] * t->x[k];
        g += t->y[i] * a;
    }
    return g;
}

/* ---------------- solution getters ---------------- */

/* PRIMAL_SOL_ITG is accepted alongside ITR and BAS because a caller written
 * against the reference asks for the integer solution under that key; this
 * solver stores one point per solve, so all three keys report it. */
int sol_key_ok(PRIMALsolt which) {
    return which == PRIMAL_SOL_ITR || which == PRIMAL_SOL_BAS || which == PRIMAL_SOL_ITG;
}

/**
 * Retrieves the primal solution vector (x).
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param xx  [out] Pre-allocated array of size numvar. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or xx is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available (has_sol=0).
 *
 * @note Returns ERR_ARG if the task has no solution (has_sol=0). The solution
 *       is only available after a successful PRIMAL_optimize call.
 *
 * @example
 * int n; PRIMAL_getnumvar(task, &n);
 * double *x = malloc(n * sizeof(double));
 * PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
 * for (int i = 0; i < n; i++) printf("x[%d] = %f\n", i, x[i]);
 */
PRIMALrescodee PRIMAL_getxx(PRIMALtask_t t, PRIMALsolt which, double *xx) {
    if (!t || !xx) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(xx, t->x, (size_t)t->numvar * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the dual solution vector (y) for constraints.
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param y   [out] Pre-allocated array of size numcon. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or y is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @example
 * int m; PRIMAL_getnumcon(task, &m);
 * double *y = malloc(m * sizeof(double));
 * PRIMAL_gety(task, PRIMAL_SOL_ITR, y);
 */
PRIMALrescodee PRIMAL_gety(PRIMALtask_t t, PRIMALsolt which, double *y) {
    if (!t || !y) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(y, t->y, (size_t)t->numcon * sizeof(double));
    return PRIMAL_RES_OK;
}

/* Store a caller-supplied primal warm-start vector in the task. */
PRIMALrescodee PRIMAL_putxx(PRIMALtask_t t, PRIMALsolt which, const double *xx) {
    if (!t || !xx) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    double *w = (double *)lazy_grow(t->warm_x, &t->warmxcap,
                                    t->numvar > 0 ? t->numvar : 1, sizeof(double));
    if (!w) return PRIMAL_RES_ERR_ALLOC;
    t->warm_x = w;
    memcpy(t->warm_x, xx, (size_t)t->numvar * sizeof(double));
    t->has_warm = 1;
    return PRIMAL_RES_OK;
}

/* Store a caller-supplied dual warm-start vector in the task. */
PRIMALrescodee PRIMAL_puty(PRIMALtask_t t, PRIMALsolt which, const double *y) {
    if (!t || !y) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    double *w = (double *)lazy_grow(t->warm_y, &t->warmycap,
                                    t->numcon > 0 ? t->numcon : 1, sizeof(double));
    if (!w) return PRIMAL_RES_ERR_ALLOC;
    t->warm_y = w;
    memcpy(t->warm_y, y, (size_t)t->numcon * sizeof(double));
    t->has_warm = 1;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the lower slack vector for constraints (slc = min(0, y)).
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param slc [out] Pre-allocated array of size numcon. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or slc is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the negative part of the dual variables (slc_i = min(0, y_i)).
 */
PRIMALrescodee PRIMAL_getslc(PRIMALtask_t t, PRIMALsolt which, double *slc) {
    if (!t || !slc) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(slc, t->slc, (size_t)t->numcon * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the upper slack vector for constraints (suc = max(0, y)).
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param suc [out] Pre-allocated array of size numcon. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or suc is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the positive part of the dual variables (suc_i = max(0, y_i)).
 */
PRIMALrescodee PRIMAL_getsuc(PRIMALtask_t t, PRIMALsolt which, double *suc) {
    if (!t || !suc) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(suc, t->suc, (size_t)t->numcon * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the lower slack vector for variables (slx = min(0, reduced_cost)).
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param slx [out] Pre-allocated array of size numvar. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or slx is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the negative part of the reduced costs (slx_j = min(0, z_j)
 *       where z is the reduced cost). For basic variables, slx_j = 0.
 */
PRIMALrescodee PRIMAL_getslx(PRIMALtask_t t, PRIMALsolt which, double *slx) {
    if (!t || !slx) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(slx, t->slx, (size_t)t->numvar * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the upper slack vector for variables (sux = max(0, reduced_cost)).
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param sux [out] Pre-allocated array of size numvar. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or sux is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the positive part of the reduced costs (sux_j = max(0, z_j)
 *       where z is the reduced cost). For basic variables, sux_j = 0.
 */
PRIMALrescodee PRIMAL_getsux(PRIMALtask_t t, PRIMALsolt which, double *sux) {
    if (!t || !sux) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    memcpy(sux, t->sux, (size_t)t->numvar * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the primal objective value.
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param pobj [out] Pointer to double receiving the primal objective value.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or pobj is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the objective value including the constant term (cfix).
 *       Requires has_sol=1 (solution must exist).
 *
 * @example
 * double obj;
 * PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
 * printf("Primal objective: %f\n", obj);
 */
PRIMALrescodee PRIMAL_getprimalobj(PRIMALtask_t t, PRIMALsolt which, double *pobj) {
    if (!t || !pobj) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol || t->result_stale) return PRIMAL_RES_ERR_ARG;
    *pobj = t->pobj;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the dual objective value.
 *
 * @param t   [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param dobj [out] Pointer to double receiving the dual objective value.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or dobj is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid or no solution available.
 *
 * @note Returns the dual objective value including the constant term (cfix).
 *       At optimality, pobj == dobj (strong duality).
 */
PRIMALrescodee PRIMAL_getdualobj(PRIMALtask_t t, PRIMALsolt which, double *dobj) {
    if (!t || !dobj) return PRIMAL_RES_ERR_NULL;
    if (t->mip_result) return PRIMAL_RES_ERR_ARG; /* use MIO_OBJ_BOUND */
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol || t->result_stale) return PRIMAL_RES_ERR_ARG;
    *dobj = t->dobj;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the solution status.
 *
 * @param t     [in]  Task handle.
 * @param which [in] Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param solsta [out] Pointer to PRIMALsolstae receiving the status.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or solsta is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid.
 *
 * @note Returns the raw solsta enum value. The numeric values match MSKsolsta:
 *       0=UNKNOWN, 1=OPTIMAL, 5=PRIM_INFEAS_CER, 6=DUAL_INFEAS_CER, 9=INTEGER_OPTIMAL.
 *       Does NOT consult has_sol (a problem status speaks about the model, not the point).
 *
 * @example
 * PRIMALsolstae sta;
 * PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
 * if (sta == PRIMAL_SOL_STA_OPTIMAL) printf("Optimal\n");
 */
PRIMALrescodee PRIMAL_getsolsta(PRIMALtask_t t, PRIMALsolt which, PRIMALsolstae *solsta) {
    if (!t || !solsta) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    /* Not every solution status has a point as its object: a *_CER member names
     * a Farkas vector, which the ray getters carry. So the answer is read from
     * solsta itself, and what keeps an unsolved task at UNKNOWN is that
     * opt_prepare resets it — not a gate on the solution buffer. */
    *solsta = t->solsta;
    return PRIMAL_RES_OK;
}

/* The reference publishes problem status and solution status as two numbers and
 * fixes their pairing in two tables (accessing the solution, Tables 7.2 for
 * continuous and 7.3 for integer problems):
 *   optimal            PRIM_AND_DUAL_FEAS + OPTIMAL
 *   primal infeasible  PRIM_INFEAS        + PRIM_INFEAS_CER
 *   dual infeasible    DUAL_INFEAS        + DUAL_INFEAS_CER
 *   integer optimal    PRIM_FEAS          + INTEGER_OPTIMAL
 *   integer feasible   PRIM_FEAS          + PRIM_FEAS
 *   no conclusion      UNKNOWN            + UNKNOWN
 * The pairing is a function of the solution status for every outcome produced
 * here, with one exception and one refinement.  The exception is an infeasible
 * mixed-integer problem: table 7.3 pairs it with UNKNOWN, which this function
 * would not derive, so it is carried in t->prosta.  The refinement is that a
 * *_CER member names a Farkas vector, and a route that found no vector must not
 * claim one: it publishes UNKNOWN here and sets t->prosta to the digit the
 * table would have shown (ray_publish on the LP/QP route, and the
 * infeasible/unbounded branches of the other routes).  So t->prosta is either
 * unset — the table rules — or an override that keeps the verdict readable
 * where the certificate is not there to name.
 *
 * Nothing here consults has_sol, and that is deliberate: a problem status
 * speaks about the model, not about a point. Gating it on the solution buffer
 * was what forced the routes that reach an infeasible or unbounded verdict to
 * raise has_sol on an all-zero x, so that the verdict would not disappear. */
static PRIMALprostae prosta_of(const PRIMALtask_t t) {
    if (t->prosta != PRIMAL_PRO_STA_UNKNOWN) return t->prosta;
    switch (t->solsta) {
        case PRIMAL_SOL_STA_OPTIMAL:         return PRIMAL_PRO_STA_PRIM_AND_DUAL_FEAS;
        case PRIMAL_SOL_STA_PRIM_FEAS:
        case PRIMAL_SOL_STA_INTEGER_OPTIMAL: return PRIMAL_PRO_STA_PRIM_FEAS;
        case PRIMAL_SOL_STA_PRIM_INFEAS_CER: return PRIMAL_PRO_STA_PRIM_INFEAS;
        case PRIMAL_SOL_STA_DUAL_INFEAS_CER: return PRIMAL_PRO_STA_DUAL_INFEAS;
        default:                             return PRIMAL_PRO_STA_UNKNOWN;
    }
}

/**
 * Retrieves the problem status.
 *
 * @param t      [in]  Task handle.
 * @param which  [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param prosta [out] Pointer to PRIMALprostae receiving the status.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or prosta is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid.
 *
 * @note The problem status is derived from solsta via the reference's pairing
 *       tables, with one exception: an infeasible MIP returns PRIM_INFEAS with
 *       solsta=UNKNOWN. The pairing is:
 *       - OPTIMAL -> PRIM_AND_DUAL_FEAS
 *       - PRIM_FEAS / INTEGER_OPTIMAL -> PRIM_FEAS
 *       - PRIM_INFEAS_CER -> PRIM_INFEAS
 *       - DUAL_INFEAS_CER -> DUAL_INFEAS
 *       - UNKNOWN -> UNKNOWN
 *       The t->prosta override handles the MIP infeasible case.
 *
 * @example
 * PRIMALprostae psta;
 * PRIMAL_getprosta(task, PRIMAL_SOL_ITR, &psta);
 * if (psta == PRIMAL_PRO_STA_PRIM_INFEAS) printf("Infeasible\n");
 */
PRIMALrescodee PRIMAL_getprosta(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *prosta) {
    if (!t || !prosta) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    *prosta = prosta_of(t);
    return PRIMAL_RES_OK;
}

/* ---------- Farkas certificates ----------
 * Both arrays are full length (numcon / numvar) and both answer ERR_ARG when
 * nothing measured: an all-zero vector would read as "no certificate" to a
 * caller that loops over the entries, and a wrong one as a proof. */
/**
 * Retrieves the dual Farkas certificate (infeasibility ray).
 *
 * @param t  [in]  Task handle.
 * @param y  [out] Pre-allocated array of size numcon. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or y is NULL,
 *         PRIMAL_RES_ERR_ARG if no dual ray available (has_dray=0).
 *
 * @note Returns the vector y such that A'y <= 0 and b'y > 0 (normalized to max|y|=1).
 *       Only available for LP/QP problems that are primal infeasible.
 *       Conic/SDP/MIP routes do NOT publish certificates (return ERR_ARG).
 *
 * @example
 * int m; PRIMAL_getnumcon(task, &m);
 * double *y = malloc(m * sizeof(double));
 * PRIMALrescodee rc = PRIMAL_getdualray(task, y);
 * if (rc == PRIMAL_RES_OK) printf("Dual ray found\n");
 */
PRIMALrescodee PRIMAL_getdualray(PRIMALtask_t t, PRIMALrealt *y) {
    if (!t || !y) return PRIMAL_RES_ERR_NULL;
    if (!t->has_dray) return PRIMAL_RES_ERR_ARG;
    memcpy(y, t->dray, (size_t)t->numcon * sizeof(double));
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the primal Farkas certificate (unbounded direction).
 *
 * @param t   [in]  Task handle.
 * @param rho [out] Pre-allocated array of size numvar. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or rho is NULL,
 *         PRIMAL_RES_ERR_ARG if no primal ray available (has_pray=0).
 *
 * @note Returns the vector rho such that rho >= 0, A*rho = 0, c'*rho < 0
 *       (normalized to max|rho|=1). Only available for LP/QP problems that
 *       are dual infeasible (unbounded). Conic/SDP/MIP routes do NOT
 *       publish certificates (return ERR_ARG).
 *
 * @example
 * int n; PRIMAL_getnumvar(task, &n);
 * double *rho = malloc(n * sizeof(double));
 * PRIMALrescodee rc = PRIMAL_getprimalray(task, rho);
 * if (rc == PRIMAL_RES_OK) printf("Primal ray found\n");
 */
PRIMALrescodee PRIMAL_getprimalray(PRIMALtask_t t, PRIMALrealt *rho) {
    if (!t || !rho) return PRIMAL_RES_ERR_NULL;
    if (!t->has_pray) return PRIMAL_RES_ERR_ARG;
    memcpy(rho, t->pray, (size_t)t->numvar * sizeof(double));
    return PRIMAL_RES_OK;
}

/* ---------------- solution quality (max violations) ---------------- */


/* First member of row i at point w, with all THREE doors a term can enter a row
 * (T97): the scalar coefficients, or the quadratic row value (which already
 * includes the linear part, so it substitutes the loop), plus coef*<A_m,X_b>
 * for every bar term on the row. One implementation for the aggregate
 * (PRIMAL_getprimalinfeas) and for the per-row getter, so a term cannot be
 * visible to one and lost by the other. */
double row_activity(const PRIMALtask_t t, int i, const double *w) {
    double ax = 0.0;
    if (t->has_qcon > 0 && t->qcon && t->qcon[i]) {
        ax = quad_row_value(t, i, w);
    } else {
        for (int j = 0; j < t->numvar; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] == i) ax += c->val[k] * w[j];
        }
    }
    for (int k = 0; k < t->nbarA; k++) {
        if (t->barA_con[k] != i) continue;
        int b = t->barA_bar[k], d = t->barDim[b], m = t->barA_sym[k];
        if (!t->barx[b]) continue;
        double tr = 0.0;
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            int p = t->sym_subi[m][e], q = t->sym_subj[m][e];
            tr += t->sym_val[m][e] * (p == q ? 1.0 : 2.0) * t->barx[b][p * d + q];
        }
        ax += t->barA_coef[k] * tr;
    }
    return ax;
}

/* Primal violation of the variable-bound side of x_j. The semi domain is
 * {0} union [l,u] and l is what its lower-bound slot holds: a DEACTIVATED
 * variable sits below its own bound by definition, so it is measured by
 * distance to the point 0 plus the capped case (T95). */
static double var_violation(const PRIMALtask_t t, int j) {
    double lo, up;
    bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
    int vt = t->vartype[j];
    int semi = (vt == PRIMAL_VAR_TYPE_SEMI_CONT || vt == PRIMAL_VAR_TYPE_SEMI_INT);
    double v = 0.0;
    if (semi && t->x[j] <= up) {
        if (t->x[j] >= lo) v = 0.0;
        else {
            v = fabs(t->x[j]);
            if (lo - t->x[j] < v) v = lo - t->x[j];
        }
    } else {
        if (t->x[j] < lo) v = lo - t->x[j];
        if (t->x[j] > up) v = t->x[j] - up;
    }
    return v;
}

/* Deviation: these are absolute max violations, not the relative figures the
 * task's tolerances speak. getprimalinfeas does NOT measure membership of the
 * point in the cones -- that is the fourth figure, printed as
 * `[cones] rel_slack=` and asserted by T90. getdualinfeas measures the dual
 * cone of every block it can read (see cone_dual_worst for what makes a block
 * readable) and does not check the signs of the row multipliers. The
 * reference's convention for either was never read, so neither was invented. */

/* Report the worst primal violation over rows and variable bounds. */
PRIMALrescodee PRIMAL_getprimalinfeas(PRIMALtask_t t, PRIMALsolt which, double *pinf) {
    (void)which;
    if (!t || !pinf) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int nvar = t->numvar, ncon = t->numcon;
    double worst = 0.0;
    for (int i = 0; i < ncon; i++) {
        double lo, up;
        bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
        /* row_activity carries the T97 reading: scalar | quad | bar terms. A row
         * of a bar model has no scalar coefficients at all, and reading its
         * left-hand side as 0 made the bound BE the violation: a barrier solve
         * converged to rel_pri = 1.7e-9 published getprimalinfeas = 1. */
        double ax = row_activity(t, i, t->x);
        double v = 0.0;
        if (ax < lo) v = lo - ax;
        if (ax > up) v = ax - up;
        if (v > worst) worst = v;
    }
    for (int j = 0; j < nvar; j++) {
        double v = var_violation(t, j);
        if (v > worst) worst = v;
    }
    *pinf = worst;
    return PRIMAL_RES_OK;
}

/* Report the worst dual violation over variables, bars and cone blocks. */
PRIMALrescodee PRIMAL_getdualinfeas(PRIMALtask_t t, PRIMALsolt which, double *dinf) {
    (void)which;
    if (!t || !dinf) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int nvar = t->numvar;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    double worst = 0.0;
    /* A variable inside a cone has no reduced cost of its own: the reduced cost
     * of a cone member IS a coordinate of the dual of that cone, and the two
     * loops below speak the LP's rules -- a nonzero reduced cost of a variable
     * strictly between its bounds, a reduced cost of the wrong sign at a bound.
     * Applying them to a cone member called a model whose pobj and dobj are
     * exact dual-infeasible by 1.67. Its dual is measured against K* below. */
    char *incon = (char *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(char));
    if (!incon) return PRIMAL_RES_ERR_ALLOC;
    /* The stationarity of a QUADRATIC objective carries Qx: c + Qx + A'y + z = 0.
     * Without it every QP looks dual-infeasible by |Qx| (measured: a 3-variable
     * QP whose dual was exact reported dinf = 3.2; T113/T117 only exercised LPs,
     * so the term was never seen).  Qx = 0 for an LP, so the LP path is unchanged. */
    double *qxv = NULL;
    if (t->has_qobj && t->qt_n > 0) {
        qxv = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
        if (qxv) task_Qx(t, t->x, qxv);
    }
    for (int k = 0; k < t->numcones; k++) {
        const int *mi = t->cone_mem[k];
        if (!mi) continue;
        for (int i = 0; i < t->cone_nmem[k]; i++)
            if (mi[i] >= 0 && mi[i] < nvar) incon[mi[i]] = 1;
    }
    /* stationarity residual: c + A'y + z = 0 (original form), and the sign
     * conditions of the slacks (complementarity structure). We report the
     * max violation of the reduced-cost sign conditions at the bounds. */
    for (int j = 0; j < nvar; j++) {
        if (incon[j]) continue;
        double av = 0.0;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) av += c->val[k] * t->y[c->sub[k]];
        double zj = -(t->c[j] + (qxv ? qxv[j] : 0.0) + av + quadratic_row_gradient(t, j));
        double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        int at_lo = (t->x[j] <= lo + 1e-7) && (lo > -INF);
        int at_up = (t->x[j] >= up - 1e-7) && (up < INF);
        int inside = (t->x[j] > lo + 1e-7) && (t->x[j] < up - 1e-7);
        double v = 0.0;
        if (at_lo && s * zj > 1e-9) v = s * zj;         /* wrong sign at lower */
        if (at_up && s * zj < -1e-9) v = -s * zj;      /* wrong sign at upper */
        if (inside && fabs(zj) > 1e-9) v = fabs(zj);   /* basic must be 0 */
        if (v > worst) worst = v;
    }
    /* stationarity residual (dual feasibility proper): r = c + A'y + z */
    for (int j = 0; j < nvar; j++) {
        if (incon[j]) continue;
        double av = 0.0;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) av += c->val[k] * t->y[c->sub[k]];
        double zj = s * (t->slx[j] + t->sux[j]);   /* min-form z */
        double r = s * (t->c[j] + (qxv ? qxv[j] : 0.0) + av + quadratic_row_gradient(t, j)) + zj;
        if (fabs(r) > worst) worst = fabs(r);
    }
    free(qxv);
    free(incon);
    /* The dual of a bar variable is a MATRIX, and the loop above cannot see it:
     * Z_j = C_j - sum_i y_i A^i_j must lie in the PSD cone, so a negative
     * eigenvalue of the published barsj[j] is exactly the dual violation this
     * getter is named for. A block nobody filled is reported as no violation
     * rather than as a false alarm. */
    for (int j = 0; j < t->numbarvar; j++) {
        int d = t->barDim[j];
        if (!t->barsj[j]) continue;
        int filled = 0;
        for (int k = 0; k < d * d && !filled; k++) if (t->barsj[j][k] != 0.0) filled = 1;
        if (!filled) continue;
        double emax = 0.0, emin = bar_min_eig(d, t->barsj[j], &emax);
        if (!isfinite(emin)) continue;
        if (-emin > worst) worst = -emin;
    }
    /* The blocks the loops above were made to skip: the reduced cost of a cone
     * member has to lie in the DUAL of that cone, which is a statement about the
     * block as a vector and not about any one of its coordinates. */
    {
        double cv = cone_dual_worst(t, s, NULL, 0);
        if (cv > worst) worst = cv;
    }
    *dinf = worst;
    return PRIMAL_RES_OK;
}

/* ---------------- solution information: violation getters ----------------
 * The reference's family (MOSEK 11.2.4, read from
 * docs.mosek.com/latest/capi/alphabetic-functionalities.html):
 *   getpviolcon/getpviolvar/getpviolbarvar/getpviolcones  (primal)
 *   getdviolcon/getdviolvar/getdviolbarvar/getdviolcones  (dual, not exposed)
 *   getsolutioninfo = the maxima of those, plus pobj/dobj.
 * The PRIMAL side is exact and independent of the dual sign convention:
 * per constraint  max(l - a'x, a'x - u) with a'x read by row_activity, and per
 * variable the bound/domain violation of var_violation. The pviolbarvar is
 * max(-lambda_min(X_j), 0), the pviolcones max(0, -cone_signed_slack), the
 * pviolitg min(x-floor(x), ceil(x)-x) -- all as the reference defines them.
 * The DUAL side is expressed in our own published y/slc/suc/slx/sux, whose sign
 * is the mirror of the reference's (README "Dual conventions"); the formulas are
 * the reference's, transposed through that mirror. Declared scope, the same
 * boundary PRIMAL_getdualinfeas draws: a variable that is a cone member carries
 * its dual inside the cone, not in a scalar reduced cost, so dviolvar leaves it
 * unmeasured (returns 0); a cone block whose member has a finite bound, or a
 * model with a quadratic row/objective, makes the cone dual unreadable and is
 * likewise left unmeasured. */

/* Test whether variable j belongs to any cone block. */
static int var_in_cone(const PRIMALtask_t t, int j) {
    for (int k = 0; k < t->numcones; k++) {
        const int *mi = t->cone_mem[k];
        if (!mi) continue;
        for (int i = 0; i < t->cone_nmem[k]; i++)
            if (mi[i] == j) return 1;
    }
    return 0;
}

/* Dual violation of row i, in the reference's own formula
 *   max( rho(s_l, l), rho(s_u, -u), |-y + s_l - s_u| ),
 *   rho(x, b) = -x if the bound b is finite else |x|,
 * read through the mirror of our published y/slc/suc (README «Dual conventions»:
 * Y_ref = -y, s_l = -slc, s_u = suc). With that substitution the third term is
 * `y + dl - du` and, because our slc/suc split y exactly, it is 0 for any point
 * this solver publishes; the two rho terms are the sign conditions, and they are
 * what carries a multiplier sitting on a side that has no bound. */
/* Compute the dual violation of row i from the published multipliers. */
static double dviol_con(const PRIMALtask_t t, int i) {
    double lo, up;
    bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
    int sense = t->sense == PRIMAL_OPTIMIZE_MAXIMIZE ? -1 : 1;
    double y = sense * t->y[i];
    double dl = fmax(0.0, -y), du = fmax(0.0, y);
    double t1 = isfinite(lo) ? -dl : fabs(dl);
    double t2 = isfinite(up) ? -du : fabs(du);
    double t3 = fabs(t->y[i] - t->slc[i] - t->suc[i]);
    double m = t1 > t2 ? t1 : t2;
    return t3 > m ? t3 : (m > 0.0 ? m : 0.0);
}

/* Dual violation of variable j: the reference's
 *   max( rho(s_l, l), rho(s_u, -u), |A'y + s_l - s_u - c| )
 * in the same mirrored convention. The stationarity term is our own residual
 * `s*c + A'y + s*(slx+sux)`, the same quantity getdualinfeas reads. A cone member
 * is left UNMEASURED (0): its dual lives inside the cone, not in a scalar reduced
 * cost -- the boundary getdualinfeas and cone_dual_worst already draw. */
/* Compute the dual violation of variable j including stationarity. */
static double dviol_var(const PRIMALtask_t t, int j, int s) {
    if (var_in_cone(t, j)) return 0.0;
    double lo, up;
    bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
    double z = s * (t->slx[j] + t->sux[j]);
    double dl = fmax(0.0, -z), du = fmax(0.0, z);
    double t1 = isfinite(lo) ? -dl : fabs(dl);
    double t2 = isfinite(up) ? -du : fabs(du);
    double av = 0.0; const Col *c = &t->cols[j];
    for (int q = 0; q < c->nz; q++) av += c->val[q] * t->y[c->sub[q]];
    double qg = 0.0;
    for (int e = 0; e < t->qt_n; e++) {
        if (t->qt_i[e] == j) qg += t->qt_v[e] * t->x[t->qt_j[e]];
        if (t->qt_j[e] == j && t->qt_i[e] != j) qg += t->qt_v[e] * t->x[t->qt_i[e]];
    }
    double t3 = fabs(t->c[j] + qg + av + quadratic_row_gradient(t, j) + t->slx[j] + t->sux[j]);
    double m = t1 > t2 ? t1 : t2;
    return t3 > m ? t3 : (m > 0.0 ? m : 0.0);
}

/* Measure the PSD dual violation of bar block j via its min eigenvalue. */
double bar_dual_viol(const PRIMALtask_t t, int j) {
    int d = t->barDim[j];
    if (!t->barsj[j]) return 0.0;
    int filled = 0;
    for (int k = 0; k < d * d && !filled; k++) if (t->barsj[j][k] != 0.0) filled = 1;
    if (!filled) return 0.0;
    double emax = 0.0, emin = bar_min_eig(d, t->barsj[j], &emax);
    if (!isfinite(emin)) return 0.0;
    return emin < 0.0 ? -emin : 0.0;
}

/* Recover a candidate decomposition of c+A'y over overlapping SOCs. The
 * native route retains each incidence's dual; the aggregate reduced cost is
 * their SUM, not the dual of each cone. Distribute any current stationarity
 * discrepancy equally, so the candidate still sums to the CURRENT c+A'y
 * (including after a user changes c or y). Membership is then measured by the
 * caller. A shape change invalidates the stored decomposition. */
double soc_dual_component(const PRIMALtask_t t, int k, int i, double raw) {
    if (!t->soc_dual) return raw;
    int p = 0;
    for (int c = 0; c < t->numcones; c++) {
        if (t->cone_type[c] != PRIMAL_CT_QUAD && t->cone_type[c] != PRIMAL_CT_RQUAD)
            return raw;
        for (int a = 0; a < t->cone_nmem[c]; a++, p++) {
            if (p >= t->nsoc_dual) return raw;
            const SocDual *v = &t->soc_dual[p];
            if (v->cone != c || v->pos != a || v->var != t->cone_mem[c][a]) return raw;
        }
    }
    if (p != t->nsoc_dual) return raw;
    int j = t->cone_mem[k][i], count = 0;
    double sum = 0.0, own = 0.0;
    for (p = 0; p < t->nsoc_dual; p++) {
        const SocDual *v = &t->soc_dual[p];
        if (v->var != j) continue;
        sum += v->value; count++;
        if (v->cone == k && v->pos == i) own = v->value;
    }
    return count > 1 ? own + (raw - sum) / count : raw;
}

/* Dual violation of cone block k. The reference spells out only the quadratic
 * cone: outside membership it returns the NORM of the whole block (inside, 0);
 * that branch is transcribed exactly. For the other cone types the page says the
 * formula is "generalized appropriately" without giving it, so the signed slack
 * `max(0, -cone_dual_signed_slack(d))` is our normalisation -- a DECLARED
 * deviation, not an invented reference number. Returns 0 both for "inside" and
 * for "not readable"; the readable-blocks detail is the [condual] trace's job. */
/* Compute the dual violation of cone block k in the given sense. */
static double dviol_cone(const PRIMALtask_t t, int k, int s) {
    if (t->has_qcon > 0 || t->has_qobj) return 0.0;
    int nk = t->cone_nmem[k]; const int *mi = t->cone_mem[k];
    if (nk <= 0 || !mi) return 0.0;
    for (int i = 0; i < nk; i++) {
        int j = mi[i]; double lo, up;
        if (j < 0 || j >= t->numvar) return 0.0;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (isfinite(lo) || isfinite(up)) return 0.0;
    }
    double *d = (double *)malloc((size_t)nk * sizeof(double));
    if (!d) return 0.0;
    for (int i = 0; i < nk; i++) {
        const Col *col = &t->cols[mi[i]];
        double av = 0.0;
        for (int q = 0; q < col->nz; q++) av += col->val[q] * t->y[col->sub[q]];
        d[i] = soc_dual_component(t, k, i, s * (t->c[mi[i]] + av));
    }
    double v = 0.0;
    int ct = t->cone_type[k];
    if (ct == PRIMAL_CT_QUAD) {
        double n2 = 0.0;
        for (int i = 1; i < nk; i++) n2 += d[i] * d[i];
        n2 = sqrt(n2);
        if (d[0] >= -n2) { double q = (n2 - d[0]) / sqrt(2.0); v = q > 0.0 ? q : 0.0; }
        else { double nn = 0.0; for (int i = 0; i < nk; i++) nn += d[i] * d[i]; v = sqrt(nn); }
    } else {
        double sl = cone_dual_signed_slack(ct, t->cone_param[k], d, nk);
        if (isfinite(sl) && -sl > v) v = -sl;
    }
    free(d);
    return v;
}

/* Worst dual violation among the readable cone blocks (the aggregate the
 * reference reports in getsolutioninfo). 0 also when nothing is readable: the
 * aggregate has one number for both facts. */
/* Return the worst dual violation over the readable cone blocks. */
static double cone_dual_viol_abs(const PRIMALtask_t t, int s) {
    double worst = 0.0;
    for (int k = 0; k < t->numcones; k++) {
        double v = dviol_cone(t, k, s);
        if (v > worst) worst = v;
    }
    return worst;
}


/* Measure the primal violation of cone block k at the published point. */
static double cone_primal_viol(const PRIMALtask_t t, int k) {
    int nk = t->cone_nmem[k]; const int *mi = t->cone_mem[k];
    if (nk <= 0 || !mi) return 0.0;
    double *v = (double *)malloc((size_t)nk * sizeof(double));
    if (!v) return 0.0;
    int ok = 1;
    for (int i = 0; i < nk; i++) {
        int j = mi[i];
        if (j < 0 || j >= t->numvar) { ok = 0; break; }
        v[i] = t->x[j];
    }
    double sl = ok ? cone_signed_slack(t->cone_type[k], t->cone_param[k], v, nk) : HUGE_VAL;
    free(v);
    if (!isfinite(sl)) return INFINITY;
    return sl < 0.0 ? -sl : 0.0;
}

/* Every per-index getter shares the same contract: a refusal (bad index, no
 * solution, null buffer) writes NOTHING, so the whole `sub` vector is validated
 * before the first store, exactly as the shape T102/T108 fixed for the readers. */
/* Validate a per-index violation query without writing on refusal. */
static PRIMALrescodee viol_precheck(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, const PRIMALrealt *viol, int dim) {
    if (!t || !viol) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    if (num < 0 || (num > 0 && !sub)) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++)
        if (sub[k] < 0 || sub[k] >= dim) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/* Report the primal violation of the listed constraints. */
PRIMALrescodee PRIMAL_getpviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numcon : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) {
        int i = sub[k];
        double lo, up;
        bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
        double ax = row_activity(t, i, t->x), v = 0.0;
        if (ax < lo) v = lo - ax;
        if (ax > up) v = ax - up;
        viol[k] = v;
    }
    return PRIMAL_RES_OK;
}

/* Report the primal violation of the listed variables. */
PRIMALrescodee PRIMAL_getpviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numvar : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) viol[k] = var_violation(t, sub[k]);
    return PRIMAL_RES_OK;
}

/* Report the primal violation of the listed bar variables. */
PRIMALrescodee PRIMAL_getpviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numbarvar : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) {
        int j = sub[k];
        if (!t->barx[j]) { viol[k] = 0.0; continue; }
        double emax = 0.0, emin = bar_min_eig(t->barDim[j], t->barx[j], &emax);
        viol[k] = isfinite(emin) && emin < 0.0 ? -emin : 0.0;
    }
    return PRIMAL_RES_OK;
}

/* Report the primal violation of the listed cones. */
PRIMALrescodee PRIMAL_getpviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numcones : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) viol[k] = cone_primal_viol(t, sub[k]);
    return PRIMAL_RES_OK;
}

/* Report the dual violation of the listed constraints. */
PRIMALrescodee PRIMAL_getdviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numcon : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) viol[k] = dviol_con(t, sub[k]);
    return PRIMAL_RES_OK;
}

/* Report the dual violation of the listed variables. */
PRIMALrescodee PRIMAL_getdviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numvar : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    for (int k = 0; k < num; k++) viol[k] = dviol_var(t, sub[k], s);
    return PRIMAL_RES_OK;
}

/* Report the dual violation of the listed bar variables. */
PRIMALrescodee PRIMAL_getdviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numbarvar : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) viol[k] = bar_dual_viol(t, sub[k]);
    return PRIMAL_RES_OK;
}

/* Report the dual violation of the listed cones. */
PRIMALrescodee PRIMAL_getdviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol) {
    PRIMALrescodee rc = viol_precheck(t, which, num, sub, viol, t ? t->numcones : 0);
    if (rc != PRIMAL_RES_OK) return rc;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    for (int k = 0; k < num; k++) viol[k] = dviol_cone(t, sub[k], s);
    return PRIMAL_RES_OK;
}

/* Fill the maxima of the primal/dual violations plus both objectives. */
PRIMALrescodee PRIMAL_getsolutioninfo(PRIMALtask_t t, PRIMALsolt which,
    PRIMALrealt *pobj, PRIMALrealt *pviolcon, PRIMALrealt *pviolvar,
    PRIMALrealt *pviolbarvar, PRIMALrealt *pviolcone, PRIMALrealt *pviolitg,
    PRIMALrealt *dobj, PRIMALrealt *dviolcon, PRIMALrealt *dviolvar,
    PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;

    if (pobj) { PRIMALrealt v = NAN; PRIMAL_getprimalobj(t, which, &v); *pobj = v; }
    if (dobj) { PRIMALrealt v = NAN; PRIMAL_getdualobj(t, which, &v); *dobj = v; }
    if (pviolcon) {
        double w = 0.0;
        for (int i = 0; i < t->numcon; i++) {
            double lo, up;
            bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
            double ax = row_activity(t, i, t->x), v = 0.0;
            if (ax < lo) v = lo - ax;
            if (ax > up) v = ax - up;
            if (v > w) w = v;
        }
        *pviolcon = w;
    }
    if (pviolvar) {
        double w = 0.0;
        for (int j = 0; j < t->numvar; j++) { double v = var_violation(t, j); if (v > w) w = v; }
        *pviolvar = w;
    }
    if (pviolbarvar) {
        double w = 0.0;
        for (int j = 0; j < t->numbarvar; j++) {
            if (!t->barx[j]) continue;
            double emax = 0.0, emin = bar_min_eig(t->barDim[j], t->barx[j], &emax);
            if (isfinite(emin) && -emin > w) w = -emin;
        }
        *pviolbarvar = w;
    }
    if (pviolcone) {
        double w = 0.0;
        for (int k = 0; k < t->numcones; k++) { double v = cone_primal_viol(t, k); if (v > w) w = v; }
        *pviolcone = w;
    }
    if (pviolitg) {
        double w = 0.0;
        for (int j = 0; j < t->numvar; j++) {
            int vt = t->vartype[j];
            if (vt != PRIMAL_VAR_TYPE_INT && vt != PRIMAL_VAR_TYPE_INT_BIN &&
                vt != PRIMAL_VAR_TYPE_SEMI_INT) continue;
            if (vt == PRIMAL_VAR_TYPE_SEMI_INT && t->x[j] == 0.0) continue;
            double fl = floor(t->x[j]), ce = ceil(t->x[j]);
            double d1 = t->x[j] - fl, d2 = ce - t->x[j];
            double v = d1 < d2 ? d1 : d2;
            if (v > w) w = v;
        }
        *pviolitg = w;
    }
    if (dviolcon) {
        double w = 0.0;
        for (int i = 0; i < t->numcon; i++) { double v = dviol_con(t, i); if (v > w) w = v; }
        *dviolcon = w;
    }
    if (dviolvar) {
        double w = 0.0;
        for (int j = 0; j < t->numvar; j++) { double v = dviol_var(t, j, s); if (v > w) w = v; }
        *dviolvar = w;
    }
    if (dviolbarvar) {
        double w = 0.0;
        for (int j = 0; j < t->numbarvar; j++) { double v = bar_dual_viol(t, j); if (v > w) w = v; }
        *dviolbarvar = w;
    }
    if (dviolcone) *dviolcone = cone_dual_viol_abs(t, s);
    return PRIMAL_RES_OK;
}

/* ---------------- feasibility repair (elastic) ---------------- */

/* Repair infeasibility by solving the elastic LP and keeping its x. */
PRIMALrescodee PRIMAL_feasrepair(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    int nvar = t->numvar, ncon = t->numcon;
    if (t->has_qobj || t->has_qcon > 0 || t->numcones > 0 || t->numbarvar > 0)
        return PRIMAL_RES_ERR_ARG;   /* LP only */

    /* elastic LP: variables x, s^-_x, s^+_x, s^-_c, s^+_c (2nvar + 2ncon)
     * objective: sum of all elastic slacks; constraints: original rows and
     * variable bounds with slack. Solve, report x only. */
    PRIMALenv_t env2 = NULL;
    PRIMALrescodee rc = PRIMAL_makeenv(&env2, NULL);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMALtask_t e = NULL;
    rc = PRIMAL_maketask(env2, 0, 0, &e);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deleteenv(&env2); return rc; }
    int nx2 = nvar, nsx = nvar, nsc = ncon;
    int ntot = nvar + nsx + nsx + nsc + nsc;   /* x, sm_x, sp_x, sm_c, sp_c */
    PRIMAL_appendvars(e, ntot);
    PRIMAL_appendcons(e, ncon > 0 ? ncon : 0);
    for (int j = 0; j < nvar; j++) {
        /* x_j: bound relaxed by the slacks: sm_x_j free>=0, sp_x_j >=0 */
        PRIMAL_putvarbound(e, j, t->bkx[j], t->blx[j], t->bux[j]);
        PRIMAL_putcj(e, nx2 + j, 1.0);            /* sm_x */
        PRIMAL_putcj(e, nx2 + nsx + j, 1.0);      /* sp_x */
        PRIMAL_putvarbound(e, nx2 + j, PRIMAL_BK_LO, 0.0, INF);
        PRIMAL_putvarbound(e, nx2 + nsx + j, PRIMAL_BK_LO, 0.0, INF);
        /* relaxed bounds: lx - sm <= x <= ux + sp implemented as two rows:
         * x + sm >= lx  (LO row),  x - sp <= ux (UP row): use nvar extra? no:
         * variable bounds cannot host slacks -> put them as extra rows on a
         * SECOND set of constraints: simplest: append 2*nvar rows. */
    }
    /* variable-bound relaxation rows: 2 per variable with finite bound */
    int nextr = 0;
    for (int j = 0; j < nvar; j++) {
        double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (lo > -INF) nextr++;
        if (up < INF) nextr++;
    }
    PRIMAL_appendcons(e, nextr);
    int row = 0;
    for (int j = 0; j < nvar; j++) {
        double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (lo > -INF) {
            int sub[2] = {j, nx2 + j};          /* x + sm >= lo */
            double v[2] = {1.0, 1.0};
            PRIMAL_putarow(e, row, 2, sub, v);
            PRIMAL_putconbound(e, row, PRIMAL_BK_LO, lo, INF);
            row++;
        }
        if (up < INF) {
            int sub[2] = {j, nx2 + nsx + j};    /* x - sp <= up */
            double v[2] = {1.0, -1.0};
            PRIMAL_putarow(e, row, 2, sub, v);
            PRIMAL_putconbound(e, row, PRIMAL_BK_UP, -INF, up);
            row++;
        }
    }
    /* row relaxation: original row i with bounds relaxed by sm_c/sp_c */
    for (int i = 0; i < ncon; i++) {
        int sub[64];
        double v[64];
        int w = 0;
        for (int j = 0; j < nvar; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] == i) { sub[w] = j; v[w] = c->val[k]; w++; }
        }
        /* + sm_c_i (LO side) and - sp_c_i (UP side) */
        sub[w] = nvar + 2 * nsx + i; v[w] = 1.0; w++;
        sub[w] = nvar + 2 * nsx + nsc + i; v[w] = -1.0; w++;
        PRIMAL_putarow(e, i, w, sub, v);
        /* relaxed bounds: [lc - sm, uc + sp]: sm/sp are free>=0, so the row
         * bounds become the ORIGINAL bounds with slack added: LO: lc - sm,
         * UP: uc + sp — with the terms above this is: row in [lc - sm, uc + sp]
         * The putconbound keeps the original bounds and the slack terms
         * relax them. */
        PRIMAL_putconbound(e, i, t->bkc[i], t->blc[i], t->buc[i]);
        PRIMAL_putcj(e, nvar + 2 * nsx + i, 1.0);
        PRIMAL_putcj(e, nvar + 2 * nsx + nsc + i, 1.0);
        PRIMAL_putvarbound(e, nvar + 2 * nsx + i, PRIMAL_BK_LO, 0.0, INF);
        PRIMAL_putvarbound(e, nvar + 2 * nsx + nsc + i, PRIMAL_BK_LO, 0.0, INF);
    }
    rc = PRIMAL_optimize(e);
    if (rc != PRIMAL_RES_OK) {
        /* infeasible relaxation cannot happen (slacks free), but guard */
        PRIMAL_deletetask(&e);
        PRIMAL_deleteenv(&env2);
        return rc;
    }
    /* copy the repaired x into the task solution */
    rc = opt_prepare(t);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&e); PRIMAL_deleteenv(&env2); return rc; }
    double *xe = (double *)malloc((size_t)(ntot > 0 ? ntot : 1) * sizeof(double));
    if (!xe) { PRIMAL_deletetask(&e); PRIMAL_deleteenv(&env2); return PRIMAL_RES_ERR_ALLOC; }
    PRIMAL_getxx(e, PRIMAL_SOL_ITR, xe);
    memcpy(t->x, xe, (size_t)nvar * sizeof(double));
    free(xe);
    double po = t->cfix;
    for (int j = 0; j < nvar; j++) po += t->c[j] * t->x[j];
    t->pobj = po;
    t->dobj = po;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;   /* repaired point, not optimal */
    PRIMAL_deletetask(&e);
    PRIMAL_deleteenv(&env2);
    return PRIMAL_RES_OK;
}

/* Read one slice of the stored solution vectors by solution item. */
PRIMALrescodee PRIMAL_getsolutionslice(PRIMALtask_t t, PRIMALsolt which, int part,
                                 int first, int last, double *values) {
    if (!t || !values) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    /* XC ("solution for the constraints") is the row activity, which is not a
     * stored vector but a reading of the model (the three doors of T97). It is
     * served here; SNX (conic multipliers per variable) is not stored and stays
     * refused -- a declared deviation. */
    if (part == PRIMAL_SOL_ITEM_XC) {
        if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
        for (int k = first; k < last; k++)
            values[k - first] = t->has_xc ? t->xc[k] : row_activity(t, k, t->x);
        return PRIMAL_RES_OK;
    }
    if (part == PRIMAL_SOL_ITEM_SNX) {
        if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
        for (int k = first; k < last; k++) values[k - first] = t->snx[k];
        return PRIMAL_RES_OK;
    }
    int len;
    const double *src;
    switch (part) {
        case PRIMAL_SOL_ITEM_XX: len = t->numvar; src = t->x; break;
        case PRIMAL_SOL_ITEM_Y:  len = t->numcon; src = t->y; break;
        case PRIMAL_SOL_ITEM_SLC: len = t->numcon; src = t->slc; break;
        case PRIMAL_SOL_ITEM_SUC: len = t->numcon; src = t->suc; break;
        case PRIMAL_SOL_ITEM_SLX: len = t->numvar; src = t->slx; break;
        case PRIMAL_SOL_ITEM_SUX: len = t->numvar; src = t->sux; break;
        default: return PRIMAL_RES_ERR_ARG;
    }
    if (first < 0 || last > len || first > last) return PRIMAL_RES_ERR_ARG;
    for (int k = first; k < last; k++) values[k - first] = src[k];
    return PRIMAL_RES_OK;
}

/* ---- solution slices and reduced costs ----
 * The reference's getxxslice/getyslice/getslcslice/getsucslice/getslxslice/
 * getsuxslice/getskxslice/getskcslice and getreducedcosts. Each is [first, last)
 * on its vector and the buffer holds last-first entries; a refusal (bad range,
 * no solution, null) writes nothing. `getreducedcosts` is the reference's
 * (s_l^x)_j - (s_u^x)_j, which through the mirror of our published slx/sux is
 * -(slx+sux): the same reduced cost (c + A'y in original form) that the
 * stationarity of getdualinfeas reads. The status-key slices follow
 * getskx/getskc -- the key exists independently of a published point. */
static PRIMALrescodee sol_slice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                const double *src, int len, PRIMALrealt *out) {
    if (!t || !out) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > len || first > last) return PRIMAL_RES_ERR_ARG;
    for (int k = first; k < last; k++) out[k - first] = src[k];
    return PRIMAL_RES_OK;
}

/**
 * Retrieves a slice of the primal solution vector.
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in]  Ending index (exclusive), first <= last <= numvar.
 * @param xx    [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or xx is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 *
 * @note The slice is [first, last). Empty slice (first == last) is valid.
 *
 * @example
 * double slice[5];
 * PRIMAL_getxxslice(task, PRIMAL_SOL_ITR, 0, 5, slice);
 * // Gets x[0]..x[4]
 */
PRIMALrescodee PRIMAL_getxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *xx) {
    return sol_slice(t, which, first, last, t ? t->x : NULL, t ? t->numvar : 0, xx);
}
/**
 * Retrieves a slice of the dual solution vector (y).
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in]  Ending index (exclusive), first <= last <= numcon.
 * @param y     [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or y is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 */
PRIMALrescodee PRIMAL_getyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *y) {
    return sol_slice(t, which, first, last, t ? t->y : NULL, t ? t->numcon : 0, y);
}
/**
 * Retrieves a slice of the lower constraint slack vector (slc).
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in]  Ending index (exclusive), first <= last <= numcon.
 * @param slc   [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or slc is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 */
PRIMALrescodee PRIMAL_getslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slc) {
    return sol_slice(t, which, first, last, t ? t->slc : NULL, t ? t->numcon : 0, slc);
}
/**
 * Retrieves a slice of the upper constraint slack vector (suc).
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in]  Ending index (exclusive), first <= last <= numcon.
 * @param suc   [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or suc is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 */
PRIMALrescodee PRIMAL_getsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *suc) {
    return sol_slice(t, which, first, last, t ? t->suc : NULL, t ? t->numcon : 0, suc);
}
/**
 * Retrieves a slice of the lower variable slack vector (slx).
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in]  Ending index (exclusive), first <= last <= numvar.
 * @param slx   [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or slx is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 */
PRIMALrescodee PRIMAL_getslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slx) {
    return sol_slice(t, which, first, last, t ? t->slx : NULL, t ? t->numvar : 0, slx);
}
/**
 * Retrieves a slice of the upper variable slack vector (sux).
 *
 * @param t     [in]  Task handle.
 * @param which [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in]  Ending index (exclusive), first <= last <= numvar.
 * @param sux   [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or sux is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 */
PRIMALrescodee PRIMAL_getsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *sux) {
    return sol_slice(t, which, first, last, t ? t->sux : NULL, t ? t->numvar : 0, sux);
}

/**
 * Retrieves a slice of the reduced costs vector.
 *
 * @param t        [in]  Task handle.
 * @param which    [in]  Which solution (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS, PRIMAL_SOL_ITG).
 * @param first    [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last     [in]  Ending index (exclusive), first <= last <= numvar.
 * @param redcosts [out] Pre-allocated array of size (last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or redcosts is NULL,
 *         PRIMAL_RES_ERR_ARG if which invalid, no solution, or indices out of bounds.
 *
 * @note The reduced cost is -(slx + sux), which equals the dual slack z_j = c_j + A'_j y.
 *       For basic variables, the reduced cost is 0.
 *
 * @example
 * double rc[5];
 * PRIMAL_getreducedcosts(task, PRIMAL_SOL_ITR, 0, 5, rc);
 * // Gets reduced costs for variables 0..4
 */
PRIMALrescodee PRIMAL_getreducedcosts(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                      PRIMALrealt *redcosts) {
    if (!t || !redcosts) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    for (int j = first; j < last; j++) redcosts[j - first] = -(t->slx[j] + t->sux[j]);
    return PRIMAL_RES_OK;
}

/* Solution-vector setters (reference putslc/putsuc/putslx/putsux and their
 * *slice forms, putxxslice/putyslice). They write into the point buffers; the
 * buffers exist after the first opt_prepare, so before a solve they answer
 * ERR_ARG. A slice is [first, last) with last-first entries. */
static PRIMALrescodee sol_set(PRIMALtask_t t, double *dst, int len, int first, int last,
                              const PRIMALrealt *src) {
    if (!t || !src) return PRIMAL_RES_ERR_NULL;
    if (!dst) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > len || first > last) return PRIMAL_RES_ERR_ARG;
    for (int k = first; k < last; k++) {
        if (src[k - first] != src[k - first]) return PRIMAL_RES_ERR_ARG;
        dst[k] = src[k - first];
    }
    return PRIMAL_RES_OK;
}
/* Overwrite the full lower constraint slack vector. */
PRIMALrescodee PRIMAL_putslc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slc) {
    (void)which;
    if (!t || !slc) return PRIMAL_RES_ERR_NULL;
    if (!t->slc) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numcon; i++) t->slc[i] = slc[i];
    return PRIMAL_RES_OK;
}
/* Overwrite the full upper constraint slack vector. */
PRIMALrescodee PRIMAL_putsuc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *suc) {
    (void)which;
    if (!t || !suc) return PRIMAL_RES_ERR_NULL;
    if (!t->suc) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numcon; i++) t->suc[i] = suc[i];
    return PRIMAL_RES_OK;
}
/* Overwrite the full lower variable slack vector. */
PRIMALrescodee PRIMAL_putslx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slx) {
    (void)which;
    if (!t || !slx) return PRIMAL_RES_ERR_NULL;
    if (!t->slx) return PRIMAL_RES_ERR_ARG;
    for (int j = 0; j < t->numvar; j++) t->slx[j] = slx[j];
    return PRIMAL_RES_OK;
}
/* Overwrite the full upper variable slack vector. */
PRIMALrescodee PRIMAL_putsux(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *sux) {
    (void)which;
    if (!t || !sux) return PRIMAL_RES_ERR_NULL;
    if (!t->sux) return PRIMAL_RES_ERR_ARG;
    for (int j = 0; j < t->numvar; j++) t->sux[j] = sux[j];
    return PRIMAL_RES_OK;
}
/* Overwrite a slice of the stored primal vector. */
PRIMALrescodee PRIMAL_putxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xx) {
    (void)which;
    return sol_set(t, t ? t->x : NULL, t ? t->numvar : 0, first, last, xx);
}
/* Overwrite a slice of the stored dual vector. */
PRIMALrescodee PRIMAL_putyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                const PRIMALrealt *y) {
    (void)which;
    return sol_set(t, t ? t->y : NULL, t ? t->numcon : 0, first, last, y);
}
/* Overwrite a slice of the stored lower variable slacks. */
PRIMALrescodee PRIMAL_putslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slx) {
    (void)which;
    return sol_set(t, t ? t->slx : NULL, t ? t->numvar : 0, first, last, slx);
}
/* Overwrite a slice of the stored upper variable slacks. */
PRIMALrescodee PRIMAL_putsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *sux) {
    (void)which;
    return sol_set(t, t ? t->sux : NULL, t ? t->numvar : 0, first, last, sux);
}
/* Overwrite a slice of the stored lower constraint slacks. */
PRIMALrescodee PRIMAL_putslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slc) {
    (void)which;
    return sol_set(t, t ? t->slc : NULL, t ? t->numcon : 0, first, last, slc);
}
/* Overwrite a slice of the stored upper constraint slacks. */
PRIMALrescodee PRIMAL_putsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *suc) {
    (void)which;
    return sol_set(t, t ? t->suc : NULL, t ? t->numcon : 0, first, last, suc);
}

/* putvarboundlistconst/putconboundlistconst: the same bound for every index
 * in the list. The whole list is validated before applying anything. */
/* Set the same variable bound on every listed index. */
PRIMALrescodee PRIMAL_putvarboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkx, PRIMALrealt blx, PRIMALrealt bux) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !sub)) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < num; k++)
        if (sub[k] < 0 || sub[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        PRIMALrescodee rc = PRIMAL_putvarbound(t, sub[k], bkx, blx, bux);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Set the same constraint bound on every listed index. */
PRIMALrescodee PRIMAL_putconboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkc, PRIMALrealt blc, PRIMALrealt buc) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !sub)) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < num; k++)
        if (sub[k] < 0 || sub[k] >= t->numcon) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        PRIMALrescodee rc = PRIMAL_putconbound(t, sub[k], bkc, blc, buc);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* Read a slice of the stored variable status keys. */
PRIMALrescodee PRIMAL_getskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  PRIMALstakeye *skx) {
    if (!t || !skx) return PRIMAL_RES_ERR_NULL;
    (void)which;
    if (!t->skx) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    for (int j = first; j < last; j++) skx[j - first] = t->skx[j];
    return PRIMAL_RES_OK;
}

/* Read a slice of the stored constraint status keys. */
PRIMALrescodee PRIMAL_getskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  PRIMALstakeye *skc) {
    if (!t || !skc) return PRIMAL_RES_ERR_NULL;
    (void)which;
    if (!t->skc) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    for (int i = first; i < last; i++) skc[i - first] = t->skc[i];
    return PRIMAL_RES_OK;
}

/* Status-key setters (reference putskcslice/putskxslice): [first,last)
 * on skc/skx, a refusal writes nothing. */
/* Overwrite a slice of the variable status keys, creating the table. */
PRIMALrescodee PRIMAL_putskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skx) {
    if (!t || !skx) return PRIMAL_RES_ERR_NULL;
    (void)which;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    if (!t->skx) {   /* the table is created on first write */
        t->skx = (PRIMALstakeye *)calloc((size_t)(t->numvar > 0 ? t->numvar : 1), sizeof(PRIMALstakeye));
        if (!t->skx) return PRIMAL_RES_ERR_ALLOC;
        t->skxcap = t->numvar;
    }
    for (int j = first; j < last; j++) t->skx[j] = skx[j - first];
    return PRIMAL_RES_OK;
}
/* Overwrite a slice of the constraint status keys, creating the table. */
PRIMALrescodee PRIMAL_putskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skc) {
    if (!t || !skc) return PRIMAL_RES_ERR_NULL;
    (void)which;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    if (!t->skc) {
        t->skc = (PRIMALstakeye *)calloc((size_t)(t->numcon > 0 ? t->numcon : 1), sizeof(PRIMALstakeye));
        if (!t->skc) return PRIMAL_RES_ERR_ALLOC;
        t->skccap = t->numcon;
    }
    for (int i = first; i < last; i++) t->skc[i] = skc[i - first];
    return PRIMAL_RES_OK;
}

/* 2-norms of the primal solution (reference getprimalsolutionnorms):
 * ||x^c|| (row activities), ||x||, ||X_bar||_F. */
/* Report the 2-norms of the published primal point and bar blocks. */
PRIMALrescodee PRIMAL_getprimalsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmxc, PRIMALrealt *nrmxx, PRIMALrealt *nrmbarx) {
    (void)which;
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    double sx = 0.0;
    for (int j = 0; j < t->numvar; j++) sx += t->x[j] * t->x[j];
    if (nrmxx) *nrmxx = sqrt(sx);
    if (nrmxc) {
        double sc = 0.0;
        for (int i = 0; i < t->numcon; i++) {
            double a = row_activity(t, i, t->x);
            sc += a * a;
        }
        *nrmxc = sqrt(sc);
    }
    if (nrmbarx) {
        double sb = 0.0;
        for (int j = 0; j < t->numbarvar; j++) {
            int d = t->barDim[j];
            if (!t->barx[j]) continue;
            for (int k = 0; k < d * d; k++) sb += t->barx[j][k] * t->barx[j][k];
        }
        *nrmbarx = sqrt(sb);
    }
    return PRIMAL_RES_OK;
}

/* 2-norms of the dual solution (reference getdualsolutionnorms). `nrmsnx`
 * (the per-variable conic multipliers) is not stored: it reads 0, the
 * same deviation as SNX. */
/* Report the 2-norms of the published dual vectors and bar duals. */
PRIMALrescodee PRIMAL_getdualsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmy, PRIMALrealt *nrmslc, PRIMALrealt *nrmsuc,
        PRIMALrealt *nrmslx, PRIMALrealt *nrmsux, PRIMALrealt *nrmsnx,
        PRIMALrealt *nrmbars) {
    (void)which;
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    double s = 0.0;
    if (nrmy) { s = 0.0; for (int i = 0; i < t->numcon; i++) s += t->y[i] * t->y[i]; *nrmy = sqrt(s); }
    if (nrmslc) { s = 0.0; for (int i = 0; i < t->numcon; i++) s += t->slc[i] * t->slc[i]; *nrmslc = sqrt(s); }
    if (nrmsuc) { s = 0.0; for (int i = 0; i < t->numcon; i++) s += t->suc[i] * t->suc[i]; *nrmsuc = sqrt(s); }
    if (nrmslx) { s = 0.0; for (int j = 0; j < t->numvar; j++) s += t->slx[j] * t->slx[j]; *nrmslx = sqrt(s); }
    if (nrmsux) { s = 0.0; for (int j = 0; j < t->numvar; j++) s += t->sux[j] * t->sux[j]; *nrmsux = sqrt(s); }
    if (nrmsnx) *nrmsnx = 0.0;   /* not stored */
    if (nrmbars) {
        s = 0.0;
        for (int j = 0; j < t->numbarvar; j++) {
            int d = t->barDim[j];
            if (!t->barsj[j]) continue;
            for (int k = 0; k < d * d; k++) s += t->barsj[j][k] * t->barsj[j][k];
        }
        *nrmbars = sqrt(s);
    }
    return PRIMAL_RES_OK;
}

/* ---- x^c: the value of the constraint variables (reference getxc/getxcslice) ----
 * For a constraint l <= a'x <= u the reported value is the left-hand side, with
 * all three doors of a term (scalar, quadratic row, bar) read by row_activity --
 * the same reading getpviolcon uses, so the two cannot disagree about a row. */
/* `x^c` is the row activity. When the user has set it by hand
 * (`putxc`/`putxcslice`), the getter returns that; otherwise it computes it. */
static double xc_value(PRIMALtask_t t, int i) {
    return t->has_xc ? t->xc[i] : row_activity(t, i, t->x);
}
/* Read the full constraint-activity vector into the caller buffer. */
PRIMALrescodee PRIMAL_getxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc) {
    if (!t || !xc) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numcon; i++) xc[i] = xc_value(t, i);
    return PRIMAL_RES_OK;
}

/* Read a slice of the constraint-activity vector. */
PRIMALrescodee PRIMAL_getxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 PRIMALrealt *xc) {
    if (!t || !xc) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    for (int i = first; i < last; i++) xc[i - first] = xc_value(t, i);
    return PRIMAL_RES_OK;
}

/* Setters for x^c and s_n^x (reference putxc/putxcslice/putsnx/putsnxslice) and
 * the s_n^x slice. s_n^x is computed by no route: it is storage only. */
/* Store a caller-supplied constraint-activity vector. */
PRIMALrescodee PRIMAL_putxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc) {
    (void)which;
    if (!t || !xc) return PRIMAL_RES_ERR_NULL;
    if (!t->xc) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numcon; i++) t->xc[i] = xc[i];
    t->has_xc = 1;
    return PRIMAL_RES_OK;
}
/* Store a caller-supplied slice of the constraint-activity vector. */
PRIMALrescodee PRIMAL_putxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xc) {
    (void)which;
    if (!t || !xc) return PRIMAL_RES_ERR_NULL;
    if (!t->xc) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    for (int i = first; i < last; i++) t->xc[i] = xc[i - first];
    t->has_xc = 1;
    return PRIMAL_RES_OK;
}
/* Store a caller-supplied conic dual vector. */
PRIMALrescodee PRIMAL_putsnx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *snx) {
    (void)which;
    if (!t || !snx) return PRIMAL_RES_ERR_NULL;
    if (!t->snx) return PRIMAL_RES_ERR_ARG;
    for (int j = 0; j < t->numvar; j++) t->snx[j] = snx[j];
    return PRIMAL_RES_OK;
}
/* Overwrite a slice of the stored conic dual vector. */
PRIMALrescodee PRIMAL_putsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *snx) {
    (void)which;
    return sol_set(t, t ? t->snx : NULL, t ? t->numvar : 0, first, last, snx);
}
/* Read a slice of the stored conic dual vector. */
PRIMALrescodee PRIMAL_getsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  PRIMALrealt *snx) {
    (void)which;
    return sol_slice(t, PRIMAL_SOL_ITR, first, last, t ? t->snx : NULL, t ? t->numvar : 0, snx);
}

/* ---- getsolution / getskn / getsnx ----
 * The reference's one-call solution reader. Every output is optional (NULL to
 * skip). `skn` (cone status keys) is SK_UNDEF for every cone: this solver keeps
 * no basis status for a conic block. `snx` (conic dual per variable) is 0: the
 * dual of a cone lives inside the block, not in a scalar column (T101 I), which
 * is exactly what the published `slx+sux` reads on those variables. Both are
 * DECLARED deviations, not invented values. When no point was published the
 * point buffers have nothing to say and answer ERR_ARG, exactly as the individual
 * getters do; the status/basis outputs are still filled. */
PRIMALrescodee PRIMAL_getskn(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skn) {
    (void)which;
    if (!t || !skn) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < t->numcones; k++) skn[k] = PRIMAL_SK_UNDEF;
    return PRIMAL_RES_OK;
}

/* Fill the per-variable conic dual vector with 0 (not stored; declared
 * deviation, like SNX). Refuses without a published point. */
PRIMALrescodee PRIMAL_getsnx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *snx) {
    if (!t || !snx) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    for (int j = 0; j < t->numvar; j++) snx[j] = 0.0;
    return PRIMAL_RES_OK;
}

/* Reference one-call solution reader: fills every non-NULL output from the
 * published point and the status/basis tables; a missing point leaves the
 * status outputs filled and makes a requested point output answer ERR_ARG. */
PRIMALrescodee PRIMAL_getsolution(PRIMALtask_t t, PRIMALsolt which,
    PRIMALprostae *problemsta, PRIMALsolstae *solutionsta,
    PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
    PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y,
    PRIMALrealt *slc, PRIMALrealt *suc, PRIMALrealt *slx, PRIMALrealt *sux,
    PRIMALrealt *snx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (problemsta) *problemsta = prosta_of(t);
    if (solutionsta) *solutionsta = t->solsta;
    if (skc) for (int i = 0; i < t->numcon; i++) skc[i] = t->skc ? t->skc[i] : PRIMAL_SK_UNDEF;
    if (skx) for (int j = 0; j < t->numvar; j++) skx[j] = t->skx ? t->skx[j] : PRIMAL_SK_UNDEF;
    if (skn) for (int k = 0; k < t->numcones; k++) skn[k] = PRIMAL_SK_UNDEF;
    if (snx) for (int j = 0; j < t->numvar; j++) snx[j] = 0.0;
    int want_point = (xc || xx || y || slc || suc || slx || sux);
    if (!t->has_sol) return want_point ? PRIMAL_RES_ERR_ARG : PRIMAL_RES_OK;
    if (xc) for (int i = 0; i < t->numcon; i++) xc[i] = row_activity(t, i, t->x);
    if (xx) memcpy(xx, t->x, (size_t)t->numvar * sizeof(double));
    if (y) memcpy(y, t->y, (size_t)t->numcon * sizeof(double));
    if (slc) memcpy(slc, t->slc, (size_t)t->numcon * sizeof(double));
    if (suc) memcpy(suc, t->suc, (size_t)t->numcon * sizeof(double));
    if (slx) memcpy(slx, t->slx, (size_t)t->numvar * sizeof(double));
    if (sux) memcpy(sux, t->sux, (size_t)t->numvar * sizeof(double));
    return PRIMAL_RES_OK;
}

/* Write the task model to a file (extension dispatch), firing the write
 * callbacks and storing the result in last_rc. */
PRIMALrescodee PRIMAL_writedata(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_WRITE);
    PRIMALrescodee rc = primalio_write(t, filename);
    cb_fire(t, PRIMAL_CALLBACK_END_WRITE);
    t->last_rc = rc;
    return rc;
}

/* Read a model from a file (extension dispatch), firing the read callbacks
 * and storing the result in last_rc. */
PRIMALrescodee PRIMAL_readdata(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_READ);
    PRIMALrescodee rc = primalio_read(t, filename);
    cb_fire(t, PRIMAL_CALLBACK_END_READ);
    t->last_rc = rc;
    return rc;
}

/* Reference forms around readdata/writedata. `readdataautoformat`
 * detects OPF from content (a file starting with '['), otherwise delegates to
 * `readdata` (extension dispatch); `readdataformat` honours the declared
 * format (0 = by extension, 1/4 MPS, 2 LP, 3 OPF, 7 CBF; 5/6/8 unread)
 * and compression (NONE/FREE accepted, GZIP/ZSTD refused -- no
 * decompression is linked, declared deviation);
 * `readtask`/`writetask` are readdata/writedata. */
PRIMALrescodee PRIMAL_readdataautoformat(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    /* Auto-detect by content: an OPF file starts with a '[' tag, whatever its
     * extension (the reference also auto-detects); anything else falls back to
     * the extension dispatch of PRIMAL_readdata. */
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    int c;
    while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {}
    fclose(f);
    if (c == '[') {
        size_t cap = 65536, len = 0;
        char *buf = (char *)malloc(cap);
        if (!buf) return PRIMAL_RES_ERR_ALLOC;
        f = fopen(filename, "r");
        if (!f) { free(buf); return PRIMAL_RES_ERR_FILE; }
        for (;;) {
            if (len + 4096 + 1 > cap) { cap *= 2; char *nb = (char *)realloc(buf, cap); if (!nb) { free(buf); fclose(f); return PRIMAL_RES_ERR_ALLOC; } buf = nb; }
            size_t got = fread(buf + len, 1, 4096, f);
            len += got;
            if (got < 4096) break;
        }
        fclose(f);
        buf[len] = 0;
        cb_fire(t, PRIMAL_CALLBACK_BEGIN_READ);
        PRIMALrescodee rc = opf_read(t, buf);
        cb_fire(t, PRIMAL_CALLBACK_END_READ);
        free(buf);
        t->last_rc = rc;
        return rc;
    }
    return PRIMAL_readdata(t, filename);
}
/* Read a model honouring the declared format and compression: GZIP/ZSTD are
 * refused because no decompression is linked (declared deviation). */
PRIMALrescodee PRIMAL_readdataformat(PRIMALtask_t t, const char *filename,
                                     PRIMALdataformate format, PRIMALcompresstypee compress) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    /* no decompression is linked: NONE and FREE are the same here, GZIP and ZSTD
     * are refused (declared deviation) */
    if (compress == PRIMAL_COMPRESS_GZIP || compress == PRIMAL_COMPRESS_ZSTD)
        return PRIMAL_RES_ERR_ARG;
    if (compress != PRIMAL_COMPRESS_NONE && compress != PRIMAL_COMPRESS_FREE)
        return PRIMAL_RES_ERR_ARG;
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_READ);
    PRIMALrescodee rc = primalio_read_format(t, filename, (int)format);
    cb_fire(t, PRIMAL_CALLBACK_END_READ);
    t->last_rc = rc;
    return rc;
}
/* Reference alias of PRIMAL_readdata. */
PRIMALrescodee PRIMAL_readtask(PRIMALtask_t t, const char *filename) {
    return PRIMAL_readdata(t, filename);
}
/* Reference alias of PRIMAL_writedata. */
PRIMALrescodee PRIMAL_writetask(PRIMALtask_t t, const char *filename) {
    return PRIMAL_writedata(t, filename);
}

/* Nonzeros of A in the rectangular piece [firsti,lasti) x [firstj,lastj)
 * (reference getapiecenumnz). */
PRIMALrescodee PRIMAL_getapiecenumnz(PRIMALtask_t t, int firsti, int lasti,
                                     int firstj, int lastj, int *numnz) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (firsti < 0 || lasti > t->numcon || firsti > lasti) return PRIMAL_RES_ERR_ARG;
    if (firstj < 0 || lastj > t->numvar || firstj > lastj) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    if (t->cols)
        for (int j = firstj; j < lastj; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] >= firsti && c->sub[k] < lasti) n++;
        }
    *numnz = n;
    return PRIMAL_RES_OK;
}

/* ---------------- optimize ---------------- */
/* Expand a bound key into a finite/infinite lower and upper interval. */
void bound_range(PRIMALboundkeye bk, double bl, double bu, double *lo, double *up) {
    switch (bk) {
        case PRIMAL_BK_LO: *lo = bl; *up = INF; break;
        case PRIMAL_BK_UP: *lo = -INF; *up = bu; break;
        case PRIMAL_BK_FR: *lo = -INF; *up = INF; break;
        case PRIMAL_BK_RA: *lo = bl; *up = bu; break;
        case PRIMAL_BK_FX: *lo = bl; *up = bl; break;
        default: *lo = -INF; *up = INF; break;
    }
}

/* Build the CSC arrays (ptr/sub/val) of the scalar constraint matrix; the
 * caller owns the three allocations. Returns 0 on allocation failure. */
int build_csc(PRIMALtask_t t, int **ptr_out, int **sub_out, double **val_out) {
    int total = 0;
    for (int j = 0; j < t->numvar; j++) total += t->cols[j].nz;
    int *ptr = (int *)malloc((size_t)(t->numvar + 1) * sizeof(int));
    int *sub = (int *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    double *val = (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double));
    if (!ptr || !sub || !val) { free(ptr); free(sub); free(val); return 0; }
    int w = 0;
    for (int j = 0; j < t->numvar; j++) {
        ptr[j] = w;
        for (int k = 0; k < t->cols[j].nz; k++) {
            sub[w] = t->cols[j].sub[k];
            val[w] = t->cols[j].val[k];
            w++;
        }
    }
    ptr[t->numvar] = w;
    *ptr_out = ptr; *sub_out = sub; *val_out = val;
    return 1;
}
