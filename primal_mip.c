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
/* primal_mip.c - MIP shadow copy, node relaxation, cut generators.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* ---- conic/quadratic node relaxation via shadow task ----
 * Builds a copy of the task with the node bounds (lx,ux) applied and
 * solves it with optimize_conic (cones) or optimize_quad (quadratic
 * terms), mapping the solution back. The caller owns `env2` (shared by
 * all node shadows of one B&B run) and frees it at the end.
 * Status mapping: rc OK -> 0, infeasible -> 1, unbounded -> 2, other -> 3. */
static PRIMALrescodee mip_shadow_copy(PRIMALtask_t t, PRIMALenv_t env2,
                                   const double *lx, const double *ux,
                                   PRIMALtask_t *sh_out) {
    PRIMALrescodee rc;
    PRIMALtask_t sh = NULL;
    rc = PRIMAL_maketask(env2, 0, 0, &sh);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMAL_appendvars(sh, t->numvar);
    PRIMAL_appendcons(sh, t->numcon);
    PRIMAL_putobjsense(sh, t->sense);
    PRIMAL_putcfix(sh, t->cfix);
    for (int j = 0; j < t->numvar; j++) {
        /* node bounds: FR/UP/LO/RA according to the (possibly inf) ends */
        PRIMALboundkeye bk;
        if (lx[j] == -INF && ux[j] == INF) bk = PRIMAL_BK_FR;
        else if (lx[j] == -INF) bk = PRIMAL_BK_UP;
        else if (ux[j] == INF) bk = PRIMAL_BK_LO;
        else bk = PRIMAL_BK_RA;
        PRIMAL_putvarbound(sh, j, bk, lx[j], ux[j]);
        PRIMAL_putcj(sh, j, t->c[j]);
    }
    for (int i = 0; i < t->numcon; i++)
        PRIMAL_putconbound(sh, i, t->bkc[i], t->blc[i], t->buc[i]);
    for (int j = 0; j < t->numvar; j++) {
        int nz = t->cols[j].nz;
        int *sub = (int *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int));
        double *val = (double *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(double));
        if (!sub || !val) { free(sub); free(val); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
        for (int k = 0; k < nz; k++) { sub[k] = t->cols[j].sub[k]; val[k] = t->cols[j].val[k]; }
        PRIMAL_putacol(sh, j, nz, sub, val);
        free(sub); free(val);
    }
    for (int k = 0; k < t->numcones; k++)
        PRIMAL_appendcone(sh, (PRIMALconetypee)t->cone_type[k], t->cone_param[k],
                       t->cone_nmem[k], t->cone_mem[k]);
    if (t->has_qobj) {
        /* copy the sparse Q triplets directly (no dense materialization) */
        PRIMAL_putqobj(sh, t->qt_n, t->qt_i, t->qt_j, t->qt_v);
    }
    for (int i = 0; i < t->numcon; i++) {
        if (!t->qcon || !t->qcon[i]) continue;
        int cap = 64, n = 0;
        int *qi = (int *)malloc((size_t)cap * sizeof(int));
        int *qj = (int *)malloc((size_t)cap * sizeof(int));
        double *qv = (double *)malloc((size_t)cap * sizeof(double));
        if (!qi || !qj || !qv) { free(qi); free(qj); free(qv); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
        for (int a = 0; a < t->numvar; a++)
            for (int b = a; b < t->numvar; b++) {
                double v = t->qcon[i][a * t->numvar + b];
                if (v == 0.0) continue;
                if (n == cap) {
                    cap *= 2;
                    /* Grow one array at a time and update the pointer right
                     * away: realloc frees the old block on success, so freeing
                     * both the old and the new pointer (as the batched form
                     * did) double-frees whichever realloc succeeded. */
                    int *i2 = (int *)realloc(qi, (size_t)cap * sizeof(int));
                    if (!i2) { free(qi); free(qj); free(qv); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
                    qi = i2;
                    int *j2 = (int *)realloc(qj, (size_t)cap * sizeof(int));
                    if (!j2) { free(qi); free(qj); free(qv); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
                    qj = j2;
                    double *v2 = (double *)realloc(qv, (size_t)cap * sizeof(double));
                    if (!v2) { free(qi); free(qj); free(qv); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
                    qv = v2;
                }
                qi[n] = a; qj[n] = b; qv[n] = v; n++;
            }
        if (n > 0) PRIMAL_putqconk(sh, i, n, qi, qj, qv);
        free(qi); free(qj); free(qv);
    }
    /* bar store (SDP MIP): symmetric registry, bar variables, barA and
     * barC. Registry ids are per-task, so the ones returned by the
     * shadow task are remapped here. */
    if (t->numbarvar > 0) {
        int *symmap = (int *)malloc((size_t)(t->nsym > 0 ? t->nsym : 1) * sizeof(int));
        int *bdim = (int *)malloc((size_t)t->numbarvar * sizeof(int));
        if (!symmap || !bdim) { free(symmap); free(bdim); PRIMAL_deletetask(&sh); return PRIMAL_RES_ERR_ALLOC; }
        for (int m = 0; m < t->nsym; m++) {
            int ni = -1;
            rc = PRIMAL_appendsparsesymmat(sh, t->sym_dim[m], t->sym_nnz[m],
                                           t->sym_subi[m], t->sym_subj[m], t->sym_val[m], &ni);
            if (rc != PRIMAL_RES_OK) { free(symmap); free(bdim); PRIMAL_deletetask(&sh); return rc; }
            symmap[m] = ni;
        }
        for (int j = 0; j < t->numbarvar; j++) bdim[j] = t->barDim[j];
        rc = PRIMAL_appendbarvars(sh, t->numbarvar, bdim);
        if (rc == PRIMAL_RES_OK)
            for (int k = 0; k < t->nbarA && rc == PRIMAL_RES_OK; k++) {
                int mi = symmap[t->barA_sym[k]];
                rc = PRIMAL_putbaraij(sh, t->barA_con[k], t->barA_bar[k], 1, &mi, &t->barA_coef[k]);
            }
        if (rc == PRIMAL_RES_OK)
            for (int k = 0; k < t->nbarC && rc == PRIMAL_RES_OK; k++) {
                int mi = symmap[t->barC_sym[k]];
                rc = PRIMAL_putbarcj(sh, t->barC_bar[k], 1, &mi, &t->barC_coef[k]);
            }
        free(symmap); free(bdim);
        if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&sh); return rc; }
    }
    *sh_out = sh;
    return PRIMAL_RES_OK;
}

/* Solve the conic/quadratic node relaxation through a shadow task.
 * Applies the node bounds, routes to the conic or quadratic path,
 * and maps the solution back. Returns 0/1/2/3 status. */
int mip_relax_conic(PRIMALtask_t t, int s, PRIMALenv_t env2,
                           const double *lx, const double *ux,
                           const double *lc, const double *uc,
                           double *xout, double *pmin, double *barX_out) {
    (void)lc; (void)uc;   /* row bounds are copied unchanged into the shadow */
    PRIMALtask_t sh = NULL;
    PRIMALrescodee rc = mip_shadow_copy(t, env2, lx, ux, &sh);
    if (rc != PRIMAL_RES_OK) return 3;
    param_copy(sh, t);
    rc = opt_prepare(sh);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&sh); return 3; }
    if (t->has_qcon > 0 || (t->has_qobj && (t->numcones > 0 || t->numbarvar > 0)))
        rc = optimize_quad(sh, s);
    else if (t->numbarvar > 0) {
        /* SDP MIP: the node is an SDP (bars + optional cones), solved by the
         * SDP IPM; if it does not answer, the same route as the dispatcher
         * (cuts on the bar alone, or the conic build when cones are present). */
        rc = optimize_sdp_ipm(sh, s);
        if (!(rc == PRIMAL_RES_OK || rc == PRIMAL_RES_ERR_INFEASIBLE ||
              rc == PRIMAL_RES_ERR_UNBOUNDED))
            rc = (sh->numcones == 0) ? optimize_sdp(sh, s) : optimize_conic(sh, s);
    } else
        rc = optimize_conic(sh, s);
    int status;
    if (rc == PRIMAL_RES_OK && sh->has_sol && sh->solsta == PRIMAL_SOL_STA_OPTIMAL) {
        memcpy(xout, sh->x, (size_t)t->numvar * sizeof(double));
        /* min-form objective from the shadow primal obj (original sense) */
        *pmin = s * sh->pobj;
        if (barX_out && sh->numbarvar == t->numbarvar) {
            int off = 0;
            for (int j = 0; j < sh->numbarvar; j++) {
                int d2 = sh->barDim[j] * sh->barDim[j];
                memcpy(barX_out + off, sh->barx[j], (size_t)d2 * sizeof(double));
                off += d2;
            }
        }
        status = 0;
    } else if (rc == PRIMAL_RES_ERR_INFEASIBLE) status = 1;
    else if (rc == PRIMAL_RES_ERR_UNBOUNDED) status = 2;
    else status = 3;
    PRIMAL_deletetask(&sh);
    return status;
}

/* Forward declaration: the cone slack lives with the conic reporting helpers. */

/* The value of a putqconk row at w, in the user's own form and with the sign
 * convention quad_encode_task uses: a'w + sgn*1/2 w'Qw, sgn = +1 on UP and -1
 * on LO. Shared by the incumbent test and by the [cones] publication measure so
 * the two cannot drift apart. */
double quad_row_value(const PRIMALtask_t t, int i, const double *w) {
    int n = t->numvar;
    double lin = 0.0, q = 0.0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < t->cols[j].nz; k++)
            if (t->cols[j].sub[k] == i) lin += t->cols[j].val[k] * w[j];
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++) q += w[a] * t->qcon[i][(size_t)a * n + b] * w[b];
    /* The row expression the USER wrote is a'x + 0.5 x'Qx for either bound
     * direction (issue #17). The stored qcon keeps the user's sign; the internal
     * flip a solver may apply to a '>=' row is not part of the exposed value. */
    return lin + 0.5 * q;
}

/* Does w measure as an integer-feasible point of THIS model: bounds, rows,
 * cones, quadratic rows, semi-continuous/semi-integer sets and SOS sets, all
 * within ftol, with integrality within itol?
 *
 * This is the check the branch-and-bound used not to have. The integrality
 * threshold declares an integer constraint satisfied when the LP value is
 * within itol of a whole number, and that declaration relaxes every row the
 * variable appears in -- for a disjunction written with big-M it relaxes the
 * disjunction itself. A leaf's LP value is therefore an incumbent candidate,
 * not an incumbent; what goes on record is the rounded point, and only if the
 * model accepts it. (MSK_DPAR_MIO_TOL_FEAS is the reference's name for ftol, and
 * its documentation says exactly this: the integer tolerance assumes a
 * constraint satisfied, feasibility is a separate question.) */
int mip_point_measures(PRIMALtask_t t, const double *lx, const double *ux,
                              const double *lc, const double *uc, const double *w,
                              double ftol, double itol) {
    int nvar = t->numvar, ncon = t->numcon;
    for (int j = 0; j < nvar; j++) {
        int vt = t->vartype[j];
        int semi = (vt == PRIMAL_VAR_TYPE_SEMI_CONT || vt == PRIMAL_VAR_TYPE_SEMI_INT);
        /* The disjunctive domain of a semi variable is {0} union [l, up], and
         * l is also what its lx slot holds (blx is the activation level), so an
         * INACTIVE semi variable sits below its own lower bound by definition.
         * Testing the bounds only when active is what makes the deactivation
         * branch publishable -- it is not a relaxation of the bound. */
        int active = !semi || (w[j] > ftol);
        if (active && (w[j] < lx[j] - ftol || w[j] > ux[j] + ftol)) return 0;
        if (semi && active && w[j] < t->blx[j] - ftol) return 0;
        int integ = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                     (vt == PRIMAL_VAR_TYPE_SEMI_INT && active));
        if (integ) {
            double fr = w[j] - floor(w[j] + 1e-9);
            if (fr > itol && 1.0 - fr > itol) return 0;
        }
    }
    double *rs = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    if (!rs) return 0;
    for (int j = 0; j < nvar; j++) {
        const Col *cj = &t->cols[j];
        for (int k = 0; k < cj->nz; k++) rs[cj->sub[k]] += cj->val[k] * w[j];
    }
    for (int i = 0; i < ncon; i++)
        if (rs[i] < lc[i] - ftol || rs[i] > uc[i] + ftol) { free(rs); return 0; }
    free(rs);

    if (t->has_qcon > 0 && t->qcon) {
        for (int i = 0; i < ncon; i++) {
            if (!t->qcon[i]) continue;
            double lo, up, v = quad_row_value(t, i, w);
            bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
            if (isfinite(up) && v > up + ftol) return 0;
            if (isfinite(lo) && v < lo - ftol) return 0;
        }
    }
    if (t->numcones > 0) {
        int maxnk = 1;
        for (int k = 0; k < t->numcones; k++)
            if (t->cone_nmem[k] > maxnk) maxnk = t->cone_nmem[k];
        double *v = (double *)calloc((size_t)maxnk, sizeof(double));
        if (!v) return 0;
        for (int k = 0; k < t->numcones; k++) {
            int nk = t->cone_nmem[k];
            const int *mi = t->cone_mem[k];
            for (int i = 0; i < nk; i++) v[i] = w[mi[i]];
            if (cone_signed_slack(t->cone_type[k], t->cone_param[k], v, nk) < -ftol)
                { free(v); return 0; }
        }
        free(v);
    }
    for (int k = 0; k < t->numsos; k++) {
        const int *mem = t->sos_mem[k];
        const double *swt = t->sos_w[k];
        int n = t->sos_n[k];
        int *idx = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
        if (!idx) return 0;
        for (int q = 0; q < n; q++) idx[q] = q;
        for (int q = 1; q < n; q++) {           /* insertion sort by weight */
            int key = idx[q];
            double kw = swt[key];
            int p = q - 1;
            while (p >= 0 && swt[idx[p]] > kw) { idx[p + 1] = idx[p]; p--; }
            idx[p + 1] = key;
        }
        int bad = 0;
        if (t->sos_type[k] == 1) {
            /* Signed support (issue #16): a negative component is nonzero. */
            int nz = 0;
            for (int q = 0; q < n; q++) if (fabs(w[mem[idx[q]]]) > ftol) nz++;
            bad = (nz > 1);
        } else {
            /* SOS2 keeps at most two ADJACENT nonzero members: a run of three
             * consecutive nonzero members violates it. */
            int run = 0;
            for (int q = 0; q < n && !bad; q++) {
                if (fabs(w[mem[idx[q]]]) > ftol) { if (++run > 2) bad = 1; }
                else run = 0;
            }
        }
        free(idx);
        if (bad) return 0;
    }
    return 1;
}

/* Total number of bar entries (d*d per block) and block offset of bar b
 * in the flat buffer; <C_bar, X> with flat X and the symmetric registry. */
int bar_tot(const PRIMALtask_t t) {
    int n = 0;
    for (int j = 0; j < t->numbarvar; j++) n += t->barDim[j] * t->barDim[j];
    return n;
}
/* Offset of bar block b inside the flat bar buffer. Sums d*d over
 * earlier blocks; used together with bar_tot. */
static int bar_off(const PRIMALtask_t t, int b) {
    int o = 0;
    for (int j = 0; j < b; j++) o += t->barDim[j] * t->barDim[j];
    return o;
}
/* Inner product of the bar objective terms with a flat bar point X.
 * Sums coef * <symmat, X_b> over every barC entry. Returns 0 for NULL. */
double barC_dot(const PRIMALtask_t t, const double *X) {
    if (!X) return 0.0;
    double v = 0.0;
    for (int k = 0; k < t->nbarC; k++) {
        int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b];
        const double *Xb = X + bar_off(t, b);
        double tr = 0.0;
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            int si = t->sym_subi[m][e], sj = t->sym_subj[m][e];
            double sv = t->sym_val[m][e];
            tr += (si == sj ? sv : 2.0 * sv) * Xb[si * d + sj];
        }
        v += t->barC_coef[k] * tr;
    }
    return v;
}

/* Conflict cut from pairwise probing (gated by GMB_MIP_CONFLICT, default off).
 * For each pair of binaries (j,k), among the first 8: fix x_j = x_k = 1 and
 * solve the LP relaxation of the ORIGINAL task (on t, not on the clone: the
 * clone does not exist yet at this point). If it is infeasible (st == 1),
 * the conflict x_j + x_k <= 1 is VALID: no feasible solution has both at 1,
 * and removing that region removes no feasible integer point. It is the
 * pairwise extension of single probing (which fixes x_j when ONE side is
 * infeasible); it covers conflicts that no single row sees (e.g. two rows
 * that together exclude the pair but neither alone). The cuts are rows
 * x_j + x_k <= 1 added to the clone like any other cut, so they enter the
 * "toxic cut" check (the relaxation with cuts must not worsen the bound)
 * and the ncuts count. Dual-safe: the MIP publishes no duals. Only INT_BIN
 * variables with task bounds [0,1] (not node bounds: this is root separation,
 * before lx/ux exist). Cones/bars/quadratics: mip_relax already handles them
 * (bound_tighten skips bar/quadratics, stdform rejects cones -> st 3, never 1:
 * no false conflict from a relaxation that gives no answer).
 * Signature: takes the row bounds lc/uc (ncon) already computed by the caller
 * and the (cidx/cval/cnnz/clo/cup, ncut/cap) lists to append to — path (a) of
 * the note: conflicts are computed BEFORE the clone and travel with the
 * other cuts. */

static int mip_conflict_cuts(PRIMALtask_t t, int s, const double *lc, const double *uc,
                             int ***cidx, double ***cval, int **cnnz,
                             double **clo, double **cup, int *ncut, int *cap) {
    int nvar = t->numvar;
    int bins[8], nbin = 0;
    for (int j = 0; j < nvar && nbin < 8; j++)
        if (t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN) bins[nbin++] = j;
    if (nbin < 2) return 0;
    double *lx = (double *)malloc((size_t)nvar * sizeof(double));
    double *ux = (double *)malloc((size_t)nvar * sizeof(double));
    double *xo = (double *)malloc((size_t)nvar * sizeof(double));
    if (!lx || !ux || !xo) { free(lx); free(ux); free(xo); return 0; }
    for (int j = 0; j < nvar; j++)
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    int nadd = 0;
    for (int a = 0; a < nbin; a++) {
        for (int b = a + 1; b < nbin; b++) {
            int j = bins[a], k = bins[b];
            double oj = lx[j], uj = ux[j], ok = lx[k], uk = ux[k];
            lx[j] = ux[j] = 1.0; lx[k] = ux[k] = 1.0;
            double z = 0.0;
            int st = mip_relax(t, s, lx, ux, lc, uc, xo, &z);
            lx[j] = oj; ux[j] = uj; lx[k] = ok; ux[k] = uk;
            if (st != 1) continue;   /* only the infeasible is a proven conflict */
            if (*ncut == *cap) {
                int nc2 = *cap ? *cap * 2 : 8;
                int **ci2 = (int **)realloc(*cidx, (size_t)nc2 * sizeof(int *));
                double **cv2 = (double **)realloc(*cval, (size_t)nc2 * sizeof(double *));
                int *cn2 = (int *)realloc(*cnnz, (size_t)nc2 * sizeof(int));
                double *cl2 = (double *)realloc(*clo, (size_t)nc2 * sizeof(double));
                double *cu2 = (double *)realloc(*cup, (size_t)nc2 * sizeof(double));
                if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) {
                    free(lx); free(ux); free(xo);
                    return nadd;
                }
                *cidx = ci2; *cval = cv2; *cnnz = cn2; *clo = cl2; *cup = cu2;
                *cap = nc2;
            }
            int q = *ncut;
            (*cidx)[q] = (int *)malloc(2 * sizeof(int));
            (*cval)[q] = (double *)malloc(2 * sizeof(double));
            if (!(*cidx)[q] || !(*cval)[q]) {
                free((*cidx)[q]); free((*cval)[q]);
                free(lx); free(ux); free(xo);
                return nadd;
            }
            (*cidx)[q][0] = j; (*cval)[q][0] = 1.0;
            (*cidx)[q][1] = k; (*cval)[q][1] = 1.0;
            (*cnnz)[q] = 2; (*clo)[q] = -INFINITY; (*cup)[q] = 1.0;
            (*ncut)++;
            nadd++;
        }
    }
    free(lx); free(ux); free(xo);
    return nadd;
}

/* Chvatal-Gomory cuts from a row. For `a'x <= u` with every row variable
 * integer and `x >= 0`, `sum_j floor(a_j) x_j <= floor(u)` is valid:
 * `floor(a_j) x_j <= a_j x_j` (x_j >= 0) gives LHS <= a'x <= u, and the LHS
 * is integer. Symmetrically `a'x >= l` gives `sum_j ceil(a_j) x_j >= ceil(l)`.
 * A row with already-integer coefficients and RHS produces the cut equal to
 * itself and is skipped (no effect on the wired corpus). The cuts are added
 * to a COPY of the task used only by the relaxations: the published model
 * (numcon, A, bounds) stays the user's one.
 * Returns the TOTAL number of cuts (conflict cuts included when the gate is
 * on; the caller separates them for the log). */
int mip_build_cuts(PRIMALtask_t t, int s,
                           const double *lc0, const double *uc0,
                           PRIMALtask_t *tc_out, int *nconf_out) {
    *tc_out = NULL;
    if (nconf_out) *nconf_out = 0;
    if (getenv("GMB_NO_MIP_CUTS")) return 0;   /* off for the tests measuring the node cap */
    int nvar = t->numvar, ncon = t->numcon;
    if (ncon <= 0 || nvar <= 0) return 0;
    int nnz = 0;
    for (int j = 0; j < nvar; j++) nnz += t->cols[j].nz;
    if (nnz <= 0) return 0;
    int *rptr = (int *)calloc((size_t)ncon + 1, sizeof(int));
    int *rsub = (int *)malloc((size_t)nnz * sizeof(int));
    double *rval = (double *)malloc((size_t)nnz * sizeof(double));
    int *fill = (int *)calloc((size_t)ncon, sizeof(int));
    if (!rptr || !rsub || !rval || !fill) { free(rptr); free(rsub); free(rval); free(fill); return 0; }
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) rptr[t->cols[j].sub[q] + 1]++;
    for (int i = 0; i < ncon; i++) rptr[i + 1] += rptr[i];
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) {
            int i = t->cols[j].sub[q], p = rptr[i] + fill[i]++;
            rsub[p] = j; rval[p] = t->cols[j].val[q];
        }
    free(fill);
    /* collect the cuts in dynamic lists. Pairwise-probing conflict cuts
     * (via (a): before the clone) travel with the others: lc0/uc0 are the
     * task row bounds (ncon), computed by the caller. */
    int ncut = 0, cap = 0;
    int **cidx = NULL; double **cval = NULL; int *cnnz = NULL;
    double *clo = NULL, *cup = NULL;
    int ok = 1;
    int nconf = 0;
    for (int i = 0; i < ncon && ok; i++) {
        int b0 = rptr[i], b1 = rptr[i + 1];
        if (b1 <= b0) continue;
        int allint = 1;
        for (int p = b0; p < b1 && allint; p++) {
            int j = rsub[p];
            if (t->vartype[j] == PRIMAL_VAR_TYPE_CONT || t->blx[j] < 0.0) allint = 0;
        }
        if (!allint) continue;
        PRIMALboundkeye bk = t->bkc[i];
        int do_up = (bk == PRIMAL_BK_UP || bk == PRIMAL_BK_RA || bk == PRIMAL_BK_FX);
        int do_lo = (bk == PRIMAL_BK_LO || bk == PRIMAL_BK_RA || bk == PRIMAL_BK_FX);
        double u = t->buc[i], l = t->blc[i];
        for (int side = 0; side < 2 && ok; side++) {
            int up = (side == 0) ? do_up : do_lo;
            if (!up) continue;
            double rhs = (side == 0) ? floor(u) : ceil(l);
            if (!isfinite(rhs)) continue;
            /* the cut is the row itself when every coefficient and the RHS
             * are already integer: skip it (no effect on the wired corpus) */
            int redundant = (side == 0) ? (rhs == u) : (rhs == l);
            if (redundant) {
                redundant = 1;
                for (int p = b0; p < b1; p++) {
                    double a = rval[p];
                    if ((side == 0 ? floor(a) : ceil(a)) != a) { redundant = 0; break; }
                }
            }
            if (redundant) continue;
            if (ncut == cap) {
                cap = cap ? cap * 2 : 8;
                int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
                double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
                int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
                double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
                double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
                if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; break; }
                cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
            }
            int k = ncut, m = b1 - b0;
            cidx[k] = (int *)malloc((size_t)m * sizeof(int));
            cval[k] = (double *)malloc((size_t)m * sizeof(double));
            if (!cidx[k] || !cval[k]) { ok = 0; break; }
            for (int p = b0; p < b1; p++) {
                cidx[k][p - b0] = rsub[p];
                cval[k][p - b0] = (side == 0) ? floor(rval[p]) : ceil(rval[p]);
            }
            cnnz[k] = m;
            clo[k] = (side == 0) ? -INF : rhs;   /* >= ceil(l) or <= floor(u) */
            cup[k] = (side == 0) ? rhs : INF;
            ncut++;
        }
    }
    /* cover cuts: for a knapsack row a'x <= u with all x binary and
     * a_j > 0, a cover C (sum_{j in C} a_j > u) gives the valid inequality
     * sum_{j in C} x_j <= |C| - 1. Greedy cover by decreasing coefficient;
     * the trivial cover (the whole row) is skipped. */
    for (int i = 0; i < ncon && ok; i++) {
        if (t->bkc[i] != PRIMAL_BK_UP && t->bkc[i] != PRIMAL_BK_RA) continue;
        double u = t->buc[i];
        if (!isfinite(u)) continue;
        int b0 = rptr[i], b1 = rptr[i + 1];
        int m = b1 - b0;
        if (m <= 1) continue;
        int allbin = 1;
        for (int p = b0; p < b1 && allbin; p++)
            if (t->vartype[rsub[p]] != PRIMAL_VAR_TYPE_INT_BIN || rval[p] <= 0.0) allbin = 0;
        if (!allbin) continue;
        int *ord = (int *)malloc((size_t)m * sizeof(int));
        if (!ord) { ok = 0; break; }
        for (int q = 0; q < m; q++) ord[q] = b0 + q;
        for (int q = 1; q < m; q++) {          /* insertion sort by descending a */
            int key = ord[q];
            int pp = q - 1;
            while (pp >= 0 && rval[ord[pp]] < rval[key]) { ord[pp + 1] = ord[pp]; pp--; }
            ord[pp + 1] = key;
        }
        double ssum = 0.0; int csize = 0;
        for (int q = 0; q < m; q++) {
            ssum += rval[ord[q]]; csize++;
            if (ssum > u) break;
        }
        int use = (ssum > u && csize < m);
        if (use) {
            if (ncut == cap) {
                cap = cap ? cap * 2 : 8;
                int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
                double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
                int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
                double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
                double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
                if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; free(ord); break; }
                cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
            }
            int k = ncut;
            cidx[k] = (int *)malloc((size_t)csize * sizeof(int));
            cval[k] = (double *)malloc((size_t)csize * sizeof(double));
            if (!cidx[k] || !cval[k]) { ok = 0; free(ord); break; }
            for (int q = 0; q < csize; q++) { cidx[k][q] = rsub[ord[q]]; cval[k][q] = 1.0; }
            cnnz[k] = csize;
            clo[k] = -INF; cup[k] = csize - 1;
            ncut++;
        }
        free(ord);
    }
    /* clique cuts: for a knapsack row a'x <= u with all x binary and
     * a_j > 0, the set C = {j : a_j > u/2} is a clique: every pair has
     * a_j + a_k > u, so sum_{j in C} x_j <= 1 is valid (and stronger than
     * the greedy cover when |C| >= 3). Skipped when |C| < 2. */
    for (int i = 0; i < ncon && ok; i++) {
        if (t->bkc[i] != PRIMAL_BK_UP && t->bkc[i] != PRIMAL_BK_RA) continue;
        double u = t->buc[i];
        if (!isfinite(u) || u < 0.0) continue;
        int b0 = rptr[i], b1 = rptr[i + 1];
        if (b1 - b0 < 2) continue;
        int allbin = 1;
        for (int p = b0; p < b1 && allbin; p++)
            if (t->vartype[rsub[p]] != PRIMAL_VAR_TYPE_INT_BIN || rval[p] <= 0.0) allbin = 0;
        if (!allbin) continue;
        int cliq = 0;
        for (int p = b0; p < b1; p++) if (rval[p] > u / 2.0) cliq++;
        if (cliq < 2) continue;
        if (ncut == cap) {
            cap = cap ? cap * 2 : 8;
            int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
            double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
            int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
            double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
            double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
            if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; break; }
            cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
        }
        int k = ncut;
        cidx[k] = (int *)malloc((size_t)cliq * sizeof(int));
        cval[k] = (double *)malloc((size_t)cliq * sizeof(double));
        if (!cidx[k] || !cval[k]) { ok = 0; break; }
        int t2 = 0;
        for (int p = b0; p < b1; p++)
            if (rval[p] > u / 2.0) { cidx[k][t2] = rsub[p]; cval[k][t2] = 1.0; t2++; }
        cnnz[k] = cliq;
        clo[k] = -INF; cup[k] = 1.0;
        ncut++;
    }
    /* Symmetry: two IDENTICAL binary columns (same ordered row pattern and
     * same cost) are interchangeable -> an optimum with x_k <= x_j exists. Add
     * x_k - x_j <= 0, valid (does not cut off every optimum). One partner
     * per column. */
    for (int j = 0; j < nvar && ok; j++) {
        if (t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) continue;
        for (int k = j + 1; k < nvar; k++) {
            if (t->vartype[k] != PRIMAL_VAR_TYPE_INT_BIN) continue;
            if (t->c[j] != t->c[k]) continue;
            if (t->cols[j].nz != t->cols[k].nz) continue;
            int same = 1;
            for (int q = 0; q < t->cols[j].nz && same; q++)
                if (t->cols[j].sub[q] != t->cols[k].sub[q] ||
                    t->cols[j].val[q] != t->cols[k].val[q]) same = 0;
            if (!same) continue;
            if (ncut == cap) {
                cap = cap ? cap * 2 : 8;
                int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
                double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
                int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
                double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
                double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
                if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; break; }
                cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
            }
            int kk = ncut;
            cidx[kk] = (int *)malloc(2 * sizeof(int));
            cval[kk] = (double *)malloc(2 * sizeof(double));
            if (!cidx[kk] || !cval[kk]) { ok = 0; break; }
            cidx[kk][0] = k; cval[kk][0] = 1.0;
            cidx[kk][1] = j; cval[kk][1] = -1.0;
            cnnz[kk] = 2;
            clo[kk] = -INF; cup[kk] = 0.0;
            ncut++;
            break;
        }
    }
    /* CMIR (complemented MIR): for a row a'x <= u with all x binary, the
     * negative coefficients are complemented (x_j -> 1-x_j) giving a'_j >= 0
     * and u' = u - sum_{a_j<0} a_j, then the MIR inequality
     *   sum_j mu(a'_j) x_j <= floor(u'),  mu(a) = floor(a) + max(0, frac(a)-f)/(1-f),
     * with f = frac(u'). Gated by GMB_MIP_CMIR (default off: validity risk). */
    if (getenv("GMB_MIP_CMIR"))
    for (int i = 0; i < ncon && ok; i++) {
        if (t->bkc[i] != PRIMAL_BK_UP && t->bkc[i] != PRIMAL_BK_RA) continue;
        double u = t->buc[i];
        if (!isfinite(u)) continue;
        int b0 = rptr[i], b1 = rptr[i + 1];
        if (b1 <= b0) continue;
        int allbin = 1;
        for (int p = b0; p < b1 && allbin; p++)
            if (t->vartype[rsub[p]] != PRIMAL_VAR_TYPE_INT_BIN) allbin = 0;
        if (!allbin) continue;
        double up = u;
        for (int p = b0; p < b1; p++) if (rval[p] < 0.0) up -= rval[p];
        double f = up - floor(up);
        if (f < 1e-9 || f > 1.0 - 1e-9) continue;
        if (ncut == cap) {
            cap = cap ? cap * 2 : 8;
            int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
            double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
            int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
            double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
            double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
            if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; break; }
            cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
        }
        int k = ncut, m = b1 - b0;
        cidx[k] = (int *)malloc((size_t)m * sizeof(int));
        cval[k] = (double *)malloc((size_t)m * sizeof(double));
        if (!cidx[k] || !cval[k]) { ok = 0; break; }
        for (int p = b0; p < b1; p++) {
            double a = rval[p] < 0.0 ? -rval[p] : rval[p];
            double fa = a - floor(a);
            double mu = floor(a) + (fa > f ? (fa - f) / (1.0 - f) : 0.0);
            cidx[k][p - b0] = rsub[p];
            cval[k][p - b0] = mu;
        }
        cnnz[k] = m;
        clo[k] = -INF; cup[k] = floor(up);
        ncut++;
    }
    /* LIPRO (lifted CG variant): like CMIR but with pure Chvatal-Gomory
     * rounding after complementing the negative coefficients
     * (x_j -> 1-x_j, a'_j >= 0, u' = u - sum_{a_j<0} a_j):
     *   sum_j floor(a'_j) x_j <= floor(u').
     * Gated by GMB_MIP_LIPRO (default off: validity risk). */
    if (getenv("GMB_MIP_LIPRO"))
    for (int i = 0; i < ncon && ok; i++) {
        if (t->bkc[i] != PRIMAL_BK_UP && t->bkc[i] != PRIMAL_BK_RA) continue;
        double u = t->buc[i];
        if (!isfinite(u)) continue;
        int b0 = rptr[i], b1 = rptr[i + 1];
        if (b1 <= b0) continue;
        int allbin = 1;
        for (int p = b0; p < b1 && allbin; p++)
            if (t->vartype[rsub[p]] != PRIMAL_VAR_TYPE_INT_BIN) allbin = 0;
        if (!allbin) continue;
        double up = u;
        for (int p = b0; p < b1; p++) if (rval[p] < 0.0) up -= rval[p];
        if (ncut == cap) {
            cap = cap ? cap * 2 : 8;
            int **ci2 = (int **)realloc(cidx, (size_t)cap * sizeof(int *));
            double **cv2 = (double **)realloc(cval, (size_t)cap * sizeof(double *));
            int *cn2 = (int *)realloc(cnnz, (size_t)cap * sizeof(int));
            double *cl2 = (double *)realloc(clo, (size_t)cap * sizeof(double));
            double *cu2 = (double *)realloc(cup, (size_t)cap * sizeof(double));
            if (!ci2 || !cv2 || !cn2 || !cl2 || !cu2) { ok = 0; break; }
            cidx = ci2; cval = cv2; cnnz = cn2; clo = cl2; cup = cu2;
        }
        int k = ncut, m = b1 - b0;
        cidx[k] = (int *)malloc((size_t)m * sizeof(int));
        cval[k] = (double *)malloc((size_t)m * sizeof(double));
        if (!cidx[k] || !cval[k]) { ok = 0; break; }
        for (int p = b0; p < b1; p++) {
            double a = rval[p] < 0.0 ? -rval[p] : rval[p];
            cidx[k][p - b0] = rsub[p];
            cval[k][p - b0] = floor(a);
        }
        cnnz[k] = m;
        clo[k] = -INF; cup[k] = floor(up);
        ncut++;
    }
    /* PARTITION-BOUND CUT + objective-box tightening */
    for (int ie = 0; ie < ncon && ok; ie++) {
        if (!(t->bkc[ie] == PRIMAL_BK_FX && t->blc[ie] == 1.0 && t->buc[ie] == 1.0)) continue;
        int e0 = rptr[ie], e1 = rptr[ie + 1];
        if (e1 - e0 < 2) continue;
        int okE = 1;
        for (int p = e0; p < e1 && okE; p++) if (t->vartype[rsub[p]] != PRIMAL_VAR_TYPE_INT_BIN || rval[p] != 1.0) okE = 0;
        if (!okE) continue;
        for (int y = 0; y < nvar && ok; y++) {
            int inE = 0; for (int p = e0; p < e1; p++) if (rsub[p] == y) inE = 1;
            if (inE) continue;
            double d = -1.0, lrhs = 0.0; int okY = 1, nseen = 0;
            for (int p = e0; p < e1 && okY; p++) {
                int x = rsub[p]; int found = 0;
                for (int i = 0; i < ncon && !found; i++) {
                    if (i == ie) continue;
                    if (t->bkc[i] != PRIMAL_BK_LO && t->bkc[i] != PRIMAL_BK_RA) continue;
                    if (!isfinite(t->blc[i])) continue;
                    int q0 = rptr[i], q1 = rptr[i + 1];
                    double cy = 0.0, cx = 0.0; int bad = 0;
                    for (int q = q0; q < q1; q++) { int jj = rsub[q]; double a = rval[q];
                        if (jj == y) { cy += a; if (cy > 1.0 + 1e-12) bad = 1; }
                        else if (jj == x) { cx += a; if (cx > 1e-12) bad = 1; }
                        else if (a > 1e-12 && t->blx[jj] < -1e-12) bad = 1; }
                    if (bad || cy != 1.0 || cx >= -1e-12) continue;
                    double dd = -cx;
                    if (nseen == 0) { d = dd; lrhs = t->blc[i]; }
                    else if (fabs(dd - d) > 1e-9 || fabs(t->blc[i] - lrhs) > 1e-9) continue;
                    found = 1;
                }
                if (!found) okY = 0; else nseen++;
            }
            if (!okY || nseen < 2 || d <= 1e-9) continue;
            /* the valid bound on y, applied to the MODEL box (the B&B reads it) */
            double y0 = lrhs + d;
            if (t->bkx[y] == PRIMAL_BK_FX && t->blx[y] >= y0 - 1e-9) continue;
            if (y0 > t->blx[y] + 1e-9) { t->blx[y] = y0; if (t->bkx[y] == PRIMAL_BK_UP) t->bkx[y] = PRIMAL_BK_RA;
                                          else if (t->bkx[y] == PRIMAL_BK_FR) t->bkx[y] = PRIMAL_BK_LO; }
            if (ncut == cap) { cap = cap ? cap*2 : 8;
                cidx=(int**)realloc(cidx,(size_t)cap*sizeof(int*)); cval=(double**)realloc(cval,(size_t)cap*sizeof(double*));
                cnnz=(int*)realloc(cnnz,(size_t)cap*sizeof(int)); clo=(double*)realloc(clo,(size_t)cap*sizeof(double)); cup=(double*)realloc(cup,(size_t)cap*sizeof(double));
                if(!cidx||!cval||!cnnz||!clo||!cup){ok=0;break;} }
            int k=ncut; cidx[k]=(int*)malloc(sizeof(int)); cval[k]=(double*)malloc(sizeof(double));
            if(!cidx[k]||!cval[k]){ok=0;break;}
            cidx[k][0]=y; cval[k][0]=1.0; cnnz[k]=1; clo[k]=y0; cup[k]=INF; ncut++;
        }
    }
    /* Pairwise-probing conflict cuts: generated LAST, so their dedup can
     * compare them against the classic cuts already collected (a conflict
     * x_j+x_k<=1 and the CG/cover of the same disjunction are the same row,
     * and a duplicated row makes the equality system rank-deficient on the
     * conic route). The order in the clone does not matter: cuts are
     * trailing rows, numbering is unchanged. With the gate off (default)
     * this block does not run and the classic path is bit-for-bit the
     * previous one. */
    int nconf_start = ncut;
    if (getenv("GMB_MIP_CONFLICT") && lc0 && uc0)
        nconf = mip_conflict_cuts(t, s, lc0, uc0, &cidx, &cval, &cnnz, &clo, &cup,
                                  &ncut, &cap);
    /* drop cuts that merely duplicate an existing row: a duplicated row makes
     * the standard-form equality system rank-deficient, and the conic IPM can
     * then lose the relaxation entirely (NaN).  A cut equal to the row it was
     * derived from is redundant by construction. */
    {
        int keep = 0;
        for (int k = 0; k < ncut; k++) {
            for (int a = 1; a < cnnz[k]; a++) {   /* insertion sort by index */
                int ii = cidx[k][a]; double vv = cval[k][a]; int b = a - 1;
                while (b >= 0 && cidx[k][b] > ii) { cidx[k][b + 1] = cidx[k][b]; cval[k][b + 1] = cval[k][b]; b--; }
                cidx[k][b + 1] = ii; cval[k][b + 1] = vv;
            }
            int dup = 0;
            for (int i = 0; i < ncon && !dup; i++) {
                int b0 = rptr[i], b1 = rptr[i + 1];
                if (b1 - b0 != cnnz[k]) continue;
                if (clo[k] != t->blc[i] || cup[k] != t->buc[i]) continue;
                int same = 1;
                for (int p = b0; p < b1 && same; p++)
                    if (rsub[p] != cidx[k][p - b0] || rval[p] != cval[k][p - b0]) same = 0;
                if (same) dup = 1;
            }
            /* only conflict cuts are also compared against the kept cuts: the
             * classic generators keep their previous behavior. */
            if (k >= nconf_start)
                for (int q = 0; q < keep && !dup; q++) {
                    if (cnnz[q] != cnnz[k]) continue;
                    if (clo[q] != clo[k] || cup[q] != cup[k]) continue;
                    int same = 1;
                    for (int p = 0; p < cnnz[k] && same; p++)
                        if (cidx[q][p] != cidx[k][p] || cval[q][p] != cval[k][p]) same = 0;
                    if (same) dup = 1;
                }
            if (dup) { if (k >= nconf_start) nconf--; free(cidx[k]); free(cval[k]); continue; }
            if (keep != k) { cidx[keep] = cidx[k]; cval[keep] = cval[k]; cnnz[keep] = cnnz[k]; clo[keep] = clo[k]; cup[keep] = cup[k]; }
            keep++;
        }
        ncut = keep;
    }
    (void)nconf;
    free(rptr); free(rsub); free(rval);
    if (!ok || ncut == 0) {
        for (int k = 0; k < ncut; k++) { free(cidx[k]); free(cval[k]); }
        free(cidx); free(cval); free(cnnz); free(clo); free(cup);
        return 0;
    }
    /* task copy + the cuts as extra rows (for the relaxations only) */
    PRIMALtask_t tc = NULL;
    if (PRIMAL_clonetask(t, &tc) != PRIMAL_RES_OK || !tc) {
        for (int k = 0; k < ncut; k++) { free(cidx[k]); free(cval[k]); }
        free(cidx); free(cval); free(cnnz); free(clo); free(cup);
        return 0;
    }
    PRIMAL_appendcons(tc, ncut);
    for (int k = 0; k < ncut; k++) {
        int row = ncon + k;
        PRIMAL_putarow(tc, row, cnnz[k], cidx[k], cval[k]);
        PRIMAL_putconbound(tc, row, PRIMAL_BK_RA, clo[k], cup[k]);
        free(cidx[k]); free(cval[k]);
    }
    free(cidx); free(cval); free(cnnz); free(clo); free(cup);
    *tc_out = tc;
    if (nconf_out) *nconf_out = nconf;   /* exact count for the log/T251 */
    return ncut;
}


/* ======================================================================
 * FRACTIONAL Gomory cuts from the tableau. Unlike rank-1 CG (a single
 * model row), here the simplex tableau is read: for every basic INTEGER
 * variable with fractional value, the row
 *     x_B + sum_{j nonbasic} a_bar_ij x_j = b_bar_i
 * gives the valid cut (modulo 1)
 *     sum_{j nonbasic} frac(a_bar_ij) x_j >= frac(b_bar_i).
 * The cut lives in STANDARD-FORM space: it is mapped back to the user
 * variables by substituting x_j = tau*(x_orig - shift) and, for slacks,
 * s = (u - a'x) [UP row], s = (a'x - l) [LO row], s = (ux_j - x_j) [VARUB].
 * Validity: every nonbasic column must be integer, which holds when the
 * model has integer variables with integer bounds and integer data -- the
 * integer-data MIP case, where row CG is degenerate instead. A single round:
 * after one Gomory cut the slacks have fractional coefficients and the pure
 * cut would no longer be valid (mixed GMI would be needed). */

int mip_gomory_applicable(PRIMALtask_t tc) {
    for (int j = 0; j < tc->numvar; j++) {
        int vt = tc->vartype[j];
        if (vt != PRIMAL_VAR_TYPE_INT && vt != PRIMAL_VAR_TYPE_INT_BIN) return 0;
        if (tc->bkx[j] == PRIMAL_BK_FR) return 0;
        if (tc->bkx[j] != PRIMAL_BK_FX && tc->blx[j] != floor(tc->blx[j])) return 0;
        if (tc->bkx[j] == PRIMAL_BK_RA && tc->bux[j] != floor(tc->bux[j])) return 0;
        if (tc->bkx[j] == PRIMAL_BK_UP && tc->bux[j] != floor(tc->bux[j])) return 0;
    }
    for (int i = 0; i < tc->numcon; i++) {
        if (tc->bkc[i] == PRIMAL_BK_FR) continue;
        double b = (tc->bkc[i] == PRIMAL_BK_UP) ? tc->buc[i] : tc->blc[i];
        if (b != floor(b)) return 0;
        if (tc->bkc[i] == PRIMAL_BK_RA && tc->buc[i] != floor(tc->buc[i])) return 0;
    }
    for (int j = 0; j < tc->numvar; j++)
        for (int q = 0; q < tc->cols[j].nz; q++)
            if (tc->cols[j].val[q] != floor(tc->cols[j].val[q])) return 0;
    return 1;
}

/* One round: solve the relaxation of `tc` and append the Gomory cuts to `tc`.
 * Returns the number added (0 when not applicable or no fractional basis). */
int mip_gomory_round(PRIMALtask_t tc, int s) {
    if (!mip_gomory_applicable(tc)) return 0;
    int nvar = tc->numvar, ncon = tc->numcon;
    double *lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *ci = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    int *ptr = NULL, *sub = NULL; double *aval = NULL;
    StdForm *sf = NULL; double *dA = NULL;
    int *basis = NULL; double *tab = NULL; double *xt = NULL, *ystd = NULL, *zst = NULL;
    int nadd = 0;
    if (!lx || !ux || !lc || !uc || !ci) goto gdone;
    for (int j = 0; j < nvar; j++) { bound_range(tc->bkx[j], tc->blx[j], tc->bux[j], &lx[j], &ux[j]); ci[j] = s * tc->c[j]; }
    for (int i = 0; i < ncon; i++) bound_range(tc->bkc[i], tc->blc[i], tc->buc[i], &lc[i], &uc[i]);
    if (!build_csc(tc, &ptr, &sub, &aval)) goto gdone;
    sf = stdform_build(nvar, ncon, ci, NULL, NULL, NULL, 0, lx, ux, lc, uc, ptr, sub, aval);
    if (!sf) goto gdone;
    dA = stdform_dense_A(sf);
    if (!dA) goto gdone;
    int m = sf->m, n = sf->n, stride = n + m + 1;
    basis = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    tab = (double *)malloc((size_t)(m + 1) * (size_t)stride * sizeof(double));
    xt = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    ystd = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    zst = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!basis || !tab || !xt || !ystd || !zst) goto gdone;
    int st = simplex_solve_std_tab(dA, m, n, sf->b, sf->c, iter_cap(tc->max_iter_simplex),
                                   xt, ystd, NULL, NULL, basis, tab);
    if (st != 0) goto gdone;
    for (int i = 0; i < m; i++) {
        int bcol = basis[i];
        if (bcol < 0 || bcol >= n) continue;   /* artificial basis: redundant row */
        double bbar = tab[(size_t)i * stride + n];
        double f0 = bbar - floor(bbar);
        if (f0 < 1e-6 || f0 > 1.0 - 1e-6) continue;
        double *coef = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
        if (!coef) break;
        double rhs = f0;
        int ok = 1;
        for (int j = 0; j < n; j++) {
            int isbasic = 0;
            for (int r2 = 0; r2 < m; r2++) if (basis[r2] == j) { isbasic = 1; break; }
            if (isbasic) continue;
            double abar = tab[(size_t)i * stride + j];
            double fj = abar - floor(abar);
            if (fabs(fj) < 1e-9) continue;
            SfrCol cj = sf->cols[j];
            if (cj.kind == SFCK_VAR) {
                coef[cj.idx] += fj * cj.tau;
                rhs += fj * cj.tau * sf->vars[cj.idx].shift;
            } else {
                SfrRow row = sf->rows[cj.idx];
                int oi = row.orig;
                if (row.kind == SFRK_VARUB) {
                    /* s = ux_j - x_j: constant fj*ux_j on the left -> rhs -= */
                    coef[oi] -= fj;
                    rhs -= fj * tc->bux[oi];
                } else if (row.kind == SFRK_UP) {
                    /* s = u - a'x: constant fj*u on the left -> rhs -= */
                    for (int k = 0; k < nvar; k++)
                        for (int q = 0; q < tc->cols[k].nz; q++)
                            if (tc->cols[k].sub[q] == oi) coef[k] -= fj * tc->cols[k].val[q];
                    rhs -= fj * tc->buc[oi];
                } else if (row.kind == SFRK_LO) {
                    /* s = a'x - l: constant -fj*l on the left -> rhs += */
                    for (int k = 0; k < nvar; k++)
                        for (int q = 0; q < tc->cols[k].nz; q++)
                            if (tc->cols[k].sub[q] == oi) coef[k] += fj * tc->cols[k].val[q];
                    rhs += fj * tc->blc[oi];
                } else { ok = 0; break; }   /* EQ: no slack */
            }
        }
        if (ok) {
            /* normalize: coef'x >= rhs, with coef and rhs at reasonable scale */
            double mx = 1.0;
            for (int k = 0; k < nvar; k++) if (fabs(coef[k]) > mx) mx = fabs(coef[k]);
            for (int k = 0; k < nvar; k++) coef[k] /= mx;
            rhs /= mx;
            int nnz = 0;
            for (int k = 0; k < nvar; k++) if (coef[k] != 0.0) nnz++;
            if (nnz > 0) {
                PRIMAL_appendcons(tc, 1);
                int row = tc->numcon - 1;
                int *idx = (int *)malloc((size_t)nnz * sizeof(int));
                double *val = (double *)malloc((size_t)nnz * sizeof(double));
                if (idx && val) {
                    int t2 = 0;
                    for (int k = 0; k < nvar; k++) if (coef[k] != 0.0) { idx[t2] = k; val[t2] = coef[k]; t2++; }
                    PRIMAL_putarow(tc, row, nnz, idx, val);
                    PRIMAL_putconbound(tc, row, PRIMAL_BK_LO, rhs, INFINITY);
                    nadd++;
                }
                free(idx); free(val);
            }
        }
        free(coef);
    }
gdone:
    free(lx); free(ux); free(lc); free(uc); free(ci);
    free(ptr); free(sub); free(aval);
    free(basis); free(tab); free(xt); free(ystd); free(zst);
    free(dA); if (sf) stdform_free(sf);
    return nadd;
}

/* parallel probing: trials on different variables are independent (each
 * solves its own relaxation with a copy of the bounds), so they can be
 * spread over threads. `fix[k]`: 1 = fix to 1, 0 = fix to 0, -1 = none. */
void *probe_worker(void *arg) {
    ProbeJob *jb = (ProbeJob *)arg;
    int nvar = jb->nvar;
    double *plx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *pux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *pout = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!plx || !pux || !pout) { free(plx); free(pux); free(pout); return NULL; }
    for (int k = jb->start; k < jb->end; k++) {
        int v = jb->bins[k];
        jb->fix[k] = -1;
        memcpy(plx, jb->lx, (size_t)nvar * sizeof(double));
        memcpy(pux, jb->ux, (size_t)nvar * sizeof(double));
        plx[v] = pux[v] = 0.0;
        double z = 0.0;
        int st0 = mip_relax(jb->t, jb->s, plx, pux, jb->lc, jb->uc, pout, &z);
        plx[v] = pux[v] = 1.0;
        int st1 = mip_relax(jb->t, jb->s, plx, pux, jb->lc, jb->uc, pout, &z);
        if (st0 == 1 && st1 == 0) jb->fix[k] = 1;
        else if (st1 == 1 && st0 == 0) jb->fix[k] = 0;
    }
    free(plx); free(pux); free(pout);
    return NULL;
}

/* parallel strong branching: the two children of each candidate are
 * independent, so scores are computed over threads; the choice (max,
 * tie-break on candidate order) is fixed, hence deterministic. */
void *sb_worker(void *arg) {
    SBJob *jb = (SBJob *)arg;
    int nvar = jb->nvar;
    double *flx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *flux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *xo = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!flx || !flux || !xo) { free(flx); free(flux); free(xo); return NULL; }
    for (int k = jb->start; k < jb->end; k++) {
        int j = jb->cand[k]; double v = jb->x[j];
        double zl = 0.0, zr = 0.0;
        memcpy(flx, jb->lx, (size_t)nvar * sizeof(double));
        memcpy(flux, jb->ux, (size_t)nvar * sizeof(double));
        flux[j] = floor(v);
        int stl = mip_relax(jb->trelax, jb->s, flx, flux, jb->lc, jb->uc, xo, &zl);
        memcpy(flx, jb->lx, (size_t)nvar * sizeof(double));
        memcpy(flux, jb->ux, (size_t)nvar * sizeof(double));
        flx[j] = ceil(v);
        int str_ = mip_relax(jb->trelax, jb->s, flx, flux, jb->lc, jb->uc, xo, &zr);
        double score;
        if (stl == 1 || str_ == 1) score = INF;
        else if (stl == 0 && str_ == 0) score = (zl < zr) ? zl : zr;
        else if (stl == 0) score = zl;
        else if (str_ == 0) score = zr;
        else score = -INF;
        jb->score[k] = score;
    }
    free(flx); free(flux); free(xo);
    return NULL;
}

/* parallel B&B: root decomposition. Solve the root, branch, and solve the
 * two children in two threads, each with a CLONE of the task (child bounds)
 * and num_threads = 1 (no recursion). The better of the two is the optimum
 * (the two children partition the root region). */
void *mip_kid_run(void *arg) {
    MipKidJob *jb = (MipKidJob *)arg;
    jb->rc = optimize_mip(jb->t, jb->s);
    return NULL;
}

