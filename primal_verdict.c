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
/* primal_verdict.c - conic verdicts, cone gate, PRIMAL_optimize dispatcher entry.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Signed slack of the block v (nk entries, parameter a) in the cone ct:
 * positive inside K, negative outside, in the same shape for every cone kind.
 * Returns HUGE_VAL for a cone kind this task cannot test. */
double cone_signed_slack(int ct, double a, const double *v, int nk) {
    switch (ct) {
    case PRIMAL_CT_QUAD: { double s = 0.0; for (int i = 1; i < nk; i++) s += v[i] * v[i];
        return v[0] - sqrt(s); }
    case PRIMAL_CT_RQUAD: { double s = 0.0; for (int i = 2; i < nk; i++) s += v[i] * v[i];
        double q = (v[0] - v[1]) * (v[0] - v[1]) + 2.0 * s;
        return (v[0] + v[1] - (q > 0.0 ? sqrt(q) : 0.0)) / sqrt(2.0); }
    /* The u = 0 face is not "no violation": the epigraph of u*exp(v/u) closes
     * onto the half-line {u = 0, v = 0, t >= 0} and on nothing else, so a block
     * sitting on that face with v anywhere outside 0 is OUTSIDE the cone. The
     * same for the t = 0 / u = 0 faces of a power cone, where the geometric mean
     * is 0 and only v = 0 belongs. Measured, not assumed: reading the face as
     * slack 0 is how a direction escaping along v passed the ray test. */
    case PRIMAL_CT_PEXP: { double t = v[0], u = v[1];
        if (!(u > 0.0)) { if (u < 0.0) return u;
            double f = -fabs(v[2]); return t < f ? t : f; }
        double G = t - u * exp(v[2] / u); return G < u ? G : u; }
    case PRIMAL_CT_DEXP: { double t = v[0], u = v[1];   /* DEXP = -PEXP */
        if (!(u < 0.0)) { if (u > 0.0) return -u;
            double f = -fabs(v[2]); return -t < f ? -t : f; }
        double G = u * exp(v[2] / u) - t; return G < -u ? G : -u; }
    case PRIMAL_CT_PPOW: case PRIMAL_CT_RPOW: { double t = v[0], u = v[1], m = t < u ? t : u;
        if (!(t > 0.0 && u > 0.0)) { double f = -fabs(v[2]); return m < f ? m : f; }
        double g = (ct == PRIMAL_CT_RPOW ? sqrt(2.0) : 1.0)
                 * pow(t, a) * pow(u, 1.0 - a) - fabs(v[2]);
        return g < m ? g : m; }
    default: return HUGE_VAL;
    }
}

/* The dual of every cone this task can hold, as a signed slack in the same shape
 * and the same units as cone_signed_slack: positive inside K*, negative outside,
 * so the normalisation opt_report_cones applies to the primal applies here too.
 * QUAD and RQUAD are self-dual. The exponential and power duals are the
 * inequalities expcone_dual_in tests, written as a QUANTITY because the caller
 * answers with a number, and their domain bounds (s0 > 0 and s2 < 0 for PEXP)
 * enter the minimum: outside them the cone has no value to compare against.
 * DEXP is -PEXP -- every member negated -- and its dual cone is that same
 * negation, so its slack is the PEXP expression read at -s. Anything else is
 * untestable and returns a positive slack, which reads as "no violation
 * measured", never as a proof. */
double cone_dual_signed_slack(int ct, double a, const double *v, int nk) {
    switch (ct) {
    case PRIMAL_CT_QUAD: case PRIMAL_CT_RQUAD:
        return cone_signed_slack(ct, a, v, nk);
    case PRIMAL_CT_PEXP: case PRIMAL_CT_DEXP: {
        if (nk < 3) return HUGE_VAL;
        /* DEXP = -PEXP (every member negated), and (MK)* = M^-T K* with M = -I
         * is the same negation: s in DEXP* <=> -s in PEXP*. */
        double s0 = (ct == PRIMAL_CT_DEXP ? -v[0] : v[0]),
               s1 = (ct == PRIMAL_CT_DEXP ? -v[1] : v[1]),
               s2 = (ct == PRIMAL_CT_DEXP ? -v[2] : v[2]);
        if (!(s0 > 0.0)) return s0;
        if (!(s2 < 0.0)) return s2;
        double q = s1 - s2 + s2 * log(-s2 / s0);
        double m = s0 < -s2 ? s0 : -s2;
        return q < m ? q : m; }
    case PRIMAL_CT_PPOW: case PRIMAL_CT_RPOW: {
        if (nk < 3 || !(a > 0.0 && a < 1.0)) return HUGE_VAL;
        double s0 = v[0], s1 = v[1];
        if (!(s0 > 0.0)) return s0;
        if (!(s1 > 0.0)) return s1;
        double g = pow(s0 / a, a) * pow(s1 / (1.0 - a), 1.0 - a)
                 - (ct == PRIMAL_CT_RPOW ? sqrt(2.0) : 1.0) * fabs(v[2]);
        double m = s0 < s1 ? s0 : s1;
        return g < m ? g : m; }
    default: return HUGE_VAL;
    }
}

/* How far the published multipliers are from a dual-feasible point AT THE CONES.
 * A member j of a cone carries the dual condition
 *      d_j = s * ( c_j + sum_k A_kj y_k )  in  K*.
 * That is the route's own min form read back: the min problem the route solves
 * has cost s*c and dual slack s*c - A'pi, and every site publishes y = -s*pi, so
 * the slack in the user's space is s*(c + A'y) -- equivalently -s*(slx+sux), the
 * same identity the stationarity loop of PRIMAL_getdualinfeas states for a scalar
 * column. Reading it as c + s*A'y instead agrees with this on a MINIMIZE model and
 * differs on a MAXIMIZE one by exactly 2*c: a MAX model whose cone members carry
 * no cost (the CBF example C.2, where this was first pinned) cannot tell the two
 * apart, while one whose dual sits ON the boundary of K* can, and C.2's block was
 * the reflection of a boundary point -- the sign of the cone, not a rounding.
 * Two shapes make a block unreadable, and it is left unmeasured rather than
 * measured wrong: a member with a finite bound, whose bound multiplier is not
 * separable from the cone's in what is published, and a model with a quadratic
 * row or objective, whose gradient term belongs in d and is not in this sum.
 * The violation is RELATIVE to the block, as opt_report_cones does for the
 * primal, because the tolerance the task declares is relative. *nmeas counts the
 * blocks measured, so "no violation" and "nothing measured" stay distinguishable. */
double cone_dual_worst(PRIMALtask_t t, int s, int *nmeas, int verb) {
    if (nmeas) *nmeas = 0;
    if (t->numcones <= 0) return 0.0;
    if (t->has_qcon > 0 || t->has_qobj) return 0.0;
    int nkmax = 1;
    for (int k = 0; k < t->numcones; k++)
        if (t->cone_nmem[k] > nkmax) nkmax = t->cone_nmem[k];
    double *d = (double *)malloc((size_t)nkmax * sizeof(double));
    if (!d) return 0.0;
    double worst = 0.0;
    int nm = 0;
    for (int k = 0; k < t->numcones; k++) {
        int nk = t->cone_nmem[k];
        const int *mi = t->cone_mem[k];
        if (nk <= 0 || !mi) continue;
        int readable = 1;
        for (int i = 0; i < nk; i++) {
            double lo, up;
            int j = mi[i];
            if (j < 0 || j >= t->numvar) { readable = 0; break; }
            bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
            if (isfinite(lo) || isfinite(up)) { readable = 0; break; }
        }
        if (!readable) continue;
        double sc = 0.0;
        for (int i = 0; i < nk; i++) {
            const Col *col = &t->cols[mi[i]];
            double av = 0.0;
            for (int q = 0; q < col->nz; q++) av += col->val[q] * t->y[col->sub[q]];
            d[i] = soc_dual_component(t, k, i, s * (t->c[mi[i]] + av));
            if (fabs(d[i]) > sc) sc = fabs(d[i]);
        }
        double sl = cone_dual_signed_slack(t->cone_type[k], t->cone_param[k], d, nk);
        if (!isfinite(sl)) continue;     /* untestable: not measured, not "measured clean" */
        nm++;
        if (verb && getenv("GMB_DBG")) {
            fprintf(stderr, "  [condual] cone %d %s rel=%.3g d=", k,
                    (t->cone_type[k] >= 0 &&
                     t->cone_type[k] < (cone_kind_count))
                        ? cone_kind_name[t->cone_type[k]] : "?",
                    sl < 0.0 ? -sl / (1.0 + sc) : 0.0);
            for (int i = 0; i < nk && i < 4; i++) fprintf(stderr, " %.12g", d[i]);
            fprintf(stderr, " slack=%.6g\n", sl);
        }
        double viol = sl < 0.0 ? -sl / (1.0 + sc) : 0.0;
        if (viol > worst) worst = viol;
    }
    free(d);
    if (nmeas) *nmeas = nm;
    return worst;
}

/* Whether a solve that is about to publish on a cone-only model may call itself
 * an answer. A cone has no rows to leave residual: the vertex of a QUAD block
 * satisfies every equation the model has, so the interior-point gap can close
 * there while the point is nowhere near the optimum -- and what says so is not
 * the primal residual (0 at the vertex) nor the gap (mu*nu, by construction) but
 * the reduced cost of the block, which has to sit in K*. Measured on
 * `min -x0` over one free 3-D QUAD block: the route returned rc=OK, solsta=
 * OPTIMAL, pobj ~ 5e-9 at x = 0 with a cone-dual violation of exactly 1.
 * So the same triple the routes judge is judged here on its fourth side, at the
 * tolerance the task DECLARES widened by the near-optimal factor -- the same
 * effective tolerance sdp_ipm's gate uses, so the two cannot disagree about a
 * point both of them could see.
 * Refusing the point is the cheap half of the decision; saying what the model
 * instead is needs a direction, so a refused point is put to the same test the
 * bar cap uses: a recession direction MEASURED on the model's own rows and cones
 * earns DUAL_INFEAS, and none leaves the solver with no verdict at all -- never
 * with the value it failed to reach (the policy T100 established, and T99's: a
 * verdict without a point publishes no point). */
/* Measure a candidate Farkas vector y (one entry per non-FR model row) for a
 * conic model: d = sum_i y_i a_i must lie in K* for every cone of the task, and
 * the signed RHS combination must be negative -- the conic Farkas condition for
 * an infeasible system. The candidate comes from an LP relaxation that CONTAINS
 * the cones, so a candidate that measures is a proof about the model and one
 * that does not proves nothing. Variable bounds are not needed: a certificate
 * that ignores them is still a certificate (bounds only shrink the feasible
 * set) and the stdform candidate carries no bound multipliers anyway. Mirrors
 * sdp_ray_measures on the dual side, sign included: the row multiplier must be
 * >= 0 on a row with an upper side and <= 0 on one with a lower side, and the
 * `d in K*` test pins the overall sign. */
static int conic_dual_ray_measures(PRIMALtask_t t, int nrowmodel, double *y) {
    if (!y || nrowmodel <= 0 || t->numcones <= 0) return 0;
    /* With bars the certificate would also carry the compressed block, which
     * `d` here does not: a `y` that measures on the scalar columns alone is not
     * a certificate of the model. The verdict stays in `prosta` (T85
     * deviation); a vector that does not measure is not published. */
    if (t->numbarvar > 0 || t->nbarA > 0 || t->nbarC > 0) return 0;
    double mx = 0.0;
    for (int r = 0; r < nrowmodel; r++) if (fabs(y[r]) > mx) mx = fabs(y[r]);
    if (!(mx > 0.0)) return 0;
    for (int r = 0; r < nrowmodel; r++) y[r] /= mx;
    int nvar = t->numvar;
    for (int sgn = 0; sgn < 2; sgn++) {
        double *d = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
        if (!d) return 0;
        double scale = 1.0, contrib = 0.0;
        int signok = 1, rr = 0;
        for (int i = 0; i < t->numcon; i++) {
            if (t->bkc[i] == PRIMAL_BK_FR) continue;
            double yi = (sgn ? -1.0 : 1.0) * y[rr++];
            if (t->bkc[i] == PRIMAL_BK_UP && yi < -1e-12) signok = 0;
            if (t->bkc[i] == PRIMAL_BK_LO && yi >  1e-12) signok = 0;
            for (int j = 0; j < nvar; j++) {
                const Col *col = &t->cols[j];
                double a = 0.0;
                for (int q = 0; q < col->nz; q++) if (col->sub[q] == i) a += col->val[q];
                if (fabs(a) > scale) scale = fabs(a);
                d[j] += yi * a;
            }
            double rhs;
            if (t->bkc[i] == PRIMAL_BK_UP) rhs = t->buc[i];
            else if (t->bkc[i] == PRIMAL_BK_LO) rhs = t->blc[i];
            else if (t->bkc[i] == PRIMAL_BK_FX) rhs = t->blc[i];
            else rhs = (yi > 0.0) ? t->buc[i] : t->blc[i];   /* RA: the side the sign picks */
            contrib += yi * rhs;
        }
        int cones_ok = 1;
        for (int k = 0; k < t->numcones && cones_ok; k++) {
            int nk = t->cone_nmem[k];
            const int *mi = t->cone_mem[k];
            if (nk <= 0 || !mi) { cones_ok = 0; break; }
            double *db = (double *)malloc((size_t)nk * sizeof(double));
            if (!db) { cones_ok = 0; break; }
            double sc = 0.0;
            for (int i = 0; i < nk; i++) { db[i] = d[mi[i]]; if (fabs(db[i]) > sc) sc = fabs(db[i]); }
            if (sc > 0.0) for (int i = 0; i < nk; i++) db[i] /= sc;
            double sl = cone_dual_signed_slack(t->cone_type[k], t->cone_param[k], db, nk);
            free(db);
            if (!isfinite(sl) || sl < -1e-8) cones_ok = 0;
        }
        free(d);
        if (signok && cones_ok && contrib < -1e-8 * (1.0 + scale)) {
            if (sgn) for (int r = 0; r < nrowmodel; r++) y[r] = -y[r];
            return 1;
        }
    }
    return 0;
}

/* Publish a measured recession direction as the user's primal ray. Only a
 * bar-free model has one in `numvar` scalars -- with bars the compressed block
 * has no image as `numvar` scalars, which is the T85 deviation -- so this
 * returns 1 when a vector the user can read was published. */
static int conic_publish_pray(PRIMALtask_t t, const double *rho) {
    if (!rho || !t->pray || t->numbarvar != 0 || t->numvar <= 0) return 0;
    for (int j = 0; j < t->numvar; j++) t->pray[j] = rho[j];
    t->has_pray = 1;
    return 1;
}

/* compressed upper-triangle coefficients of the matrix store (off-diagonal
 * doubled), the form `sdp_bar_row`/`model_lp_witness` expect. Needed to ask the
 * model's own rows on a task with bar terms; NULL on allocation failure. */
static double **build_symPq(PRIMALtask_t t) {
    double **symPq = (double **)malloc((size_t)(t->nsym > 0 ? t->nsym : 1) * sizeof(double *));
    if (!symPq) return NULL;
    for (int m = 0; m < t->nsym; m++) symPq[m] = NULL;
    for (int m = 0; m < t->nsym; m++) {
        int d = t->sym_dim[m];
        double *M = (double *)calloc((size_t)d * d, sizeof(double));
        if (!M) { for (int q = 0; q < m; q++) free(symPq[q]); free(symPq); return NULL; }
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            int si = t->sym_subi[m][e], sj = t->sym_subj[m][e]; double v = t->sym_val[m][e];
            M[si * d + sj] += v; if (si != sj) M[sj * d + si] += v;
        }
        int pq = d * (d + 1) / 2;
        symPq[m] = (double *)malloc((size_t)pq * sizeof(double));
        if (!symPq[m]) { free(M); for (int q = 0; q <= m; q++) free(symPq[q]); free(symPq); return NULL; }
        for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
            symPq[m][bar_pack(d, p, q)] = (p == q) ? M[p * d + q] : 2.0 * M[p * d + q];
        free(M);
    }
    return symPq;
}
/* Release the compressed matrix store built by build_symPq. */
static void free_symPq(PRIMALtask_t t, double **symPq) {
    if (!symPq) return;
    for (int m = 0; m < t->nsym; m++) free(symPq[m]);
    free(symPq);
}

/* Judge a conic solve's own answer on the fourth face: the reduced cost of
 * every readable cone and bar block must lie in its dual cone. On a violation
 * the point is discarded and the model is asked for a recession direction. */
static PRIMALrescodee conic_dual_verdict(PRIMALtask_t t, int s) {
    int nm = 0;
    double viol = cone_dual_worst(t, s, &nm, 1);
    /* The fourth face of a PSD block is the dual cone as for a cone: the
     * reduced cost published in `barsj` must lie in S_+. The same measure as
     * `getdualinfeas` (`bar_dual_viol`), so the verdict and the figure read by
     * the user cannot diverge. */
    for (int j = 0; j < t->numbarvar; j++) {
        double bv = bar_dual_viol(t, j);
        if (t->barsj[j]) nm++;
        if (bv > viol) viol = bv;
    }
    double near = (t->tol_near_rel > 1.0) ? t->tol_near_rel : 1.0;
    /* nm == 0 says "nothing could be read at the cones", not "the cones are
     * dual feasible" -- with nothing measured the route's own verdict stands. */
    if (nm <= 0 || !(viol > t->tol_co_dfeas * near)) return PRIMAL_RES_OK;

    char cb[160];
    snprintf(cb, sizeof cb, "cone dual: published reduced cost outside K* by %.3g"
             " (tol %.3g)\n", viol, t->tol_co_dfeas * near);
    tlog(t, cb);
    double *rho = (double *)malloc((size_t)(t->numvar > 0 ? t->numvar : 1) * sizeof(double));
    double **symPq = build_symPq(t);
    int rayed = model_lp_witness(t, s, symPq, 0, "cone dual", rho, NULL);
    free_symPq(t, symPq);
    t->has_sol = 0;
    if (rayed < 0) { free(rho); return PRIMAL_RES_ERR_ALLOC; }
    if (rayed > 0) {
        t->prosta = PRIMAL_PRO_STA_DUAL_INFEAS;
        /* A `*_CER` names a vector: it goes out only with the ray beside it. */
        t->solsta = conic_publish_pray(t, rho) ? PRIMAL_SOL_STA_DUAL_INFEAS_CER
                                               : PRIMAL_SOL_STA_UNKNOWN;
        tlog(t, t->has_pray ? "dual infeasible (unbounded): primal ray published\n"
                            : "dual infeasible (unbounded): recession direction measured,"
                              " no ray in the user's space\n");
        free(rho);
        return PRIMAL_RES_ERR_UNBOUNDED;
    }
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    free(rho);
    tlog(t, "conic answer not dual feasible: no recession direction measured,"
            " no verdict\n");
    return PRIMAL_RES_TRM_MAX_ITER;
}

/* A conic route that runs out of iterations has said something about its own
 * trajectory and nothing about the model, and TRM_MAX_ITER is the answer that
 * leaves the user with. Two questions about the model are still settled here,
 * each on the relaxation that makes its own answer sound:
 *  - infeasibility needs a set that CONTAINS the cones (mode 1), because an LP
 *    infeasible on a subset of the model proves nothing;
 *  - unboundedness needs a set INSIDE them (mode 0), because a direction of a
 *    wider set is not a direction of the model.
 * Both are statements the reference makes with a Farkas vector beside them, and
 * this solver has none on a conic route: the conic/SDP/MIP deviation T85 declares
 * is that the ray getters answer ERR_ARG there. So the verdict goes out as
 * prosta = PRIM_INFEAS / DUAL_INFEAS with solsta = UNKNOWN and no point, which is
 * exactly the family T98 established for a verdict without a certificate, and the
 * reason PRIMAL_getprosta exists beside PRIMAL_getsolsta. */
static PRIMALrescodee conic_no_answer_verdict(PRIMALtask_t t, int s,
                                              PRIMALrescodee r) {
    int nrowmodel = 0;
    for (int i = 0; i < t->numcon; i++) if (t->bkc[i] != PRIMAL_BK_FR) nrowmodel++;
    double *dual = (double *)calloc((size_t)(nrowmodel > 0 ? nrowmodel : 1), sizeof(double));
    if (!dual) return PRIMAL_RES_ERR_ALLOC;
    double **symPq = build_symPq(t);
    int infea = model_lp_witness(t, s, symPq, 1, "cone faces", NULL, dual);
    if (infea < 0) { free_symPq(t, symPq); free(dual); return PRIMAL_RES_ERR_ALLOC; }
    free_symPq(t, symPq);
    if (infea > 0) {
        t->prosta = PRIMAL_PRO_STA_PRIM_INFEAS;
        t->has_sol = 0;
        /* The relaxation said infeasible; the candidate dual vector goes out
         * only if it measures in the MODEL's own cones (d in K* on every block,
         * the signed RHS combination negative), which is what makes it a
         * certificate of the model and not of the relaxation. */
        if (conic_dual_ray_measures(t, nrowmodel, dual) && t->dray) {
            int rr = 0;
            for (int i = 0; i < t->numcon; i++)
                t->dray[i] = (t->bkc[i] == PRIMAL_BK_FR) ? 0.0 : dual[rr++];
            t->has_dray = 1;
            t->solsta = PRIMAL_SOL_STA_PRIM_INFEAS_CER;
            tlog(t, "primal infeasible: dual ray published (d in K* on every cone)\n");
        } else {
            t->solsta = PRIMAL_SOL_STA_UNKNOWN;
            tlog(t, "primal infeasible: no dual vector measured in the model's cones\n");
        }
        free(dual);
        return PRIMAL_RES_ERR_INFEASIBLE;
    }
    free(dual);
    double *rho = (double *)malloc((size_t)(t->numvar > 0 ? t->numvar : 1) * sizeof(double));
    double **symPqR = build_symPq(t);
    int rayed = model_lp_witness(t, s, symPqR, 0, "cone recession", rho, NULL);
    free_symPq(t, symPqR);
    if (rayed < 0) { free(rho); return PRIMAL_RES_ERR_ALLOC; }
    if (rayed > 0) {
        t->prosta = PRIMAL_PRO_STA_DUAL_INFEAS;
        t->solsta = conic_publish_pray(t, rho) ? PRIMAL_SOL_STA_DUAL_INFEAS_CER
                                               : PRIMAL_SOL_STA_UNKNOWN;
        t->has_sol = 0;
        tlog(t, t->has_pray ? "dual infeasible (unbounded): primal ray published\n"
                            : "dual infeasible (unbounded): recession direction measured,"
                              " no ray in the user's space\n");
        free(rho);
        return PRIMAL_RES_ERR_UNBOUNDED;
    }
    free(rho);
    tlog(t, "conic route gave no answer: neither the faces nor a recession"
            " direction measured, no verdict\n");
    return r;
}

/* Smallest eigenvalue of the dense symmetric d x d matrix A (Jacobi), and its
 * largest in *pmax when pmax is non-NULL. */
double bar_min_eig(int d, const double *A, double *pmax) {
    double *Ac = (double *)malloc((size_t)d * d * sizeof(double));
    double *ev = (double *)malloc((size_t)d * sizeof(double));
    double *evec = (double *)malloc((size_t)d * d * sizeof(double));
    double lo = HUGE_VAL, hi = -HUGE_VAL;
    if (Ac && ev && evec) { memcpy(Ac, A, sizeof(double) * (size_t)d * d);
        dmat_eig_jacobi(d, Ac, ev, evec);
        for (int k = 0; k < d; k++) { if (ev[k] < lo) lo = ev[k]; if (ev[k] > hi) hi = ev[k]; } }
    free(Ac); free(ev); free(evec);
    if (pmax) *pmax = (hi > -HUGE_VAL && hi > 0.0) ? hi : 0.0;
    return lo;
}
/* Print the worst relative cone/bound slack of the published point to stderr,
 * ranked by slack/(1+scale), with the location and the numbers involved. */
static void opt_report_cones(PRIMALtask_t t) {
    /* Ranked by the RELATIVE violation: the tolerance the task declares is
     * relative, so a slack of 6e-8 on a block of size 1e-3 and one of 6e-8 on
     * a block of size 5 are not the same verdict. Normalised by 1 + (size of
     * the object), the same way rel_pri normalises by 1 + |b|. */
    double worst = HUGE_VAL, wabs = 0.0;
    char where[32] = "", vals[96] = "", w[32], s[96];
#define WORST(sl, sc, what, numbers) do { double rel_ = (sl) / (1.0 + (sc)); \
        if (rel_ < worst) { worst = rel_; wabs = (sl); \
        snprintf(where, sizeof where, "%s", (what)); \
        snprintf(vals, sizeof vals, "%s", (numbers)); } } while (0)
    for (int j = 0; j < t->numvar; j++) { double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (!isfinite(lo)) continue;
        char key = (t->bkx[j] == PRIMAL_BK_FX) ? 'X' : (t->bkx[j] == PRIMAL_BK_RA) ? 'R' : 'L';
        double sl = t->x[j] - lo;
        int vt = t->vartype[j];
        if (vt == PRIMAL_VAR_TYPE_SEMI_CONT || vt == PRIMAL_VAR_TYPE_SEMI_INT) {
            /* {0} union [l,u], the same reading PRIMAL_getprimalinfeas uses */
            double d0 = -fabs(t->x[j]);
            if (d0 > sl) sl = d0;
            if (t->x[j] > up && up - t->x[j] > sl) sl = up - t->x[j];
        }
        snprintf(w, sizeof w, "bound %d %c", j, key);
        snprintf(s, sizeof s, "x=%.12g lo=%.12g", t->x[j], lo);
        WORST(sl, fabs(t->x[j]) > fabs(lo) ? fabs(t->x[j]) : fabs(lo), w, s); }
    int maxnk = 1;
    for (int k = 0; k < t->numcones; k++) if (t->cone_nmem[k] > maxnk) maxnk = t->cone_nmem[k];
    double *v = (double *)calloc((size_t)maxnk, sizeof(double));
    if (v) {
        for (int k = 0; k < t->numcones; k++) { int nk = t->cone_nmem[k]; const int *mi = t->cone_mem[k];
            double sc = 0.0;
            for (int i = 0; i < nk; i++) { v[i] = t->x[mi[i]];
                if (fabs(v[i]) > sc) sc = fabs(v[i]); }
            int ct = t->cone_type[k];
            const char *nm = (ct >= 0 && ct < (cone_kind_count))
                           ? cone_kind_name[ct] : "?";
            snprintf(w, sizeof w, "cone %d %s", k, nm);
            snprintf(s, sizeof s, "a=%.12g{%.12g %.12g %.12g%s}", t->cone_param[k], v[0],
                     nk > 1 ? v[1] : 0.0, nk > 2 ? v[2] : 0.0, nk > 3 ? " ..." : "");
            WORST(cone_signed_slack(ct, t->cone_param[k], v, nk), sc, w, s); }
        free(v);
    }
    /* A `putqconk` row is not a cone of THIS task (the RQUAD block lives in the
     * shadow model the encoder builds), so its membership has to be measured in
     * the user's own form, with the sign convention quad_encode_task uses:
     * a'x + sgn*1/2 x'Qx against the row bound, sgn = +1 on UP, -1 on LO. */
    if (t->has_qcon > 0 && t->qcon) {
        for (int i = 0; i < t->numcon; i++) {
            if (!t->qcon[i]) continue;
            double lo, up, val = quad_row_value(t, i, t->x);
            bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
            double sl = HUGE_VAL, sc = fabs(val);
            if (isfinite(up) && up - val < sl) sl = up - val;
            if (isfinite(lo) && val - lo < sl) sl = val - lo;
            if (isfinite(up) && fabs(up) > sc) sc = fabs(up);
            if (isfinite(lo) && fabs(lo) > sc) sc = fabs(lo);
            if (sl == HUGE_VAL) continue;
            snprintf(w, sizeof w, "qrow %d", i);
            snprintf(s, sizeof s, "val=%.12g lo=%.12g up=%.12g", val, lo, up);
            WORST(sl, sc, w, s); }
    }
    for (int j = 0; j < t->numbarvar; j++) { double emax = 0.0;
        double e = bar_min_eig(t->barDim[j], t->barx[j], &emax);
        snprintf(w, sizeof w, "bar %d", j);
        snprintf(s, sizeof s, "dim=%d min_eig=%.6g", t->barDim[j], e);
        WORST(e, emax, w, s); }
#undef WORST
    if (isfinite(worst)) fprintf(stderr,
        "  [cones] task rel_slack=%.3g pri_slack=%.3g where=%s %s\n",
        worst, wabs, where, vals);
}

/* The worst RELATIVE slack of the published point at the model's cones, in the
 * same units opt_report_cones prints: negative when outside. The conic route's
 * three figures (rel_pri/rel_dual/rel_gap) do not see membership, so a point can
 * satisfy them while sitting outside a cone -- measured: `min x0` with `x0 = -1`
 * and `(x0,x1,x2) in QUAD` answered OPTIMAL at x = (-1,0,0), outside by 0.5.
 * What decides the answer must include this side too. */
static double conic_primal_cone_worst(PRIMALtask_t t) {
    double worst = 0.0;
    if (!t->x) return 0.0;
    for (int k = 0; k < t->numcones; k++) {
        int nk = t->cone_nmem[k];
        const int *mi = t->cone_mem[k];
        if (nk <= 0 || !mi) continue;
        double *v = (double *)malloc((size_t)nk * sizeof(double));
        if (!v) continue;
        double sc = 0.0;
        for (int i = 0; i < nk; i++) { v[i] = t->x[mi[i]]; if (fabs(v[i]) > sc) sc = fabs(v[i]); }
        double sl = cone_signed_slack(t->cone_type[k], t->cone_param[k], v, nk);
        free(v);
        if (!isfinite(sl)) continue;
        double rel = sl / (1.0 + sc);
        if (rel < worst) worst = rel;
    }
    return worst;
}

/* Bound tightening from constraints: for each row l <= a'x <= u and each j
 * with a_ij != 0, the rest of the row lies in [tmin-amax, tmax-amin] (min/max
 * of the terms over the other bounds), so a_ij x_j <= u - smin and
 * a_ij x_j >= l - smax give implied bounds on x_j. The LOCAL arrays lx/ux are
 * tightened: the user's model (getvarbound) does not change, and the published
 * point stays valid for the original because the implied bounds follow from the
 * constraints. */
int bound_tighten(PRIMALtask_t t, int nvar, int ncon,
                         double *lx, double *ux, const double *lc, const double *uc,
                         int *lo_row, double *lo_coef, int *up_row, double *up_coef) {
    int tightened = 0;
    for (int j = 0; j < nvar; j++) {
        if (lo_row) lo_row[j] = -1;
        if (up_row) up_row[j] = -1;
        if (lo_coef) lo_coef[j] = 0.0;
        if (up_coef) up_coef[j] = 0.0;
    }
    /* CSR of the linear A once (row -> (column, value) entries): both row-scan
     * passes below are O(nnz), not O(m*nnz). */
    int *rp = (int *)calloc((size_t)(ncon + 1), sizeof(int));
    if (!rp) return 0;
    int nnz = 0;
    for (int j = 0; j < nvar; j++) nnz += t->cols[j].nz;
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) { int i = t->cols[j].sub[q]; if (i >= 0 && i < ncon) rp[i + 1]++; }
    for (int i = 0; i < ncon; i++) rp[i + 1] += rp[i];
    int *rs = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    double *rv = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    int *fr = (int *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(int));
    if (!rs || !rv || !fr) { free(rp); free(rs); free(rv); free(fr); return 0; }
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) {
            int i = t->cols[j].sub[q];
            if (i >= 0 && i < ncon) { int p = rp[i] + fr[i]++; rs[p] = j; rv[p] = t->cols[j].val[q]; }
        }
    free(fr);
    for (int i = 0; i < ncon; i++) {
        int flo = isfinite(lc[i]), fup = isfinite(uc[i]);
        if (!flo && !fup) continue;
        /* a row with bar or quadratic terms is not a scalar row: `t->cols`
         * does not carry them, so the rest is not [tmin,tmax] and the implied
         * bound would be false. Skip. */
        int skip = 0;
        for (int k = 0; k < t->nbarA && !skip; k++) if (t->barA_con[k] == i) skip = 1;
        if (!skip && t->has_qcon > 0 && t->qcon && t->qcon[i]) skip = 1;
        /* an equality row fixes the variables: tightening it moves the
         * infeasibility from the constraint to the bound and the Farkas ray
         * (which lives on the rows) no longer measures. Skip. */
        if (t->bkc[i] == PRIMAL_BK_FX) skip = 1;
        if (skip) continue;
        /* sum of the min/max of the terms, counting the infinities: this way a
         * FREE variable does not poison the implied bound of the others (its
         * term is excluded) and can itself receive a bound. */
        double tmin = 0.0, tmax = 0.0;
        int tmin_inf = 0, tmax_inf = 0;
        for (int p = rp[i]; p < rp[i + 1]; p++) {
                int k = rs[p]; double a = rv[p];
                double lo = lx[k], up = ux[k];
                double amin = (a > 0.0) ? a * lo : a * up;
                double amax = (a > 0.0) ? a * up : a * lo;
                if (amin == -INF) tmin_inf++; else tmin += amin;
                if (amax == INF) tmax_inf++; else tmax += amax;
            }
        for (int p = rp[i]; p < rp[i + 1]; p++) {
                int j = rs[p]; double a = rv[p];
                if (a == 0.0) continue;
                double lo = lx[j], up = ux[j];
                double amin = (a > 0.0) ? a * lo : a * up;
                double amax = (a > 0.0) ? a * up : a * lo;
                int jmin_inf = (amin == -INF) ? 1 : 0;
                int jmax_inf = (amax == INF) ? 1 : 0;
                double smin = (tmin_inf - jmin_inf > 0) ? -INF
                              : tmin - (jmin_inf ? 0.0 : amin);
                double smax = (tmax_inf - jmax_inf > 0) ? INF
                              : tmax - (jmax_inf ? 0.0 : amax);
                /* Implied bound as BOUND FIXING (gated GMB_MIP_IMPLIED_BOUND):
                 * for an INTEGER variable the implied bound is rounded to the
                 * useful integer part. An implied upper bound 0.8 on a binary
                 * fixes it to 0, an implied lower bound 0.2 fixes it to 1.
                 * Default OFF: the node-cap test models are solved first. */
                int jint = getenv("GMB_MIP_IMPLIED_BOUND") &&
                           (t->vartype[j] == PRIMAL_VAR_TYPE_INT ||
                            t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN ||
                            t->vartype[j] == PRIMAL_VAR_TYPE_SEMI_INT);
                if (fup) {
                    double rhs = uc[i] - smin;
                    double nb = rhs / a;
                    double nbu = jint ? floor(nb) : nb;
                    double nbl = jint ? ceil(nb) : nb;
                    if (a > 0.0) {
                        if (nbu < ux[j] && nbu >= lx[j]) { ux[j] = nbu; tightened++;
                            if (up_row) up_row[j] = i; if (up_coef) up_coef[j] = a; }
                    } else {
                        if (nbl > lx[j] && nbl <= ux[j]) { lx[j] = nbl; tightened++;
                            if (lo_row) lo_row[j] = i; if (lo_coef) lo_coef[j] = a; }
                    }
                }
                if (flo) {
                    double rhs = lc[i] - smax;
                    double nb = rhs / a;
                    double nbu = jint ? floor(nb) : nb;
                    double nbl = jint ? ceil(nb) : nb;
                    if (a > 0.0) {
                        if (nbl > lx[j] && nbl <= ux[j]) { lx[j] = nbl; tightened++;
                            if (lo_row) lo_row[j] = i; if (lo_coef) lo_coef[j] = a; }
                    } else {
                        if (nbu < ux[j] && nbu >= lx[j]) { ux[j] = nbu; tightened++;
                            if (up_row) up_row[j] = i; if (up_coef) up_coef[j] = a; }
                    }
                }
            }
    }
    free(rp); free(rs); free(rv);
    return tightened;
}

/* Redundant rows: a row a'x <= u with max a'x <= u is implied by the bounds,
 * and its dual is 0 at the optimum. Removing it from the standard form (lc/uc
 * set to +-INF) is dual-safe: the dual of the removed row is 0, so the KKT of
 * the original model still holds. Rows with bar/quadratic terms are skipped. */
int row_redundant(PRIMALtask_t t, int nvar, int ncon,
                         const double *lx, const double *ux, double *lc, double *uc) {
    /* Build a CSR of the linear A once (row -> its (column, value) entries) so
     * the row-activity pass is O(nnz) instead of O(m*nnz). */
    int *rp = (int *)calloc((size_t)(ncon + 1), sizeof(int));
    if (!rp) return 0;
    int nnz = 0;
    for (int j = 0; j < nvar; j++) nnz += t->cols[j].nz;
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) { int i = t->cols[j].sub[q]; if (i >= 0 && i < ncon) rp[i + 1]++; }
    for (int i = 0; i < ncon; i++) rp[i + 1] += rp[i];
    int *rs = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    double *rv = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    int *fr = (int *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(int));
    if (!rs || !rv || !fr) { free(rp); free(rs); free(rv); free(fr); return 0; }
    for (int j = 0; j < nvar; j++)
        for (int q = 0; q < t->cols[j].nz; q++) {
            int i = t->cols[j].sub[q];
            if (i >= 0 && i < ncon) { int p = rp[i] + fr[i]++; rs[p] = j; rv[p] = t->cols[j].val[q]; }
        }
    free(fr);
    int removed = 0;
    for (int i = 0; i < ncon; i++) {
        int flo = isfinite(lc[i]), fup = isfinite(uc[i]);
        if (!flo && !fup) continue;
        int skip = 0;
        for (int k = 0; k < t->nbarA && !skip; k++) if (t->barA_con[k] == i) skip = 1;
        if (!skip && t->has_qcon > 0 && t->qcon && t->qcon[i]) skip = 1;
        /* an equality row fixes the variables: tightening it moves the
         * infeasibility from the constraint to the bound and the Farkas ray
         * (which lives on the rows) no longer measures. Skip. */
        if (t->bkc[i] == PRIMAL_BK_FX) skip = 1;
        if (skip) continue;
        double tmin = 0.0, tmax = 0.0;
        for (int p = rp[i]; p < rp[i + 1]; p++) {
            int j = rs[p]; double a = rv[p];
            tmin += (a > 0.0) ? a * lx[j] : a * ux[j];
            tmax += (a > 0.0) ? a * ux[j] : a * lx[j];
        }
        if (fup && tmax <= uc[i] + 1e-9 * (1.0 + fabs(uc[i]))) { uc[i] = INF; removed++; }
        if (flo && tmin >= lc[i] - 1e-9 * (1.0 + fabs(lc[i]))) { lc[i] = -INF; removed++; }
    }
    free(rp); free(rs); free(rv);
    return removed;
}

/* Conic presolve: a redundant row (implied by the bounds) has dual 0, so it is
 * temporarily removed (bkc = FR) before `optimize_conic` and restored after --
 * the published model (getconbound) does not change. */
void conic_presolve_apply(PRIMALtask_t t, SavedRows *sv) {
    sv->n = 0; sv->idx = NULL; sv->bkc = NULL; sv->blc = sv->buc = NULL;
    int nvar = t->numvar, ncon = t->numcon;
    double *lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    if (!lx || !ux || !lc || !uc) { free(lx); free(ux); free(lc); free(uc); return; }
    for (int j = 0; j < nvar; j++) bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    for (int i = 0; i < ncon; i++) bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc[i], &uc[i]);
    if (!getenv("GMB_NO_PRESOLVE_ROWS"))
        row_redundant(t, nvar, ncon, lx, ux, lc, uc);
    int cnt = 0;
    for (int i = 0; i < ncon; i++)
        if (!isfinite(lc[i]) && !isfinite(uc[i]) &&
            (isfinite(t->blc[i]) || isfinite(t->buc[i]))) cnt++;
    if (cnt > 0) {
        sv->idx = (int *)malloc((size_t)cnt * sizeof(int));
        sv->bkc = (int *)malloc((size_t)cnt * sizeof(int));
        sv->blc = (double *)malloc((size_t)cnt * sizeof(double));
        sv->buc = (double *)malloc((size_t)cnt * sizeof(double));
        if (sv->idx && sv->bkc && sv->blc && sv->buc) {
            int q = 0;
            for (int i = 0; i < ncon; i++)
                if (!isfinite(lc[i]) && !isfinite(uc[i]) &&
                    (isfinite(t->blc[i]) || isfinite(t->buc[i]))) {
                    sv->idx[q] = i; sv->bkc[q] = t->bkc[i];
                    sv->blc[q] = t->blc[i]; sv->buc[q] = t->buc[i]; q++;
                    t->bkc[i] = PRIMAL_BK_FR; t->blc[i] = -INF; t->buc[i] = INF;
                }
            sv->n = q;
            if (getenv("GMB_DBG")) {
                char cb[64];
                snprintf(cb, sizeof cb, "conic: %d redundant rows\n", q);
                tlog(t, cb);
            }
        } else { free(sv->idx); free(sv->bkc); free(sv->blc); free(sv->buc);
                 sv->idx = NULL; sv->bkc = NULL; sv->blc = sv->buc = NULL; }
    }
    free(lx); free(ux); free(lc); free(uc);
}
/* Restore the rows hidden by conic_presolve_apply from the saved copy. */
void conic_presolve_restore(PRIMALtask_t t, SavedRows *sv) {
    for (int q = 0; q < sv->n; q++) {
        int i = sv->idx[q];
        t->bkc[i] = sv->bkc[q]; t->blc[i] = sv->blc[q]; t->buc[i] = sv->buc[q];
    }
    free(sv->idx); free(sv->bkc); free(sv->blc); free(sv->buc);
    sv->idx = NULL; sv->bkc = NULL; sv->blc = sv->buc = NULL; sv->n = 0;
}

/* Crossover IPM -> basis (MSK_IPAR_INTPNT_BASIS = ALWAYS): from an INTERIOR
 * point it identifies a basis. A variable is NONBASIC at a bound if x_j is at
 * that bound (within tol), otherwise BASIC; a constraint is NONBASIC (active)
 * if its activity is at a bound, otherwise BASIC. It does not guarantee
 * non-singularity (declared deviation: no crash/simplex cleanup). */
static void intpnt_identify_basis(PRIMALtask_t t)
{
    if (!t || !t->has_sol) return;
    int nvar = t->numvar, ncon = t->numcon;
    if (nvar <= 0 || ncon <= 0) return;
    if (!t->skx) t->skx = (PRIMALstakeye *)calloc((size_t)nvar, sizeof(PRIMALstakeye));
    if (!t->skc) t->skc = (PRIMALstakeye *)calloc((size_t)ncon, sizeof(PRIMALstakeye));
    if (!t->skx || !t->skc) return;
    double tol = 1e-7;
    for (int j = 0; j < nvar; j++) {
        double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (isfinite(lo) && t->x[j] <= lo + tol * (1.0 + fabs(lo))) t->skx[j] = PRIMAL_SK_LOW;
        else if (isfinite(up) && t->x[j] >= up - tol * (1.0 + fabs(up))) t->skx[j] = PRIMAL_SK_UPR;
        else t->skx[j] = PRIMAL_SK_BAS;
    }
    double *act = (double *)calloc((size_t)ncon, sizeof(double));
    if (!act) return;
    for (int j = 0; j < nvar; j++) {
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) act[c->sub[k]] += c->val[k] * t->x[j];
    }
    for (int i = 0; i < ncon; i++) {
        double lo, up;
        bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
        if (isfinite(lo) && act[i] <= lo + tol * (1.0 + fabs(lo))) t->skc[i] = PRIMAL_SK_LOW;
        else if (isfinite(up) && act[i] >= up - tol * (1.0 + fabs(up))) t->skc[i] = PRIMAL_SK_UPR;
        else t->skc[i] = PRIMAL_SK_BAS;
    }
    free(act);
}

/* Crossover cleanup: the basis identified from the interior point may not be
 * square or non-singular (degenerate LPs). It checks the basis (|basic| == m
 * and B non-singular via LU); if it does not hold, it REBUILDS it with the
 * crash basis over the general-form columns (variables A_j + slacks e_i).
 * Deviation: it does not re-optimize with the simplex, it publishes only a
 * feasible basis. */
static void intpnt_crossover_cleanup(PRIMALtask_t t)
{
    if (!t || !t->skx || !t->skc) return;
    int nvar = t->numvar, ncon = t->numcon, m = ncon;
    if (nvar <= 0 || m <= 0) return;
    int nb = 0;
    for (int j = 0; j < nvar; j++) if (t->skx[j] == PRIMAL_SK_BAS) nb++;
    for (int i = 0; i < ncon; i++) if (t->skc[i] == PRIMAL_SK_BAS) nb++;
    if (nb == m) {
        double *B = (double *)calloc((size_t)m * (size_t)m, sizeof(double));
        if (B) {
            int col = 0;
            for (int j = 0; j < nvar && col < m; j++) {
                if (t->skx[j] != PRIMAL_SK_BAS) continue;
                const Col *c = &t->cols[j];
                for (int k = 0; k < c->nz; k++) B[(size_t)c->sub[k] * m + col] += c->val[k];
                col++;
            }
            for (int i = 0; i < ncon && col < m; i++) {
                if (t->skc[i] != PRIMAL_SK_BAS) continue;
                B[(size_t)i * m + col] = 1.0;
                col++;
            }
            LuFact *f = (col == m) ? dmat_lu_factor(B, m) : NULL;
            if (f) { dmat_lu_free(f); free(B); return; }   /* valid basis */
            free(B);
        }
    }
    /* invalid basis: crash basis over the general-form columns (A_j | e_i) */
    int nc = nvar + ncon;
    double *C = (double *)calloc((size_t)m * (size_t)nc, sizeof(double));
    int *bas = (int *)malloc((size_t)m * sizeof(int));
    if (!C || !bas) { free(C); free(bas); return; }
    for (int j = 0; j < nvar; j++) {
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) C[(size_t)c->sub[k] * nc + j] += c->val[k];
    }
    for (int i = 0; i < ncon; i++) C[(size_t)i * nc + nvar + i] = 1.0;
    if (simplex_crash_basis(C, m, nc, bas)) {
        for (int j = 0; j < nvar; j++) t->skx[j] = PRIMAL_SK_LOW;
        for (int i = 0; i < ncon; i++) t->skc[i] = PRIMAL_SK_LOW;
        for (int b = 0; b < m; b++) {
            int k = bas[b];
            if (k < nvar) t->skx[k] = PRIMAL_SK_BAS;
            else t->skc[k - nvar] = PRIMAL_SK_BAS;
        }
    }
    free(C); free(bas);
}

/**
 * Main optimization routine: solves the optimization problem.
 *
 * @param t [in] Task handle containing the model to solve.
 *
 * @return PRIMAL_RES_OK on success (optimal solution found),
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if the model is invalid (e.g., invalid bounds),
 *         PRIMAL_RES_TRM_MAX_ITER if iteration limit reached,
 *         PRIMAL_RES_TRM_MAX_TIME if time limit reached,
 *         PRIMAL_RES_TRM_OBJECTIVE_RANGE if objective cut triggered,
 *         PRIMAL_RES_ERR_INFEASIBLE if primal infeasible (certificate available via getdualray),
 *         PRIMAL_RES_ERR_UNBOUNDED if dual infeasible (certificate available via getprimalray),
 *         Other error codes for specific failure modes.
 *
 * @note This is the main entry point for solving. The function dispatches to
 *       the appropriate solver based on the model structure:
 *       - MIP (integer variables, semi-continuous, SOS): branch & bound
 *       - SDP (PSD variables): primal-dual IPM with tangent-cut fallback
 *       - QP with cones / quadratic constraints: RQUAD encoding + conic IPM
 *       - Pure conic (SOC/exp-power): unified conic IPM (native or tangent-cut)
 *       - LP/QP: Mehrotra IPM or dense simplex
 *
 *       The solver evaluates wall-clock time limits (PRIMAL_DPAR_OPTIMIZER_MAX_TIME,
 *       PRIMAL_DPAR_MIO_MAX_TIME), objective cuts (PRIMAL_DPAR_LOWER_OBJ_CUT,
 *       PRIMAL_DPAR_UPPER_OBJ_CUT), and iteration limits.
 *
 *       After a successful solve, the solution is available via getxx, gety, etc.
 *       For infeasible/unbounded problems, certificates are available via
 *       getdualray/getprimalray (LP/QP only; conic/SDP/MIP do not publish certificates).
 *
 *       The conic/SDP path includes a "fourth face" check: the published point
 *       must satisfy cone membership (PSD, SOC, exp-power) within tolerances.
 *       If not, the point is refused and the model is interrogated for a
 *       certificate (unbounded direction or infeasibility ray).
 *
 * @example
 * PRIMALrescodee rc = PRIMAL_optimize(task);
 * if (rc == PRIMAL_RES_OK) {
 *     double *x = malloc(task->numvar * sizeof(double));
 *     PRIMAL_getxx(task, x);
 *     printf("Optimal value: %f\n", task->pobj);
 * }
 */
PRIMALrescodee PRIMAL_optimize(PRIMALtask_t t) {
    clock_t opt_t0 = clock();
    /* Wall-clock cap: PRIMAL_DPAR_OPTIMIZER_MAX_TIME seconds (<0 = no limit). */
    double deadline = (t && t->optimizer_max_time >= 0.0)
        ? (double)opt_t0 + t->optimizer_max_time * (double)CLOCKS_PER_SEC : -1.0;
    double mip_deadline = (t && t->mio_max_time >= 0.0)
        ? (double)opt_t0 + t->mio_max_time * (double)CLOCKS_PER_SEC : -1.0;
    /* The B&B obeys the tighter of the two caps. */
    if (deadline >= 0.0 && (mip_deadline < 0.0 || deadline < mip_deadline)) mip_deadline = deadline;
    if (t) { t->opt_deadline = deadline; t->mip_deadline = mip_deadline; }
    ipm_set_deadline(deadline);
    ipm_set_max_cor(t ? t->intpnt_max_cor : -1);
    /* Objective cuts in min-space: for a minimization the two cuts are as
     * declared; for a maximization the mirrored pair would need the dual side
     * (not wired, so both are disabled). */
    if (t && t->sense != PRIMAL_OPTIMIZE_MAXIMIZE)
        ipm_set_obj_cuts(t->lower_obj_cut, t->upper_obj_cut);
    else
        ipm_set_obj_cuts(-1e308, 1e308);
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_OPTIMIZER);
    PRIMALrescodee r = opt_routes(t);
    int cut_hit = ipm_obj_cut_hit();
    ipm_set_deadline(-1.0);
    ipm_set_max_cor(1);
    ipm_set_obj_cuts(-1e308, 1e308);
    /* A fired cap is its own termination code; the conic verdicts below are
     * skipped (a timed-out solve has no answer to judge). */
    if (cut_hit) r = PRIMAL_RES_TRM_OBJECTIVE_RANGE;
    else if (r == PRIMAL_RES_TRM_MAX_ITER
        && ((deadline >= 0.0 && (double)clock() >= deadline)
            || (mip_deadline >= 0.0 && (double)clock() >= mip_deadline)))
        r = PRIMAL_RES_TRM_MAX_TIME;
    cb_fire(t, PRIMAL_CALLBACK_END_OPTIMIZER);
    if (r != PRIMAL_RES_OK && t && t->respfn) t->respfn(t, t->resphandle, r);
    /* The fourth face applies to every conic block, PSD included: the gate no
     * longer consults only bar-free models, and `conic_dual_verdict` measures
     * the bar blocks too. Quadratic terms remain out, since neither
     * `cone_dual_worst` nor `model_lp_witness` carries them. */
    int conic_only = t && (t->numcones > 0 || t->numbarvar > 0) &&
                     !t->has_qcon && !t->has_qobj;
    int s = (t && t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    /* A cone-only model has one more side to judge than the rows do, and it is
     * judged here rather than in a route because this is the only place the
     * model the USER wrote is answered: a shadow task is solved by calling a
     * route directly, and its cones are the encoder's, not the model's. */
    if (r == PRIMAL_RES_OK && conic_only && t->has_sol) {
        double near = (t->tol_near_rel > 1.0) ? t->tol_near_rel : 1.0;
        if (conic_primal_cone_worst(t) < -t->tol_co_pfeas * near) {
            /* The published point is outside the model's cones: it is not a
             * solution, and the route's own triple did not see it. Refuse it and
             * ask the model, the same way a route that ran out of iterations is
             * asked. */
            t->has_sol = 0;
            t->solsta = PRIMAL_SOL_STA_UNKNOWN;
            r = conic_no_answer_verdict(t, s, PRIMAL_RES_TRM_MAX_ITER);
        } else {
            r = conic_dual_verdict(t, s);
        }
    }
    /* And when a route gave up, the model is still asked what it is. */
    if (r == PRIMAL_RES_TRM_MAX_ITER && conic_only)
        r = conic_no_answer_verdict(t, s, r);
    /* One line per solve that published something into a conic/SDP model: the
     * triple says how well the equations were met, this says whether the point
     * is in the cones at all. */
    if (r == PRIMAL_RES_OK && getenv("GMB_DBG") && t &&
        (t->numcones > 0 || t->numbarvar > 0 || t->has_qcon > 0) && t->has_sol) opt_report_cones(t);
    /* Crossover IPM->basis: on a solved LP it publishes a basis identified from
     * the interior point (declared deviation: no cleanup with the simplex). */
    if (r == PRIMAL_RES_OK && t && t->has_sol && !t->has_qobj && t->has_qcon == 0 &&
        t->numcones == 0 && t->numbarvar == 0) {
        intpnt_identify_basis(t);
        intpnt_crossover_cleanup(t);
    }
    if (t) { t->last_rc = r; t->opt_time = (double)(clock() - opt_t0) / (double)CLOCKS_PER_SEC; }
    return r;
}

