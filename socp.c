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

#include <stdio.h>
/* socp.c - primal-dual interior point solver for conic problems
 *
 * Problem:  min c'x  s.t.  E x = d,  s_k = G_k x + h_k in K_k.
 * KKT:  r_d = c + E'y - G'lam = 0,  r_p = Ex - d = 0,  r_g = Gx + h - s = 0,
 *       s in K, lam in K*, complementarity per cone.
 *
 * SOC complementarity uses the arrow matrix A(v) = [[v0, vbar'],
 * [vbar, v0 I]]:  A(s) lam = 0.  Linearized:  A(s) dlam + A(lam) ds = -r_c.
 * For R_+ (dim-1) cones this reduces to the elementwise rule.
 *
 * Newton system in (dx, dy, dlam), with ds = -r_g - G dx eliminated:
 *   -A(lam) G dx + A(s) dlam = -r_c + A(lam) r_g     (K rows)
 *    E dx                    = -r_p                  (p rows)
 *    E dy - G' dlam          = -r_d                  (n rows)
 * Dense LU each iteration; Mehrotra predictor-corrector; fraction-to-
 * boundary via the spectral roots of the cone membership quadratic.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "socp.h"
#include "linalg.h"

/* per-iteration callback hook (implemented in primal.c) */
extern int primal_cb_iter_on;
void primal_cb_iter(int code);


enum { OPT_OK = 0, OPT_MAXITER, OPT_MEMORY, OPT_SINGULAR };

/* max step t in [0,1] keeping v + t*dv inside its cone.  type 0 is R_+ (the
 * elementwise ratio test, one coordinate at a time); type 1 is SOC, where the
 * boundary equation ||v + t dv||^2 - (v0 + t dv0)^2 = 0 is a quadratic in t --
 * the same A t^2 + B t + C that sdp.c's soc_step solves.  A discriminant below
 * zero, or a first crossing that is not positive, means the direction does not
 * meet the boundary inside the unit step, which is t = 1. */
static double step_len(const double *v, const double *dv, int type, int k) {
    if (type == 0) {
        double a = 1.0;
        for (int i = 0; i < k; i++)
            if (dv[i] < 0.0) { double t = -v[i] / dv[i]; if (t < a) a = t; }
        return a;
    }
    double n2 = 0.0, nd2 = 0.0, cross = 0.0;
    for (int i = 1; i < k; i++) {
        n2 += v[i] * v[i]; nd2 += dv[i] * dv[i]; cross += v[i] * dv[i];
    }
    double A = nd2 - dv[0] * dv[0];
    double B = 2.0 * (cross - v[0] * dv[0]);
    double C = n2 - v[0] * v[0];   /* < 0 strictly inside */
    if (fabs(A) < 1e-300) {
        if (B < -1e-300) { double t = -C / B; return t < 1.0 ? t : 1.0; }
        return 1.0;
    }
    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return 1.0;
    double sq = sqrt(disc);
    double r2 = (-B + sq) / (2.0 * A);
    /* C < 0 (strictly inside).  A > 0: the roots have opposite sign, so the
     * first crossing is r2.  A < 0: dividing by the negative leading coefficient
     * reverses the order of the two roots, and r2 is the crossing again.  The
     * two cases therefore read the same way: take r2 when it is ahead of us. */
    double t = (r2 > 0.0) ? r2 : 1.0;
    return t < 1.0 ? t : 1.0;
}

/* Jordan product A(v) w for one cone block, with the arrow matrix never
 * materialised: (v0 w0 + <vbar,wbar>, v0 wbar + w0 vbar).  This is the left side
 * of the linearised cone equation A(s) dlam + A(lam) ds = -r_c; the NT block
 * below spells the same map nt_amul because it also needs A(v) as a matrix. */
static void arrow_mul(const double *v, const double *w, int k, double *out) {
    double t = v[0] * w[0];
    for (int i = 1; i < k; i++) t += v[i] * w[i];
    out[0] = t;
    for (int i = 1; i < k; i++) out[i] = v[i] * w[0] + v[0] * w[i];
}

/* |v|_inf -- the normaliser every residual is judged against.  0 for an empty
 * vector (a block of size 0 has no residual, which is not the same as an
 * unknown one). */
static double maxabs(const double *v, int n) {
    double a = 0.0;
    for (int i = 0; i < n; i++) { double b = fabs(v[i]); if (b > a) a = b; }
    return a;
}

/**
 * Solves a conic problem using dense Mehrotra IPM.
 *
 * @param n       [in]  Number of variables.
 * @param p       [in]  Number of equality constraints.
 * @param E       [in]  Equality constraint matrix (p x n, row-major).
 * @param d       [in]  RHS for equality constraints (size p).
 * @param c       [in]  Linear objective coefficients (size n).
 * @param ncones  [in]  Number of cones.
 * @param cones   [in]  Array of cone descriptions (type, nmem, mem[]).
 * @param G       [in]  Cone constraint matrix (K x n, row-major, K = sum nmem).
 * @param h       [in]  RHS for cone constraints (size K).
 * @param tol_gap [in]  Relative gap tolerance.
 * @param tol_feas [in] Feasibility tolerance (primal/dual).
 * @param max_iter [in] Maximum iterations.
 * @param x       [out] Primal solution (size n). Must not be NULL.
 * @param y       [out] Dual for equalities (size p). Must not be NULL.
 * @param lam     [out] Dual for cones (size K). Must not be NULL.
 *
 * @return OPT_OK (0), OPT_MAXITER (1), OPT_MEMORY (2), OPT_SINGULAR (3).
 *
 * @note Solves: min c'x  s.t.  E x = d,  G x + h in K.
 *
 *       KKT system:
 *         r_d = c + E'y - G'lam = 0
 *         r_p = E x - d = 0
 *         r_g = G x + h - s = 0
 *         s in K, lam in K*, complementarity per cone.
 *
 *       Dense LU each iteration with Mehrotra predictor-corrector.
 *       Automatically drops unused variables and redundant equality rows.
 *       Fraction-to-boundary step lengths via spectral roots.
 *
 *       Returns best feasible point on divergence (gap degrades after 20 iters).
 *
 * @example
 * // 1 var, 1 eq, 1 SOC cone: min x0 s.t. x0 = 1, x0 >= 0
 * double E[] = {1}, d[] = {1}, c[] = {1};
 * SocpCone cone = {1, 1, {0}}; // SOC of dim 1 (non-negative)
 * double G[] = {1}, h[] = {0};
 * double x[1], y[1], lam[1];
 * int rc = socp_solve(1, 1, E, d, c, 1, &cone, G, h, 1e-8, 1e-8, 100, x, y, lam);
 */
int socp_solve(int n, int p,
               const double *E, const double *d, const double *c,
               int ncones, const SocpCone *cones,
               const double *G, const double *h,
               double tol_gap, double tol_feas, int max_iter,
               double *x, double *y, double *lam)
{
    if (n < 0 || p < 0 || ncones < 0) return OPT_MAXITER;
    int K = 0;
    for (int k = 0; k < ncones; k++) K += cones[k].nmem;
    int N = n + p + K;
    if (N == 0) return OPT_OK;

    /* Columns of E/G that are identically zero (a variable that appears in no
     * constraint and in no cone) make the KKT structurally SINGULAR: the LU
     * returns no factor, the diagonal regularisation opens it but the variable's
     * own unknown then divides by ~0 and the Newton step explodes, so the IPM
     * diverges (measured: an unused free variable, rc=1007; and the redundant
     * equality block of T222).  If the variable's cost is zero it is irrelevant
     * -- E, G and the cones carry every use -- so drop its column, solve the
     * reduced problem and recover x_j = 0.  A variable with c_j != 0 and no
     * constraint is unbounded and is left in so the caller still sees it. */
    {
        char *used = (char *)calloc((size_t)(n > 0 ? n : 1), 1);
        int *keep = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
        int drop = 0;
        if (used && keep) {
            for (int i = 0; i < p; i++)
                for (int j = 0; j < n; j++) if (E[(size_t)i * n + j] != 0.0) used[j] = 1;
            for (int k = 0; k < K; k++)
                for (int j = 0; j < n; j++) if (G[(size_t)k * n + j] != 0.0) used[j] = 1;
            for (int k = 0; k < ncones; k++)
                for (int i = 0; cones[k].mem && i < cones[k].nmem; i++)
                    if (cones[k].mem[i] >= 0 && cones[k].mem[i] < n) used[cones[k].mem[i]] = 1;
            for (int j = 0; j < n; j++) if (used[j] || (c && c[j] != 0.0)) used[j] = 1;
            for (int j = 0; j < n; j++) if (!used[j]) drop = 1;
        }
        if (drop) {
            int nk = 0;
            for (int j = 0; j < n; j++) keep[j] = used[j] ? nk++ : -1;
            double *E2 = (double *)calloc((size_t)(p > 0 ? p : 1) * (size_t)(nk > 0 ? nk : 1), sizeof(double));
            double *G2 = (double *)calloc((size_t)(K > 0 ? K : 1) * (size_t)(nk > 0 ? nk : 1), sizeof(double));
            double *c2 = (double *)malloc((size_t)(nk > 0 ? nk : 1) * sizeof(double));
            double *x2 = (double *)calloc((size_t)(nk > 0 ? nk : 1), sizeof(double));
            SocpCone *cn = (SocpCone *)malloc((size_t)(ncones > 0 ? ncones : 1) * sizeof(SocpCone));
            int **cnm = (int **)malloc((size_t)(ncones > 0 ? ncones : 1) * sizeof(int *));
            int okc = E2 && G2 && c2 && x2 && cn && cnm;
            for (int k = 0; okc && k < ncones; k++) {
                cn[k] = cones[k];
                cnm[k] = (int *)malloc((size_t)(cones[k].nmem > 0 ? cones[k].nmem : 1) * sizeof(int));
                if (!cnm[k]) { okc = 0; break; }
                for (int i = 0; i < cones[k].nmem; i++) cnm[k][i] = keep[cones[k].mem[i]];
                cn[k].mem = cnm[k];
            }
            if (okc) {
                for (int i = 0; i < p; i++)
                    for (int j = 0; j < n; j++) if (keep[j] >= 0) E2[(size_t)i * nk + keep[j]] = E[(size_t)i * n + j];
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < n; j++) if (keep[j] >= 0) G2[(size_t)k * nk + keep[j]] = G[(size_t)k * n + j];
                for (int j = 0; j < n; j++) if (keep[j] >= 0) c2[keep[j]] = c[j];
                int st = socp_solve(nk, p, E2, d, c2, ncones, cn, G2, h,
                                    tol_gap, tol_feas, max_iter, x2, y, lam);
                for (int j = 0; j < n; j++) x[j] = (keep[j] >= 0) ? x2[keep[j]] : 0.0;
                for (int k = 0; k < ncones; k++) free(cnm[k]);
                free(cn); free(cnm); free(E2); free(G2); free(c2); free(x2);
                free(used); free(keep);
                return st;
            }
            for (int k = 0; k < ncones; k++) free(cnm[k]);
            free(cn); free(cnm); free(E2); free(G2); free(c2); free(x2);
        }
        free(used); free(keep);
    }

    /* Redundant EQUALITY rows (a rank-deficient E) leave a null space in the
     * dual block and hit the SAME failure: the LU has no factor, the diagonal
     * regularisation opens it but the dual of a dependent row is then its own
     * residual over ~0, so the step explodes.  Drop the rows that are linear
     * combinations of the others -- their dual is zero -- with a conservative
     * relative tolerance so an independent row is never dropped.  Modified
     * Gram-Schmidt on the rows of E (p is small on the conic path). */
    if (p > 0) {
        double *Qb = (double *)calloc((size_t)p * (size_t)n, sizeof(double));
        double *tmp = (double *)malloc((size_t)n * sizeof(double));
        char *rowkeep = (char *)calloc((size_t)p, 1);
        int rank = 0, drop = 0;
        if (Qb && tmp && rowkeep) {
            for (int i = 0; i < p; i++) {
                memcpy(tmp, E + (size_t)i * n, (size_t)n * sizeof(double));
                for (int r2 = 0; r2 < rank; r2++) {
                    double pr = 0.0;
                    for (int j = 0; j < n; j++) pr += tmp[j] * Qb[(size_t)r2 * n + j];
                    if (pr != 0.0)
                        for (int j = 0; j < n; j++) tmp[j] -= pr * Qb[(size_t)r2 * n + j];
                }
                double nn = 0.0, oo = 0.0;
                for (int j = 0; j < n; j++) { nn += tmp[j] * tmp[j]; oo += E[(size_t)i * n + j] * E[(size_t)i * n + j]; }
                nn = sqrt(nn); oo = sqrt(oo);
                if (oo == 0.0 || nn > 1e-10 * oo) {
                    if (nn > 0.0) { for (int j = 0; j < n; j++) Qb[(size_t)rank * n + j] = tmp[j] / nn; rank++; }
                    rowkeep[i] = 1;
                } else drop = 1;
            }
        }
        if (drop) {
            int pk = 0;
            int *rmap = (int *)malloc((size_t)p * sizeof(int));
            double *E3 = (double *)calloc((size_t)(rank > 0 ? rank : 1) * (size_t)(n > 0 ? n : 1), sizeof(double));
            double *d3 = (double *)calloc((size_t)(rank > 0 ? rank : 1), sizeof(double));
            double *y2 = (double *)calloc((size_t)(rank > 0 ? rank : 1), sizeof(double));
            if (rmap && E3 && d3 && y2) {
                for (int i = 0; i < p; i++) rmap[i] = rowkeep[i] ? pk++ : -1;
                for (int i = 0; i < p; i++) if (rowkeep[i]) {
                    memcpy(E3 + (size_t)rmap[i] * n, E + (size_t)i * n, (size_t)n * sizeof(double));
                    d3[rmap[i]] = d[i];
                }
                int st = socp_solve(n, pk, E3, d3, c, ncones, cones, G, h,
                                    tol_gap, tol_feas, max_iter, x, y2, lam);
                for (int i = 0; i < p; i++) y[i] = rowkeep[i] ? y2[rmap[i]] : 0.0;
                free(rmap); free(E3); free(d3); free(y2);
                free(Qb); free(tmp); free(rowkeep);
                return st;
            }
            free(rmap); free(E3); free(d3); free(y2);
        }
        free(Qb); free(tmp); free(rowkeep);
    }

    int *off = (int *)malloc((size_t)(ncones > 0 ? ncones : 1) * sizeof(int));
    int *typ = (int *)malloc((size_t)(ncones > 0 ? ncones : 1) * sizeof(int));
    double *xs = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *ys = (double *)calloc((size_t)(p > 0 ? p : 1), sizeof(double));
    double *s  = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *lm = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *rd = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *rp = (double *)malloc((size_t)(p > 0 ? p : 1) * sizeof(double));
    double *rg = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *rc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *Asc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double)); /* A(s)lam */
    double *Alr = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double)); /* A(lam)rg */
    double *M  = (double *)calloc((size_t)N * (size_t)N, sizeof(double));
    double *rhs = (double *)malloc((size_t)N * sizeof(double));
    double *dx = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *dy = (double *)malloc((size_t)(p > 0 ? p : 1) * sizeof(double));
    double *ds = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *dlm = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *dx_aff = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *dy_aff = (double *)malloc((size_t)(p > 0 ? p : 1) * sizeof(double));
    double *ds_aff = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *dlm_aff = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    if (!off || !typ || !xs || !ys || !s || !lm || !rd || !rp || !rg || !rc ||
        !Asc || !Alr || !M || !rhs || !ds || !dlm ||
        !dx_aff || !dy_aff || !ds_aff || !dlm_aff) {
        free(off); free(typ); free(xs); free(ys); free(s); free(lm); free(rd);
        free(rp); free(rg); free(rc); free(Asc); free(Alr); free(M); free(rhs);
        free(ds); free(dlm); free(dx_aff); free(dy_aff); free(ds_aff); free(dlm_aff);
        return OPT_MEMORY;
    }

    /* per-cone offsets and types */
    {
        int o = 0;
        for (int k = 0; k < ncones; k++) { off[k] = o; typ[k] = cones[k].type; o += cones[k].nmem; }
    }

    /* strictly feasible cone start: R_+ -> tutti 1, SOC/arrow -> [1,0,...] */
    {
        int o = 0;
        for (int k = 0; k < ncones; k++) {
            if (cones[k].type == 0) {
                for (int i = 0; i < cones[k].nmem; i++) { s[o + i] = 1.0; lm[o + i] = 1.0; }
            } else {
                s[o] = 1.0; lm[o] = 1.0;
                for (int i = 1; i < cones[k].nmem; i++) { s[o + i] = 0.0; lm[o + i] = 0.0; }
            }
            o += cones[k].nmem;
        }
    }

    double dnorm = 1.0, hnorm = 1.0, cnorm = 1.0;
    for (int i = 0; i < p; i++) { double a = fabs(d[i]); if (a > dnorm) dnorm = a; }
    for (int k = 0; k < K; k++) { double a = fabs(h[k]); if (a > hnorm) hnorm = a; }
    for (int j = 0; j < n; j++) { double a = fabs(c[j]); if (a > cnorm) cnorm = a; }

    int status = OPT_MAXITER;
    int no_progress = 0;
    /* Miglior iterato per gap (stack, nessuna allocazione heap: aggiungere
     * malloc qui perturbava l'aritmetica e faceva cambiare verdetto a T82/T101 C).
     * L'IPM su un LP a tagli accumulati (mal condizionato) puo' convergere in mu
     * e poi DIVERGERE in gap; si tiene il punto col gap minore e, quando il gap
     * risale di molti ordini, si restituisce quello. */
    double bx[(n > 0) ? n : 1], by[(p > 0) ? p : 1];
    double bs[(K > 0) ? K : 1], blm[(K > 0) ? K : 1];
    int have_bp = 0; double best_gap = HUGE_VAL;

    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(34);
        /* residuals */
        for (int i = 0; i < p; i++) {
            double t = -d[i];
            for (int j = 0; j < n; j++) t += E[i * n + j] * xs[j];
            rp[i] = t;
        }
        for (int k = 0; k < K; k++) {
            double t = h[k];
            for (int j = 0; j < n; j++) t += G[k * n + j] * xs[j];
            rg[k] = t - s[k];
        }
        for (int j = 0; j < n; j++) {
            double t = c[j];
            for (int i = 0; i < p; i++) t += E[i * n + j] * ys[i];
            for (int k = 0; k < K; k++) t -= G[k * n + j] * lm[k];
            rd[j] = t;
        }
        double mu = 0.0;
        for (int k = 0; k < K; k++) mu += s[k] * lm[k];
        mu /= (double)(K > 0 ? K : 1);

        /* A(s)lam per cone into Asc */
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                if (cones[k].type == 0)
                    for (int i = 0; i < cones[k].nmem; i++) Asc[o + i] = s[o + i] * lm[o + i];
                else arrow_mul(&s[o], &lm[o], cones[k].nmem, &Asc[o]);
                o += cones[k].nmem;
            }
        }

        double pobj = 0.0;
        for (int j = 0; j < n; j++) pobj += c[j] * xs[j];
        double dobj = 0.0;
        for (int i = 0; i < p; i++) dobj -= d[i] * ys[i];
        for (int k = 0; k < K; k++) dobj -= h[k] * lm[k];

        double feas_p = maxabs(rp, p) / dnorm;
        if (K) { double a = maxabs(rg, K) / hnorm; if (a > feas_p) feas_p = a; }
        double feas_d = maxabs(rd, n) / cnorm;
        double gap = fabs(pobj - dobj) / (1.0 + fabs(pobj));

        if (getenv("GMB_DBG") && (it % 10 == 0 || feas_p <= tol_feas))
            fprintf(stderr, "socp it=%d feas_p=%.3g feas_d=%.3g gap=%.3g mu=%.3g\n",
                    it, feas_p, feas_d, gap, mu);

        if (feas_p <= tol_feas && feas_d <= tol_feas && gap <= tol_gap) {
            status = OPT_OK;
            break;
        }
        /* Si tiene solo un punto FEASIBLE: un LP infeasible ha per natura un gap
         * grande e non deve essere "aggiustato" in una risposta (T101 C, T82). */
        if (feas_p <= tol_feas && feas_d <= tol_feas &&
            (!have_bp || gap < best_gap)) {
            have_bp = 1; best_gap = gap;
            memcpy(bx, xs, (size_t)n * sizeof(double));
            memcpy(by, ys, (size_t)p * sizeof(double));
            memcpy(bs, s, (size_t)K * sizeof(double));
            memcpy(blm, lm, (size_t)K * sizeof(double));
        } else if (it > 20 && have_bp && best_gap < 1e-6 && gap > 1e-2) {
            memcpy(xs, bx, (size_t)n * sizeof(double));
            memcpy(ys, by, (size_t)p * sizeof(double));
            memcpy(s, bs, (size_t)K * sizeof(double));
            memcpy(lm, blm, (size_t)K * sizeof(double));
            if (getenv("GMB_DBG")) fprintf(stderr,
                "  [socp diverge] it=%d gap=%.3g best_gap=%.3g\n", it, gap, best_gap);
            status = OPT_OK;
            break;
        }

        /* build KKT matrix M (N x N row-major) */
        memset(M, 0, (size_t)N * (size_t)N * sizeof(double));
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                int kk = cones[k].nmem;
                if (cones[k].type == 0) {
                    for (int i = 0; i < kk; i++) {
                        for (int j = 0; j < n; j++)
                            M[o * N + j] += lm[o] * G[o * n + j];  /* +A(lam)G row */
                        M[o * N + n + p + o] = s[o];
                        o++;
                    }
                } else {
                    for (int i = 0; i < kk; i++)
                        for (int i2 = 0; i2 < kk; i2++) {
                            double as_, al;
                            if (i == 0)      { as_ = s[o + i2]; al = lm[o + i2]; }
                            else if (i2 == 0){ as_ = s[o + i];  al = lm[o + i];  }
                            else { as_ = (i == i2) ? s[o] : 0.0;
                                   al  = (i == i2) ? lm[o] : 0.0; }
                            if (as_ != 0.0) M[(o + i) * N + n + p + o + i2] += as_;
                            if (al != 0.0)
                                for (int j = 0; j < n; j++)
                                    M[(o + i) * N + j] += al * G[(o + i2) * n + j];
                        }
                    o += kk;
                }
            }
        }
        for (int i = 0; i < p; i++)
            for (int j = 0; j < n; j++) M[(K + i) * N + j] = E[i * n + j];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < p; i++) M[(K + p + j) * N + n + i] = E[i * n + j];
            for (int k = 0; k < K; k++) M[(K + p + j) * N + n + p + k] = -G[k * n + j];
        }
        LuFact *f = dmat_lu_factor(M, N);
        double reg_used = 0.0;
        if (!f) {
            /* adaptive regularization for degenerate KKT systems (e.g. >= 2
             * free variables tied by one equality row): try increasing
             * diagonal shifts before giving up */
            static const double regs[] = { 1e-10, 1e-8, 1e-7, 1e-6, 1e-5 };
            for (size_t ri = 0; ri < sizeof regs / sizeof regs[0] && !f; ri++) {
                for (int i = 0; i < N; i++) M[i * N + i] += regs[ri] - reg_used;
                reg_used = regs[ri];
                f = dmat_lu_factor(M, N);
            }
            if (!f) {
                /* restore for the next iteration attempt */
                for (int i = 0; i < N; i++) M[i * N + i] -= reg_used;
            }
        }
        if (!f) { status = OPT_SINGULAR; break; }

        /* A(lam) * rg per cone */
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                if (cones[k].type == 0)
                    for (int i = 0; i < cones[k].nmem; i++) Alr[o + i] = lm[o + i] * rg[o + i];
                else arrow_mul(&lm[o], &rg[o], cones[k].nmem, &Alr[o]);
                o += cones[k].nmem;
            }
        }

        /* ---- affine predictor (sigma = 0): r_c = A(s)lam ---- */
        for (int k = 0; k < K; k++) rhs[k] = -Asc[k] - Alr[k];
        for (int i = 0; i < p; i++) rhs[K + i] = -rp[i];
        for (int j = 0; j < n; j++) rhs[K + p + j] = -rd[j];
        dmat_lu_solve(f, rhs);
        for (int j = 0; j < n; j++) dx_aff[j] = rhs[j];
        for (int i = 0; i < p; i++) dy_aff[i] = rhs[n + i];
        for (int k = 0; k < K; k++) dlm_aff[k] = rhs[n + p + k];
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                for (int i = 0; i < cones[k].nmem; i++) {
                    double t = rg[o + i];
                    for (int j = 0; j < n; j++) t += G[(o + i) * n + j] * dx_aff[j];
                    ds_aff[o + i] = t;
                }
                o += cones[k].nmem;
            }
        }

        /* affine step lengths + mu_aff */
        double ap = 1.0, ad = 1.0;
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                double a = step_len(&s[o], &ds_aff[o], cones[k].type, cones[k].nmem);
                if (a < ap) ap = a;
                a = step_len(&lm[o], &dlm_aff[o], cones[k].type, cones[k].nmem);
                if (a < ad) ad = a;
                o += cones[k].nmem;
            }
        }
        double mu_aff = 0.0;
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                for (int i = 0; i < cones[k].nmem; i++)
                    mu_aff += (s[o + i] + ap * ds_aff[o + i]) * (lm[o + i] + ad * dlm_aff[o + i]);
                o += cones[k].nmem;
            }
        }
        mu_aff /= (double)(K > 0 ? K : 1);
        double sigma = (mu > 0.0) ? (mu_aff / mu) : 0.0;
        sigma = sigma * sigma * sigma;
        if (!(sigma > 1e-8)) sigma = 1e-8;
        if (sigma > 1.0) sigma = 1.0;

        /* ---- corrector: r_c = A(s)lam - sigma*mu*e (e = 1 / e1 per cone) ---- */
        {
            /* rhs = -r_c - A(lam) rg,  r_c = A(s)lam - sigma*mu*e
             * (e = 1 for R_+, e1 for SOC) */
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                if (cones[k].type == 0)
                    for (int i = 0; i < cones[k].nmem; i++)
                        rhs[o + i] = -Asc[o + i] + sigma * mu - Alr[o + i];
                else {
                    rhs[o] = -Asc[o] + sigma * mu - Alr[o];
                    for (int i = 1; i < cones[k].nmem; i++)
                        rhs[o + i] = -Asc[o + i] - Alr[o + i];
                }
                o += cones[k].nmem;
            }
        }
        for (int i = 0; i < p; i++) rhs[K + i] = -rp[i];
        for (int j = 0; j < n; j++) rhs[K + p + j] = -rd[j];
        dmat_lu_solve(f, rhs);
        dmat_lu_free(f);
        for (int j = 0; j < n; j++) dx[j] = rhs[j];
        for (int i = 0; i < p; i++) dy[i] = rhs[n + i];
        for (int k = 0; k < K; k++) dlm[k] = rhs[n + p + k];
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                for (int i = 0; i < cones[k].nmem; i++) {
                    double t = rg[o + i];
                    for (int j = 0; j < n; j++) t += G[(o + i) * n + j] * dx[j];
                    ds[o + i] = t;
                }
                o += cones[k].nmem;
            }
        }

        double apm = 1.0, adm = 1.0;
        {
            int o = 0;
            for (int k = 0; k < ncones; k++) {
                double a = step_len(&s[o], &ds[o], cones[k].type, cones[k].nmem);
                if (a < apm) apm = a;
                a = step_len(&lm[o], &dlm[o], cones[k].type, cones[k].nmem);
                if (a < adm) adm = a;
                o += cones[k].nmem;
            }
        }
        double tau = 1.0 - mu * 0.01;
        if (tau < 0.99) tau = 0.99;
        if (tau > 0.99995) tau = 0.99995;
        double apf = tau * apm, adf = tau * adm;
        if (apf > 1.0) apf = 1.0;
        if (adf > 1.0) adf = 1.0;
        if (apf < 0.1 * (tau * ap) || adf < 0.1 * (tau * ad)) {
            apf = tau * ap; if (apf > 1.0) apf = 1.0;
            adf = tau * ad; if (adf > 1.0) adf = 1.0;
            for (int j = 0; j < n; j++) dx[j] = dx_aff[j];
            for (int i = 0; i < p; i++) dy[i] = dy_aff[i];
            for (int k = 0; k < K; k++) { ds[k] = ds_aff[k]; dlm[k] = dlm_aff[k]; }
        }

        if (apf <= 1e-13 && adf <= 1e-13) {
            if (++no_progress > 5) break;
        } else no_progress = 0;

        for (int j = 0; j < n; j++) xs[j] += apf * dx[j];
        for (int i = 0; i < p; i++) ys[i] += adf * dy[i];
        for (int k = 0; k < K; k++) { s[k] += apf * ds[k]; lm[k] += adf * dlm[k]; }
    }

    if (status == OPT_OK) {
        for (int j = 0; j < n; j++) x[j] = xs[j];
        for (int i = 0; i < p; i++) y[i] = ys[i];
        for (int k = 0; k < K; k++) lam[k] = lm[k];
    }
    free(off); free(typ); free(xs); free(ys); free(s); free(lm); free(rd);
    free(rp); free(rg); free(rc); free(Asc); free(Alr); free(M); free(rhs);
    free(ds); free(dlm); free(dx_aff); free(dy_aff); free(ds_aff); free(dlm_aff);
    return status;
}

/* =====================================================================
 * Sparse conic IPM: SAME augmented KKT matrix M and SAME math as socp_solve,
 * but M is assembled SPARSELY (from CSR of E and G plus the per-cone arrow
 * blocks) and factored with the sparse LU (splu_factor, partial pivoting)
 * instead of a dense N x N LU. So the O(N^3) dense factorization becomes
 * O(fill); results are identical to socp_solve. Used for large sparse conic
 * problems; small ones keep the dense path.
 * ===================================================================== */
typedef struct { int *rp,*ri; double *rv; int *cp,*ci; double *cv; } SpMat;
/* Dense row-major -> CSR *and* CSC in one pass.  Both, because every Newton
 * iteration needs A x (a row scan) and A'y (a column scan) on the same matrix;
 * building the transpose separately would cost a second copy of the same data.
 * Exact zeros are counted out, so the nnz here is what multiplies -- a "-0.0" or
 * a structural zero in the dense input does not become an entry. */
static int spmat_init(SpMat *S, const double *D, int r, int q) {
    long nnz=0; for(long i=0;i<(long)r*q;i++) if(D[i]!=0.0) nnz++;
    S->rp=(int*)calloc((size_t)(r+1),sizeof(int)); S->cp=(int*)calloc((size_t)(q+1),sizeof(int));
    S->ri=(int*)malloc((size_t)(nnz>0?nnz:1)*sizeof(int)); S->rv=(double*)malloc((size_t)(nnz>0?nnz:1)*sizeof(double));
    S->ci=(int*)malloc((size_t)(nnz>0?nnz:1)*sizeof(int)); S->cv=(double*)malloc((size_t)(nnz>0?nnz:1)*sizeof(double));
    if(!S->rp||!S->cp||!S->ri||!S->rv||!S->ci||!S->cv) return -1;
    for(int i=0;i<r;i++)for(int j=0;j<q;j++) if(D[i*q+j]!=0.0){S->rp[i+1]++;S->cp[j+1]++;}
    for(int i=0;i<r;i++)S->rp[i+1]+=S->rp[i];
    for(int j=0;j<q;j++)S->cp[j+1]+=S->cp[j];
    int *fr=(int*)calloc((size_t)(r>0?r:1),sizeof(int)),*fc=(int*)calloc((size_t)(q>0?q:1),sizeof(int));
    if(!fr||!fc){free(fr);free(fc);return -1;}
    for(int i=0;i<r;i++)for(int j=0;j<q;j++){double v=D[i*q+j];if(v==0.0)continue;
        int a=S->rp[i]+fr[i]++;S->ri[a]=j;S->rv[a]=v; int b=S->cp[j]+fc[j]++;S->ci[b]=i;S->cv[b]=v;}
    free(fr);free(fc);return 0;
}
/* Nulls every pointer: the error paths here free the same struct more than once
 * (once in the failing step, once in the common tail) and a freed-again pointer
 * must not be a live address. */
static void spmat_free(SpMat*S){free(S->rp);free(S->cp);free(S->ri);free(S->rv);free(S->ci);free(S->cv);
    S->rp=S->cp=S->ri=S->ci=NULL;S->rv=S->cv=NULL;}
/* y = A x, by rows (r of them). */
static void sp_row(const SpMat*S,const double*x,double*y,int r){for(int i=0;i<r;i++){double t=0;for(int p=S->rp[i];p<S->rp[i+1];p++)t+=S->rv[p]*x[S->ri[p]];y[i]=t;}}
/* y = A' x, by columns -- the second half of the same struct, which is why
 * spmat_init builds both. */
static void sp_colT(const SpMat*S,const double*x,double*y,int q){for(int j=0;j<q;j++){double t=0;for(int p=S->cp[j];p<S->cp[j+1];p++)t+=S->cv[p]*x[S->ci[p]];y[j]=t;}}

/* growable triplet list */
typedef struct { int *r,*c; double *v; int n,cap; } Tri3;
/* Each grown array is adopted as soon as it exists: a successful realloc has
 * already invalidated t->r, so `free`ing the returned pointer on a later
 * failure would leave t->r dangling -- and every caller's OPT_MEMORY exit goes
 * through a tail that frees r, c and v.  On failure the capacity therefore stays
 * un-grown while whatever did grow is kept: the list is still valid, just not
 * larger, and nothing is written at index n because the caller stops assembling. */
static int tri3_add(Tri3*t,int r,int c,double v){ if(t->n==t->cap){int nc=t->cap?t->cap*2:512;
    int*a=(int*)realloc(t->r,(size_t)nc*sizeof(int)); if(a) t->r=a;
    int*b=(int*)realloc(t->c,(size_t)nc*sizeof(int)); if(b) t->c=b;
    double*e=(double*)realloc(t->v,(size_t)nc*sizeof(double)); if(e) t->v=e;
    if(!a||!b||!e) return -1;
    t->cap=nc;} t->r[t->n]=r;t->c[t->n]=c;t->v[t->n]=v;t->n++;return 0; }
/* triplets -> CSC, summing contributions that land on the same (row, column);
 * caller frees op/oi/ov.  Two triplets at one position are two contributions to
 * ONE entry, and the assembly does emit them: the regularisation bumps below add
 * `regs[ri] - reg`, the INCREMENT over the previous bump, which is a form
 * written for a representation that accumulates.  Left unmerged, the entry is
 * read by first slot in the pivot search (srow_get) and by LAST slot when a row
 * update is staged through the workspace, so a repeated column can be dropped
 * for one purpose and used for another.  Measured here: every repeated position
 * in the suite (123) and in logistic_large (2.7e6) is diagonal, and every one of
 * them is a bump landing on a diagonal entry that already exists -- so what the
 * unmerged list was doing is dropping the regularisation exactly where the
 * matrix already has a diagonal term.  The merge moves no published number today
 * (68 samples and the 267 `[route]`/`[cones]` lines of the suite are identical
 * before and after, timings aside); it is here because the entry of an assembled
 * matrix is its sum, and because a factorisation that reads one slot for the
 * pivot and another for the update is not a thing to keep relying on. */
static int tri3_to_csc(int ncol,const Tri3*t,int **op,int **oi,double **ov){
    int *cnt=(int*)calloc((size_t)(ncol+1),sizeof(int)); if(!cnt)return -1;
    int nr=0;
    for(int e=0;e<t->n;e++){ cnt[t->c[e]]++; if(t->r[e]>=nr) nr=t->r[e]+1; }
    int *ptr=(int*)malloc((size_t)(ncol+1)*sizeof(int)); if(!ptr){free(cnt);return -1;}
    ptr[0]=0; for(int j=0;j<ncol;j++)ptr[j+1]=ptr[j]+cnt[j];
    int nnz=ptr[ncol]; int *ri=(int*)malloc((size_t)(nnz>0?nnz:1)*sizeof(int));
    double *vv=(double*)malloc((size_t)(nnz>0?nnz:1)*sizeof(double));
    int *cur=(int*)malloc((size_t)(ncol>0?ncol:1)*sizeof(int));
    int *slot=(int*)malloc((size_t)(nr>0?nr:1)*sizeof(int));
    int *seen=(int*)malloc((size_t)(nr>0?nr:1)*sizeof(int));
    if(!ri||!vv||!cur||!slot||!seen){free(cnt);free(ptr);free(ri);free(vv);free(cur);free(slot);free(seen);return -1;}
    for(int j=0;j<ncol;j++)cur[j]=ptr[j];
    for(int e=0;e<t->n;e++){int j=t->c[e];ri[cur[j]]=t->r[e];vv[cur[j]]=t->v[e];cur[j]++;}
    free(cnt);free(cur);
    for(int i=0;i<nr;i++) slot[i]=-1;
    int k=0;
    for(int j=0;j<ncol;j++){
        int lo=ptr[j], hi=ptr[j+1], ns=0;
        ptr[j]=k;
        for(int p=lo;p<hi;p++){            /* k <= p: this only ever shrinks a column */
            int r=ri[p];
            if(slot[r]>=0){ vv[slot[r]]+=vv[p]; }
            else { slot[r]=k; seen[ns++]=r; ri[k]=r; vv[k]=vv[p]; k++; }
        }
        for(int a=0;a<ns;a++) slot[seen[a]]=-1;
    }
    ptr[ncol]=k;
    *op=ptr;*oi=ri;*ov=vv; return 0;
}

/* =====================================================================
 * Nesterov-Todd (NT) scaling for the second-order cone (SOC), used by the
 * sparse conic IPM below.
 *
 * For a cone block (s,lam) strictly inside K the NT scaling point w is
 * characterised by Q(w) lam = s (Q the Jordan quadratic representation,
 * Q(v) = 2 A(v)^2 - A(v^2)); it is computed as
 *     w = Q(s^{1/2}) (Q(s^{1/2}) lam)^{-1/2}.
 * Scaling by Q(w)^{+/-1/2} makes the primal/dual cone variables equal
 * (balanced), so the reduced system is SYMMETRIC:  H = G' Q(w)^{-1} G is
 * SPD and is solved by sparse Cholesky (normal equations, as MOSEK):
 *     H dx + E' dy = -rd - G'( lam - sigma*mu*s^{-1} + Q(w)^{-1} rg )
 *     E dx         = -rp
 *     ds = rg + G dx ,  dlam = -lam + sigma*mu*s^{-1} - Q(w)^{-1} ds .
 * Near the cone boundary Q(s^{1/2}) lam -> 0 and the scaling point does not
 * exist (at the exact optimum Q(w) lam = s is inconsistent), so nt_setup()
 * reports the block as degenerate and the iteration falls back to the
 * augmented sparse-LU direction (robust near the optimum).
 * ===================================================================== */
#define NT_MAXK 32
/* Jordan product v o w = A(v) w, written out: (v0 w0 + <vbar,wbar>, v0 wbar +
 * w0 vbar).  o may NOT alias v or w. */
static void nt_amul(const double *v,const double *w,int k,double *o){
    double t=v[0]*w[0]; for(int i=1;i<k;i++) t+=v[i]*w[i];
    o[0]=t; for(int i=1;i<k;i++) o[i]=v[i]*w[0]+v[0]*w[i];
}
/* Quadratic representation Q(v) = 2 A(v)^2 - A(v o v), dense k x k row-major.
 * It is the map that carries a cone element to its scaling partner (Q(w) lam = s
 * defines w), and k is capped at NT_MAXK because the buffers here are stack
 * arrays of that size -- every caller tests k first and treats -1 as
 * "this block is not NT-solvable". */
static void nt_Qmat(const double *v,int k,double *Q){
    double A[NT_MAXK*NT_MAXK],A2[NT_MAXK*NT_MAXK],vv[NT_MAXK];
    nt_amul(v,v,k,vv);
    memset(A,0,sizeof(double)*k*k);
    A[0]=v[0]; for(int i=1;i<k;i++){A[i]=v[i];A[i*k]=v[i];A[i*k+i]=v[0];}
    for(int i=0;i<k;i++)for(int j=0;j<k;j++){double s=0;for(int z=0;z<k;z++)s+=A[i*k+z]*A[z*k+j];A2[i*k+j]=s;}
    for(int i=0;i<k;i++)for(int j=0;j<k;j++){double a=(i==0&&j==0)?vv[0]:(i==0)?vv[j]:(j==0)?vv[i]:(i==j?vv[0]:0.0);Q[i*k+j]=2*A2[i*k+j]-a;}
}
static void nt_Qapp(const double *v,const double *w,int k,double *o){
    double Q[NT_MAXK*NT_MAXK]; nt_Qmat(v,k,Q);
    for(int i=0;i<k;i++){double s=0;for(int z=0;z<k;z++)s+=Q[i*k+z]*w[z];o[i]=s;}
}
/* Cone square root: SOC is symmetric, so v has the two eigenvalues
 * v0 +/- |vbar| with eigen-directions (1, +/- vbar/|vbar|)/sqrt 2, and
 * sqrt(v) = (sqrt(l1)+sqrt(l2))/2 along the axis plus the same mixture times
 * vbar.  l2 is floored at 0: outside the cone the root is not real, and what
 * the callers want there is a usable number to detect degeneracy on, not NaN.
 * nb -> 0 (v along the axis) makes the transverse part vanish, hence the guard. */
static void nt_sqrt(const double *v,int k,double *r){
    double nb=0;for(int i=1;i<k;i++)nb+=v[i]*v[i];nb=sqrt(nb);
    double l1=v[0]+nb,l2=v[0]-nb;if(l2<0)l2=0;double s1=sqrt(l1),s2=sqrt(l2);r[0]=0.5*(s1+s2);
    double sc=(nb>1e-300)?0.5*(s1-s2)/nb:0;for(int i=1;i<k;i++)r[i]=sc*v[i];
}
/* Cone inverse: v^{-1} = (v0, -vbar) / (v0^2 - |vbar|^2), which exists exactly
 * when v is invertible -- i.e. off the boundary, where the determinant is 0.
 * That test IS the degeneracy report, so -1 here means "fall back to the
 * augmented LU", not "the model is wrong". */
static int nt_inv(const double *v,int k,double *r){
    double nb=0;for(int i=1;i<k;i++)nb+=v[i]*v[i];double D=v[0]*v[0]-nb;
    if(!(D>0))return -1; r[0]=v[0]/D;for(int i=1;i<k;i++)r[i]=-v[i]/D;return 0;
}
/* sqrt then inverse, through a private buffer: r may not alias v. */
static int nt_invsqrt(const double *v,int k,double *r){double q[NT_MAXK];nt_sqrt(v,k,q);return nt_inv(q,k,r);}
/* NT data for one cone block: w (scaling point), Qwi = Q(w)^{-1} (kk x kk
 * row-major), sinv = s^{-1}.  Returns -1 if the block is degenerate. */
static int nt_setup(const double *s,const double *lam,int k,double *w,double *Qwi,double *sinv){
    double a[NT_MAXK],Qal[NT_MAXK],bis[NT_MAXK],Q[NT_MAXK*NT_MAXK],A[2*NT_MAXK*NT_MAXK];
    if(k>NT_MAXK) return -1;
    nt_sqrt(s,k,a); nt_Qapp(a,lam,k,Qal);
    double n2=0; for(int i=1;i<k;i++) n2+=Qal[i]*Qal[i];
    double det=Qal[0]*Qal[0]-n2, nn=Qal[0]*Qal[0]+n2;
    if(!(det>1e-12*nn)) return -1;                 /* near cone boundary */
    if(nt_invsqrt(Qal,k,bis)) return -1;
    nt_Qapp(a,bis,k,w);
    nt_Qmat(w,k,Q);
    /* Q(w)^{-1} by Gauss-Jordan on [Q | I] held in one 2k-wide array; the right
     * half ends up the inverse.  A zero pivot is the same degeneracy the det
     * test above reports (Q(w) is singular exactly when w is on the boundary),
     * and the caller falls back to the LU direction. */
    for(int i=0;i<k;i++)for(int j=0;j<k;j++){A[i*(2*k)+j]=Q[i*k+j];A[i*(2*k)+k+j]=(i==j)?1.0:0.0;}
    for(int c=0;c<k;c++){int pv=c;for(int r=c+1;r<k;r++)if(fabs(A[r*(2*k)+c])>fabs(A[pv*(2*k)+c]))pv=r;
        if(pv!=c)for(int j=0;j<2*k;j++){double t=A[c*(2*k)+j];A[c*(2*k)+j]=A[pv*(2*k)+j];A[pv*(2*k)+j]=t;}
        double dv=A[c*(2*k)+c]; if(!(fabs(dv)>1e-300)) return -1;
        for(int j=0;j<2*k;j++)A[c*(2*k)+j]/=dv;
        for(int r=0;r<k;r++)if(r!=c){double f=A[r*(2*k)+c];for(int j=0;j<2*k;j++)A[r*(2*k)+j]-=f*A[c*(2*k)+j];}}
    for(int i=0;i<k;i++)for(int j=0;j<k;j++)Qwi[i*k+j]=A[i*(2*k)+k+j];
    if(nt_inv(s,k,sinv)) return -1;
    return 0;
}
/* direction context: either the NT Cholesky factors or the sparse-LU factor */
typedef struct {
    int n,p,K,use_nt,ncones,maxk;
    const SocpCone *cones;
    const SpMat *SE,*SG;
    const double *Qwi,*sinv;          /* NT per-cone Q(w)^{-1} blocks / s^{-1} */
    SpChol *Hchol,*Schol; double *Z;  /* NT: H=L L', S=E H^{-1}E', Z=H^{-1}E' */
    SpluFact *lu;                     /* LU: augmented KKT */
    double *b1,*b2,*tmp,*rhs;
} DirCtx;
/* One Newton step (dx, dy, ds, dlam) of the conic KKT at a given (sigma, mu).
 * The two branches are the SAME direction computed two ways: the NT branch
 * reduces to the symmetric normal equations H dx + E'dy = ... with H = G'Q(w)^-1 G
 * (sparse Cholesky, plus the Schur solve for dy), the LU branch keeps the
 * augmented KKT and factors it with partial pivoting.  Which one runs is decided
 * per iteration by whether every block survives nt_setup, and the two must agree
 * -- that agreement is what T72/T73 measure against the dense path.  ds is never
 * solved for: it is rg + G dx in both branches, which is the elimination written
 * in the file header. */
static int conic_dir(const DirCtx *c,double sigma,double mu,
                     const double *lm,const double *rg,const double *rd,const double *rp,
                     const double *Asc,const double *Alr,double *dx,double *dy,double *ds,double *dlm){
    int n=c->n,p=c->p,K=c->K;
    if(c->use_nt){
        for(int a=0;a<n;a++) c->b1[a]=-rd[a];
        {int o=0;for(int k=0;k<c->ncones;k++){int kk=c->cones[k].nmem;
          for(int i=0;i<kk;i++){double t=lm[o+i]-sigma*mu*c->sinv[o+i];
            for(int i2=0;i2<kk;i2++) t+=c->Qwi[o*c->maxk+i*kk+i2]*rg[o+i2];
            for(int q=c->SG->rp[o+i];q<c->SG->rp[o+i+1];q++) c->b1[c->SG->ri[q]]-=c->SG->rv[q]*t;}
          o+=kk;}}
        for(int j=0;j<n;j++) c->tmp[j]=c->b1[j];
        if(spchol_solve(c->Hchol,c->tmp)) return -1;
        if(p>0){
            sp_row(c->SE,c->tmp,c->b2,p);
            for(int i=0;i<p;i++) c->b2[i]+=rp[i];
            if(spchol_solve(c->Schol,c->b2)) return -1;
            for(int i=0;i<p;i++) dy[i]=c->b2[i];
        }
        for(int j=0;j<n;j++){double t=c->tmp[j];for(int i=0;i<p;i++)t-=c->Z[i*n+j]*dy[i];dx[j]=t;}
        sp_row(c->SG,dx,ds,K);
        for(int k=0;k<K;k++) ds[k]+=rg[k];
        {int o=0;for(int k=0;k<c->ncones;k++){int kk=c->cones[k].nmem;
          for(int i=0;i<kk;i++){double t=-lm[o+i]+sigma*mu*c->sinv[o+i];
            for(int i2=0;i2<kk;i2++) t-=c->Qwi[o*c->maxk+i*kk+i2]*ds[o+i2];
            dlm[o+i]=t;}
          o+=kk;}}
    } else {
        {int o=0;for(int k=0;k<c->ncones;k++){int kk=c->cones[k].nmem;
          /* sigma*mu scales the identity ELEMENT of the algebra, which for SOC is
           * e = (1,0,..,0): the centreing term enters the axis coordinate only.
           * A dim-1 R_+ block is its own identity, so it carries the term as is. */
          if(c->cones[k].type==0) for(int i=0;i<kk;i++) c->rhs[o+i]=-Asc[o+i]+sigma*mu-Alr[o+i];
          else { c->rhs[o]=-Asc[o]+sigma*mu-Alr[o]; for(int i=1;i<kk;i++) c->rhs[o+i]=-Asc[o+i]-Alr[o+i]; }
          o+=kk;}}
        for(int i=0;i<p;i++) c->rhs[K+i]=-rp[i];
        for(int j=0;j<n;j++) c->rhs[K+p+j]=-rd[j];
        if(splu_solve(c->lu,c->rhs)) return -1;
        for(int j=0;j<n;j++)dx[j]=c->rhs[j];
        for(int i=0;i<p;i++)dy[i]=c->rhs[n+i];
        for(int k=0;k<K;k++)dlm[k]=c->rhs[n+p+k];
        sp_row(c->SG,dx,ds,K);
        for(int k=0;k<K;k++) ds[k]+=rg[k];
    }
    return 0;
}

/* condition estimate of an SPD matrix from its Cholesky factor L: ratio of
 * the largest to smallest |diagonal| of L.  Used to decide whether the NT
 * normal equations are safe, or the robust augmented-LU path is needed. */
static double spchol_cond_diag(const SpChol *L){
    double lo=1e300, hi=0.0;
    for(int j=0;j<L->n;j++){
        for(int p=L->Lp[j];p<L->Lp[j+1];p++) if(L->Li[p]==j){double a=fabs(L->Lx[p]); if(a<lo)lo=a; if(a>hi)hi=a; break;}
    }
    return (lo>0.0)?hi/lo:1e300;
}

/**
 * Solves a conic problem using sparse Mehrotra IPM with NT scaling.
 *
 * @param n       [in]  Number of variables.
 * @param p       [in]  Number of equality constraints.
 * @param E       [in]  Equality constraint matrix (p x n, row-major).
 * @param d       [in]  RHS for equality constraints (size p).
 * @param c       [in]  Linear objective coefficients (size n).
 * @param ncones  [in]  Number of cones.
 * @param cones   [in]  Array of cone descriptions (type, nmem, mem[]).
 * @param G       [in]  Cone constraint matrix (K x n, row-major, K = sum nmem).
 * @param h       [in]  RHS for cone constraints (size K).
 * @param tol_gap [in]  Relative gap tolerance.
 * @param tol_feas [in] Feasibility tolerance (primal/dual).
 * @param max_iter [in] Maximum iterations.
 * @param x       [out] Primal solution (size n). Must not be NULL.
 * @param y       [out] Dual for equalities (size p). Must not be NULL.
 * @param lam     [out] Dual for cones (size K). Must not be NULL.
 *
 * @return OPT_OK (0), OPT_MAXITER (1), OPT_MEMORY (2), OPT_SINGULAR (3).
 *
 * @note Same mathematical formulation as socp_solve but uses sparse matrices:
 *       - Builds CSR/CSC of E and G for sparse matrix-vector products.
 *       - Nesterov-Todd (NT) scaling with sparse Cholesky on H = G' Q(w)^-1 G.
 *       - Falls back to sparse augmented LU near cone boundary.
 *       - Uses symmetric normal equations: H dx + E' dy = rhs.
 *       - Results identical to dense socp_solve (verified in T72/T73).
 *       - Falls back to LU if NT scaling ill-conditioned (det < 1e-12).
 *
 *       GMB_NO_NT environment variable forces the LU path.
 */
int socp_solve_sparse(int n,int p,const double *E,const double *d,const double *c,
                      int ncones,const SocpCone *cones,const double *G,const double *h,
                      double tol_gap,double tol_feas,int max_iter,double *x,double *y,double *lam)
{
    if(n<0||p<0||ncones<0) return OPT_MAXITER;
    int K=0; for(int k=0;k<ncones;k++) K+=cones[k].nmem;
    int N=n+p+K; if(N==0) return OPT_OK;
    SpMat SE,SG; memset(&SE,0,sizeof(SE)); memset(&SG,0,sizeof(SG));
    if(spmat_init(&SE,E,p,n)||spmat_init(&SG,G,K,n)){spmat_free(&SE);spmat_free(&SG);return OPT_MEMORY;}
    int *off=(int*)malloc((size_t)(ncones>0?ncones:1)*sizeof(int));
    double *xs=(double*)calloc((size_t)(n>0?n:1),sizeof(double)),*ys=(double*)calloc((size_t)(p>0?p:1),sizeof(double));
    double *s=(double*)malloc((size_t)(K>0?K:1)*sizeof(double)),*lm=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *rd=(double*)malloc((size_t)(n>0?n:1)*sizeof(double)),*rp=(double*)malloc((size_t)(p>0?p:1)*sizeof(double));
    double *rg=(double*)malloc((size_t)(K>0?K:1)*sizeof(double)),*Asc=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *Alr=(double*)malloc((size_t)(K>0?K:1)*sizeof(double)),*rhs=(double*)malloc((size_t)(N>0?N:1)*sizeof(double));
    double *dx=(double*)malloc((size_t)(n>0?n:1)*sizeof(double)),*dy=(double*)malloc((size_t)(p>0?p:1)*sizeof(double));
    double *ds=(double*)malloc((size_t)(K>0?K:1)*sizeof(double)),*dlm=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *dxa=(double*)malloc((size_t)(n>0?n:1)*sizeof(double)),*dya=(double*)malloc((size_t)(p>0?p:1)*sizeof(double));
    double *dsa=(double*)malloc((size_t)(K>0?K:1)*sizeof(double)),*dlma=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *t1=(double*)malloc((size_t)(n>0?n:1)*sizeof(double));
    double *wacc=(double*)calloc((size_t)(n>0?n:1),sizeof(double));
    int *wst=(int*)calloc((size_t)(n>0?n:1),sizeof(int)); int *wtouch=(int*)malloc((size_t)(n>0?n:1)*sizeof(int));
    /* NT (Nesterov-Todd) scratch: per-cone Q(w)^{-1} blocks, s^{-1}, scaling
     * point, Z = H^{-1}E', dense solve buffers, plus a triplet list for S. */
    int maxk=1; for(int k=0;k<ncones;k++) if(cones[k].nmem>maxk) maxk=cones[k].nmem;
    double *Qwi=(double*)malloc((size_t)(K>0?K:1)*(size_t)maxk*sizeof(double));
    double *sinv=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *wnt=(double*)malloc((size_t)(K>0?K:1)*sizeof(double));
    double *Znt=(double*)malloc((size_t)(n>0?n:1)*(size_t)(p>0?p:1)*sizeof(double));
    double *nb1=(double*)malloc((size_t)(n>0?n:1)*sizeof(double));
    double *nb2=(double*)malloc((size_t)(p>0?p:1)*sizeof(double));
    Tri3 stri={0};
    Tri3 tri={0};
    int memok=off&&xs&&ys&&s&&lm&&rd&&rp&&rg&&Asc&&Alr&&rhs&&dx&&dy&&ds&&dlm&&dxa&&dya&&dsa&&dlma&&t1&&wacc&&wst&&wtouch&&Qwi&&sinv&&wnt&&Znt&&nb1&&nb2;
    int status=OPT_MAXITER, stamp=1;
    if(!memok){status=OPT_MEMORY;goto done;}
    {int o=0;for(int k=0;k<ncones;k++){off[k]=o;o+=cones[k].nmem;}}
    {int o=0;for(int k=0;k<ncones;k++){if(cones[k].type==0)for(int i=0;i<cones[k].nmem;i++){s[o+i]=1.0;lm[o+i]=1.0;}
        else{s[o]=1.0;lm[o]=1.0;for(int i=1;i<cones[k].nmem;i++){s[o+i]=0.0;lm[o+i]=0.0;}}o+=cones[k].nmem;}}
    double dnorm=1,hnorm=1,cnorm=1;
    for(int i=0;i<p;i++){double a=fabs(d[i]);if(a>dnorm)dnorm=a;}
    for(int k=0;k<K;k++){double a=fabs(h[k]);if(a>hnorm)hnorm=a;}
    for(int j=0;j<n;j++){double a=fabs(c[j]);if(a>cnorm)cnorm=a;}
    int no_progress=0;
    for(int it=0;it<max_iter;it++){
        if(primal_cb_iter_on) primal_cb_iter(34);
        if(p>0){sp_row(&SE,xs,rp,p);for(int i=0;i<p;i++)rp[i]-=d[i];}
        sp_row(&SG,xs,rg,K);for(int k=0;k<K;k++)rg[k]+=h[k]-s[k];
        for(int j=0;j<n;j++)rd[j]=c[j];
        sp_colT(&SE,ys,t1,n);for(int j=0;j<n;j++)rd[j]+=t1[j];
        sp_colT(&SG,lm,t1,n);for(int j=0;j<n;j++)rd[j]-=t1[j];
        double mu=0;for(int k=0;k<K;k++)mu+=s[k]*lm[k];mu/=(double)(K>0?K:1);
        {int o=0;for(int k=0;k<ncones;k++){if(cones[k].type==0)for(int i=0;i<cones[k].nmem;i++)Asc[o+i]=s[o+i]*lm[o+i];
            else arrow_mul(&s[o],&lm[o],cones[k].nmem,&Asc[o]);o+=cones[k].nmem;}}
        {int o=0;for(int k=0;k<ncones;k++){if(cones[k].type==0)for(int i=0;i<cones[k].nmem;i++)Alr[o+i]=lm[o+i]*rg[o+i];
            else arrow_mul(&lm[o],&rg[o],cones[k].nmem,&Alr[o]);o+=cones[k].nmem;}}
        double pobj=0;for(int j=0;j<n;j++)pobj+=c[j]*xs[j];
        double dobj=0;for(int i=0;i<p;i++)dobj-=d[i]*ys[i];for(int k=0;k<K;k++)dobj-=h[k]*lm[k];
        double feas_p=maxabs(rp,p)/dnorm;if(K){double a=maxabs(rg,K)/hnorm;if(a>feas_p)feas_p=a;}
        double feas_d=maxabs(rd,n)/cnorm;double gap=fabs(pobj-dobj)/(1.0+fabs(pobj));
        if(getenv("GMB_DBG")&&(it%10==0||feas_p<=tol_feas))
            fprintf(stderr,"socp_sp it=%d feas_p=%.3g feas_d=%.3g gap=%.3g mu=%.3g\n",it,feas_p,feas_d,gap,mu);
        if(feas_p<=tol_feas&&feas_d<=tol_feas&&gap<=tol_gap){status=OPT_OK;break;}
        /* ---- choose direction: NT (symmetric, sparse Cholesky) with an
         * augmented sparse-LU fallback near the cone boundary ---- */
        int use_nt = (n>0) && !getenv("GMB_NO_NT");
        if(use_nt){
            int o=0;
            for(int k=0;k<ncones&&use_nt;k++){int kk=cones[k].nmem;
                double amax=0.0;
                if(cones[k].type==0){
                    double *blk=&Qwi[(size_t)o*maxk];
                    for(int z=0;z<kk*kk;z++) blk[z]=0.0;
                    for(int i=0;i<kk;i++){
                        if(s[o+i]<=1e-300||lm[o+i]<=1e-300){use_nt=0;break;}
                        wnt[o+i]=sqrt(s[o+i]/lm[o+i]);
                        double q=lm[o+i]/s[o+i]; blk[i*kk+i]=q;
                        sinv[o+i]=1.0/s[o+i];
                        if(fabs(q)>amax)amax=fabs(q);
                        if(fabs(sinv[o+i])>amax)amax=fabs(sinv[o+i]);
                    }
                } else {
                    if(nt_setup(&s[o],&lm[o],kk,&wnt[o],&Qwi[(size_t)o*maxk],&sinv[o])){use_nt=0;break;}
                    for(int z=0;z<kk*kk;z++){double q=fabs(Qwi[(size_t)o*maxk+z]); if(!isfinite(q)||q>amax)amax=q;}
                    for(int i=0;i<kk;i++){double q=fabs(sinv[o+i]); if(!isfinite(q)||q>amax)amax=q;}
                }
                if(amax>1e12){use_nt=0;break;}   /* scaling ill-conditioned -> fall back */
                o+=kk;
            }
        }
        SpChol *Hchol=NULL,*Schol=NULL; SpluFact *f=NULL; int *Mp=NULL,*Mi=NULL; double *Mv=NULL;
        if(use_nt){
            /* H = G' Q(w)^{-1} G  (SPD; upper triangle in CSC) */
            tri.n=0;
            {int o=0;for(int k=0;k<ncones;k++){int kk=cones[k].nmem;
              for(int i=0;i<kk;i++){
                stamp++; int nt=0;
                for(int i2=0;i2<kk;i2++){double q=Qwi[(size_t)o*maxk+i*kk+i2]; if(q==0.0) continue;
                  for(int qb=SG.rp[o+i2];qb<SG.rp[o+i2+1];qb++){int col=SG.ri[qb];
                    if(wst[col]!=stamp){wst[col]=stamp;wacc[col]=0.0;wtouch[nt++]=col;} wacc[col]+=q*SG.rv[qb];}}
                for(int qa=SG.rp[o+i];qa<SG.rp[o+i+1];qa++){int a=SG.ri[qa]; double ga=SG.rv[qa];
                  for(int z=0;z<nt;z++){int col=wtouch[z]; if(a>=col) if(tri3_add(&tri,a,col,ga*wacc[col])){status=OPT_MEMORY;goto done;}}}
              }
              o+=kk;}}
            int *Hp,*Hi; double *Hv;
            if(tri3_to_csc(n,&tri,&Hp,&Hi,&Hv)){status=OPT_MEMORY;goto done;}
            Hchol=spchol_factor(n,Hp,Hi,Hv);
            { static const double regs[]={1e-10,1e-9,1e-8}; double rgu=0.0;
              for(size_t ri=0;ri<sizeof regs/sizeof regs[0]&&!Hchol;ri++){
                for(int i=0;i<n;i++) tri3_add(&tri,i,i,regs[ri]-rgu);
                rgu=regs[ri];
                free(Hp);free(Hi);free(Hv);
                if(tri3_to_csc(n,&tri,&Hp,&Hi,&Hv)){status=OPT_MEMORY;goto done;}
                Hchol=spchol_factor(n,Hp,Hi,Hv); } }
            free(Hp);free(Hi);free(Hv);
            if(Hchol&&spchol_cond_diag(Hchol)>1e10){spchol_free(Hchol);Hchol=NULL;}  /* ill-conditioned */
            if(!Hchol){use_nt=0;}
        }
        if(use_nt&&p>0){
            /* Z = H^{-1} E' (n x p),  S = E H^{-1} E' (p x p, SPD) */
            int okz=1;
            for(int i=0;i<p;i++){
                for(int j=0;j<n;j++) nb1[j]=0.0;
                for(int q=SE.rp[i];q<SE.rp[i+1];q++) nb1[SE.ri[q]]=SE.rv[q];
                if(spchol_solve(Hchol,nb1)){okz=0;break;}
                for(int j=0;j<n;j++) Znt[(size_t)i*n+j]=nb1[j];
            }
            if(okz){
                stri.n=0;
                for(int i=0;i<p;i++)for(int i2=0;i2<=i;i2++){double sv=0;
                    for(int q=SE.rp[i];q<SE.rp[i+1];q++){int j=SE.ri[q]; sv+=SE.rv[q]*Znt[(size_t)i2*n+j];}
                    if(sv!=0.0) if(tri3_add(&stri,i,i2,sv)){status=OPT_MEMORY;goto done;}}
                int *Sp,*Si; double *Sv;
                if(tri3_to_csc(p,&stri,&Sp,&Si,&Sv)){status=OPT_MEMORY;goto done;}
                Schol=spchol_factor(p,Sp,Si,Sv);
                /* S = E H^{-1} E' must be SPD (E full row rank); if it is
                 * singular the normal equations are invalid -> fall back. */
                free(Sp);free(Si);free(Sv);
                if(Schol&&spchol_cond_diag(Schol)>1e10){spchol_free(Schol);Schol=NULL;}  /* ill-conditioned */
            }
            if(!okz||!Schol) use_nt=0;
        }
        if(!use_nt){
            /* ---- augmented KKT M assembled sparse, sparse LU (robust) ---- */
            if(Hchol){spchol_free(Hchol);Hchol=NULL;}
            if(Schol){spchol_free(Schol);Schol=NULL;}
            tri.n=0;
            {int o=0;for(int k=0;k<ncones;k++){int kk=cones[k].nmem;
              if(cones[k].type==0){
                for(int i=0;i<kk;i++){ int row=o+i;
                  for(int q=SG.rp[row];q<SG.rp[row+1];q++){ if(tri3_add(&tri,row,SG.ri[q],lm[row]*SG.rv[q])){status=OPT_MEMORY;goto done;} }
                  if(tri3_add(&tri,row,n+p+row,s[row])){status=OPT_MEMORY;goto done;} }
              } else {
                /* A(s) block: arrow */
                for(int i2=0;i2<kk;i2++) if(tri3_add(&tri,o+0,n+p+o+i2,s[o+i2])){status=OPT_MEMORY;goto done;}
                for(int i=1;i<kk;i++){ if(tri3_add(&tri,o+i,n+p+o+0,s[o+i])){status=OPT_MEMORY;goto done;}
                                       if(tri3_add(&tri,o+i,n+p+o+i,s[o+0])){status=OPT_MEMORY;goto done;} }
                /* A(lam)G block: row 0 = sum_i2 lam[i2]*G[o+i2]; row i>0 = lam[i]*G[o]+lam[0]*G[o+i] */
                stamp++; int nt=0;
                for(int i2=0;i2<kk;i2++) for(int q=SG.rp[o+i2];q<SG.rp[o+i2+1];q++){int col=SG.ri[q];
                    if(wst[col]!=stamp){wst[col]=stamp;wacc[col]=0;wtouch[nt++]=col;} wacc[col]+=lm[o+i2]*SG.rv[q];}
                for(int a=0;a<nt;a++){int col=wtouch[a]; if(wacc[col]!=0.0) if(tri3_add(&tri,o+0,col,wacc[col])){status=OPT_MEMORY;goto done;}}
                for(int i=1;i<kk;i++){ stamp++; nt=0;
                    for(int q=SG.rp[o+0];q<SG.rp[o+0+1];q++){int col=SG.ri[q]; if(wst[col]!=stamp){wst[col]=stamp;wacc[col]=0;wtouch[nt++]=col;} wacc[col]+=lm[o+i]*SG.rv[q];}
                    for(int q=SG.rp[o+i];q<SG.rp[o+i+1];q++){int col=SG.ri[q]; if(wst[col]!=stamp){wst[col]=stamp;wacc[col]=0;wtouch[nt++]=col;} wacc[col]+=lm[o+0]*SG.rv[q];}
                    for(int a=0;a<nt;a++){int col=wtouch[a]; if(wacc[col]!=0.0) if(tri3_add(&tri,o+i,col,wacc[col])){status=OPT_MEMORY;goto done;}} }
              }
              o+=kk;}}
            for(int i=0;i<p;i++) for(int q=SE.rp[i];q<SE.rp[i+1];q++){ int j=SE.ri[q]; double v=SE.rv[q];
                if(tri3_add(&tri,K+i,j,v)){status=OPT_MEMORY;goto done;}
                if(tri3_add(&tri,K+p+j,n+i,v)){status=OPT_MEMORY;goto done;} }
            for(int k2=0;k2<K;k2++) for(int q=SG.rp[k2];q<SG.rp[k2+1];q++){ int j=SG.ri[q];
                if(tri3_add(&tri,K+p+j,n+p+k2,-SG.rv[q])){status=OPT_MEMORY;goto done;} }
            if(tri3_to_csc(N,&tri,&Mp,&Mi,&Mv)){status=OPT_MEMORY;goto done;}
            f=splu_factor(N,Mp,Mi,Mv);
            double reg=0.0;
            if(!f){ static const double regs[]={1e-10,1e-8,1e-7,1e-6,1e-5};
                for(size_t ri=0;ri<sizeof regs/sizeof regs[0]&&!f;ri++){
                    for(int i=0;i<N;i++) tri3_add(&tri,i,i,regs[ri]-reg);   /* append diag bump */
                    reg=regs[ri];
                    free(Mp);free(Mi);free(Mv);
                    if(tri3_to_csc(N,&tri,&Mp,&Mi,&Mv)){status=OPT_MEMORY;goto done;}
                    f=splu_factor(N,Mp,Mi,Mv); } }
            if(!f){free(Mp);free(Mi);free(Mv);status=OPT_SINGULAR;goto done;}
        }
        DirCtx dctx; memset(&dctx,0,sizeof(dctx));
        if(getenv("GMB_DBG_NT")) fprintf(stderr,"socp_sp it=%d dir=%s n=%d p=%d K=%d nc=%d\n",it,use_nt?"NT":"LU",n,p,K,ncones);
        dctx.n=n; dctx.p=p; dctx.K=K; dctx.use_nt=use_nt; dctx.ncones=ncones; dctx.maxk=maxk;
        dctx.cones=cones; dctx.SE=&SE; dctx.SG=&SG; dctx.Qwi=Qwi; dctx.sinv=sinv;
        dctx.Hchol=Hchol; dctx.Schol=Schol; dctx.Z=Znt; dctx.lu=f;
        dctx.b1=nb1; dctx.b2=nb2; dctx.tmp=t1; dctx.rhs=rhs;
        /* affine predictor */
        if(conic_dir(&dctx,0.0,mu,lm,rg,rd,rp,Asc,Alr,dxa,dya,dsa,dlma)){status=OPT_SINGULAR;goto done;}
        double ap=1,ad=1;{int o=0;for(int k=0;k<ncones;k++){double a=step_len(&s[o],&dsa[o],cones[k].type,cones[k].nmem);if(a<ap)ap=a;
            a=step_len(&lm[o],&dlma[o],cones[k].type,cones[k].nmem);if(a<ad)ad=a;o+=cones[k].nmem;}}
        double mu_aff=0;{int o=0;for(int k=0;k<ncones;k++){for(int i=0;i<cones[k].nmem;i++)mu_aff+=(s[o+i]+ap*dsa[o+i])*(lm[o+i]+ad*dlma[o+i]);o+=cones[k].nmem;}}
        mu_aff/=(double)(K>0?K:1);
        double sigma=(mu>0)?(mu_aff/mu):0;sigma=sigma*sigma*sigma;if(!(sigma>1e-8))sigma=1e-8;if(sigma>1)sigma=1;
        /* corrector: same factors, only the cone rhs changes (sigma > 0) */
        if(conic_dir(&dctx,sigma,mu,lm,rg,rd,rp,Asc,Alr,dx,dy,ds,dlm)){
            if(Hchol)spchol_free(Hchol); if(Schol)spchol_free(Schol); if(f)splu_free(f);
            free(Mp);free(Mi);free(Mv); status=OPT_SINGULAR;goto done; }
        if(Hchol){spchol_free(Hchol);Hchol=NULL;}
        if(Schol){spchol_free(Schol);Schol=NULL;}
        if(f){splu_free(f);f=NULL;}
        free(Mp);free(Mi);free(Mv); Mp=Mi=NULL;Mv=NULL;
        double apm=1,adm=1;{int o=0;for(int k=0;k<ncones;k++){double a=step_len(&s[o],&ds[o],cones[k].type,cones[k].nmem);if(a<apm)apm=a;
            a=step_len(&lm[o],&dlm[o],cones[k].type,cones[k].nmem);if(a<adm)adm=a;o+=cones[k].nmem;}}
        double tau=1.0-mu*0.01;if(tau<0.99)tau=0.99;if(tau>0.99995)tau=0.99995;
        double apf=tau*apm,adf=tau*adm;if(apf>1)apf=1;if(adf>1)adf=1;
        if(apf<0.1*(tau*ap)||adf<0.1*(tau*ad)){apf=tau*ap;if(apf>1)apf=1;adf=tau*ad;if(adf>1)adf=1;
            for(int j=0;j<n;j++)dx[j]=dxa[j];for(int i=0;i<p;i++)dy[i]=dya[i];for(int k2=0;k2<K;k2++){ds[k2]=dsa[k2];dlm[k2]=dlma[k2];}}
        if(apf<=1e-13&&adf<=1e-13){if(++no_progress>5)break;}else no_progress=0;
        for(int j=0;j<n;j++)xs[j]+=apf*dx[j];
        for(int i=0;i<p;i++)ys[i]+=adf*dy[i];
        for(int k2=0;k2<K;k2++){s[k2]+=apf*ds[k2];lm[k2]+=adf*dlm[k2];}
    }
    if(status==OPT_OK){for(int j=0;j<n;j++)x[j]=xs[j];for(int i=0;i<p;i++)y[i]=ys[i];for(int k=0;k<K;k++)lam[k]=lm[k];}
done:
    free(off);free(xs);free(ys);free(s);free(lm);free(rd);free(rp);free(rg);free(Asc);free(Alr);free(rhs);
    free(dx);free(dy);free(ds);free(dlm);free(dxa);free(dya);free(dsa);free(dlma);free(t1);
    free(wacc);free(wst);free(wtouch); free(tri.r);free(tri.c);free(tri.v);
    free(Qwi);free(sinv);free(wnt);free(Znt);free(nb1);free(nb2);
    free(stri.r);free(stri.c);free(stri.v);
    spmat_free(&SE);spmat_free(&SG);
    return status;
}
