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

/* ipm.c - primal-dual interior point (Mehrotra predictor-corrector)
 *
 * Standard form: min 1/2 x'Qx + c'x, A x = b, x >= 0 (Q possibly NULL).
 * KKT residuals:
 *   r_d = Qx + c - A'y - z = 0        (z >= 0 multiplier of x >= 0)
 *   r_p = A x - b = 0
 *   x_j z_j = 0
 * Newton (target sigma*mu):
 *   (Q + X^-1 Z) dx - A' dy = -r_d - X^-1 r_c
 *   A dx = -r_p
 *   dz = -(r_c + z.*dx)./x
 * Solved via the regularized symmetric augmented system
 *   [ -(Q+D+delta)   A' ] [dx]   [  r_d + r_c./x ]
 *   [     A        -delta ] [dy] = [   -r_p        ]
 * with D = diag(z./x), dense LU with partial pivoting; delta increased on
 * factorization failure.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "ipm.h"
#include "linalg.h"

/* per-iteration callback hook (implemented in primal.c) */
extern int primal_cb_iter_on;
void primal_cb_iter(int code);


enum { IPM_OPTIMAL = 0, IPM_MAXITER, IPM_MEMORY, IPM_SINGULAR };

/* Wall-clock deadline for the current solve (absolute clock() ticks; <0 = no
 * limit). Set once by PRIMAL_optimize from PRIMAL_DPAR_OPTIMIZER_MAX_TIME and
 * read by every IPM loop. A file-scope value is correct here: the cap is wall
 * time, so concurrent solvers sharing it all stop at the same instant. */
static double g_ipm_deadline = -1.0;
/**
 * Sets the absolute wall-clock deadline for the current IPM solve.
 *
 * @param abs_deadline [in] Absolute clock() deadline (ticks since process start).
 *                        Negative means no limit.
 *
 * @note Called by PRIMAL_optimize from PRIMAL_DPAR_OPTIMIZER_MAX_TIME.
 *       File-scope because the cap is wall time -- concurrent solvers sharing
 *       it all stop at the same instant.
 */
void ipm_set_deadline(double abs_deadline) { g_ipm_deadline = abs_deadline; }
static int ipm_past_deadline(void) {
    return g_ipm_deadline >= 0.0 && (double)clock() >= g_ipm_deadline;
}

/* Objective lower cut (min-space; -1e308 = none) and the flag a fired cut sets.
 * PRIMAL_optimize publishes PRIMAL_DPAR_LOWER_OBJ_CUT here (negated for a
 * maximization) and reads the flag back to report TRM_OBJECTIVE_RANGE. The cut
 * fires only on a PRIMAL-FEASIBLE point whose objective is below it -- that is
 * the only way a primal point proves the optimum is below the cut. */
static double g_obj_lower = -1e308;
static double g_obj_upper = 1e308;
static int g_obj_cut_hit = 0;
/**
 * Sets the objective cuts for the interior point method.
 *
 * @param lower [in] Lower objective cut (primal, min-space).
 * @param upper [in] Upper objective cut (dual, min-space).
 *
 * @note Called by PRIMAL_optimize from PRIMAL_DPAR_LOWER_OBJ_CUT /
 *       PRIMAL_DPAR_UPPER_OBJ_CUT. Resets the cut-hit flag.
 */
void ipm_set_obj_cuts(double lower, double upper) { g_obj_lower = lower; g_obj_upper = upper; g_obj_cut_hit = 0; }
/**
 * Returns whether an objective cut was hit during the last solve.
 *
 * @return 1 if a cut was hit, 0 otherwise.
 *
 * @note The cut-hit flag is set when a feasible point exceeds the objective
 *       cut. Used by PRIMAL_optimize to report TRM_OBJECTIVE_RANGE.
 */
int ipm_obj_cut_hit(void) { return g_obj_cut_hit; }

/* Numero di correttori (MSK_IPAR_INTPNT_MAX_NUM_COR): >= 1. Il primo e' il
 * correttore di Mehrotra, i successivi sono higher-order. Default 1. */
static int g_intpnt_max_cor = 1;
/**
 * Sets the maximum number of higher-order correctors (Mehrotra + higher-order).
 *
 * @param ncor [in] Number of correctors. Must be >= 1. Default is 1.
 *
 * @note The first corrector is Mehrotra's (uses affine step cross-term).
 *       Subsequent correctors use the previous corrector's cross-term
 *       (higher-order). Controlled by MSK_IPAR_INTPNT_MAX_NUM_COR.
 */
void ipm_set_max_cor(int ncor) { g_intpnt_max_cor = (ncor >= 1) ? ncor : 1; }

/* build and factor the augmented system; returns NULL on failure */
static LuFact *build_factor(const double *A, const double *Q, const double *D,
                            int m, int n, double delta) {
    int N = n + m;
    double *K = (double *)calloc((size_t)N * (size_t)N, sizeof(double));
    if (!K) return NULL;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++)
            K[j * N + k] = -(Q ? Q[j * n + k] : 0.0);
        K[j * N + j] -= D[j] + delta;
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double a = A[i * n + j];
            if (a != 0.0) {
                K[j * N + n + i] = a;
                K[(n + i) * N + j] = a;
            }
        }
    for (int i = 0; i < m; i++) K[(n + i) * N + n + i] = -delta;
    LuFact *f = dmat_lu_factor(K, N);
    free(K);
    return f;
}

/**
 * Solves an LP/QP in standard form using Mehrotra's primal-dual interior point.
 *
 * @param A         [in]  Constraint matrix in row-major (m x n).
 * @param Q         [in]  Quadratic objective matrix (n x n), or NULL for LP.
 * @param m         [in]  Number of constraints.
 * @param n         [in]  Number of variables.
 * @param b         [in]  RHS vector (size m).
 * @param c         [in]  Linear objective coefficients (size n).
 * @param tol_gap   [in]  Relative gap tolerance.
 * @param tol_pfeas [in]  Primal feasibility tolerance.
 * @param tol_dfeas [in]  Dual feasibility tolerance.
 * @param max_iter  [in]  Maximum iterations.
 * @param x         [out] Primal solution (size n). Must not be NULL.
 * @param y         [out] Dual solution (size m). Must not be NULL.
 * @param z         [out] Dual slack (size n). Must not be NULL.
 * @param x0        [in]  Optional warm start x (size n). NaN entries -> default.
 * @param y0        [in]  Optional warm start y (size m). NaN entries -> default.
 *
 * @return IPM_OPTIMAL (0), IPM_MAXITER (1), IPM_MEMORY (2), IPM_SINGULAR (3).
 *
 * @note Solves: min 1/2 x'Qx + c'x  s.t.  Ax = b, x >= 0.
 *
 *       KKT system (primal-dual):
 *         r_d = Qx + c - A'y - z = 0
 *         r_p = Ax - b = 0
 *         x_j z_j = 0
 *
 *       Mehrotra predictor-corrector:
 *       - Affine step (sigma = 0): compute dxa, dya, dza
 *       - mu_aff, sigma = min(1, max(1e-8, (mu_aff/mu)^3))
 *       - Corrector(s): first uses dxa*dza cross-term, higher-order use
 *         previous corrector's cross-term (g_intpnt_max_cor)
 *
 *       Augmented system solved via dense LU with partial pivoting:
 *         [ -(Q+D+delta)   A' ] [dx] = [ r_d + r_c./x ]
 *         [     A        -delta ] [dy]   [    -r_p      ]
 *       with D = diag(z./x). Delta escalated on factorization failure.
 *
 *       Objective cuts: g_obj_lower (primal) and g_obj_upper (dual, LP only).
 *       Wall-clock deadline enforced via ipm_past_deadline().
 *
 *       Warm start: x0, y0 projected to interior (z derived from stationarity).
 *       If x0/y0 contain NaN, uses default start (x=1, z=1, y=0).
 *
 * @example
 * double A[] = {1, 0, 0, 1}; // 2x2
 * double b[] = {1, 1};
 * double c[] = {1, 1};
 * double x[2], y[2], z[2];
 * int rc = ipm_solve_std(A, NULL, 2, 2, b, c, 1e-8, 1e-8, 1e-8, 100, x, y, z, NULL, NULL);
 * if (rc == IPM_OPTIMAL) printf("Optimal: %f\n", c[0]*x[0] + c[1]*x[1]);
 */
int ipm_solve_std(const double *A, const double *Q, int m, int n,
                  const double *b, const double *c,
                  double tol_gap, double tol_pfeas, double tol_dfeas,
                  int max_iter,
                  double *x, double *y, double *z,
                  const double *x0, const double *y0)
{
    if (n <= 0) return IPM_OPTIMAL;
    int N = n + m;

    double *rp = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *rd = (double *)calloc((size_t)n, sizeof(double));
    double *rc = (double *)calloc((size_t)n, sizeof(double));
    double *qx = (double *)calloc((size_t)n, sizeof(double));
    double *rhs = (double *)calloc((size_t)N, sizeof(double));
    double *dxa = (double *)calloc((size_t)n, sizeof(double));
    double *dya = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *dza = (double *)calloc((size_t)n, sizeof(double));
    double *dx = (double *)calloc((size_t)n, sizeof(double));
    double *dy = (double *)calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *dz = (double *)calloc((size_t)n, sizeof(double));
    if (!rp || !rd || !rc || !qx || !rhs || !dxa || !dya || !dza || !dx || !dy || !dz) {
        free(rp); free(rd); free(rc); free(qx); free(rhs);
        free(dxa); free(dya); free(dza); free(dx); free(dy); free(dz);
        return IPM_MEMORY;
    }

    /* warm start: parte dal punto fornito, proiettato nell'interno;
     * z derivato dalla stazionarieta' c + Qx - A'y */
    if (x0) {
        for (int j = 0; j < n; j++)
            if (x0[j] != x0[j]) { x0 = NULL; break; }  /* NaN -> default */
        for (int i = 0; i < m; i++)
            if (y0 && y0[i] != y0[i]) { y0 = NULL; break; }
    }
    if (x0) {
        for (int j = 0; j < n; j++) x[j] = x0[j] > 1e-8 ? x0[j] : 1e-8;
        if (y0) for (int i = 0; i < m; i++) y[i] = y0[i];
        for (int j = 0; j < n; j++) {
            double s2 = c[j];
            if (Q) for (int k = 0; k < n; k++) s2 += Q[j * n + k] * x[k];
            for (int i = 0; i < m; i++) s2 -= A[i * n + j] * y[i];
            z[j] = s2 > 1e-2 ? s2 : 1e-2;
        }
    } else {
        for (int j = 0; j < n; j++) { x[j] = 1.0; z[j] = 1.0; }
        for (int i = 0; i < m; i++) y[i] = 0.0;
    }

    double bnorm = 1.0, cnorm = 1.0;
    for (int i = 0; i < m; i++) { double a = fabs(b[i]); if (a > bnorm) bnorm = a; }
    for (int j = 0; j < n; j++) { double a = fabs(c[j]); if (a > cnorm) cnorm = a; }
    if (Q)
        for (int j = 0; j < n * n; j++) { double a = fabs(Q[j]); if (a > cnorm) cnorm = a; }
    int is_qp = (Q != NULL);   /* the dual objective b'y below is an LP bound only */

    int status = IPM_MAXITER;
    int no_progress = 0;

    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(90);
        if (ipm_past_deadline()) { status = IPM_MAXITER; break; }
        /* residuals and measures */
        for (int i = 0; i < m; i++) {
            double s = -b[i];
            for (int j = 0; j < n; j++) s += A[i * n + j] * x[j];
            rp[i] = s;
        }
        if (Q)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) s += Q[j * n + k] * x[k];
                qx[j] = s;
            }
        else
            for (int j = 0; j < n; j++) qx[j] = 0.0;
        double xz = 0.0;
        for (int j = 0; j < n; j++) {
            double s = c[j] + qx[j] - z[j];
            for (int i = 0; i < m; i++) s -= A[i * n + j] * y[i];
            rd[j] = s;
            xz += x[j] * z[j];
        }
        double mu = xz / (double)n;
        double pobj = 0.0;
        for (int j = 0; j < n; j++) pobj += c[j] * x[j];
        if (Q) { double qq = 0.0; for (int j = 0; j < n; j++) qq += qx[j] * x[j]; pobj += 0.5 * qq; }

        double feas_p = 0.0, feas_d = 0.0;
        for (int i = 0; i < m; i++) { double a = fabs(rp[i]); if (a > feas_p) feas_p = a; }
        for (int j = 0; j < n; j++) { double a = fabs(rd[j]); if (a > feas_d) feas_d = a; }
        feas_p /= bnorm;
        feas_d /= cnorm;
        double gap = fabs(xz) / (1.0 + fabs(pobj));
        if (feas_p <= tol_pfeas && pobj < g_obj_lower) { g_obj_cut_hit = 1; status = IPM_MAXITER; break; }
        /* Upper cut on the DUAL side (LP only): a dual-feasible point whose dual
         * objective b'y is above the cut proves the optimum is above it. The QP's
         * dual objective differs, so the QP route does not enforce it. */
        if (!is_qp) {
            double db = 0.0;
            for (int i = 0; i < m; i++) db += b[i] * y[i];
            if (feas_d <= tol_dfeas && db > g_obj_upper) { g_obj_cut_hit = 1; status = IPM_MAXITER; break; }
        }

        if (feas_p <= tol_pfeas && feas_d <= tol_dfeas && gap <= tol_gap) {
            status = IPM_OPTIMAL;
            break;
        }

        /* D = z./x (x,z stay strictly positive by construction) */
        double *D = (double *)malloc((size_t)n * sizeof(double));
        if (!D) { status = IPM_MEMORY; break; }
        for (int j = 0; j < n; j++) {
            double xj = x[j] > 1e-300 ? x[j] : 1e-300;
            D[j] = z[j] / xj;
        }

        /* factor (retry with larger regularization on failure) */
        LuFact *f = NULL;
        double delta = 1e-8;
        for (int try = 0; try < 5; try++) {
            f = build_factor(A, Q, D, m, n, delta);
            if (f) break;
            delta *= 100.0;
        }
        if (!f) { free(D); status = IPM_SINGULAR; break; }

        /* ---- affine predictor (sigma = 0) ---- */
        for (int j = 0; j < n; j++) rc[j] = x[j] * z[j];
        for (int j = 0; j < n; j++) {
            double xj = x[j] > 1e-300 ? x[j] : 1e-300;
            rhs[j] = rd[j] + rc[j] / xj;
        }
        for (int i = 0; i < m; i++) rhs[n + i] = -rp[i];
        dmat_lu_solve(f, rhs);
        for (int j = 0; j < n; j++) dxa[j] = rhs[j];
        for (int i = 0; i < m; i++) dya[i] = rhs[n + i];
        for (int j = 0; j < n; j++) {
            double xj = x[j] > 1e-300 ? x[j] : 1e-300;
            dza[j] = -(rc[j] + z[j] * dxa[j]) / xj;
        }

        /* affine step lengths */
        double ap_aff = 1.0, ad_aff = 1.0;
        for (int j = 0; j < n; j++) {
            if (dxa[j] < 0.0) { double t = -x[j] / dxa[j]; if (t < ap_aff) ap_aff = t; }
            if (dza[j] < 0.0) { double t = -z[j] / dza[j]; if (t < ad_aff) ad_aff = t; }
        }
        double mu_aff = 0.0;
        for (int j = 0; j < n; j++)
            mu_aff += (x[j] + ap_aff * dxa[j]) * (z[j] + ad_aff * dza[j]);
        mu_aff /= (double)n;
        double sigma = (mu > 0.0) ? (mu_aff / mu) : 0.0;
        sigma = sigma * sigma * sigma;
        if (!(sigma > 1e-8)) sigma = 1e-8;
        if (sigma > 1.0) sigma = 1.0;

        /* ---- correttori (Mehrotra; g_intpnt_max_cor >= 1) ----
         * Il primo e' il correttore di Mehrotra (termine del second'ordine
         * affine dxa*dza); i successivi usano il termine del correttore
         * precedente (higher-order corrector, MSK_IPAR_INTPNT_MAX_NUM_COR). */
        for (int cor = 0; cor < g_intpnt_max_cor; cor++) {
            for (int j = 0; j < n; j++)
                rc[j] = x[j] * z[j] - sigma * mu +
                        (cor == 0 ? dxa[j] * dza[j] : dx[j] * dz[j]);
            for (int j = 0; j < n; j++) {
                double xj = x[j] > 1e-300 ? x[j] : 1e-300;
                rhs[j] = rd[j] + rc[j] / xj;
            }
            for (int i = 0; i < m; i++) rhs[n + i] = -rp[i];
            dmat_lu_solve(f, rhs);
            for (int j = 0; j < n; j++) dx[j] = rhs[j];
            for (int i = 0; i < m; i++) dy[i] = rhs[n + i];
            for (int j = 0; j < n; j++) {
                double xj = x[j] > 1e-300 ? x[j] : 1e-300;
                dz[j] = -(rc[j] + z[j] * dx[j]) / xj;
            }
        }
        dmat_lu_free(f);
        free(D);

        /* step lengths with fraction-to-boundary */
        double ap_max = 1.0, ad_max = 1.0;
        for (int j = 0; j < n; j++) {
            if (dx[j] < 0.0) { double t = -x[j] / dx[j]; if (t < ap_max) ap_max = t; }
            if (dz[j] < 0.0) { double t = -z[j] / dz[j]; if (t < ad_max) ad_max = t; }
        }
        double tau = 1.0 - mu;
        if (tau < 0.99) tau = 0.99;
        if (tau > 0.999999) tau = 0.999999;
        double ap = tau * ap_max, ad = tau * ad_max;
        if (ap > 1.0) ap = 1.0;
        if (ad > 1.0) ad = 1.0;

        /* safeguard: if corrector step collapsed, fall back to affine */
        if (ap < 0.1 * (tau * ap_aff) || ad < 0.1 * (tau * ad_aff)) {
            ap = tau * ap_aff; if (ap > 1.0) ap = 1.0;
            ad = tau * ad_aff; if (ad > 1.0) ad = 1.0;
            for (int j = 0; j < n; j++) { dx[j] = dxa[j]; dz[j] = dza[j]; }
            for (int i = 0; i < m; i++) dy[i] = dya[i];
        }

        if (ap <= 1e-13 && ad <= 1e-13) {
            if (++no_progress > 5) break;
        } else no_progress = 0;

        for (int j = 0; j < n; j++) { x[j] += ap * dx[j]; z[j] += ad * dz[j]; }
        for (int i = 0; i < m; i++) y[i] += ad * dy[i];
    }

    free(rp); free(rd); free(rc); free(qx); free(rhs);
    free(dxa); free(dya); free(dza); free(dx); free(dy); free(dz);
    return status;
}


/* =====================================================================
 * Sparse LP interior point (Mehrotra) via normal equations.
 *
 * min c'x, Ax = b, x >= 0. Residuals (same convention as ipm_solve_std):
 *   rp = Ax - b,  rd = c - A'y - z,  mu = x'z/n
 * Newton (Mehrotra corrector):
 *   g  = (sigma*mu - x*z - dxa*dza)/(z+dd) - theta*rd,  theta = (x+dp)/(z+dd)
 *   K dy = -rp - A g,   K = A Theta A' + delta I   (sparse Cholesky)
 *   dx = g + theta*(A'dy),   dz = rd - A'dy
 * Primal-dual regularization (Vanderbei) keeps theta bounded; delta is
 * escalated on factorization failure. Starting point: Mehrotra's
 * least-squares init (x = 1 + A'(AA')^-1(b - A1), y = (AA')^-1 A c) with
 * positivity shifts.
 * ===================================================================== */

/* build K = A Theta A' + delta I (lower CSC) and factor it.
 * Kd is a dense m x m scratch; Kp has m+1 ints; the arrays Ki and Kx are
 * (re)allocated to *kcap. Returns the factor or NULL. */
static SpChol *factor_K(const int *Aptr, const int *Arow, const double *Aval,
                        const double *theta, int m, int n, double delta,
                        double *Kd, int *Kp, int *kcap, int **Ki, double **Kx) {
    memset(Kd, 0, (size_t)m * (size_t)m * sizeof(double));
    for (int j = 0; j < n; j++) {
        double th = theta[j];
        for (int p = Aptr[j]; p < Aptr[j + 1]; p++) {
            int i1 = Arow[p];
            double v1 = Aval[p] * th;
            for (int p2 = p; p2 < Aptr[j + 1]; p2++) {
                int i2 = Arow[p2];
                double prod = v1 * Aval[p2];
                if (i1 >= i2) Kd[i1 * m + i2] += prod;
                else          Kd[i2 * m + i1] += prod;
            }
        }
    }
    for (int i = 0; i < m; i++) Kd[i * m + i] += delta;
    Kp[0] = 0;
    for (int i2 = 0; i2 < m; i2++) {
        int cnt = 0;
        for (int i1 = i2; i1 < m; i1++)
            if (Kd[i1 * m + i2] != 0.0) cnt++;
        Kp[i2 + 1] = Kp[i2] + cnt;
    }
    if (Kp[m] > *kcap) {
        *kcap = Kp[m];
        free(*Ki); free(*Kx);
        *Ki = (int *)malloc((size_t)(*kcap > 0 ? *kcap : 1) * sizeof(int));
        *Kx = (double *)malloc((size_t)(*kcap > 0 ? *kcap : 1) * sizeof(double));
        if (!*Ki || !*Kx) return NULL;
    }
    for (int i2 = 0; i2 < m; i2++) {
        int w = Kp[i2];
        for (int i1 = i2; i1 < m; i1++) {
            double v = Kd[i1 * m + i2];
            if (v != 0.0) { (*Ki)[w] = i1; (*Kx)[w] = v; w++; }
        }
    }
    return spchol_factor_ord(m, Kp, *Ki, *Kx);
}

/**
 * Solves an LP in CSC format using sparse Mehrotra IPM (normal equations).
 *
 * @param Aptr      [in]  Column pointers for A (size n+1, CSC format).
 * @param Arow      [in]  Row indices for A (size Aptr[n]).
 * @param Aval      [in]  Values for A (size Aptr[n]).
 * @param m         [in]  Number of constraints.
 * @param n         [in]  Number of variables.
 * @param b         [in]  RHS vector (size m).
 * @param c         [in]  Objective coefficients (size n).
 * @param tol_gap   [in]  Relative gap tolerance.
 * @param tol_pfeas [in]  Primal feasibility tolerance.
 * @param tol_dfeas [in]  Dual feasibility tolerance.
 * @param max_iter  [in]  Maximum iterations.
 * @param x         [out] Primal solution (size n). Must not be NULL.
 * @param y         [out] Dual solution (size m). Must not be NULL.
 * @param z         [out] Dual slack (size n). Must not be NULL.
 * @param x0        [in]  Optional warm start x (size n). NaN entries -> default.
 * @param y0        [in]  Optional warm start y (size m). NaN entries -> default.
 *
 * @return IPM_OPTIMAL (0), IPM_MAXITER (1), IPM_MEMORY (2), IPM_SINGULAR (3).
 *
 * @note Solves LP: min c'x  s.t.  Ax = b, x >= 0 using sparse normal equations.
 *
 *       K = A Theta A' + delta I, where Theta_j = (x_j + dp) / (z_j + dd)
 *       with Vanderbei regularization dp, dd.
 *
 *       Solves via sparse Cholesky (spchol_factor/spchol_solve) on K.
 *       Builds CSR of A on entry for Ag = A g products.
 *
 *       Starting point: if x0/y0 provided, projects to interior; else uses
 *       Mehrotra least-squares init: x = 1 + A'(AA'+dI)^-1(b-A1),
 *       y = (AA'+dI)^-1 A c, z = c - A'y with positivity shifts.
 *
 *       Mehrotra predictor-corrector with fraction-to-boundary step lengths.
 *       Objective cuts and wall-clock deadline same as ipm_solve_std.
 *
 * @example
 * int Aptr[] = {0, 2, 4};
 * int Arow[] = {0, 1, 0, 1};
 * double Aval[] = {1, 1, 1, 1};
 * double b[] = {1, 1}, c[] = {1, 1};
 * double x[2], y[2], z[2];
 * int rc = ipm_solve_std_csc(Aptr, Arow, Aval, 2, 2, b, c, 1e-8, 1e-8, 1e-8, 100, x, y, z, NULL, NULL);
 */
int ipm_solve_std_csc(const int *Aptr, const int *Arow, const double *Aval,
                      int m, int n,
                      const double *b, const double *c,
                      double tol_gap, double tol_pfeas, double tol_dfeas,
                      int max_iter,
                      double *x, double *y, double *z,
                      const double *x0, const double *y0)
{
    if (n <= 0) return IPM_OPTIMAL;
    if (m <= 0) return IPM_MAXITER;

    double *rp    = (double *)calloc((size_t)m, sizeof(double));
    double *rd    = (double *)calloc((size_t)n, sizeof(double));
    double *tmpn  = (double *)calloc((size_t)n, sizeof(double));   /* A'v / g */
    double *tmpm  = (double *)calloc((size_t)m, sizeof(double));   /* A v / rhs */
    double *dyA   = (double *)calloc((size_t)m, sizeof(double));
    double *dxA   = (double *)calloc((size_t)n, sizeof(double));
    double *dzA   = (double *)calloc((size_t)n, sizeof(double));
    double *dxC   = (double *)calloc((size_t)n, sizeof(double));
    double *dzC   = (double *)calloc((size_t)n, sizeof(double));
    double *theta = (double *)calloc((size_t)n, sizeof(double));
    double *g     = (double *)calloc((size_t)n, sizeof(double));
    double *Ag    = (double *)calloc((size_t)m, sizeof(double));
    double *Kd    = (double *)calloc((size_t)m * (size_t)m, sizeof(double));
    int *rptr = (int *)calloc((size_t)(m + 1), sizeof(int));
    int nnzA = Aptr[n];
    int *ridx = (int *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(int));
    double *rval = (double *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(double));
    int *Kp = (int *)malloc((size_t)(m + 1) * sizeof(int));
    int *Ki = NULL;
    double *Kx = NULL;
    int kcap = 0;
    int memfail = 0;
    if (!rp || !rd || !tmpn || !tmpm || !dyA || !dxA || !dzA || !dxC || !dzC ||
        !theta || !g || !Ag || !Kd || !rptr || !ridx || !rval || !Kp) memfail = 1;

#define IPMSP_FREE() do { \
    free(rp); free(rd); free(tmpn); free(tmpm); free(dyA); \
    free(dxA); free(dzA); free(dxC); free(dzC); free(theta); free(g); \
    free(Ag); free(Kd); free(rptr); free(ridx); free(rval); free(Kp); \
    free(Ki); free(Kx); } while (0)

    if (memfail) { IPMSP_FREE(); return IPM_MEMORY; }

    /* CSC -> CSR */
    for (int j = 0; j < n; j++)
        for (int p = Aptr[j]; p < Aptr[j + 1]; p++) rptr[Arow[p] + 1]++;
    for (int i = 0; i < m; i++) rptr[i + 1] += rptr[i];
    {
        int *fill = (int *)calloc((size_t)m, sizeof(int));
        if (!fill) { IPMSP_FREE(); return IPM_MEMORY; }
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) {
                int i = Arow[p];
                int pos = rptr[i] + fill[i]++;
                ridx[pos] = j; rval[pos] = Aval[p];
            }
        free(fill);
    }

    /* ---------------- starting point ---------------- */
    if (x0) {
        for (int j = 0; j < n; j++)
            if (x0[j] != x0[j]) { x0 = NULL; break; }
    }
    if (x0) {
        for (int j = 0; j < n; j++) x[j] = x0[j] > 1e-8 ? x0[j] : 1e-8;
        for (int i = 0; i < m; i++) y[i] = y0 ? y0[i] : 0.0;
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * y[Arow[p]];
        for (int j = 0; j < n; j++) {
            double s2 = c[j] - tmpn[j];
            z[j] = s2 > 1e-2 ? s2 : 1e-2;
        }
    } else {
        /* Mehrotra least-squares init with Theta = I */
        for (int j = 0; j < n; j++) theta[j] = 1.0;
        SpChol *L0 = NULL;
        double delta0 = 1e-9;
        for (int tries = 0; tries < 6 && !L0; tries++) {
            L0 = factor_K(Aptr, Arow, Aval, theta, m, n, delta0, Kd, Kp, &kcap, &Ki, &Kx);
            if (!L0) delta0 *= 100.0;
        }
        if (!L0) { IPMSP_FREE(); return IPM_SINGULAR; }
        /* x_hat = 1 + A' u,  (AA'+dI) u = b - A*1 */
        for (int i = 0; i < m; i++) {
            double s = b[i];
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s -= rval[p];
            tmpm[i] = s;
        }
        if (spchol_solve_ord(L0, tmpm) != 0) { spchol_free(L0); IPMSP_FREE(); return IPM_SINGULAR; }
        for (int j = 0; j < n; j++) x[j] = 1.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                x[j] += Aval[p] * tmpm[Arow[p]];
        /* y0: (AA'+dI) y = A c ; z0 = c - A'y */
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * c[ridx[p]];
            tmpm[i] = s;
        }
        if (spchol_solve_ord(L0, tmpm) != 0) { spchol_free(L0); IPMSP_FREE(); return IPM_SINGULAR; }
        for (int i = 0; i < m; i++) y[i] = tmpm[i];
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * y[Arow[p]];
        for (int j = 0; j < n; j++) z[j] = c[j] - tmpn[j];
        spchol_free(L0);
        /* positivity shifts */
        double minx = 0.0, minz = 0.0;
        for (int j = 0; j < n; j++) { if (x[j] < minx) minx = x[j]; if (z[j] < minz) minz = z[j]; }
        double sx = 0.0, sz = 0.0;
        if (minx < 0.0) { sx = -1.5 * minx; }
        if (minz < 0.0) { sz = -1.5 * minz; }
        for (int j = 0; j < n; j++) { x[j] += sx; z[j] += sz; }
        double xzsum = 0.0, zsum = 0.0, xsum = 0.0;
        for (int j = 0; j < n; j++) { xzsum += x[j] * z[j]; zsum += z[j]; xsum += x[j]; }
        double dxg = (zsum > 0.0) ? 0.5 * xzsum / zsum : 1.0;
        double dzg = (xsum > 0.0) ? 0.5 * xzsum / xsum : 1.0;
        for (int j = 0; j < n; j++) { x[j] += dxg + 1e-6; z[j] += dzg + 1e-6; }
    }

    double bnorm = 1.0, cnorm = 1.0;
    for (int i = 0; i < m; i++) { double a = fabs(b[i]); if (a > bnorm) bnorm = a; }
    for (int j = 0; j < n; j++) { double a = fabs(c[j]); if (a > cnorm) cnorm = a; }
    int is_qp = 0;   /* LP-only loop */

    int status = IPM_MAXITER;
    int no_progress = 0;

    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(90);
        if (ipm_past_deadline()) { status = IPM_MAXITER; break; }
        /* residuals */
        for (int i = 0; i < m; i++) rp[i] = -b[i];
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++)
                rp[i] += rval[p] * x[ridx[p]];
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * y[Arow[p]];
        double xz = 0.0;
        for (int j = 0; j < n; j++) {
            rd[j] = c[j] - tmpn[j] - z[j];
            xz += x[j] * z[j];
        }
        double mu = xz / (double)n;
        double pobj = 0.0;
        for (int j = 0; j < n; j++) pobj += c[j] * x[j];
        double feas_p = 0.0, feas_d = 0.0;
        for (int i = 0; i < m; i++) { double a = fabs(rp[i]); if (a > feas_p) feas_p = a; }
        for (int j = 0; j < n; j++) { double a = fabs(rd[j]); if (a > feas_d) feas_d = a; }
        feas_p /= bnorm; feas_d /= cnorm;
        double gap = fabs(xz) / (1.0 + fabs(pobj));
        if (feas_p <= tol_pfeas && pobj < g_obj_lower) { g_obj_cut_hit = 1; status = IPM_MAXITER; break; }
        if (!is_qp) {   /* LP: the dual objective b'y is a valid bound */
            double db = 0.0;
            for (int i = 0; i < m; i++) db += b[i] * y[i];
            if (feas_d <= tol_dfeas && db > g_obj_upper) { g_obj_cut_hit = 1; status = IPM_MAXITER; break; }
        }
        if (getenv("GMB_DBG") && (it % 5 == 0))
            fprintf(stderr, "spipm it=%d feas_p=%.3g feas_d=%.3g gap=%.3g mu=%.3g\n",
                    it, feas_p, feas_d, gap, mu);
        if (feas_p <= tol_pfeas && feas_d <= tol_dfeas && gap <= tol_gap) {
            status = IPM_OPTIMAL;
            break;
        }

        /* regularized theta (Vanderbei: dp = dd = delta) */
        double mx = 0.0, mz = 0.0;
        for (int j = 0; j < n; j++) { mx += x[j]; mz += z[j]; }
        mx /= (double)n; mz /= (double)n;
        double delta = 1e-6;
        double dp_reg = delta * (1.0 + mx), dd_reg = delta * (1.0 + mz);
        for (int j = 0; j < n; j++)
            theta[j] = (x[j] + dp_reg) / (z[j] + dd_reg);

        /* ---- factor K (delta escalated) ---- */
        SpChol *L = NULL;
        for (int tries = 0; tries < 6 && !L; tries++) {
            L = factor_K(Aptr, Arow, Aval, theta, m, n, delta, Kd, Kp, &kcap, &Ki, &Kx);
            if (!L) { delta *= 100.0; if (delta > 1e4) break; }
        }
        if (!L) { status = IPM_SINGULAR; break; }

        /* ---- affine predictor (sigma = 0) ---- */
        for (int j = 0; j < n; j++)
            g[j] = -x[j] * z[j] / (z[j] + dd_reg) - theta[j] * rd[j];
        for (int i = 0; i < m; i++) Ag[i] = 0.0;
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++)
                Ag[i] += rval[p] * g[ridx[p]];
        for (int i = 0; i < m; i++) tmpm[i] = -rp[i] - Ag[i];
        if (spchol_solve_ord(L, tmpm) != 0) { spchol_free(L); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) dyA[i] = tmpm[i];
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * dyA[Arow[p]];
        for (int j = 0; j < n; j++) {
            dxA[j] = g[j] + theta[j] * tmpn[j];
            dzA[j] = rd[j] - tmpn[j];
        }
        double ap_aff = 1.0, ad_aff = 1.0;
        for (int j = 0; j < n; j++) {
            if (dxA[j] < 0.0) { double t = -x[j] / dxA[j]; if (t < ap_aff) ap_aff = t; }
            if (dzA[j] < 0.0) { double t = -z[j] / dzA[j]; if (t < ad_aff) ad_aff = t; }
        }
        double mu_aff = 0.0;
        for (int j = 0; j < n; j++)
            mu_aff += (x[j] + ap_aff * dxA[j]) * (z[j] + ad_aff * dzA[j]);
        mu_aff /= (double)n;
        double sigma = (mu > 0.0) ? (mu_aff / mu) : 0.0;
        sigma = sigma * sigma * sigma;
        if (!(sigma > 1e-8)) sigma = 1e-8;
        if (sigma > 1.0) sigma = 1.0;

        /* ---- corrector ---- */
        for (int j = 0; j < n; j++)
            g[j] = (sigma * mu - x[j] * z[j] - dxA[j] * dzA[j]) /
                   (z[j] + dd_reg) - theta[j] * rd[j];
        for (int i = 0; i < m; i++) Ag[i] = 0.0;
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++)
                Ag[i] += rval[p] * g[ridx[p]];
        for (int i = 0; i < m; i++) tmpm[i] = -rp[i] - Ag[i];
        if (spchol_solve_ord(L, tmpm) != 0) { spchol_free(L); status = IPM_SINGULAR; break; }
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * tmpm[Arow[p]];
        for (int j = 0; j < n; j++) {
            dxC[j] = g[j] + theta[j] * tmpn[j];
            dzC[j] = rd[j] - tmpn[j];
        }
        spchol_free(L);

        double ap_max = 1.0, ad_max = 1.0;
        for (int j = 0; j < n; j++) {
            if (dxC[j] < 0.0) { double t = -x[j] / dxC[j]; if (t < ap_max) ap_max = t; }
            if (dzC[j] < 0.0) { double t = -z[j] / dzC[j]; if (t < ad_max) ad_max = t; }
        }
        double tau = 1.0 - mu;
        if (tau < 0.99) tau = 0.99;
        if (tau > 0.999999) tau = 0.999999;
        double ap = tau * ap_max, ad = tau * ad_max;
        if (ap > 1.0) ap = 1.0;
        if (ad > 1.0) ad = 1.0;
        if (ap < 0.1 * (tau * ap_aff) || ad < 0.1 * (tau * ad_aff)) {
            ap = tau * ap_aff; if (ap > 1.0) ap = 1.0;
            ad = tau * ad_aff; if (ad > 1.0) ad = 1.0;
            for (int j = 0; j < n; j++) { dxC[j] = dxA[j]; dzC[j] = dzA[j]; }
            for (int i = 0; i < m; i++) tmpm[i] = dyA[i];   /* dyC = dyA */
        }
        if (ap <= 1e-13 && ad <= 1e-13) {
            if (++no_progress > 5) break;
        } else no_progress = 0;

        if (getenv("GMB_DBG")) {
            double mdy = 0.0, mdx = 0.0, mdz = 0.0;
            for (int i = 0; i < m; i++) { double a = fabs(tmpm[i]); if (a > mdy) mdy = a; }
            for (int j = 0; j < n; j++) { double a = fabs(dxC[j]); if (a > mdx) mdx = a; }
            for (int j = 0; j < n; j++) { double a = fabs(dzC[j]); if (a > mdz) mdz = a; }
            double minx = 1e300, minz = 1e300, maxth = 0.0;
            for (int j = 0; j < n; j++) {
                if (x[j] < minx) minx = x[j];
                if (z[j] < minz) minz = z[j];
                if (theta[j] > maxth) maxth = theta[j];
            }
            fprintf(stderr, "  step it=%d ap=%.4g ad=%.4g delta=%.2g max|dy|=%.3g max|dx|=%.3g max|dz|=%.3g sigma=%.3g minx=%.3g minz=%.3g maxth=%.3g\n",
                    it, ap, ad, delta, mdy, mdx, mdz, sigma, minx, minz, maxth);
        }

        for (int j = 0; j < n; j++) { x[j] += ap * dxC[j]; z[j] += ad * dzC[j]; }
        for (int i = 0; i < m; i++) y[i] += ad * tmpm[i];
    }

    IPMSP_FREE();
#undef IPMSP_FREE
    return status;
}

/* factor a dense symmetric m x m matrix (lower triangle read) + already-added
 * delta on the diagonal, via the sparse Cholesky. Diagonal is always emitted. */
static SpChol *spchol_from_dense(const double *Kd, int m,
                                 int *Kp, int *kcap, int **Ki, double **Kx) {
    Kp[0] = 0;
    for (int j = 0; j < m; j++) {
        int cnt = 1;
        for (int i = j + 1; i < m; i++) if (Kd[(size_t)i * m + j] != 0.0) cnt++;
        Kp[j + 1] = Kp[j] + cnt;
    }
    if (Kp[m] > *kcap) {
        *kcap = Kp[m];
        free(*Ki); free(*Kx);
        *Ki = (int *)malloc((size_t)(*kcap > 0 ? *kcap : 1) * sizeof(int));
        *Kx = (double *)malloc((size_t)(*kcap > 0 ? *kcap : 1) * sizeof(double));
        if (!*Ki || !*Kx) return NULL;
    }
    for (int j = 0; j < m; j++) {
        int w = Kp[j];
        (*Ki)[w] = j; (*Kx)[w] = Kd[(size_t)j * m + j]; w++;
        for (int i = j + 1; i < m; i++) {
            double v = Kd[(size_t)i * m + j];
            if (v != 0.0) { (*Ki)[w] = i; (*Kx)[w] = v; w++; }
        }
    }
    return spchol_factor_ord(m, Kp, *Ki, *Kx);
}

/**
 * Solves a QP in CSC format using sparse Mehrotra IPM with Q + D regularization.
 *
 * @param Aptr      [in]  Column pointers for A (size n+1, CSC).
 * @param Arow      [in]  Row indices for A (size Aptr[n]).
 * @param Aval      [in]  Values for A (size Aptr[n]).
 * @param Qptr      [in]  Column pointers for Q (size n+1, CSC).
 * @param Qrow      [in]  Row indices for Q (size Qptr[n]).
 * @param Qval      [in]  Values for Q (size Qptr[n]).
 * @param m         [in]  Number of constraints.
 * @param n         [in]  Number of variables.
 * @param b         [in]  RHS vector (size m).
 * @param c         [in]  Linear objective (size n).
 * @param tol_gap   [in]  Relative gap tolerance.
 * @param tol_pfeas [in]  Primal feasibility tolerance.
 * @param tol_dfeas [in]  Dual feasibility tolerance.
 * @param max_iter  [in]  Maximum iterations.
 * @param x         [out] Primal solution (size n). Must not be NULL.
 * @param y         [out] Dual solution (size m). Must not be NULL.
 * @param z         [out] Dual slack (size n). Must not be NULL.
 * @param x0        [in]  Optional warm start x (size n). NaN entries -> default.
 * @param y0        [in]  Optional warm start y (size m). NaN entries -> default.
 *
 * @return IPM_OPTIMAL (0), IPM_MAXITER (1), IPM_MEMORY (2), IPM_SINGULAR (3).
 *
 * @note Solves: min 1/2 x'Qx + c'x  s.t.  Ax = b, x >= 0.
 *
 *       Uses sparse Cholesky on:
 *       - M = Q + D (n x n), where D = (z+dd)/(x+dp) (Vanderbei)
 *       - K = A W + delta I (m x m), where W = M^-1 A'
 *
 *       Solves two linear systems per iteration via sparse Cholesky:
 *       1. M u = -(xz - rd) / (x+dp)  (affine predictor on Q-space)
 *       2. K dy = -rp - A W u           (affine dual step)
 *       Then recovers dx = u + W dy, dz = Q dx - A' dy + rd.
 *
 *       Requires explicit diagonal entries in Q (caller must ensure).
 *       M is refactored each iteration; K is refactored with delta escalation.
 *
 * @example
 * // See ipm.c for CSC format description
 */
int ipm_solve_qp_csc(const int *Aptr, const int *Arow, const double *Aval,
                     const int *Qptr, const int *Qrow, const double *Qval,
                     int m, int n, const double *b, const double *c,
                     double tol_gap, double tol_pfeas, double tol_dfeas,
                     int max_iter, double *x, double *y, double *z,
                     const double *x0, const double *y0)
{
    if (n <= 0) return IPM_OPTIMAL;
    if (m <= 0) return IPM_MAXITER;
    int nnzA = Aptr[n], nnzQ = Qptr[n];

    double *rp   = (double *)calloc((size_t)m, sizeof(double));
    double *rd   = (double *)calloc((size_t)n, sizeof(double));
    double *Qx   = (double *)calloc((size_t)n, sizeof(double));
    double *tmpn = (double *)calloc((size_t)n, sizeof(double));
    double *tmpm = (double *)calloc((size_t)m, sizeof(double));
    double *dyA  = (double *)calloc((size_t)m, sizeof(double));
    double *dxA  = (double *)calloc((size_t)n, sizeof(double));
    double *dzA  = (double *)calloc((size_t)n, sizeof(double));
    double *dxC  = (double *)calloc((size_t)n, sizeof(double));
    double *dzC  = (double *)calloc((size_t)n, sizeof(double));
    double *D    = (double *)calloc((size_t)n, sizeof(double));
    double *u    = (double *)calloc((size_t)n, sizeof(double));
    double *Atdy = (double *)calloc((size_t)n, sizeof(double));
    double *Mval = (double *)malloc((size_t)(nnzQ > 0 ? nnzQ : 1) * sizeof(double));
    double *W    = (double *)malloc((size_t)n * (size_t)m * sizeof(double));
    double *Kd   = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    int *rptr  = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *ridx  = (int *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(int));
    double *rval = (double *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(double));
    int *diagpos = (int *)malloc((size_t)n * sizeof(int));
    int *Kp = (int *)malloc((size_t)(m + 1) * sizeof(int));
    int *Ki = NULL; double *Kx = NULL; int kcap = 0;
    int memfail = 0;
    if (!rp||!rd||!Qx||!tmpn||!tmpm||!dyA||!dxA||!dzA||!dxC||!dzC||!D||!u||!Atdy||
        !Mval||!W||!Kd||!rptr||!ridx||!rval||!diagpos||!Kp) memfail = 1;

#define IPMQP_FREE() do { free(rp);free(rd);free(Qx);free(tmpn);free(tmpm);free(dyA);\
    free(dxA);free(dzA);free(dxC);free(dzC);free(D);free(u);free(Atdy);free(Mval);\
    free(W);free(Kd);free(rptr);free(ridx);free(rval);free(diagpos);free(Kp);free(Ki);free(Kx);} while(0)
    if (memfail) { IPMQP_FREE(); return IPM_MEMORY; }

    /* CSR of A */
    for (int j = 0; j < n; j++)
        for (int p = Aptr[j]; p < Aptr[j + 1]; p++) rptr[Arow[p] + 1]++;
    for (int i = 0; i < m; i++) rptr[i + 1] += rptr[i];
    { int *fill = (int *)calloc((size_t)m, sizeof(int));
      if (!fill) { IPMQP_FREE(); return IPM_MEMORY; }
      for (int j = 0; j < n; j++)
        for (int p = Aptr[j]; p < Aptr[j + 1]; p++) {
            int i = Arow[p], pos = rptr[i] + fill[i]++;
            ridx[pos] = j; rval[pos] = Aval[p];
        }
      free(fill); }

    /* Q diagonal positions (caller guarantees an explicit diagonal entry) */
    for (int j = 0; j < n; j++) {
        diagpos[j] = -1;
        for (int p = Qptr[j]; p < Qptr[j + 1]; p++) if (Qrow[p] == j) { diagpos[j] = p; break; }
        if (diagpos[j] < 0) { IPMQP_FREE(); return IPM_SINGULAR; }
    }

    /* ---- starting point (Mehrotra LS, Theta = I) ---- */
    if (x0) { for (int j = 0; j < n; j++) if (x0[j] != x0[j]) { x0 = NULL; break; } }
    if (x0) {
        for (int j = 0; j < n; j++) x[j] = x0[j] > 1e-8 ? x0[j] : 1e-8;
        for (int i = 0; i < m; i++) y[i] = y0 ? y0[i] : 0.0;
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) tmpn[j] += Aval[p] * y[Arow[p]];
        for (int j = 0; j < n; j++) Qx[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
                int i = Qrow[p]; Qx[i] += Qval[p] * x[j];
                if (i != j) Qx[j] += Qval[p] * x[i];
            }
        for (int j = 0; j < n; j++) { double s2 = c[j] + Qx[j] - tmpn[j]; z[j] = s2 > 1e-2 ? s2 : 1e-2; }
    } else {
        for (int j = 0; j < n; j++) D[j] = 1.0;   /* theta = I */
        SpChol *L0 = NULL; double d0 = 1e-9;
        for (int tr = 0; tr < 6 && !L0; tr++) {
            L0 = factor_K(Aptr, Arow, Aval, D, m, n, d0, Kd, Kp, &kcap, &Ki, &Kx);
            if (!L0) d0 *= 100.0;
        }
        if (!L0) { IPMQP_FREE(); return IPM_SINGULAR; }
        for (int i = 0; i < m; i++) { double s = b[i];
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s -= rval[p]; tmpm[i] = s; }
        if (spchol_solve_ord(L0, tmpm)) { spchol_free(L0); IPMQP_FREE(); return IPM_SINGULAR; }
        for (int j = 0; j < n; j++) x[j] = 1.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) x[j] += Aval[p] * tmpm[Arow[p]];
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * c[ridx[p]]; tmpm[i] = s; }
        if (spchol_solve_ord(L0, tmpm)) { spchol_free(L0); IPMQP_FREE(); return IPM_SINGULAR; }
        for (int i = 0; i < m; i++) y[i] = tmpm[i];
        spchol_free(L0);
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) tmpn[j] += Aval[p] * y[Arow[p]];
        /* z = c + Qx - A'y */
        for (int j = 0; j < n; j++) Qx[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
                int i = Qrow[p]; Qx[i] += Qval[p] * x[j];
                if (i != j) Qx[j] += Qval[p] * x[i];
            }
        for (int j = 0; j < n; j++) z[j] = c[j] + Qx[j] - tmpn[j];
        double minx = 0.0, minz = 0.0;
        for (int j = 0; j < n; j++) { if (x[j] < minx) minx = x[j]; if (z[j] < minz) minz = z[j]; }
        double sx = minx < 0.0 ? -1.5 * minx : 0.0, sz = minz < 0.0 ? -1.5 * minz : 0.0;
        for (int j = 0; j < n; j++) { x[j] += sx; z[j] += sz; }
        double xzsum = 0.0, zsum = 0.0, xsum = 0.0;
        for (int j = 0; j < n; j++) { xzsum += x[j]*z[j]; zsum += z[j]; xsum += x[j]; }
        double dxg = zsum > 0.0 ? 0.5*xzsum/zsum : 1.0, dzg = xsum > 0.0 ? 0.5*xzsum/xsum : 1.0;
        for (int j = 0; j < n; j++) { x[j] += dxg + 1e-6; z[j] += dzg + 1e-6; }
    }

    double bnorm = 1.0, cnorm = 1.0;
    for (int i = 0; i < m; i++) { double a = fabs(b[i]); if (a > bnorm) bnorm = a; }
    for (int j = 0; j < n; j++) { double a = fabs(c[j]); if (a > cnorm) cnorm = a; }
    for (int p = 0; p < nnzQ; p++) { double a = fabs(Qval[p]); if (a > cnorm) cnorm = a; }

    int status = IPM_MAXITER, no_progress = 0;
    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(90);
        if (ipm_past_deadline()) { status = IPM_MAXITER; break; }
        /* residuals */
        for (int i = 0; i < m; i++) rp[i] = -b[i];
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++) rp[i] += rval[p] * x[ridx[p]];
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) tmpn[j] += Aval[p] * y[Arow[p]];
        for (int j = 0; j < n; j++) Qx[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
                int i = Qrow[p]; Qx[i] += Qval[p] * x[j];
                if (i != j) Qx[j] += Qval[p] * x[i];
            }
        double xz = 0.0;
        for (int j = 0; j < n; j++) { rd[j] = Qx[j] + c[j] - tmpn[j] - z[j]; xz += x[j] * z[j]; }
        double mu = xz / (double)n;
        double pobj = 0.0;
        for (int j = 0; j < n; j++) pobj += c[j] * x[j] + 0.5 * Qx[j] * x[j];
        double feas_p = 0.0, feas_d = 0.0;
        for (int i = 0; i < m; i++) { double a = fabs(rp[i]); if (a > feas_p) feas_p = a; }
        for (int j = 0; j < n; j++) { double a = fabs(rd[j]); if (a > feas_d) feas_d = a; }
        feas_p /= bnorm; feas_d /= cnorm;
        double gap = fabs(xz) / (1.0 + fabs(pobj));
        if (feas_p <= tol_pfeas && pobj < g_obj_lower) { g_obj_cut_hit = 1; status = IPM_MAXITER; break; }
        if (getenv("GMB_DBG") && (it % 5 == 0))
            fprintf(stderr, "spqp it=%d feas_p=%.3g feas_d=%.3g gap=%.3g mu=%.3g\n", it, feas_p, feas_d, gap, mu);
        if (feas_p <= tol_pfeas && feas_d <= tol_dfeas && gap <= tol_gap) { status = IPM_OPTIMAL; break; }

        /* M = Q + D, D = (z+dd)/(x+dp) (Vanderbei) */
        double mx = 0.0, mz = 0.0;
        for (int j = 0; j < n; j++) { mx += x[j]; mz += z[j]; }
        mx /= (double)n; mz /= (double)n;
        double delta = 1e-6, dp = delta * (1.0 + mx), dd = delta * (1.0 + mz);
        for (int j = 0; j < n; j++) D[j] = (z[j] + dd) / (x[j] + dp);
        for (int p = 0; p < nnzQ; p++) Mval[p] = Qval[p];
        for (int j = 0; j < n; j++) Mval[diagpos[j]] += D[j];
        SpChol *Lm = spchol_factor(n, Qptr, Qrow, Mval);
        if (!Lm) { status = IPM_SINGULAR; break; }

        /* W = M^-1 A' (column i = M^-1 (row i of A)) */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) tmpn[j] = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) tmpn[ridx[p]] = rval[p];
            if (spchol_solve(Lm, tmpn)) { spchol_free(Lm); status = IPM_SINGULAR; break; }
            for (int j = 0; j < n; j++) W[(size_t)j * m + i] = tmpn[j];
        }
        if (status == IPM_SINGULAR) break;
        /* Kd = A W + delta I */
        for (int i = 0; i < m; i++)
            for (int k = 0; k < m; k++) Kd[(size_t)i * m + k] = (i == k) ? delta : 0.0;
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++) {
                int j = ridx[p]; double a = rval[p]; const double *Wj = &W[(size_t)j * m];
                for (int k = 0; k < m; k++) Kd[(size_t)i * m + k] += a * Wj[k];
            }
        SpChol *Lk = spchol_from_dense(Kd, m, Kp, &kcap, &Ki, &Kx);
        if (!Lk) { spchol_free(Lm); status = IPM_SINGULAR; break; }

        /* ---- affine predictor (target = -xz) ---- */
        for (int j = 0; j < n; j++) u[j] = -x[j] * z[j] / (x[j] + dp) - rd[j];
        if (spchol_solve(Lm, u)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * u[ridx[p]];
            tmpm[i] = -rp[i] - s; }
        if (spchol_solve(Lk, tmpm)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) dyA[i] = tmpm[i];
        for (int j = 0; j < n; j++) { double s = u[j];
            for (int i = 0; i < m; i++) s += W[(size_t)j * m + i] * dyA[i]; dxA[j] = s; }
        /* dzA = Q dxA - A'dyA + rd */
        for (int j = 0; j < n; j++) Atdy[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) Atdy[j] += Aval[p] * dyA[Arow[p]];
        for (int j = 0; j < n; j++) Qx[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
                int i = Qrow[p]; Qx[i] += Qval[p] * dxA[j];
                if (i != j) Qx[j] += Qval[p] * dxA[i];
            }
        for (int j = 0; j < n; j++) dzA[j] = Qx[j] - Atdy[j] + rd[j];

        double ap_aff = 1.0, ad_aff = 1.0;
        for (int j = 0; j < n; j++) {
            if (dxA[j] < 0.0) { double t = -x[j] / dxA[j]; if (t < ap_aff) ap_aff = t; }
            if (dzA[j] < 0.0) { double t = -z[j] / dzA[j]; if (t < ad_aff) ad_aff = t; }
        }
        double mu_aff = 0.0;
        for (int j = 0; j < n; j++) mu_aff += (x[j] + ap_aff*dxA[j]) * (z[j] + ad_aff*dzA[j]);
        mu_aff /= (double)n;
        double sigma = (mu > 0.0) ? (mu_aff / mu) : 0.0; sigma = sigma*sigma*sigma;
        if (!(sigma > 1e-8)) sigma = 1e-8; if (sigma > 1.0) sigma = 1.0;

        /* ---- corrector ---- */
        for (int j = 0; j < n; j++) u[j] = (sigma*mu - x[j]*z[j] - dxA[j]*dzA[j]) / (x[j] + dp) - rd[j];
        if (spchol_solve(Lm, u)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * u[ridx[p]];
            tmpm[i] = -rp[i] - s; }
        if (spchol_solve(Lk, tmpm)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int j = 0; j < n; j++) { double s = u[j];
            for (int i = 0; i < m; i++) s += W[(size_t)j * m + i] * tmpm[i]; dxC[j] = s; }
        for (int j = 0; j < n; j++) Atdy[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) Atdy[j] += Aval[p] * tmpm[Arow[p]];
        for (int j = 0; j < n; j++) Qx[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Qptr[j]; p < Qptr[j + 1]; p++) {
                int i = Qrow[p]; Qx[i] += Qval[p] * dxC[j];
                if (i != j) Qx[j] += Qval[p] * dxC[i];
            }
        for (int j = 0; j < n; j++) dzC[j] = Qx[j] - Atdy[j] + rd[j];
        spchol_free(Lm); spchol_free(Lk);

        double ap_max = 1.0, ad_max = 1.0;
        for (int j = 0; j < n; j++) {
            if (dxC[j] < 0.0) { double t = -x[j] / dxC[j]; if (t < ap_max) ap_max = t; }
            if (dzC[j] < 0.0) { double t = -z[j] / dzC[j]; if (t < ad_max) ad_max = t; }
        }
        double tau = 1.0 - mu; if (tau < 0.99) tau = 0.99; if (tau > 0.999999) tau = 0.999999;
        double ap = tau*ap_max, ad = tau*ad_max; if (ap > 1.0) ap = 1.0; if (ad > 1.0) ad = 1.0;
        if (ap < 0.1*(tau*ap_aff) || ad < 0.1*(tau*ad_aff)) {
            ap = tau*ap_aff; if (ap > 1.0) ap = 1.0;
            ad = tau*ad_aff; if (ad > 1.0) ad = 1.0;
            for (int j = 0; j < n; j++) { dxC[j] = dxA[j]; dzC[j] = dzA[j]; }
            for (int i = 0; i < m; i++) tmpm[i] = dyA[i];
        }
        if (ap <= 1e-13 && ad <= 1e-13) { if (++no_progress > 5) break; } else no_progress = 0;
        for (int j = 0; j < n; j++) { x[j] += ap*dxC[j]; z[j] += ad*dzC[j]; }
        for (int i = 0; i < m; i++) y[i] += ad*tmpm[i];
    }
    IPMQP_FREE();
#undef IPMQP_FREE
    return status;
}
