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
/* Reports whether the shared wall-clock deadline has passed.
 * Reads the file-scope deadline set by ipm_set_deadline. */
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

/* Number of correctors (MSK_IPAR_INTPNT_MAX_NUM_COR): >= 1. The first is the
 * Mehrotra corrector, the later ones are higher-order. Default 1. */
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

typedef struct {
    LuFact *reduced, *full;
    int *map, *row;
    double *weight;
    int kept;
} QpKkt;

static void qp_kkt_free(QpKkt *K) {
    dmat_lu_free(K->reduced); dmat_lu_free(K->full);
    free(K->map); free(K->row); free(K->weight);
    memset(K, 0, sizeof(*K));
}

/* Eliminate zero-Hessian columns with at most one equality entry.
 * Their diagonal pivots change only the corresponding dual diagonal. */
static int qp_kkt_reduce(QpKkt *K, const double *A, const double *Q,
                         const double *D, int m, int n, double delta) {
    if (!Q || n < 16) return 0;
    K->map = malloc((size_t)n * sizeof(int));
    K->row = malloc((size_t)n * sizeof(int));
    K->weight = malloc((size_t)n * sizeof(double));
    if (!K->map || !K->row || !K->weight) return 0;
    for (int j = 0; j < n; j++) {
        int eligible = D[j] + delta > 0.0 && isfinite(D[j] + delta);
        int row = -1;
        for (int k = 0; k < n && eligible; k++) eligible = Q[(size_t)j*n+k] == 0.0;
        for (int i = 0; i < m && eligible; i++) {
            if (A[(size_t)i*n+j] == 0.0) continue;
            if (row >= 0) eligible = 0;
            row = i;
        }
        K->map[j] = eligible ? -1 : K->kept++;
        K->row[j] = row;
        K->weight[j] = eligible ? 1.0 / (D[j] + delta) : 0.0;
    }
    if (n - K->kept < n / 4 || K->kept == 0) return 0;
    int r = K->kept + m;
    double *matrix = calloc((size_t)r*r, sizeof(double));
    if (!matrix) return 0;
    for (int i = 0; i < m; i++) matrix[(size_t)(K->kept+i)*r+K->kept+i] = -delta;
    for (int j = 0; j < n; j++) {
        int col = K->map[j];
        if (col < 0) {
            int i = K->row[j];
            if (i >= 0) {
                int k = K->kept + i;
                double a = A[(size_t)i*n+j];
                matrix[(size_t)k*r+k] += a * K->weight[j] * a;
            }
            continue;
        }
        for (int k = 0; k < n; k++)
            if (K->map[k] >= 0) matrix[(size_t)col*r+K->map[k]] = -Q[(size_t)j*n+k];
        matrix[(size_t)col*r+col] -= D[j] + delta;
        for (int i = 0; i < m; i++) {
            int k = K->kept + i;
            matrix[(size_t)col*r+k] = matrix[(size_t)k*r+col] = A[(size_t)i*n+j];
        }
    }
    K->reduced = dmat_lu_factor(matrix, r);
    free(matrix);
    return K->reduced != NULL;
}

static int qp_kkt_residual(const double *A, const double *Q, const double *D,
                           int m, int n, double delta, const double *rhs,
                           const double *x) {
    for (int j = 0; j < n; j++) {
        double value = -(D[j] + delta) * x[j];
        double scale = 1.0 + fabs(rhs[j]) + fabs(value);
        for (int k = 0; k < n; k++) {
            double term = Q[(size_t)j*n+k] * x[k];
            value -= term; scale += fabs(term);
        }
        for (int i = 0; i < m; i++) {
            double term = A[(size_t)i*n+j] * x[n+i];
            value += term; scale += fabs(term);
        }
        if (!isfinite(scale) || !(fabs(value-rhs[j]) <= 1e-9*scale)) return 0;
    }
    for (int i = 0; i < m; i++) {
        double value = -delta * x[n+i];
        double scale = 1.0 + fabs(rhs[n+i]) + fabs(value);
        for (int j = 0; j < n; j++) {
            double term = A[(size_t)i*n+j] * x[j];
            value += term; scale += fabs(term);
        }
        if (!isfinite(scale) || !(fabs(value-rhs[n+i]) <= 1e-9*scale)) return 0;
    }
    return 1;
}

static int qp_kkt_solve(QpKkt *K, const double *A, const double *Q,
                        const double *D, int m, int n, double delta, double *rhs) {
    if (K->full) return dmat_lu_solve(K->full, rhs);
    int r = K->kept + m;
    double *work = malloc(((size_t)r+n+m) * sizeof(double));
    if (work) {
        double *full = work + r;
        memcpy(work + K->kept, rhs + n, (size_t)m * sizeof(double));
        for (int j = 0; j < n; j++) {
            if (K->map[j] >= 0) work[K->map[j]] = rhs[j];
            else if (K->row[j] >= 0) {
                int i = K->row[j];
                work[K->kept+i] += A[(size_t)i*n+j] * K->weight[j] * rhs[j];
            }
        }
        if (dmat_lu_solve(K->reduced, work) == 0) {
            memcpy(full + n, work + K->kept, (size_t)m * sizeof(double));
            for (int j = 0; j < n; j++) {
                int i = K->row[j];
                full[j] = K->map[j] >= 0 ? work[K->map[j]] :
                    ((i >= 0 ? A[(size_t)i*n+j] * full[n+i] : 0.0) - rhs[j]) * K->weight[j];
            }
            if (qp_kkt_residual(A, Q, D, m, n, delta, rhs, full)) {
                memcpy(rhs, full, (size_t)(n+m) * sizeof(double));
                free(work);
                return 0;
            }
        }
        free(work);
    }
    K->full = build_factor(A, Q, D, m, n, delta);
    return K->full ? dmat_lu_solve(K->full, rhs) : -1;
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

    /* warm start: starts from the given point, projected into the interior;
     * z derived from stationarity c + Qx - A'y */
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
        QpKkt factor = {0};
        double delta = 1e-8;
        for (int try = 0; try < 5; try++) {
            if (qp_kkt_reduce(&factor, A, Q, D, m, n, delta)) break;
            qp_kkt_free(&factor);
            factor.full = build_factor(A, Q, D, m, n, delta);
            if (factor.full) break;
            delta *= 100.0;
        }
        if (!factor.full && !factor.reduced) { free(D); status = IPM_SINGULAR; break; }

        /* ---- affine predictor (sigma = 0) ---- */
        for (int j = 0; j < n; j++) rc[j] = x[j] * z[j];
        for (int j = 0; j < n; j++) {
            double xj = x[j] > 1e-300 ? x[j] : 1e-300;
            rhs[j] = rd[j] + rc[j] / xj;
        }
        for (int i = 0; i < m; i++) rhs[n + i] = -rp[i];
        if (qp_kkt_solve(&factor, A, Q, D, m, n, delta, rhs)) {
            qp_kkt_free(&factor); free(D); status = IPM_SINGULAR; break;
        }
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

        /* ---- correctors (Mehrotra; g_intpnt_max_cor >= 1) ----
         * The first is the Mehrotra corrector (affine second-order term
         * dxa*dza); the later ones use the previous corrector's term
         * (higher-order corrector, MSK_IPAR_INTPNT_MAX_NUM_COR). */
        int solve_failed = 0;
        for (int cor = 0; cor < g_intpnt_max_cor; cor++) {
            for (int j = 0; j < n; j++)
                rc[j] = x[j] * z[j] - sigma * mu +
                        (cor == 0 ? dxa[j] * dza[j] : dx[j] * dz[j]);
            for (int j = 0; j < n; j++) {
                double xj = x[j] > 1e-300 ? x[j] : 1e-300;
                rhs[j] = rd[j] + rc[j] / xj;
            }
            for (int i = 0; i < m; i++) rhs[n + i] = -rp[i];
            if (qp_kkt_solve(&factor, A, Q, D, m, n, delta, rhs)) {
                solve_failed = 1;
                break;
            }
            for (int j = 0; j < n; j++) dx[j] = rhs[j];
            for (int i = 0; i < m; i++) dy[i] = rhs[n + i];
            for (int j = 0; j < n; j++) {
                double xj = x[j] > 1e-300 ? x[j] : 1e-300;
                dz[j] = -(rc[j] + z[j] * dx[j]) / xj;
            }
        }
        qp_kkt_free(&factor);
        free(D);
        if (solve_failed) { status = IPM_SINGULAR; break; }

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
                        double *Kd, int *Kp, int *kcap, int **Ki, double **Kx,
                        int **Kperm, int **Kpat, int *Kpatn) {
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
    /* Reuse the fill-reducing ordering while K's pattern is unchanged.  K =
     * A Theta A' + delta I has a fixed structure (Theta and delta touch values
     * only), so AMD is paid once per solve instead of once per factor_K call --
     * measured: sym_amd dominates this IPM (lp_400x200 profile). */
    {
        int nnz = Kp[m];
        int same = (*Kperm) && (*Kpat) && (*Kpatn) == nnz &&
                   memcmp(*Kpat, *Ki, (size_t)nnz * sizeof(int)) == 0;
        if (!same) {
            free(*Kperm); *Kperm = NULL; free(*Kpat); *Kpat = NULL;
            *Kperm = spchol_order(m, Kp, *Ki);
            if (*Kperm) {
                *Kpat = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
                if (*Kpat) { memcpy(*Kpat, *Ki, (size_t)nnz * sizeof(int)); *Kpatn = nnz; }
                else { free(*Kperm); *Kperm = NULL; }
            }
        }
        if (*Kperm) return spchol_factor_perm(m, Kp, *Ki, *Kx, *Kperm);
    }
    return spchol_factor_ord(m, Kp, *Ki, *Kx);
}

/* A bound row contains one shared column and one private slack. The
 * eliminated block is diagonal. Form its Schur complement by combining
 * the two positive weights, without subtracting nearly equal products. */
typedef struct {
    SpChol *factor, *fallback;
    int m, n, reduced, eliminated;
    int *map, *rows;
    double *diagonal, *coupling;
    const int *Ap, *Ai;
    const double *Av, *theta;
    double delta;
} LpNormal;

static void lp_normal_free(LpNormal *F) {
    if (!F) return;
    spchol_free(F->factor); spchol_free(F->fallback);
    free(F->map); free(F->rows);
    free(F->diagonal); free(F->coupling);
    free(F);
}

static LpNormal *lp_normal_reduce(const int *Ap, const int *Ai, const double *Av,
                                 const double *theta, int m, int n, double delta,
                                 const int *rp, const int *ri, const double *rv) {
    LpNormal *F = calloc(1, sizeof(*F));
    int *owner = malloc((size_t)n * sizeof(int));
    int *private_col = malloc((size_t)m * sizeof(int));
    double *a = malloc((size_t)m * sizeof(double));
    double *slack = malloc((size_t)m * sizeof(double));
    double *dense = NULL;
    if (!F || !owner || !private_col || !a || !slack) goto fail;
    F->m = m; F->n = n; F->Ap = Ap; F->Ai = Ai;
    F->Av = Av; F->theta = theta; F->delta = delta;
    F->map = calloc((size_t)m, sizeof(int));
    F->rows = malloc((size_t)m * sizeof(int));
    if (!F->map || !F->rows) goto fail;
    for (int j = 0; j < n; j++) owner[j] = -1;
    for (int i = m - 1; i >= 0; i--) {
        if (rp[i+1] - rp[i] != 2) continue;
        int p = rp[i], q = p + 1;
        if (Ap[ri[q]+1] - Ap[ri[q]] != 1) {
            if (Ap[ri[p]+1] - Ap[ri[p]] != 1) continue;
            int tmp = p; p = q; q = tmp;
        }
        int col = ri[p], priv = ri[q], e = F->eliminated;
        if (owner[col] >= 0 || rv[p] == 0.0 || rv[q] == 0.0) continue;
        owner[col] = e; private_col[e] = priv;
        a[e] = rv[p]; slack[e] = rv[q]; F->rows[e] = i;
        F->map[i] = -1; F->eliminated++;
    }
    if (F->eliminated == 0 || F->eliminated < m / 3 || F->eliminated == m) goto fail;
    for (int i = 0; i < m; i++)
        if (F->map[i] != -1) F->map[i] = F->reduced++;
    int r = F->reduced, count = F->eliminated;
    /* The reduced factor is dense. Leave larger systems on sparse Cholesky. */
    if (r > 256) goto fail;
    dense = calloc((size_t)r * r, sizeof(double));
    F->diagonal = malloc((size_t)count * sizeof(double));
    F->coupling = calloc((size_t)count * r, sizeof(double));
    if (!dense || !F->diagonal || !F->coupling) goto fail;
    for (int j = 0; j < n; j++) {
        double weight = theta[j];
        int e = owner[j];
        if (e >= 0) {
            double rest = theta[private_col[e]] * slack[e] * slack[e] + delta;
            double diagonal = theta[j] * a[e] * a[e] + rest;
            if (!(diagonal > 0.0) || !isfinite(diagonal)) goto fail;
            F->diagonal[e] = diagonal;
            double ratio = theta[j] * a[e] / diagonal;
            weight = theta[j] * (rest / diagonal);
            for (int p = Ap[j]; p < Ap[j+1]; p++) {
                int i = F->map[Ai[p]];
                if (i >= 0) F->coupling[(size_t)e*r+i] += ratio * Av[p];
            }
        }
        if (!(weight > 0.0) || !isfinite(weight)) goto fail;
        for (int p = Ap[j]; p < Ap[j+1]; p++) {
            int i = F->map[Ai[p]];
            if (i < 0) continue;
            double value = Av[p] * weight;
            for (int q = p; q < Ap[j+1]; q++) {
                int k = F->map[Ai[q]];
                if (k < 0) continue;
                if (i >= k) dense[(size_t)i*r+k] += value * Av[q];
                else dense[(size_t)k*r+i] += value * Av[q];
            }
        }
    }
    for (int i = 0; i < r; i++) dense[(size_t)i*r+i] += delta;
    F->factor = spchol_factor_dense(r, dense);
    if (!F->factor) goto fail;
    free(owner); free(private_col); free(a); free(slack); free(dense);
    return F;
fail:
    free(owner); free(private_col); free(a); free(slack); free(dense);
    lp_normal_free(F);
    return NULL;
}

static LpNormal *lp_normal_factor(const int *Ap, const int *Ai, const double *Av,
                                  const double *theta, int m, int n, double delta,
                                  double *Kd, int *Kp, int *cap, int **Ki, double **Kx,
                                  int **perm, int **pattern, int *pattern_n,
                                  const int *rp, const int *ri, const double *rv) {
    LpNormal *F = lp_normal_reduce(Ap, Ai, Av, theta, m, n, delta, rp, ri, rv);
    if (F) return F;
    F = calloc(1, sizeof(*F));
    if (!F) return NULL;
    F->factor = factor_K(Ap, Ai, Av, theta, m, n, delta, Kd, Kp, cap, Ki, Kx,
                         perm, pattern, pattern_n);
    if (!F->factor) { free(F); return NULL; }
    F->m = m;
    return F;
}

/* Keep the original right-hand side until the reconstructed solve passes.
 * A failed residual check retries through the full normal equations. */
static int lp_normal_full_solve(LpNormal *F, double *rhs) {
    if (!F->fallback) {
        double *dense = calloc((size_t)F->m * F->m, sizeof(double));
        int *p = malloc((size_t)(F->m + 1) * sizeof(int));
        int *i = NULL, *perm = NULL, *pattern = NULL, cap = 0, pattern_n = -1;
        double *values = NULL;
        if (dense && p)
            F->fallback = factor_K(F->Ap, F->Ai, F->Av, F->theta, F->m, F->n,
                                    F->delta, dense, p, &cap, &i, &values,
                                    &perm, &pattern, &pattern_n);
        free(dense); free(p); free(i); free(values); free(perm); free(pattern);
    }
    return F->fallback ? spchol_solve_ord(F->fallback, rhs) : -1;
}

static int lp_normal_solve(LpNormal *F, double *rhs) {
    if (!F->eliminated) return spchol_solve_ord(F->factor, rhs);
    if (F->fallback) return spchol_solve_ord(F->fallback, rhs);
    int m = F->m, n = F->n, r = F->reduced;
    double *work = calloc((size_t)n + 3*(size_t)m + r, sizeof(double));
    if (!work) return lp_normal_full_solve(F, rhs);
    double *solution = work, *reduced = solution + m;
    double *aty = reduced + r, *residual = aty + n, *scale = residual + m;
    for (int i = 0; i < m; i++)
        if (F->map[i] >= 0) reduced[F->map[i]] = rhs[i];
    for (int e = 0; e < F->eliminated; e++) {
        double value = rhs[F->rows[e]];
        for (int i = 0; i < r; i++) reduced[i] -= F->coupling[(size_t)e*r+i] * value;
    }
    int rc = spchol_solve_ord(F->factor, reduced);
    if (rc == 0) {
        for (int i = 0; i < m; i++)
            if (F->map[i] >= 0) solution[i] = reduced[F->map[i]];
        for (int e = 0; e < F->eliminated; e++) {
            double value = rhs[F->rows[e]] / F->diagonal[e];
            for (int i = 0; i < r; i++) value -= F->coupling[(size_t)e*r+i] * reduced[i];
            solution[F->rows[e]] = value;
        }
        for (int i = 0; i < m; i++) {
            residual[i] = F->delta * solution[i] - rhs[i];
            scale[i] = 1.0 + fabs(rhs[i]) + fabs(F->delta * solution[i]);
        }
        for (int j = 0; j < n; j++) {
            for (int p = F->Ap[j]; p < F->Ap[j+1]; p++)
                aty[j] += F->Av[p] * solution[F->Ai[p]];
            for (int p = F->Ap[j]; p < F->Ap[j+1]; p++) {
                int i = F->Ai[p];
                double value = F->Av[p] * F->theta[j] * aty[j];
                residual[i] += value; scale[i] += fabs(value);
            }
        }
        for (int i = 0; i < m; i++)
            if (!isfinite(solution[i]) || !isfinite(residual[i]) || !isfinite(scale[i]) ||
                fabs(residual[i]) > 1e-9 * scale[i]) rc = -1;
    }
    if (rc == 0) memcpy(rhs, solution, (size_t)m * sizeof(double));
    free(work);
    return rc == 0 ? 0 : lp_normal_full_solve(F, rhs);
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
    int *Kperm = NULL, *Kpat = NULL;   /* cached AMD ordering of K's pattern */
    int Kpatn = -1;
    int memfail = 0;
    if (!rp || !rd || !tmpn || !tmpm || !dyA || !dxA || !dzA || !dxC || !dzC ||
        !theta || !g || !Ag || !Kd || !rptr || !ridx || !rval || !Kp) memfail = 1;

#define IPMSP_FREE() do { \
    free(rp); free(rd); free(tmpn); free(tmpm); free(dyA); \
    free(dxA); free(dzA); free(dxC); free(dzC); free(theta); free(g); \
    free(Ag); free(Kd); free(rptr); free(ridx); free(rval); free(Kp); \
    free(Ki); free(Kx); free(Kperm); free(Kpat); } while (0)

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
        LpNormal *L0 = NULL;
        double delta0 = 1e-9;
        for (int tries = 0; tries < 6 && !L0; tries++) {
            L0 = lp_normal_factor(Aptr, Arow, Aval, theta, m, n, delta0, Kd, Kp, &kcap, &Ki, &Kx,
                          &Kperm, &Kpat, &Kpatn, rptr, ridx, rval);
            if (!L0) delta0 *= 100.0;
        }
        if (!L0) { IPMSP_FREE(); return IPM_SINGULAR; }
        if (L0->eliminated && getenv("GMB_DBG"))
            fprintf(stderr, "spipm normal rows=%d reduced=%d\n", m, L0->reduced);
        /* x_hat = 1 + A' u,  (AA'+dI) u = b - A*1 */
        for (int i = 0; i < m; i++) {
            double s = b[i];
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s -= rval[p];
            tmpm[i] = s;
        }
        if (lp_normal_solve(L0, tmpm) != 0) { lp_normal_free(L0); IPMSP_FREE(); return IPM_SINGULAR; }
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
        if (lp_normal_solve(L0, tmpm) != 0) { lp_normal_free(L0); IPMSP_FREE(); return IPM_SINGULAR; }
        for (int i = 0; i < m; i++) y[i] = tmpm[i];
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * y[Arow[p]];
        for (int j = 0; j < n; j++) z[j] = c[j] - tmpn[j];
        lp_normal_free(L0);
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
        LpNormal *L = NULL;
        for (int tries = 0; tries < 6 && !L; tries++) {
            L = lp_normal_factor(Aptr, Arow, Aval, theta, m, n, delta, Kd, Kp, &kcap, &Ki, &Kx,
                         &Kperm, &Kpat, &Kpatn, rptr, ridx, rval);
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
        if (lp_normal_solve(L, tmpm) != 0) { lp_normal_free(L); status = IPM_SINGULAR; break; }
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
        if (lp_normal_solve(L, tmpm) != 0) { lp_normal_free(L); status = IPM_SINGULAR; break; }
        for (int j = 0; j < n; j++) tmpn[j] = 0.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++)
                tmpn[j] += Aval[p] * tmpm[Arow[p]];
        for (int j = 0; j < n; j++) {
            dxC[j] = g[j] + theta[j] * tmpn[j];
            dzC[j] = rd[j] - tmpn[j];
        }
        lp_normal_free(L);

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
 * delta on the diagonal, via the sparse Cholesky. Diagonal is always emitted.
 * The factor may carry a fill-reducing ordering, so it is solved with
 * spchol_solve_ord, which undoes it. */
/* Dense Hessian block followed by diagonal-only slack columns. */
static SpChol *qp_chol_dense_block(int n, int d, const int *p, const int *row,
                                    const double *values) {
    double *dense = calloc((size_t)d*d, sizeof(double));
    if (!dense) return NULL;
    for (int j = 0; j < d; j++)
        for (int k = p[j]; k < p[j+1]; k++)
            if (row[k] < d) dense[(size_t)row[k]*d+j] = values[k];
    SpChol *L = spchol_factor_dense(d, dense);
    free(dense);
    if (!L) return NULL;
    int start = L->Lp[d], total = start + n - d;
    int *lp = realloc(L->Lp, ((size_t)n+1) * sizeof(int));
    if (lp) L->Lp = lp;
    int *li = realloc(L->Li, (size_t)total * sizeof(int));
    if (li) L->Li = li;
    double *lx = realloc(L->Lx, (size_t)total * sizeof(double));
    if (lx) L->Lx = lx;
    if (!lp || !li || !lx) { spchol_free(L); return NULL; }
    for (int j = d; j < n; j++) {
        double diagonal = 0.0;
        for (int k = p[j]; k < p[j+1]; k++)
            if (row[k] == j) diagonal += values[k];
        if (!(diagonal > 1e-300) || !isfinite(diagonal)) { spchol_free(L); return NULL; }
        int k = start + j - d;
        L->Lp[j] = k;
        L->Li[k] = j;
        L->Lx[k] = sqrt(diagonal);
    }
    L->Lp[n] = total;
    L->n = n;
    return L;
}

static SpChol *spchol_from_dense(const double *Kd, int m,
                                 int *Kp, int *kcap, int **Ki, double **Kx,
                                 int **Kperm, int **Kpat, int *Kpatn) {
    /* Count the lower triangle first: a full one needs no fill-reducing order
     * and no sparse machinery, just the dense factor. */
    {
        long long full = (long long)m * (m + 1) / 2, cnt = 0;
        for (int j = 0; j < m; j++) {
            cnt++;
            for (int i = j + 1; i < m; i++) if (Kd[(size_t)i * m + j] != 0.0) cnt++;
        }
        if (cnt == full) return spchol_factor_dense(m, Kd);
    }
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
    /* K = A M^-1 A' + delta I is DENSE whenever M^-1 is (a dense Q makes it so),
     * and AMD on a complete graph reduces no fill while costing a full ordering
     * pass every iteration -- measured at 42% of a dense QP's runtime.  Order
     * only when the pattern actually has gaps to exploit. */
    long long half = (long long)m * (m + 1) / 2;
    if ((long long)Kp[m] * 4 > half)
        return spchol_factor(m, Kp, *Ki, *Kx);
    {
        int nnz = Kp[m];
        int same = (*Kperm) && (*Kpat) && (*Kpatn) == nnz &&
                   memcmp(*Kpat, *Ki, (size_t)nnz * sizeof(int)) == 0;
        if (!same) {
            free(*Kperm); *Kperm = NULL; free(*Kpat); *Kpat = NULL;
            *Kperm = spchol_order(m, Kp, *Ki);
            if (*Kperm) {
                *Kpat = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
                if (*Kpat) { memcpy(*Kpat, *Ki, (size_t)nnz * sizeof(int)); *Kpatn = nnz; }
                else { free(*Kperm); *Kperm = NULL; }
            }
        }
        if (*Kperm) return spchol_factor_perm(m, Kp, *Ki, *Kx, *Kperm);
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
    int *Kperm = NULL, *Kpat = NULL;   /* cached AMD ordering of K's pattern */
    int Kpatn = -1;
    int memfail = 0;
    if (!rp||!rd||!Qx||!tmpn||!tmpm||!dyA||!dxA||!dzA||!dxC||!dzC||!D||!u||!Atdy||
        !Mval||!W||!Kd||!rptr||!ridx||!rval||!diagpos||!Kp) memfail = 1;

#define IPMQP_FREE() do { free(rp);free(rd);free(Qx);free(tmpn);free(tmpm);free(dyA);\
    free(dxA);free(dzA);free(dxC);free(dzC);free(D);free(u);free(Atdy);free(Mval);\
    free(W);free(Kd);free(rptr);free(ridx);free(rval);free(diagpos);free(Kp);free(Ki);free(Kx);free(Kperm);free(Kpat);} while(0)
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
            L0 = factor_K(Aptr, Arow, Aval, D, m, n, d0, Kd, Kp, &kcap, &Ki, &Kx,
                          &Kperm, &Kpat, &Kpatn);
            if (!L0) d0 *= 100.0;
        }
        if (!L0) { IPMQP_FREE(); return IPM_SINGULAR; }
        for (int i = 0; i < m; i++) { double s = b[i];
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s -= rval[p];
            tmpm[i] = s; }
        if (spchol_solve_ord(L0, tmpm)) { spchol_free(L0); IPMQP_FREE(); return IPM_SINGULAR; }
        for (int j = 0; j < n; j++) x[j] = 1.0;
        for (int j = 0; j < n; j++)
            for (int p = Aptr[j]; p < Aptr[j + 1]; p++) x[j] += Aval[p] * tmpm[Arow[p]];
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * c[ridx[p]];
            tmpm[i] = s; }
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

    int dense_q = 0;
    size_t qentries = 0;
    for (int j = 0; j < n; j++) for (int k = Qptr[j]; k < Qptr[j+1]; k++) {
        if (Qval[k] == 0.0) continue;
        if (Qrow[k] >= dense_q) dense_q = Qrow[k] + 1;
        qentries++;
    }
    if (dense_q < 32 || dense_q > 512 ||
        qentries < 0.4 * dense_q * (dense_q + 1)) dense_q = 0;

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
        SpChol *Lm = dense_q ? qp_chol_dense_block(n, dense_q, Qptr, Qrow, Mval) : NULL;
        if (!Lm) Lm = spchol_factor(n, Qptr, Qrow, Mval);
        if (!Lm) { status = IPM_SINGULAR; break; }

        /* W = M^-1 A' (column i = M^-1 (row i of A)): all m right-hand sides
         * in one pass over the factor.  The batched solve keeps the per-RHS
         * operation order, so the result is what m spchol_solve calls gave. */
        for (size_t q = 0; q < (size_t)n * (size_t)m; q++) W[q] = 0.0;
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++) W[(size_t)ridx[p] * m + i] = rval[p];
        if (spchol_solve_all(Lm, W, m)) { spchol_free(Lm); status = IPM_SINGULAR; break; }
        /* Kd = A W + delta I */
        for (int i = 0; i < m; i++)
            for (int k = 0; k < m; k++) Kd[(size_t)i * m + k] = (i == k) ? delta : 0.0;
        for (int i = 0; i < m; i++)
            for (int p = rptr[i]; p < rptr[i + 1]; p++) {
                int j = ridx[p]; double a = rval[p]; const double *Wj = &W[(size_t)j * m];
                for (int k = 0; k < m; k++) Kd[(size_t)i * m + k] += a * Wj[k];
            }
        SpChol *Lk = spchol_from_dense(Kd, m, Kp, &kcap, &Ki, &Kx, &Kperm, &Kpat, &Kpatn);
        if (!Lk) { spchol_free(Lm); status = IPM_SINGULAR; break; }

        /* ---- affine predictor (target = -xz) ---- */
        for (int j = 0; j < n; j++) u[j] = -x[j] * z[j] / (x[j] + dp) - rd[j];
        if (spchol_solve(Lm, u)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * u[ridx[p]];
            tmpm[i] = -rp[i] - s; }
        if (spchol_solve_ord(Lk, tmpm)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) dyA[i] = tmpm[i];
        for (int j = 0; j < n; j++) { double s = u[j];
            for (int i = 0; i < m; i++) s += W[(size_t)j * m + i] * dyA[i];
            dxA[j] = s; }
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
        if (!(sigma > 1e-8)) sigma = 1e-8;
        if (sigma > 1.0) sigma = 1.0;

        /* ---- corrector ---- */
        for (int j = 0; j < n; j++) u[j] = (sigma*mu - x[j]*z[j] - dxA[j]*dzA[j]) / (x[j] + dp) - rd[j];
        if (spchol_solve(Lm, u)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int i = 0; i < m; i++) { double s = 0.0;
            for (int p = rptr[i]; p < rptr[i + 1]; p++) s += rval[p] * u[ridx[p]];
            tmpm[i] = -rp[i] - s; }
        if (spchol_solve_ord(Lk, tmpm)) { spchol_free(Lm); spchol_free(Lk); status = IPM_SINGULAR; break; }
        for (int j = 0; j < n; j++) { double s = u[j];
            for (int i = 0; i < m; i++) s += W[(size_t)j * m + i] * tmpm[i];
            dxC[j] = s; }
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
