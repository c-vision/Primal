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

/* simplex.c - dense two-phase primal simplex
 *
 * Tableau layout: (m+1) x (n+m+1): rows 0..m-1 constraints, row m = reduced
 * cost row; columns 0..n-1 structural (incl. slack columns of the std form),
 * n..n+m-1 artificials, column n+m = rhs. Cost row holds (red | -z).
 *
 * Phase 1: artificials form the initial basis, cost 1 each; entering columns
 * restricted to structural ones. Termination with z1 > 0 proves infeasibility
 * (Farkas certificate: y = c_B B^-1, y'A <= 0, y'b = z1 > 0).
 * Artificials leaving the basis are driven out (pivot on any nonzero entry);
 * rows where this is impossible are redundant (artificial stays basic at 0).
 *
 * Phase 2: cost row rebuilt exactly from c for the current basis. Anti
 * cycling: Dantzig rule with fallback to Bland's rule after stalling.
 * Duals: y_std[r] = -costrow[n+r] (phase-2 artificial cost is 0).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "simplex.h"
#include "linalg.h"

/* per-iteration callback hook (implemented in primal.c) */
extern int primal_cb_iter_on;
void primal_cb_iter(int code);


#define PIV_EPS     1e-9
#define RED_EPS     1e-9
#define STALL_LIMIT 40

enum { SIMP_OPTIMAL = 0, SIMP_INFEASIBLE, SIMP_UNBOUNDED, SIMP_MAXITER, SIMP_MEMORY };

typedef struct {
    int m, n, ntot, stride;
    double *T;
    int *basis;
} Tab;

/**
 * Pivots on element (lev, ent) in the tableau.
 *
 * @param t   [in/out] Tableau structure.
 * @param lev [in]    Leaving row index.
 * @param ent [in]    Entering column index.
 *
 * @note Updates the basis and normalizes the pivot row. Performs row operations
 *       to zero out the entering column in all other rows.
 */
static void tab_pivot(Tab *t, int lev, int ent) {
    int stride = t->stride, ntot = t->ntot;
    double *T = t->T;
    double piv = T[lev * stride + ent];
    double *rl = T + lev * stride;
    for (int j = 0; j <= ntot; j++) rl[j] /= piv;
    rl[ent] = 1.0;
    for (int i = 0; i <= t->m; i++) {
        if (i == lev) continue;
        double *ri = T + i * stride;
        double f = ri[ent];
        if (f == 0.0) continue;
        for (int j = 0; j <= ntot; j++) ri[j] -= f * rl[j];
        ri[ent] = 0.0;
    }
    t->basis[lev] = ent;
}

/**
 * Chooses the entering column using Dantzig or Bland rule.
 *
 * @param t     [in] Tableau structure.
 * @param bland [in] If non-zero, use Bland's rule (smallest index with negative
 *                    reduced cost); otherwise use Dantzig (most negative).
 *
 * @return Column index of entering variable, or -1 if optimal (no negative
 *         reduced cost).
 *
 * @note If GMB_SIMPLEX_STEEPEST env var is set, uses steepest-edge rule
 *       (maximizes red^2 / (1 + ||col||^2)) instead of Dantzig.
 */
static int enter_col(const Tab *t, int bland) {
    const double *red = t->T + t->m * t->stride;
    int n = t->n;
    if (bland) {
        for (int j = 0; j < n; j++)
            if (red[j] < -RED_EPS) return j;
        return -1;
    }
    /* Steepest-edge (gated GMB_SIMPLEX_STEEPEST): massimizza red^2 / gamma con
     * gamma = 1 + ||colonna_j||^2 del tableau. Pivot piu' lunghi ma meno
     * iterazioni; il default resta Dantzig (costo O(n) per iterazione). */
    if (getenv("GMB_SIMPLEX_STEEPEST")) {
        int best = -1; double bestscore = 0.0;
        for (int j = 0; j < n; j++) {
            if (red[j] >= -RED_EPS) continue;
            double nrm = 1.0;
            for (int i = 0; i < t->m; i++) {
                double a = t->T[i * t->stride + j];
                nrm += a * a;
            }
            double sc = red[j] * red[j] / nrm;
            if (best < 0 || sc > bestscore) { best = j; bestscore = sc; }
        }
        return best;
    }
    int best = -1;
    for (int j = 0; j < n; j++)
        if (red[j] < -RED_EPS && (best < 0 || red[j] < red[best])) best = j;
    return best;
}

/**
 * Chooses the leaving row using the minimum ratio test.
 *
 * @param t        [in]  Tableau structure.
 * @param ent      [in]  Entering column index.
 * @param bland    [in]  If non-zero, use Bland's tie-breaking rule.
 * @param pure     [in]  If non-zero, force pure Bland (smallest basis index)
 *                       regardless of pivot magnitude.
 * @param theta_out [out] Optional pointer to receive the minimum ratio (step length).
 *
 * @return Row index of leaving variable, or -1 if unbounded (no positive
 *         entry in entering column).
 *
 * @note Uses stability-first tie-breaking: among rows with tied minimum ratio,
 *       prefers the row with the LARGER pivot element (1.5x threshold) to
 *       avoid numerically destructive tiny pivots. Pure Bland is used after
 *       3*STALL_LIMIT iterations.
 */
static int leave_row(const Tab *t, int ent, int bland, int pure, double *theta_out) {
    const double *T = t->T;
    int best = -1;
    double best_ratio = 0.0, best_piv = 0.0;
    for (int i = 0; i < t->m; i++) {
        double a = T[i * t->stride + ent];
        if (a <= PIV_EPS) continue;
        double v = T[i * t->stride + t->ntot];
        if (v < 0.0) v = 0.0;
        double r = v / a;
        if (best < 0) { best = i; best_ratio = r; best_piv = a; continue; }
        if (r < best_ratio - 1e-12) {
            best = i; best_ratio = r; best_piv = a;
        } else if (r <= best_ratio + 1e-12) {
            /* Stability FIRST: at a tied ratio the LARGER pivot wins outright.
             * A 1e-9 pivot where a 5.9e4 one exists multiplies the tableau by
             * ~1e9 and destroys the phase-1 cost (netlib blend: 6.2e4 -> 5.8e13,
             * negative RHS, z1 != 0 with no artificial basic). Bland's
             * smallest-index rule stays the tie-break only among pivots of
             * comparable magnitude, so anti-cycling is preserved. */
            int take;
            if (pure) {
                /* Pure Bland: smallest basis index wins outright.  The ratio
                 * test is then lexicographic and cycling is impossible.  Only
                 * used after a very long stall, because a tiny pivot is
                 * numerically destructive (netlib blend). */
                take = (t->basis[i] < t->basis[best]);
            } else if (a > best_piv * 1.5) {
                take = 1;
            } else if (bland) {
                take = (a > best_piv / 1.5) && (t->basis[i] < t->basis[best]);
            } else {
                take = (a > best_piv * (1.0 + 1e-9));
            }
            if (take) { best = i; best_ratio = r; best_piv = a; }
        }
    }
    if (theta_out) *theta_out = best_ratio;
    return best;
}

/**
 * Runs one phase of the simplex algorithm (Phase 1 or Phase 2).
 *
 * @param t        [in/out] Tableau with cost row already built.
 * @param max_iter [in]    Maximum iterations for this phase.
 * @param ent_out  [out]   Optional pointer to receive entering column if
 *                         unbounded.
 *
 * @return SIMP_OPTIMAL (0), SIMP_UNBOUNDED (2), or SIMP_MAXITER (3).
 *
 * @note Uses Dantzig rule with fallback to Bland's rule after STALL_LIMIT
 *       iterations without progress. Forces pure Bland after 3*STALL_LIMIT
 *       stalls.
 */
static int run_phase(Tab *t, int max_iter, int *ent_out) {
    int bland = 0, stall = 0, pure = 0;
    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(93);
        double *red = t->T + t->m * t->stride;
        int ent = enter_col(t, bland);
        if (ent < 0) return SIMP_OPTIMAL;
        int lev = leave_row(t, ent, bland, pure, NULL);
        if (lev < 0) { if (ent_out) *ent_out = ent; return SIMP_UNBOUNDED; }
        double zprev = red[t->ntot];
        tab_pivot(t, lev, ent);
        double znow = red[t->ntot];
        if (fabs(znow - zprev) > 1e-11 * (1.0 + fabs(znow))) { stall = 0; bland = 0; }
        else if (++stall > STALL_LIMIT) bland = 1;
        if (stall > 3 * STALL_LIMIT) pure = 1;   /* long stall: force Bland */
    }
    return SIMP_MAXITER;
}

int simplex_solve_std_tab(const double *A, int m, int n,
                          const double *b, const double *c,
                          int max_iter, double *x, double *y,
                          double *dray, double *pray,
                          int *basis_out, double *tab_out)
{
    Tab t;
    t.m = m; t.n = n;
    t.ntot = n + m;
    t.stride = t.ntot + 1;
    t.T = (double *)calloc((size_t)(m + 1) * (size_t)t.stride, sizeof(double));
    t.basis = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    if (!t.T || !t.basis) { free(t.T); free(t.basis); return SIMP_MEMORY; }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) t.T[i * t.stride + j] = A[i * n + j];
        t.T[i * t.stride + n + i] = 1.0;
        double bv = b[i];
        if (bv < 0.0 && bv > -1e-9) bv = 0.0;
        t.T[i * t.stride + t.ntot] = bv;
        t.basis[i] = n + i;
    }
    double *red = t.T + m * t.stride;

    /* ---------- phase 1 ---------- */
    for (int j = 0; j < t.ntot; j++) {
        double s = (j >= n) ? 1.0 : 0.0;
        for (int i = 0; i < m; i++) s -= t.T[i * t.stride + j]; /* all basic costs are 1 */
        red[j] = s;
    }
    {
        double s = 0.0;
        for (int i = 0; i < m; i++) s -= t.T[i * t.stride + t.ntot];
        red[t.ntot] = s;
    }
    int status = run_phase(&t, max_iter, NULL);
    if (status == SIMP_MAXITER) goto done;
    if (status == SIMP_UNBOUNDED) { status = SIMP_MAXITER; goto done; } /* cannot happen */

    {
        double z1 = -red[t.ntot];
        double bscale = 1.0;
        for (int i = 0; i < m; i++) { double a = fabs(b[i]); if (a > bscale) bscale = a; }
        if (z1 > 1e-7 * bscale) {
            /* the phase-1 cost row is still in place: red[n+r] = 1 - y_r */
            if (dray) for (int r = 0; r < m; r++) dray[r] = 1.0 - red[n + r];
            status = SIMP_INFEASIBLE; goto done;
        }
    }

    /* drive artificials out of the basis */
    for (int i = 0; i < m; i++) {
        if (t.basis[i] < n) continue;
        int bj = -1;
        double bmax = 1e-7;
        for (int j = 0; j < n; j++) {
            double a = fabs(t.T[i * t.stride + j]);
            if (a > bmax) { bmax = a; bj = j; }
        }
        if (bj >= 0) tab_pivot(&t, i, bj);
        /* else: redundant row, artificial stays basic at ~0 */
    }

    /* ---------- phase 2: rebuild cost row for current basis ---------- */
    for (int j = 0; j < t.ntot; j++) {
        double s = (j < n) ? c[j] : 0.0;
        for (int i = 0; i < m; i++) {
            int bi = t.basis[i];
            double cb = (bi < n) ? c[bi] : 0.0;
            if (cb != 0.0) s -= cb * t.T[i * t.stride + j];
        }
        red[j] = s;
    }
    {
        double s = 0.0;
        for (int i = 0; i < m; i++) {
            int bi = t.basis[i];
            double cb = (bi < n) ? c[bi] : 0.0;
            if (cb != 0.0) s -= cb * t.T[i * t.stride + t.ntot];
        }
        red[t.ntot] = s;
    }
    int uent = -1;
    status = run_phase(&t, max_iter, &uent);

    /* ---------- extract ---------- */
    if (status == SIMP_UNBOUNDED && pray && uent >= 0) {
        /* x = x* + theta*d with d[ent] = 1 and d[basis[i]] = -(B^-1 A_ent)_i.
         * No leaving row exists, so every tableau entry of that column is
         * <= 0: d >= 0, A d = 0 and c'd = red[ent] < 0. */
        for (int j = 0; j < n; j++) pray[j] = 0.0;
        pray[uent] = 1.0;
        for (int i = 0; i < m; i++) {
            int bi = t.basis[i];
            if (bi >= 0 && bi < n) pray[bi] = -t.T[i * t.stride + uent];
        }
    }
    if (status == SIMP_OPTIMAL || status == SIMP_UNBOUNDED) {
        for (int j = 0; j < n; j++) x[j] = 0.0;
        for (int i = 0; i < m; i++) {
            int bi = t.basis[i];
            double v = t.T[i * t.stride + t.ntot];
            if (v < 0.0 && v > -1e-7) v = 0.0;
            if (bi < n) x[bi] = v;
        }
    }
    for (int r = 0; r < m; r++) y[r] = -red[n + r];

done:
    if (basis_out) for (int i = 0; i < m; i++) basis_out[i] = t.basis[i];
    if (tab_out) memcpy(tab_out, t.T, (size_t)(m + 1) * (size_t)t.stride * sizeof(double));
    free(t.T);
    free(t.basis);
    return status;
}

/**
 * Simplified wrapper for simplex_solve_std_tab without basis/tab output.
 *
 * @param A       [in]  Constraint matrix in row-major (m x n).
 * @param m       [in]  Number of constraints.
 * @param n       [in]  Number of variables.
 * @param b       [in]  RHS vector (size m).
 * @param c       [in]  Objective coefficients (size n).
 * @param max_iter [in] Maximum simplex iterations.
 * @param x       [out] Primal solution (size n). May be NULL.
 * @param y       [out] Dual solution (size m). May be NULL.
 * @param dray    [out] Optional dual Farkas ray (size m). May be NULL.
 * @param pray    [out] Optional primal Farkas ray (size n). May be NULL.
 *
 * @return Same as simplex_solve_std_tab.
 *
 * @note Convenience wrapper that omits basis_out and tab_out.
 */
int simplex_solve_std(const double *A, int m, int n,
                      const double *b, const double *c,
                      int max_iter, double *x, double *y,
                      double *dray, double *pray) {
    return simplex_solve_std_tab(A, m, n, b, c, max_iter, x, y, dray, pray, NULL, NULL);
}

/**
 * Dual simplex method with explicit basis inverse.
 *
 * @param A        [in]  Constraint matrix in row-major (m x n).
 * @param m        [in]  Number of constraints.
 * @param n        [in]  Number of variables.
 * @param b        [in]  RHS vector (size m).
 * @param c        [in]  Objective coefficients (size n).
 * @param basis    [in]  Initial basis (size m) -- column indices of basic variables.
 * @param max_iter [in]  Maximum iterations.
 * @param x        [out] Primal solution (size n). Must not be NULL.
 * @param basis_out [out] Optional final basis (size m). May be NULL.
 * @param yout     [out] Optional dual solution (size m). May be NULL.
 *
 * @return 0 = optimal, 1 = primal infeasible (basis), 2 = unbounded,
 *         3 = max iterations, 4 = error.
 *
 * @note Starts from a dual-feasible basis and maintains dual feasibility
 *       while driving primal feasibility. Computes B^-1 via Gauss-Jordan
 *       on [B | I] at each iteration (not updated incrementally).
 *
 *       Dual variables y = cB^T B^-1 are computed at the end via LU on B^T.
 *
 * @note Ported from gmbortools `gor_lp_dual_simplex`. Fixed heap-buffer-overflow
 *       in M allocation (was m*m, now m*2m).
 */
int simplex_dual_solve_std(const double *A, int m, int n,
                           const double *b, const double *c,
                           const int *basis, int max_iter, double *x,
                           int *basis_out, double *yout) {
    int ncols = n + 1, i, j, k, it, rc = 0;
    double *T, *red, *cB, *M;
    int *bas;
    if (m < 1 || n < 1 || !A || !b || !c || !basis || !x) return 4;
    T = (double *)malloc((size_t)m * ncols * sizeof(double));
    red = (double *)malloc((size_t)n * sizeof(double));
    cB = (double *)malloc((size_t)m * sizeof(double));
    /* NB: M e' la [B | I] di m righe per 2m colonne (il codice di gmbortools la
     * allocava m*m -> heap-buffer-overflow, trovato da ASan nel porting). */
    M = (double *)malloc((size_t)m * 2 * m * sizeof(double));   /* [B | I] -> B^-1 */
    bas = (int *)malloc((size_t)m * sizeof(int));
    if (!T || !red || !cB || !M || !bas) { free(T); free(red); free(cB); free(M); free(bas); return 4; }
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) M[(size_t)i * 2 * m + j] = A[(size_t)i * n + basis[j]];
        for (j = 0; j < m; j++) M[(size_t)i * 2 * m + m + j] = (i == j) ? 1.0 : 0.0;
    }
    {   /* Gauss-Jordan su [B | I] per ottenere B^-1 nella meta' destra */
        double *W = (double *)malloc((size_t)m * 2 * m * sizeof(double));
        if (!W) { free(T); free(red); free(cB); free(M); free(bas); return 4; }
        for (i = 0; i < m; i++) for (j = 0; j < 2 * m; j++) W[(size_t)i * 2 * m + j] = M[(size_t)i * 2 * m + j];
        for (k = 0; k < m; k++) {
            int piv = k; double mx = fabs(W[(size_t)k * 2 * m + k]);
            for (i = k + 1; i < m; i++) { double v = fabs(W[(size_t)i * 2 * m + k]); if (v > mx) { mx = v; piv = i; } }
            if (mx < 1e-12) { free(W); free(T); free(red); free(cB); free(M); free(bas); return 1; }
            if (piv != k) for (j = 0; j < 2 * m; j++) { double t = W[(size_t)k*2*m+j]; W[(size_t)k*2*m+j] = W[(size_t)piv*2*m+j]; W[(size_t)piv*2*m+j] = t; }
            { double p = W[(size_t)k * 2 * m + k]; for (j = 0; j < 2 * m; j++) W[(size_t)k * 2 * m + j] /= p; }
            for (i = 0; i < m; i++) {
                double f;
                if (i == k) continue;
                f = W[(size_t)i * 2 * m + k];
                if (f == 0.0) continue;
                for (j = 0; j < 2 * m; j++) W[(size_t)i * 2 * m + j] -= f * W[(size_t)k * 2 * m + j];
            }
        }
        for (i = 0; i < m; i++) for (j = 0; j < m; j++) M[(size_t)i * 2 * m + j] = W[(size_t)i * 2 * m + m + j];
        free(W);
    }
    for (i = 0; i < m; i++) bas[i] = basis[i];
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double s = 0.0;
            for (k = 0; k < m; k++) s += M[(size_t)i * 2 * m + k] * A[(size_t)k * n + j];
            T[(size_t)i * ncols + j] = s;
        }
        { double s = 0.0; for (k = 0; k < m; k++) s += M[(size_t)i * 2 * m + k] * b[k]; T[(size_t)i * ncols + n] = s; }
    }
    for (it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(36);
        int r = -1, ent = -1;
        double worst = -1e-9, bestratio = 1e300;
        for (i = 0; i < m; i++) cB[i] = c[bas[i]];
        for (j = 0; j < n; j++) {
            double s = c[j];
            for (i = 0; i < m; i++) s -= cB[i] * T[(size_t)i * ncols + j];
            red[j] = s;
        }
        for (i = 0; i < m; i++) if (T[(size_t)i * ncols + n] < worst) { worst = T[(size_t)i * ncols + n]; r = i; }
        if (r < 0) { rc = 0; break; }
        for (j = 0; j < n; j++) {
            int isbasic = 0;
            for (i = 0; i < m; i++) if (bas[i] == j) isbasic = 1;
            if (isbasic) continue;
            if (T[(size_t)r * ncols + j] < -1e-9) {
                double ratio = red[j] / (-T[(size_t)r * ncols + j]);
                if (ratio < bestratio) { bestratio = ratio; ent = j; }
            }
        }
        if (ent < 0) { rc = 1; break; }
        {   /* pivot (r, ent) */
            double p = T[(size_t)r * ncols + ent];
            for (j = 0; j < ncols; j++) T[(size_t)r * ncols + j] /= p;
            for (i = 0; i < m; i++) {
                double f;
                if (i == r) continue;
                f = T[(size_t)i * ncols + ent];
                if (f == 0.0) continue;
                for (j = 0; j < ncols; j++) T[(size_t)i * ncols + j] -= f * T[(size_t)r * ncols + j];
            }
            bas[r] = ent;
        }
        if (it == max_iter - 1) rc = 3;
    }
    for (i = 0; i < n; i++) x[i] = 0.0;
    for (i = 0; i < m; i++) if (bas[i] >= 0 && bas[i] < n) x[bas[i]] = T[(size_t)i * ncols + n];
    if (basis_out) for (i = 0; i < m; i++) basis_out[i] = bas[i];
    /* duali y = cB^T B^-1: risolve B^T y = cB con LU su B = A[:, bas] (il tableau
     * non conserva B^-1 aggiornata, quindi si ricostruisce dalla base finale). */
    if (yout) {
        double *Bt = (double *)malloc((size_t)m * m * sizeof(double));
        double *rhs = (double *)malloc((size_t)m * sizeof(double));
        if (Bt && rhs) {
            for (i = 0; i < m; i++) rhs[i] = c[bas[i]];
            for (j = 0; j < m; j++)
                for (i = 0; i < m; i++) Bt[(size_t)j * m + i] = A[(size_t)i * n + bas[j]];  /* B^T */
            LuFact *f = dmat_lu_factor(Bt, m);
            if (f && dmat_lu_solve(f, rhs) == 0) for (i = 0; i < m; i++) yout[i] = rhs[i];
            if (f) dmat_lu_free(f);
        }
        free(Bt); free(rhs);
    }
    free(T); free(red); free(cB); free(M); free(bas);
    return rc;
}

/**
 * Revised simplex method with eta-matrix basis inverse updates.
 *
 * @param A        [in]  Constraint matrix in row-major (m x n).
 * @param m        [in]  Number of constraints.
 * @param n        [in]  Number of variables.
 * @param b        [in]  RHS vector (size m).
 * @param c        [in]  Objective coefficients (size n).
 * @param basis    [in]  Initial basis (size m) -- column indices of basic variables.
 * @param max_iter [in]  Maximum iterations.
 * @param x        [out] Primal solution (size n). Must not be NULL.
 * @param yout     [out] Optional dual solution (size m). May be NULL.
 *
 * @return 0 = optimal, 1 = primal infeasible (basis), 2 = unbounded,
 *         3 = max iterations, 4 = error.
 *
 * @note Maintains B^-1 explicitly with eta-matrix updates (rank-1 updates).
 *       More memory-efficient than full tableau but more complex.
 *
 *       Steps:
 *       1. Compute B^-1 from initial basis via Gauss-Jordan on [B | I].
 *       2. Check primal feasibility of initial basis (xB >= -1e-7).
 *       3. Iterate: compute dual y = cB^T B^-1, find entering variable
 *          (most negative reduced cost), compute d = B^-1 A_ent,
 *          ratio test on xB/d, eta-update B^-1.
 *
 *       Returns dual variables y = cB^T B^-1 if yout provided.
 *
 * @note Ported from gmbortools `gor_lp_revised_simplex` (from athityakumar/or_lab).
 */
int simplex_revised_solve_std(const double *A, int m, int n,
                              const double *b, const double *c,
                              const int *basis, int max_iter, double *x,
                              double *yout) {
    double *Binv, *xB, *y, *d, *cB;
    int *bas;
    int i, j, k, it, rc = 0;
    if (m < 1 || n < 1 || !A || !b || !c || !basis || !x) return 4;
    Binv = (double *)malloc((size_t)m * m * sizeof(double));
    xB = (double *)malloc((size_t)m * sizeof(double));
    y = (double *)malloc((size_t)m * sizeof(double));
    d = (double *)malloc((size_t)m * sizeof(double));
    cB = (double *)malloc((size_t)m * sizeof(double));
    bas = (int *)malloc((size_t)m * sizeof(int));
    if (!Binv || !xB || !y || !d || !cB || !bas) {
        free(Binv); free(xB); free(y); free(d); free(cB); free(bas); return 4;
    }
    {   /* [B | I] -> B^-1 nella meta' destra */
        double *W = (double *)malloc((size_t)m * 2 * m * sizeof(double));
        if (!W) { free(Binv); free(xB); free(y); free(d); free(cB); free(bas); return 4; }
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) W[(size_t)i * 2 * m + j] = A[(size_t)i * n + basis[j]];
            for (j = 0; j < m; j++) W[(size_t)i * 2 * m + m + j] = (i == j) ? 1.0 : 0.0;
        }
        for (k = 0; k < m; k++) {
            int piv = k; double mx = fabs(W[(size_t)k * 2 * m + k]);
            for (i = k + 1; i < m; i++) { double v = fabs(W[(size_t)i * 2 * m + k]); if (v > mx) { mx = v; piv = i; } }
            if (mx < 1e-12) { free(W); free(Binv); free(xB); free(y); free(d); free(cB); free(bas); return 4; }
            if (piv != k) for (j = 0; j < 2 * m; j++) { double t = W[(size_t)k*2*m+j]; W[(size_t)k*2*m+j] = W[(size_t)piv*2*m+j]; W[(size_t)piv*2*m+j] = t; }
            { double p = W[(size_t)k * 2 * m + k]; for (j = 0; j < 2 * m; j++) W[(size_t)k * 2 * m + j] /= p; }
            for (i = 0; i < m; i++) {
                double f;
                if (i == k) continue;
                f = W[(size_t)i * 2 * m + k];
                if (f == 0.0) continue;
                for (j = 0; j < 2 * m; j++) W[(size_t)i * 2 * m + j] -= f * W[(size_t)k * 2 * m + j];
            }
        }
        for (i = 0; i < m; i++) for (j = 0; j < m; j++) Binv[(size_t)i * m + j] = W[(size_t)i * 2 * m + m + j];
        free(W);
    }
    for (i = 0; i < m; i++) bas[i] = basis[i];
    for (i = 0; i < m; i++) { double s = 0.0; for (k = 0; k < m; k++) s += Binv[(size_t)i * m + k] * b[k]; xB[i] = s; }
    for (i = 0; i < m; i++) if (xB[i] < -1e-7) { free(Binv); free(xB); free(y); free(d); free(cB); free(bas); return 1; }
    for (it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(93);
        int ent = -1, lev = -1;
        double bestred = -1e-9, bestratio = 1e300;
        for (i = 0; i < m; i++) cB[i] = c[bas[i]];
        for (j = 0; j < m; j++) { double s = 0.0; for (i = 0; i < m; i++) s += cB[i] * Binv[(size_t)i * m + j]; y[j] = s; }
        for (j = 0; j < n; j++) {
            int isbasic = 0; double red = c[j];
            for (i = 0; i < m; i++) if (bas[i] == j) isbasic = 1;
            if (isbasic) continue;
            for (k = 0; k < m; k++) red -= y[k] * A[(size_t)k * n + j];
            if (red < bestred) { bestred = red; ent = j; }
        }
        if (ent < 0) { rc = 0; break; }
        for (i = 0; i < m; i++) { double s = 0.0; for (k = 0; k < m; k++) s += Binv[(size_t)i * m + k] * A[(size_t)k * n + ent]; d[i] = s; }
        for (i = 0; i < m; i++) if (d[i] > 1e-9) { double r = xB[i] / d[i]; if (r < bestratio - 1e-12) { bestratio = r; lev = i; } }
        if (lev < 0) { rc = 2; break; }
        {   /* eta update: B^-1 <- E B^-1 */
            double piv = d[lev];
            for (j = 0; j < m; j++) Binv[(size_t)lev * m + j] /= piv;
            for (i = 0; i < m; i++) {
                double f;
                if (i == lev) continue;
                f = d[i];
                if (f == 0.0) continue;
                for (j = 0; j < m; j++) Binv[(size_t)i * m + j] -= f * Binv[(size_t)lev * m + j];
            }
        }
        bas[lev] = ent;
        for (i = 0; i < m; i++) { double s = 0.0; for (k = 0; k < m; k++) s += Binv[(size_t)i * m + k] * b[k]; xB[i] = s; }
        if (it == max_iter - 1) rc = 3;
    }
    for (i = 0; i < n; i++) x[i] = 0.0;
    for (i = 0; i < m; i++) if (bas[i] >= 0 && bas[i] < n) x[bas[i]] = xB[i];
    if (yout) for (j = 0; j < m; j++) yout[j] = y[j];   /* duali y = cB^T B^-1 */
    free(Binv); free(xB); free(y); free(d); free(cB); free(bas);
    return rc;
}

/**
 * Finds an initial crash basis (m linearly independent columns of A).
 *
 * @param A     [in]  Constraint matrix in row-major (m x n).
 * @param m     [in]  Number of constraints.
 * @param n     [in]  Number of variables (must have n >= m).
 * @param basis [out] Output array of size m receiving column indices of basic variables.
 *
 * @return 1 if successful (rank(A) == m), 0 if rank < m or error.
 *
 * @note Uses Gaussian elimination with column pivoting (complete pivoting).
 *       The matrix A is NOT modified (copied internally).
 *
 * @example
 * int basis[2];
 * double A[] = {1, 2, 3, 4}; // 2x2 matrix
 * if (simplex_crash_basis(A, 2, 2, basis)) {
 *     // basis[0], basis[1] are the independent column indices
 * }
 */
int simplex_crash_basis(const double *A, int m, int n, int *basis)
{
    if (m < 1 || n < m || !A || !basis) return 0;
    double *W = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
    int *col = (int *)malloc((size_t)n * sizeof(int));
    if (!W || !col) { free(W); free(col); return 0; }
    memcpy(W, A, (size_t)m * (size_t)n * sizeof(double));
    for (int j = 0; j < n; j++) col[j] = j;
    int ok = 1;
    for (int r = 0; r < m; r++) {
        int pi = r, pj = r; double mx = 0.0;
        for (int i = r; i < m; i++)
            for (int c = r; c < n; c++) { double v = fabs(W[(size_t)i * n + c]); if (v > mx) { mx = v; pi = i; pj = c; } }
        if (mx < 1e-9) { ok = 0; break; }   /* rank < m */
        if (pj != r) {
            for (int i = 0; i < m; i++) { double t = W[(size_t)i * n + r]; W[(size_t)i * n + r] = W[(size_t)i * n + pj]; W[(size_t)i * n + pj] = t; }
            int tc = col[r]; col[r] = col[pj]; col[pj] = tc;
        }
        if (pi != r) for (int c = 0; c < n; c++) { double t = W[(size_t)r * n + c]; W[(size_t)r * n + c] = W[(size_t)pi * n + c]; W[(size_t)pi * n + c] = t; }
        for (int i = r + 1; i < m; i++) {
            double f = W[(size_t)i * n + r] / W[(size_t)r * n + r];
            if (f != 0.0) for (int c = r; c < n; c++) W[(size_t)i * n + c] -= f * W[(size_t)r * n + c];
        }
    }
    if (ok) for (int r = 0; r < m; r++) basis[r] = col[r];
    free(W); free(col);
    return ok;
}

/**
 * Computes reduced costs: red = c - A'y.
 *
 * @param A    [in]  Constraint matrix in row-major (m x n).
 * @param m    [in]  Number of constraints.
 * @param n    [in]  Number of variables.
 * @param c    [in]  Objective coefficients (size n).
 * @param y    [in]  Dual variables (size m).
 * @param red  [out] Reduced costs (size n). Must not be NULL.
 *
 * @note Computes red[j] = c[j] - sum_i y[i] * A[i,j] for all j.
 *       If m < 1 or n < 1 or any pointer is NULL, returns silently.
 *
 * @note Ported from gmbortools `gor_lp_reduced_costs`.
 */
void simplex_reduced_costs(const double *A, int m, int n, const double *c,
                           const double *y, double *red)
{
    int i, j;
    if (!A || !c || !y || !red || m < 1 || n < 1) return;
    for (j = 0; j < n; j++) {
        double s = c[j];
        for (i = 0; i < m; i++) s -= y[i] * A[(size_t)i * n + j];
        red[j] = s;
    }
}
