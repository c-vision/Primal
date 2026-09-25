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

/* linalg.c - dense linear algebra for primal */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "linalg.h"

/* sub-step callback hook (implemented in primal.c) */
extern int primal_cb_iter_on;
void primal_cb_iter(int code);

/**
 * Creates a new dense matrix (row-major).
 *
 * @param m [in] Number of rows.
 * @param n [in] Number of columns.
 *
 * @return Pointer to new DMat on success, NULL on allocation failure.
 *
 * @note Allocates a matrix of m rows, each row is a contiguous array of n doubles.
 *       All entries are zero-initialized. Rows are allocated separately for
 *       cache efficiency. Zero fill is intentional -- callers build matrices
 *       by writing known entries and read back the whole square.
 *
 * @example
 * DMat *A = dmat_new(3, 3);
 * if (!A) return ALLOC_FAILURE;
 * A->v[0][0] = 1.0; A->v[0][1] = 2.0;
 * // ...
 * dmat_free(A);
 */
DMat *dmat_new(int m, int n) {
    DMat *A = (DMat *)malloc(sizeof(DMat));
    if (!A) return NULL;
    A->m = m; A->n = n;
    A->v = (double **)calloc((size_t)(m > 0 ? m : 1), sizeof(double *));
    if (!A->v) { free(A); return NULL; }
    for (int i = 0; i < m; i++) {
        A->v[i] = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
        if (!A->v[i]) { dmat_free(A); return NULL; }
    }
    return A;
}

/**
 * Frees a dense matrix.
 *
 * @param A [in] Matrix to free.
 *
 * @note NULL-safe. Frees all rows and the matrix struct itself.
 */
void dmat_free(DMat *A) {
    if (!A) return;
    if (A->v) {
        for (int i = 0; i < A->m; i++) free(A->v[i]);
        free(A->v);
    }
    free(A);
}

/**
 * Solves a dense linear system A x = b via LU with partial pivoting (destroys A).
 *
 * @param A [in/out] Dense matrix (n x n, row-major). On output, L in strict lower
 *       triangle (unit diagonal implicit), U in upper triangle + diagonal.
 * @param b [in/out] RHS vector (size n). On output, solution x.
 * @param n [in]    Matrix dimension.
 *
 * @return 0 on success, 1 if pivot < 1e-14 (absolute threshold).
 *
 * @note Destroys the input matrix A (stores LU factorization in-place).
 *       Uses row pointer swaps for pivoting. Absolute pivot threshold 1e-14
 *       means matrices scaled around 1e-20 read as singular.
 *       For multiple RHS with same matrix, use dmat_lu_factor + dmat_lu_solve.
 *
 * @example
 * DMat *A = dmat_new(3, 3);
 * // fill A...
 * double b[3] = {1, 2, 3};
 * int rc = dmat_solve_lu(A, b, 3);
 * if (rc == 0) { // b now contains solution }
 */
int dmat_solve_lu(DMat *A, double *b, int n) {
    for (int k = 0; k < n; k++) {
        int piv = k;
        double maxv = fabs(A->v[k][k]);
        for (int i = k + 1; i < n; i++)
            if (fabs(A->v[i][k]) > maxv) { maxv = fabs(A->v[i][k]); piv = i; }
        if (maxv < 1e-14) return 1;
        if (piv != k) { double *t = A->v[piv]; A->v[piv] = A->v[k]; A->v[k] = t; }
        for (int i = k + 1; i < n; i++) {
            double f = A->v[i][k] / A->v[k][k];
            A->v[i][k] = f;
            for (int j = k + 1; j < n; j++) A->v[i][j] -= f * A->v[k][j];
        }
    }
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= A->v[i][j] * b[j];
        b[i] = s;
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= A->v[i][j] * b[j];
        b[i] = s / A->v[i][i];
    }
    return 0;
}

/**
 * Computes Cholesky factorization L L' of a symmetric positive definite matrix.
 *
 * @param A [in/out] Dense matrix (n x n, row-major). Lower triangle read,
 *       overwritten with L (strict lower = L, diagonal = diag(L)).
 *       Upper triangle zeroed on output.
 * @param n [in] Matrix dimension.
 *
 * @return 0 on success, 1 if diagonal <= 1e-300 (not numerically SPD).
 *
 * @note No pivoting, no permutation. Reads only lower triangle of A.
 *       Overwrites lower triangle with L, zeros upper triangle.
 *       On failure, A is left half-factored.
 *       Use dmat_chol_solve for solving with the factor.
 *
 * @example
 * DMat *A = dmat_new(3, 3);
 * // fill A with symmetric PD matrix...
 * int rc = dmat_cholesky(A, 3);
 * if (rc == 0) { double b[3] = {1,2,3}; dmat_chol_solve(A, b, 3); }
 */
int dmat_cholesky(DMat *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A->v[i][j];
            for (int k = 0; k < j; k++) s -= A->v[i][k] * A->v[j][k];
            if (i == j) {
                if (s <= 1e-300) return 1;
                A->v[i][i] = sqrt(s);
            } else {
                A->v[i][j] = s / A->v[j][j];
            }
        }
        for (int j = i + 1; j < n; j++) A->v[i][j] = 0.0;
    }
    return 0;
}

/**
 * Solves L L' x = b using a Cholesky factor from dmat_cholesky.
 *
 * @param L [in]    Cholesky factor (lower triangle from dmat_cholesky).
 * @param b [in/out] RHS vector (size n). Overwritten with solution x.
 * @param n [in]    Matrix dimension.
 *
 * @return 0 always (singularity checked in dmat_cholesky).
 *
 * @note Reads L's lower triangle (forward sweep L v = b) and upper triangle
 *       (backward sweep L' u = v). The upper triangle must be zero (as left
 *       by dmat_cholesky). Zero diagonal in L causes division by zero.
 *
 * @example
 * dmat_cholesky(A, n);
 * dmat_chol_solve(A, b, n); // b now contains solution
 */
int dmat_chol_solve(const DMat *L, double *b, int n) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= L->v[i][j] * b[j];
        b[i] = s / L->v[i][i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= L->v[j][i] * b[j];
        b[i] = s / L->v[i][i];
    }
    return 0;
}

/**
 * Computes y = a*x + y (BLAS-1 axpy).
 *
 * @param n [in]    Vector length.
 * @param a [in]    Scalar multiplier.
 * @param x [in]    Source vector (size n).
 * @param y [in/out] Destination vector (size n). Updated in-place.
 *
 * @note A length of 0 is a no-op that returns immediately.
 */
void dvec_axpy(int n, double a, const double *x, double *y) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

/**
 * Computes dot product x'y.
 *
 * @param n [in] Vector length.
 * @param x [in] First vector (size n).
 * @param y [in] Second vector (size n).
 *
 * @return x'y = sum_i x_i * y_i.
 *
 * @note A length of 0 returns 0.
 */
double dvec_dot(int n, const double *x, const double *y) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

/**
 * Computes Euclidean (L2) norm of a vector.
 *
 * @param n [in] Vector length.
 * @param x [in] Vector (size n).
 *
 * @return sqrt(sum_i x_i^2).
 */
double dvec_norm2(int n, const double *x) {
    return sqrt(dvec_dot(n, x, x));
}

/**
 * Computes infinity (L-infinity) norm of a vector.
 *
 * @param n [in] Vector length.
 * @param x [in] Vector (size n).
 *
 * @return max_i |x_i|.
 */
double dvec_norm_inf(int n, const double *x) {
    double m = 0.0;
    for (int i = 0; i < n; i++) { double a = fabs(x[i]); if (a > m) m = a; }
    return m;
}

/* ---------------- LuFact ---------------- */

/**
 * Factors a dense matrix into LU with partial pivoting (copy-based).
 *
 * @param A [in] Dense matrix (n x n, row-major). Not modified.
 * @param n [in] Matrix dimension.
 *
 * @return LuFact* on success, NULL on singular or allocation failure.
 *
 * @note Creates a COPY of A and factors it. The original A is unchanged.
 *       Uses partial pivoting with absolute threshold 1e-300 (catches exact
 *       zero pivots; regularization is caller's responsibility).
 *       Returns NULL if any pivot < 1e-300 (singular).
 *       The factor can be used with dmat_lu_solve for multiple RHS.
 *
 * @example
 * LuFact *f = dmat_lu_factor(A_data, n);
 * if (f) { dmat_lu_solve(f, b, n); dmat_lu_free(f); }
 */
LuFact *dmat_lu_factor(const double *A, int n) {
    if (n <= 0) return NULL;
    if (primal_cb_iter_on) primal_cb_iter(79);
    LuFact *f = (LuFact *)malloc(sizeof(LuFact));
    if (!f) return NULL;
    f->n = n;
    f->lu = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    f->piv = (int *)malloc((size_t)n * sizeof(int));
    if (!f->lu || !f->piv) { dmat_lu_free(f); return NULL; }
    memcpy(f->lu, A, (size_t)n * (size_t)n * sizeof(double));
    f->ok = 1;
    for (int k = 0; k < n; k++) {
        int p = k;
        double maxv = fabs(f->lu[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double a = fabs(f->lu[i * n + k]);
            if (a > maxv) { maxv = a; p = i; }
        }
        if (maxv < 1e-300) { f->ok = 0; break; }
        f->piv[k] = p;
        if (p != k)
            for (int j = 0; j < n; j++) {
                double t = f->lu[k * n + j];
                f->lu[k * n + j] = f->lu[p * n + j];
                f->lu[p * n + j] = t;
            }
        double pivv = f->lu[k * n + k];
        for (int i = k + 1; i < n; i++) {
            double mlt = f->lu[i * n + k] / pivv;
            f->lu[i * n + k] = mlt;
            if (mlt != 0.0)
                for (int j = k + 1; j < n; j++)
                    f->lu[i * n + j] -= mlt * f->lu[k * n + j];
        }
    }
    if (!f->ok) { dmat_lu_free(f); return NULL; }
    return f;
}

/* Compensated dot product (Neumaier summation + FMA product residual), mirrored
 * from sdp.c's residual helper of the same name.  On a degenerate system the
 * triangular sweeps below mix terms of wildly different magnitude (a free
 * variable split into two nonnegatives that both drift to ~1e6 while their
 * difference stays O(1) puts both scales in every row of L/U) and plain
 * double summation loses exactly the digits the factorisation kept.  Two
 * doubles of precision for the price of a few FMA per entry. */
static double nsum_prod(const double *a, const double *b, int n) {
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double p = a[i] * b[i];
        double e = fma(a[i], b[i], -p);
        double t = s + p;
        double bb = (fabs(s) >= fabs(p)) ? ((s - t) + p) : ((p - t) + s);
        c += bb + e;
        s = t;
    }
    return s + c;
}

/**
 * Solves A x = b using a precomputed LU factorization.
 *
 * @param f   [in]  LuFact from dmat_lu_factor.
 * @param rhs [in/out] RHS vector (size n). Overwritten with solution x.
 *
 * @return 0 on success, -1 if factor is NULL or singular.
 *
 * @note Reads rhs in ORIGINAL row ordering, writes solution in same ordering.
 *       Pivot swaps are applied internally. Factor is const (multiple RHS OK).
 *       Does NOT return error for near-singular systems -- returns large answer.
 *
 * @note For iterative refinement, use dmat_lu_solve_comp instead.
 *
 * @example
 * LuFact *f = dmat_lu_factor(A, n);
 * if (f) { dmat_lu_solve(f, b, n); // b now has solution }
 */
int dmat_lu_solve(const LuFact *f, double *rhs) {
    if (!f || !f->ok) return -1;
    int n = f->n;
    /* apply row swaps */
    for (int k = 0; k < n; k++) {
        int p = f->piv[k];
        if (p != k) { double t = rhs[k]; rhs[k] = rhs[p]; rhs[p] = t; }
    }
    /* forward: L y = P b (L unit lower) */
    for (int i = 0; i < n; i++) {
        double s = rhs[i];
        for (int j = 0; j < i; j++) s -= f->lu[i * n + j] * rhs[j];
        rhs[i] = s;
    }
    /* back: U x = y */
    for (int i = n - 1; i >= 0; i--) {
        double s = rhs[i];
        for (int j = i + 1; j < n; j++) s -= f->lu[i * n + j] * rhs[j];
        rhs[i] = s / f->lu[i * n + i];
    }
    return 0;
}

/* Same solve, triangular sweeps in compensated (Neumaier+FMA) arithmetic --
 * see nsum_prod above.  A plain dmat_lu_solve loses digits exactly where a
 * row mixes wildly different magnitudes, which is what a degenerate SDP's
 * augmented KKT does (a free variable split into two nonnegatives that both
 * drift to ~1e6 while their difference stays O(1) puts both scales in every
 * row of L/U).  Kept as a SEPARATE function rather than changed in place:
 * dmat_lu_solve is also the LP/QP simplex basis solve and the SOC/exp-power
 * conic path (socp.c, ipm.c, and sdp.c's own exp/power rows), all validated
 * as they stand, and the extra compensation there only pays for the flops --
 * measured on T36/T81 (DEXP), the changed rounding shifts a step ratio enough
 * to move the accepted point by 1.6e-4, outside those tests' tolerance. */
int dmat_lu_solve_comp(const LuFact *f, double *rhs) {
    if (!f || !f->ok) return -1;
    int n = f->n;
    for (int k = 0; k < n; k++) {
        int p = f->piv[k];
        if (p != k) { double t = rhs[k]; rhs[k] = rhs[p]; rhs[p] = t; }
    }
    for (int i = 0; i < n; i++)
        rhs[i] -= nsum_prod(f->lu + i * n, rhs, i);
    for (int i = n - 1; i >= 0; i--) {
        double t = nsum_prod(f->lu + i * n + i + 1, rhs + i + 1, n - i - 1);
        rhs[i] = (rhs[i] - t) / f->lu[i * n + i];
    }
    return 0;
}

/* Also the rollback inside dmat_lu_factor, which calls it after allocating one
 * of the two arrays -- hence the per-field freedom rather than one release. */
void dmat_lu_free(LuFact *f) {
    if (!f) return;
    free(f->lu);
    free(f->piv);
    free(f);
}

/* ---------------- Jacobi eigenvalue (symmetric) ---------------- */
/* Autodecomposizione simmetrica con rotazioni di Jacobi (cyclic, sweep).
 * Convergenza: off-diagonale scende quadraticamente; tolleranza 1e-14. */
void dmat_eig_jacobi(int n, const double *A, double *eval, double *evec) {
    int nn = n > 0 ? n : 1;
    double *W = (double *)calloc((size_t)nn * (size_t)nn, sizeof(double));
    double *V = (double *)calloc((size_t)nn * (size_t)nn, sizeof(double));
    if (!W || !V) { free(W); free(V); return; }
    for (int i = 0; i < n * n; i++) W[i] = A[i];
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) off += W[p * n + q] * W[p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = W[p * n + q];
                if (fabs(apq) < 1e-32) continue;
                double theta = (W[q * n + q] - W[p * n + p]) / (2.0 * apq);
                double t = (theta >= 0.0 ? 1.0 : -1.0) /
                           (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), sn = t * c;
                for (int k = 0; k < n; k++) {   /* righe (J^T W) */
                    double wp = W[p * n + k], wq = W[q * n + k];
                    W[p * n + k] = c * wp - sn * wq;
                    W[q * n + k] = sn * wp + c * wq;
                }
                for (int k = 0; k < n; k++) {   /* colonne (W J) */
                    double wp = W[k * n + p], wq = W[k * n + q];
                    W[k * n + p] = c * wp - sn * wq;
                    W[k * n + q] = sn * wp + c * wq;
                }
                for (int k = 0; k < n; k++) {   /* V J */
                    double vp = V[k * n + p], vq = V[k * n + q];
                    V[k * n + p] = c * vp - sn * vq;
                    V[k * n + q] = sn * vp + c * vq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) eval[i] = W[i * n + i];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) evec[i * n + j] = V[i * n + j];
    free(W); free(V);
}

/* ---------------- sparse Cholesky (left-looking) ----------------
 * K = L L', K simmetrica definita positiva data in CSC (triangolo
 * inferiore: colonna j con righe i >= j). Per ogni colonna j:
 *   w = K(:,j) - sum_{k<j, L(j,k)!=0} L(j,k) * L(:,k)
 *   L(j,j) = sqrt(w_j),  L(i,j) = w_i / L(j,j)
 * L'adiacenza per riga di L (lista dei k<j con L(j,k) != 0) e' mantenuta
 * incrementalmente: quando la colonna k e' completata, ogni (i,k) viene
 * accodato alla lista della riga i. Costo ~ O(sum_k nnz(L(:,k))^2) con
 * fill-in dinamico. */
static int *sym_amd(int n, const int *Ap, const int *Ai);

static SpChol *spchol_factor_nat(int n, const int *Kp, const int *Ki, const double *Kx) {
    if (n <= 0 || !Kp || !Ki || !Kx) return NULL;
    int **ci = (int **)calloc((size_t)n, sizeof(int *));
    double **cv = (double **)calloc((size_t)n, sizeof(double *));
    int *cn = (int *)calloc((size_t)n, sizeof(int));
    int *cc = (int *)calloc((size_t)n, sizeof(int));
    int **ri = (int **)calloc((size_t)n, sizeof(int *));     /* L' row adjacency */
    double **rv = (double **)calloc((size_t)n, sizeof(double *));
    int *rn = (int *)calloc((size_t)n, sizeof(int));
    int *rc = (int *)calloc((size_t)n, sizeof(int));
    double *w = (double *)calloc((size_t)n, sizeof(double));
    int *mark = (int *)calloc((size_t)n, sizeof(int));
    int *touched = (int *)malloc((size_t)n * sizeof(int));
    if (!ci || !cv || !cn || !cc || !ri || !rv || !rn || !rc || !w || !mark || !touched) {
        for (int j = 0; j < n; j++) { free(ci[j]); free(cv[j]); free(ri[j]); free(rv[j]); }
        free(ci); free(cv); free(cn); free(cc);
        free(ri); free(rv); free(rn); free(rc);
        free(w); free(mark); free(touched);
        return NULL;
    }

    for (int j = 0; j < n; j++) {
        int nt = 0;
        /* scatter del triangolo inferiore di K(:,j) */
        for (int p = Kp[j]; p < Kp[j + 1]; p++) {
            int i = Ki[p];
            if (i < j) continue;
            if (mark[i] != j + 1) { mark[i] = j + 1; touched[nt++] = i; w[i] = 0.0; }
            w[i] += Kx[p];
        }
        /* sottrai i contributi delle colonne precedenti */
        for (int q = 0; q < rn[j]; q++) {
            int k = ri[j][q];
            double ljk = rv[j][q];
            for (int p = 0; p < cn[k]; p++) {
                int i = ci[k][p];
                if (i < j) continue;
                if (mark[i] != j + 1) { mark[i] = j + 1; touched[nt++] = i; w[i] = 0.0; }
                w[i] -= ljk * cv[k][p];
            }
        }
        double dj = w[j];
        if (!(dj > 1e-300)) {   /* non definita positiva */
            for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
            free(ci); free(cv); free(cn); free(cc);
            free(ri); free(rv); free(rn); free(rc);
            free(w); free(mark); free(touched);
            return NULL;
        }
        double ljj = sqrt(dj);
        if (cc[j] < nt + 1) {
            int cap = nt + 2;
            int *ti = (int *)realloc(ci[j], (size_t)cap * sizeof(int));
            double *tv = (double *)realloc(cv[j], (size_t)cap * sizeof(double));
            if (!ti || !tv) { free(ti); free(tv);
                for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
                free(ci); free(cv); free(cn); free(cc);
                free(ri); free(rv); free(rn); free(rc);
                free(w); free(mark); free(touched);
                return NULL; }
            ci[j] = ti; cv[j] = tv; cc[j] = cap;
        }
        ci[j][0] = j; cv[j][0] = ljj; cn[j] = 1;
        for (int q = 0; q < nt; q++) {
            int i = touched[q];
            double v = w[i];
            w[i] = 0.0;
            if (i == j) continue;
            v /= ljj;
            if (v == 0.0) continue;
            ci[j][cn[j]] = i; cv[j][cn[j]] = v; cn[j]++;
            if (rc[i] == rn[i]) {
                int cap = rc[i] ? rc[i] * 2 : 4;
                int *ti = (int *)realloc(ri[i], (size_t)cap * sizeof(int));
                double *tv = (double *)realloc(rv[i], (size_t)cap * sizeof(double));
                if (!ti || !tv) { free(ti); free(tv);
                    for (int q2 = 0; q2 < n; q2++) { free(ci[q2]); free(cv[q2]); free(ri[q2]); free(rv[q2]); }
                    free(ci); free(cv); free(cn); free(cc);
                    free(ri); free(rv); free(rn); free(rc);
                    free(w); free(mark); free(touched);
                    return NULL; }
                ri[i] = ti; rv[i] = tv; rc[i] = cap;
            }
            ri[i][rn[i]] = j; rv[i][rn[i]] = v; rn[i]++;
        }
        w[j] = 0.0;
    }

    /* compatta in CSC */
    SpChol *L = (SpChol *)malloc(sizeof(SpChol));
    if (!L) {
        for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
        free(ci); free(cv); free(cn); free(cc);
        free(ri); free(rv); free(rn); free(rc);
        free(w); free(mark); free(touched);
        return NULL;
    }
    int total = 0;
    for (int j = 0; j < n; j++) total += cn[j];
    if (getenv("GMB_DBG_SPCHOL")) fprintf(stderr,"spchol n=%d nnzK=%d nnzL=%d fill=%.2f\n",n,Kp[n],total,(double)total/(Kp[n]>0?Kp[n]:1));
    L->n = n; L->perm = NULL;
    L->Lp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    L->Li = (int *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    L->Lx = (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double));
    if (!L->Lp || !L->Li || !L->Lx) {
        free(L->Lp); free(L->Li); free(L->Lx); free(L);
        L = NULL;
    } else {
        int off = 0;
        for (int j = 0; j < n; j++) {
            L->Lp[j] = off;
            for (int p = 0; p < cn[j]; p++) { L->Li[off] = ci[j][p]; L->Lx[off] = cv[j][p]; off++; }
        }
        L->Lp[n] = off;
    }
    for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
    free(ci); free(cv); free(cn); free(cc);
    free(ri); free(rv); free(rn); free(rc);
    free(w); free(mark); free(touched);
    return L;
}

/* K u = rhs through the two triangular sweeps, rhs overwritten.  The diagonal
 * is read by a scan of the column rather than assumed at slot 0: the factor
 * does write ci[j][0] = j, and the non-zero off-diagonal rows land after it in
 * the order the column was assembled (scatter, then one block per previously
 * completed column of L' that touches this row) -- which is NOT sorted by row
 * index.  Solve therefore depends on neither the slot convention nor the
 * ordering, and each sweep is O(nnz(L)) with a linear search inside it.
 * Returns -1 when a diagonal is missing or exactly zero; a factor that
 * spchol_factor accepted cannot normally reach that, and the test is there
 * because the alternative is dividing by zero and returning a number that looks
 * like an answer. */
static int spchol_solve_nat(const SpChol *L, double *rhs) {
    if (!L || !rhs) return -1;
    int n = L->n;
    /* forward: L v = rhs (colonne ascendenti) */
    for (int k = 0; k < n; k++) {
        int p0 = L->Lp[k], p1 = L->Lp[k + 1];
        double lkk = 0.0;
        for (int p = p0; p < p1; p++)
            if (L->Li[p] == k) { lkk = L->Lx[p]; break; }
        if (lkk == 0.0) return -1;
        rhs[k] /= lkk;
        double vk = rhs[k];
        for (int p = p0; p < p1; p++) {
            int i = L->Li[p];
            if (i > k) rhs[i] -= L->Lx[p] * vk;
        }
    }
    /* backward: L' u = v (colonne discendenti) */
    for (int k = n - 1; k >= 0; k--) {
        int p0 = L->Lp[k], p1 = L->Lp[k + 1];
        double lkk = 0.0, sum = 0.0;
        for (int p = p0; p < p1; p++) {
            if (L->Li[p] == k) lkk = L->Lx[p];
            else sum += L->Lx[p] * rhs[L->Li[p]];
        }
        if (lkk == 0.0) return -1;
        rhs[k] = (rhs[k] - sum) / lkk;
    }
    return 0;
}

SpChol *spchol_factor(int n, const int *Kp, const int *Ki, const double *Kx) {
    return spchol_factor_nat(n, Kp, Ki, Kx);
}

int spchol_solve(const SpChol *L, double *rhs) {
    return spchol_solve_nat(L, rhs);
}

/* Fill-reducing wrapper: apply AMD to K, factor the permuted matrix, and keep
 * the permutation so spchol_solve can undo it.  A quasi-definite KKT factored
 * in natural order can carry orders-of-magnitude more fill than the same matrix
 * ordered (Clarabel/qdldl use exactly this: static AMD + sparse factor). */
SpChol *spchol_factor_ord(int n, const int *Kp, const int *Ki, const double *Kx) {
    if (n <= 0 || !Kp || !Ki || !Kx) return NULL;
    int *perm = sym_amd(n, Kp, Ki);
    if (!perm) return spchol_factor_nat(n, Kp, Ki, Kx);   /* natural order */
    int *iperm = (int *)malloc((size_t)n * sizeof(int));
    int *pKp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!iperm || !pKp) { free(iperm); free(pKp); free(perm); return NULL; }
    for (int k = 0; k < n; k++) iperm[perm[k]] = k;
    int nnz = Kp[n];
    int *pKi = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    double *pKx = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (!pKi || !pKx) { free(pKi); free(pKx); free(iperm); free(pKp); free(perm); return NULL; }
    /* count the permuted lower triangle */
    for (int k = 0; k <= n; k++) pKp[k] = 0;
    for (int j = 0; j < n; j++)
        for (int p = Kp[j]; p < Kp[j + 1]; p++) {
            int i = Ki[p];
            if (i < j) continue;
            int a = iperm[i], b = iperm[j];
            pKp[(a < b ? a : b) + 1]++;
        }
    for (int k = 0; k < n; k++) pKp[k + 1] += pKp[k];
    {   int *w = (int *)malloc((size_t)n * sizeof(int));
        if (!w) { free(pKi); free(pKx); free(iperm); free(pKp); free(perm); return NULL; }
        for (int k = 0; k < n; k++) w[k] = pKp[k];
        for (int j = 0; j < n; j++)
            for (int p = Kp[j]; p < Kp[j + 1]; p++) {
                int i = Ki[p];
                if (i < j) continue;
                int a = iperm[i], b = iperm[j];
                int col = a < b ? a : b, row = a < b ? b : a;
                pKi[w[col]] = row; pKx[w[col]] = Kx[p]; w[col]++;
            }
        free(w);
    }
    SpChol *L = spchol_factor_nat(n, pKp, pKi, pKx);
    free(pKi); free(pKx); free(iperm); free(pKp);
    if (L) L->perm = perm; else free(perm);
    return L;
}

int spchol_solve_ord(const SpChol *L, double *rhs) {
    if (!L || !rhs) return -1;
    if (!L->perm) return spchol_solve_nat(L, rhs);
    int n = L->n;
    double *t = (double *)malloc((size_t)n * sizeof(double));
    if (!t) return -1;
    for (int k = 0; k < n; k++) t[k] = rhs[L->perm[k]];
    int rc = spchol_solve_nat(L, t);
    for (int k = 0; k < n; k++) rhs[L->perm[k]] = t[k];
    free(t);
    return rc;
}

void spchol_free(SpChol *L) {
    if (!L) return;
    free(L->Lp); free(L->Li); free(L->Lx);
    free(L);
}

/* ===================== sparse LU (partial pivoting) =====================
 * Right-looking Gaussian elimination on a sparse CSC matrix with partial
 * (row) pivoting. The working matrix is kept as sparse rows; L multipliers are
 * stored in-place in the strictly-lower positions so a row swap exchanges L and
 * U parts together. Fill-in is handled with a dense marker (w/stamp) per axpy.
 * Result: P A = L U with L unit-lower (CSC) and U upper (CSR by position-row). */
typedef struct { int nz, cap; int *ci; double *cv; } SRow;

/* A row of the working matrix is an unordered list of (column, value) slots,
 * a single namespace for both halves of the factor: slots at a column already
 * passed hold the L multipliers, slots ahead hold the working U row, which is
 * what lets a pivot swap exchange L and U content together.
 * srow_get finds the FIRST slot with a given column, and that is a contract
 * rather than an accident of the loop: a row carrying the same column twice
 * would be read as its first copy by the pivot search while the axpy update
 * stages the row as w[col] = rows[i].cv[a], which keeps the LAST copy.  The
 * factorisation would then be of a matrix that is neither copy, and no error
 * would say so.  Rows are built from the input CSC one entry at a time with no
 * summation, so the distinct-positions requirement of the header is enforced by
 * the caller: the conic KKT comes through tri3_to_csc, which merges repeated
 * (row,column) triplets into their sum before handing the matrix over. */
static int srow_get(const SRow *r, int col) {
    for (int q = 0; q < r->nz; q++) if (r->ci[q] == col) return q;
    return -1;
}

/* ---- fill-reducing ordering (minimum degree) ---- */
/* Set insert with a linear duplicate scan: returns 0 when x was already in the
 * list (so an adjacency list really is a set) and -1 only when the growth
 * failed.  The dedup is what lets the ORDERING tolerate a sparsity pattern that
 * repeats an edge -- the factorisation cannot, see srow_get.  A caller that
 * gets -1 abandons the ordering instead of truncating it, and sym_amd returns
 * NULL, which sends splu_factor back to the natural column order: a worse fill,
 * never a wrong answer. */
static int amd_add(int **a, int *cnt, int *cap, int x) {
    for (int q = 0; q < *cnt; q++) if ((*a)[q] == x) return 0;
    if (*cnt == *cap) {
        int nc = *cap ? *cap * 2 : 8;
        int *na = (int *)realloc(*a, (size_t)nc * sizeof(int));
        if (!na) return -1;
        *a = na; *cap = nc;
    }
    (*a)[(*cnt)++] = x;
    return 0;
}

/* Minimum-degree fill-reducing ordering of a symmetric sparsity pattern.
 * (Ap,Ai) is CSC; each entry (i,j) is treated as an undirected edge i-j.
 * Returns perm[n] (perm[k] = original index placed at position k), or NULL.
 * Explicit-fill minimum degree with a linear min search; used to order the
 * augmented conic KKT system before the sparse LU (the natural order can give
 * 80x fill-in there). */
static int *sym_amd(int n, const int *Ap, const int *Ai) {
    if (n <= 0 || !Ap || !Ai) return NULL;
    if (primal_cb_iter_on) primal_cb_iter(84);
    int **adj = (int **)calloc((size_t)n, sizeof(int *));
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    int *cap = (int *)calloc((size_t)n, sizeof(int));
    int *rem = (int *)calloc((size_t)n, sizeof(int));
    int *seen = (int *)calloc((size_t)n, sizeof(int));
    int *nbr = (int *)malloc((size_t)n * sizeof(int));
    int *perm = (int *)malloc((size_t)n * sizeof(int));
    if (!adj || !deg || !cap || !rem || !seen || !nbr || !perm) {
        for (int i = 0; i < n; i++) free(adj[i]);
        free(adj); free(deg); free(cap); free(rem); free(seen); free(nbr); free(perm);
        return NULL;
    }
    int ok = 1;
    for (int j = 0; j < n && ok; j++)
        for (int p = Ap[j]; p < Ap[j + 1] && ok; p++) {
            int i = Ai[p];
            if (i == j || i < 0 || i >= n) continue;
            if (amd_add(&adj[i], &deg[i], &cap[i], j) || amd_add(&adj[j], &deg[j], &cap[j], i)) ok = 0;
        }
    int sst = 0;
    for (int step = 0; step < n && ok; step++) {
        int v = -1;
        for (int i = 0; i < n; i++) if (!rem[i] && (v < 0 || deg[i] < deg[v])) v = i;
        if (v < 0) { ok = 0; break; }
        perm[step] = v; rem[v] = 1;
        int nn = 0;
        for (int q = 0; q < deg[v]; q++) { int u = adj[v][q]; if (!rem[u]) nbr[nn++] = u; }
        for (int a = 0; a < nn; a++) { int u = nbr[a];   /* detach v */
            for (int q = 0; q < deg[u]; q++) if (adj[u][q] == v) { adj[u][q] = adj[u][deg[u] - 1]; deg[u]--; break; } }
        for (int a = 0; a < nn && ok; a++) { int u = nbr[a];   /* fill: neighbourhood -> clique */
            sst++;
            for (int q = 0; q < deg[u]; q++) seen[adj[u][q]] = sst;
            for (int b = 0; b < nn; b++) { int w = nbr[b];
                if (w == u || seen[w] == sst) continue;
                if (amd_add(&adj[u], &deg[u], &cap[u], w) || amd_add(&adj[w], &deg[w], &cap[w], u)) { ok = 0; break; } }
        }
    }
    for (int i = 0; i < n; i++) free(adj[i]);
    free(adj); free(deg); free(cap); free(rem); free(seen); free(nbr);
    if (!ok) { free(perm); return NULL; }
    return perm;
}

/* Factor A in CSC with partial (row) pivoting and a fill-reducing COLUMN
 * ordering.  The ordering is applied while the rows are scattered, not as a
 * later renumbering: the working matrix is already A with its columns
 * permuted, and the row swaps move whole SRow records, so the factor held on
 * return is of P A Q, with perm[] naming the rows and qperm[] the columns.
 * splu_solve is what undoes both, so a caller hands in an UNPERMUTED rhs and
 * reads an unpermuted solution -- the permutations are internal bookkeeping,
 * never a contract the caller has to remember.
 * Failure is always NULL and never a partial factor: either the column has no
 * pivot left above 1e-300 (singular, and the caller regularises and retries) or
 * a row could not grow to hold its fill.  The second mode is the one to know
 * about, because it breaks out with the working rows half updated -- an
 * inconsistent factor -- which is why every exit frees `rows` and why nothing
 * is published on that path.  On success the rows are scanned once more to
 * split the factor into L (CSC, unit diagonal implicit) and U (CSR by
 * position row), and from there the struct owns every array. */
SpluFact *splu_factor(int n, const int *Ap, const int *Ai, const double *Ax) {
    if (n < 0 || !Ap) return NULL;
    if (primal_cb_iter_on) primal_cb_iter(79);
    SpluFact *F = (SpluFact *)calloc(1, sizeof(SpluFact));
    SRow *rows = (SRow *)calloc((size_t)(n > 0 ? n : 1), sizeof(SRow));
    int *rc = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    if (!F || !rows || !rc) { free(F); free(rows); free(rc); return NULL; }
    F->n = n;
    /* fill-reducing column ordering (falls back to the natural order) */
    int *qperm = sym_amd(n, Ap, Ai);
    int *iperm = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!iperm) { free(qperm); free(F); free(rows); free(rc); return NULL; }
    if (qperm) { for (int k = 0; k < n; k++) iperm[qperm[k]] = k; }
    else {
        qperm = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
        if (!qperm) { free(iperm); free(F); free(rows); free(rc); return NULL; }
        for (int k = 0; k < n; k++) { qperm[k] = k; iperm[k] = k; }
    }
    F->qperm = qperm;
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) rc[Ai[p]]++;
    for (int i = 0; i < n; i++) {
        rows[i].cap = rc[i] > 0 ? rc[i] : 1; rows[i].nz = 0;
        rows[i].ci = (int *)malloc((size_t)rows[i].cap * sizeof(int));
        rows[i].cv = (double *)malloc((size_t)rows[i].cap * sizeof(double));
        if (!rows[i].ci || !rows[i].cv) { free(rc); for (int q=0;q<=i;q++){free(rows[q].ci);free(rows[q].cv);} free(rows); free(F); return NULL; }
    }
    for (int j = 0; j < n; j++) for (int p = Ap[j]; p < Ap[j + 1]; p++) {
        int i = Ai[p]; rows[i].ci[rows[i].nz] = iperm[j]; rows[i].cv[rows[i].nz] = Ax[p]; rows[i].nz++;
    }
    free(iperm); free(rc);
    F->perm = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    double *w = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    int *wst = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int *touch = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!F->perm || !w || !wst || !touch) { free(w);free(wst);free(touch); for(int i=0;i<n;i++){free(rows[i].ci);free(rows[i].cv);} free(rows); splu_free(F); return NULL; }
    for (int i = 0; i < n; i++) F->perm[i] = i;
    int stamp = 0, singular = 0;
    for (int k = 0; k < n; k++) {
        int pk = srow_get(&rows[k], k);
        int p = k; double best = pk >= 0 ? fabs(rows[k].cv[pk]) : 0.0;
        for (int i = k + 1; i < n; i++) { int q = srow_get(&rows[i], k); double v = q >= 0 ? fabs(rows[i].cv[q]) : 0.0; if (v > best) { best = v; p = i; } }
        if (best < 1e-300) { singular = 1; break; }
        if (p != k) { SRow tr = rows[k]; rows[k] = rows[p]; rows[p] = tr; int tp = F->perm[k]; F->perm[k] = F->perm[p]; F->perm[p] = tp; }
        int qk = srow_get(&rows[k], k);
        double piv = rows[k].cv[qk];
        for (int i = k + 1; i < n; i++) {
            int qi = srow_get(&rows[i], k);
            if (qi < 0) continue;
            double aik = rows[i].cv[qi];
            if (aik == 0.0) { continue; }
            double mult = aik / piv;
            rows[i].cv[qi] = mult;                 /* store L[i][k] in place */
            /* rows[i](cols>k) -= mult * rows[k](cols>k), with fill via marker */
            stamp++;
            int nt = 0;
            for (int a = 0; a < rows[i].nz; a++) { int col = rows[i].ci[a]; if (col > k) { w[col] = rows[i].cv[a]; wst[col] = stamp; touch[nt++] = col; } }
            for (int a = 0; a < rows[k].nz; a++) {
                int col = rows[k].ci[a]; if (col <= k) continue;
                double sub = mult * rows[k].cv[a];
                if (wst[col] != stamp) { wst[col] = stamp; w[col] = 0.0; touch[nt++] = col; }
                w[col] -= sub;
            }
            /* rebuild rows[i]: keep cols<=k (L part + the mult at col k), then cols>k from w */
            int newnz = 0;
            for (int a = 0; a < rows[i].nz; a++) if (rows[i].ci[a] <= k) { rows[i].ci[newnz] = rows[i].ci[a]; rows[i].cv[newnz] = rows[i].cv[a]; newnz++; }
            int need = newnz + nt;
            if (need > rows[i].cap) {
                int nc = need * 2 + 4;
                int *ni = (int *)realloc(rows[i].ci, (size_t)nc * sizeof(int));
                double *nv = (double *)realloc(rows[i].cv, (size_t)nc * sizeof(double));
                if (!ni || !nv) { free(ni); free(nv); singular = 2; break; }
                rows[i].ci = ni; rows[i].cv = nv; rows[i].cap = nc;
            }
            for (int a = 0; a < nt; a++) { int col = touch[a]; double v = w[col]; if (v != 0.0) { rows[i].ci[newnz] = col; rows[i].cv[newnz] = v; newnz++; } }
            rows[i].nz = newnz;
        }
        if (singular) break;
    }
    free(w); free(wst); free(touch);
    if (singular) {
        for (int i = 0; i < n; i++) { free(rows[i].ci); free(rows[i].cv); }
        free(rows); splu_free(F);
        return NULL;
    }
    /* extract L (CSC, unit diagonal implicit) and U (CSR by position-row) */
    int *lcnt = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int *ucnt = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    if (!lcnt || !ucnt) { free(lcnt);free(ucnt); for(int i=0;i<n;i++){free(rows[i].ci);free(rows[i].cv);} free(rows); splu_free(F); return NULL; }
    for (int i = 0; i < n; i++)
        for (int a = 0; a < rows[i].nz; a++) { int col = rows[i].ci[a]; if (col < i) lcnt[col]++; else ucnt[i]++; }
    F->Lp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    F->Up = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!F->Lp || !F->Up) { free(lcnt);free(ucnt); for(int i=0;i<n;i++){free(rows[i].ci);free(rows[i].cv);} free(rows); splu_free(F); return NULL; }
    F->Lp[0] = 0; for (int j = 0; j < n; j++) F->Lp[j + 1] = F->Lp[j] + lcnt[j];
    F->Up[0] = 0; for (int i = 0; i < n; i++) F->Up[i + 1] = F->Up[i] + ucnt[i];
    int lnz = F->Lp[n], unz = F->Up[n];
    if (getenv("GMB_DBG_SPCHOL")) fprintf(stderr,"splu n=%d nnzA=%d nnzLU=%d fill=%.2f\n",n,Ap[n],lnz+unz,(double)(lnz+unz)/(Ap[n]>0?Ap[n]:1));
    F->Li = (int *)malloc((size_t)(lnz > 0 ? lnz : 1) * sizeof(int));
    F->Lx = (double *)malloc((size_t)(lnz > 0 ? lnz : 1) * sizeof(double));
    F->Ui = (int *)malloc((size_t)(unz > 0 ? unz : 1) * sizeof(int));
    F->Ux = (double *)malloc((size_t)(unz > 0 ? unz : 1) * sizeof(double));
    if (!F->Li || !F->Lx || !F->Ui || !F->Ux) { free(lcnt);free(ucnt); for(int i=0;i<n;i++){free(rows[i].ci);free(rows[i].cv);} free(rows); splu_free(F); return NULL; }
    int *lc = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    int *uc = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!lc || !uc) { free(lc);free(uc);free(lcnt);free(ucnt); for(int i=0;i<n;i++){free(rows[i].ci);free(rows[i].cv);} free(rows); splu_free(F); return NULL; }
    for (int j = 0; j < n; j++) lc[j] = F->Lp[j];
    for (int i = 0; i < n; i++) uc[i] = F->Up[i];
    for (int i = 0; i < n; i++)
        for (int a = 0; a < rows[i].nz; a++) {
            int col = rows[i].ci[a]; double v = rows[i].cv[a];
            if (col < i) { F->Li[lc[col]] = i; F->Lx[lc[col]] = v; lc[col]++; }
            else { F->Ui[uc[i]] = col; F->Ux[uc[i]] = v; uc[i]++; }
        }
    free(lc); free(uc); free(lcnt); free(ucnt);
    for (int i = 0; i < n; i++) { free(rows[i].ci); free(rows[i].cv); }
    free(rows);
    return F;
}

/* Solves A x = rhs, rhs overwritten with x.  A scratch copy receives the row
 * permutation, so the caller's buffer is read once (through perm) and written
 * once (through qperm) and the factor stays const -- socp.c factors the
 * augmented sparse system once per Newton iteration and solves it twice, the
 * affine direction and the corrector.  The forward sweep updates
 * the scratch in place as it walks the columns of L, which is why the unit
 * diagonal is never stored.
 * Returns -1 if some row of U has no diagonal slot; the caller's rhs is then
 * UNTOUCHED, still holding the right-hand side, so a -1 must never be read as
 * an answer. */
int splu_solve(const SpluFact *F, double *rhs) {
    if (!F || !rhs) return -1;
    int n = F->n;
    double *pb = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!pb) return -1;
    for (int k = 0; k < n; k++) pb[k] = rhs[F->perm[k]];
    /* forward: L y = pb (L unit-lower, CSC) */
    for (int j = 0; j < n; j++) {
        double yj = pb[j];
        for (int p = F->Lp[j]; p < F->Lp[j + 1]; p++) pb[F->Li[p]] -= F->Lx[p] * yj;
    }
    /* back: U x = y (U upper, CSR by position-row) */
    for (int i = n - 1; i >= 0; i--) {
        double diag = 0.0, sum = 0.0;
        for (int p = F->Up[i]; p < F->Up[i + 1]; p++) {
            if (F->Ui[p] == i) diag = F->Ux[p];
            else sum += F->Ux[p] * pb[F->Ui[p]];
        }
        if (diag == 0.0) { free(pb); return -1; }
        pb[i] = (pb[i] - sum) / diag;
    }
    for (int k = 0; k < n; k++) rhs[F->qperm[k]] = pb[k];
    free(pb);
    return 0;
}

/* NULL-safe, and safe on a HALF-BUILT struct: splu_factor calloc's the record
 * and calls this on its error paths, where only some of the arrays exist.  A
 * new field added to SpluFact has to be released here as well, or every one of
 * those paths leaks it. */
void splu_free(SpluFact *F) {
    if (!F) return;
    free(F->perm); free(F->qperm); free(F->Lp); free(F->Li); free(F->Lx); free(F->Up); free(F->Ui); free(F->Ux);
    free(F);
}

/* ============================ sparse LDL' ============================
 * K = L D L', K symmetric (lower triangle in CSC).  L unit lower triangular,
 * D block-diagonal with 1x1 entries of mixed sign and 2x2 blocks that absorb a
 * (numerically) zero diagonal -- the case the conic/SDP KKT presents on its
 * equality rows.  The 2x2 pivots are chosen STATICALLY: a greedy matching pairs
 * each zero-diagonal node with an off-diagonal neighbour, then a permutation
 * (stored in L->perm) makes every pair adjacent; the left-looking elimination
 * treats pairs as rank-2 blocks.  Nodes with a nonzero diagonal stay 1x1 (a
 * quasi-definite K therefore factors exactly as the 1x1 version did).  No
 * dynamic pivoting.  K is the lower triangle of a symmetric matrix (strictly
 * upper entries are ignored, so a full symmetric CSC is accepted too).
 * For the algorithm see Duff-Pralet-style static 2x2 pivoting; the oracle is
 * the backward error ||Kx-b||/(||K|||x||+||b||) on random indefinite matrices
 * with structural zero diagonals (test_primal.c T253). */
SpLdl *spldl_factor(int n, const int *Kp, const int *Ki, const double *Kx) {
    /* --- greedy symmetric matching on off-diagonal nonzeros --- */
    int *pair = (int *)malloc((size_t)n * sizeof(int));
    int *adjc = (int *)calloc((size_t)(n + 1), sizeof(int));
    for (int j = 0; j < n; j++) for (int p = Kp[j]; p < Kp[j+1]; p++) { int i = Ki[p]; if (i > j) { adjc[j+1]++; adjc[i+1]++; } }
    /* adjacency CSR, fully symmetric (both directions of each edge) */
    int *arp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    arp[0] = 0; for (int j = 0; j < n; j++) arp[j+1] = arp[j] + adjc[j+1];
    int *ari = (int *)malloc((size_t)(arp[n] > 0 ? arp[n] : 1) * sizeof(int));
    int *cur = (int *)calloc((size_t)n, sizeof(int));
    for (int j = 0; j < n; j++) for (int p = Kp[j]; p < Kp[j+1]; p++) { int i = Ki[p]; if (i > j) { ari[arp[j] + cur[j]++] = i; ari[arp[i] + cur[i]++] = j; } }
    free(cur);
    for (int k = 0; k < n; k++) pair[k] = -1;
    if (!getenv("NO2")) {
        double *diag = (double *)calloc((size_t)n, sizeof(double)), maxd = 0.0;
        for (int j = 0; j < n; j++) for (int p = Kp[j]; p < Kp[j+1]; p++) if (Ki[p] == j) { diag[j] = Kx[p]; if (fabs(Kx[p]) > maxd) maxd = fabs(Kx[p]); }
        double thr = 1e-12 * (maxd > 0 ? maxd : 1.0);
        for (int u = 0; u < n; u++) {
            if (pair[u] != -1) continue;
            if (fabs(diag[u]) > thr) continue;      /* nonzero diagonal -> 1x1 is fine */
            for (int q = arp[u]; q < arp[u+1]; q++) {
                int v = ari[q];
                if (v != u && pair[v] == -1) { pair[u] = v; pair[v] = u; break; }
            }
        }
        free(diag);
    }
    free(ari); free(arp); free(adjc);

    /* --- perm: pairs first (adjacent), then singletons --- */
    int *perm = (int *)malloc((size_t)n * sizeof(int));
    int *ip = (int *)malloc((size_t)n * sizeof(int));
    int pos = 0;
    for (int u = 0; u < n; u++) if (pair[u] != -1 && u < pair[u]) { perm[pos++] = u; perm[pos++] = pair[u]; }
    for (int u = 0; u < n; u++) if (pair[u] == -1) perm[pos++] = u;
    for (int k = 0; k < n; k++) ip[perm[k]] = k;
    /* static pair in permuted order: is k the first of a 2x2? */
    int *piv2 = (int *)calloc((size_t)n, sizeof(int));
    for (int k = 0; k + 1 < n; k++) {
        int u = perm[k], v = perm[k+1];
        if (pair[u] == v) { piv2[k] = 1; k++; }
    }

    /* --- permuted lower CSC of K --- */
    int *Pp = (int *)calloc((size_t)(n + 1), sizeof(int));
    for (int j = 0; j < n; j++) for (int p = Kp[j]; p < Kp[j+1]; p++) {
        int a = ip[Ki[p]], b = ip[j];      /* original (row=Ki[p]>=j) */
        int col = a < b ? a : b; Pp[col+1]++;
    }
    for (int j = 0; j < n; j++) Pp[j+1] += Pp[j];
    int pn = Pp[n];
    int *Pi = (int *)malloc((size_t)(pn > 0 ? pn : 1) * sizeof(int));
    double *Pv = (double *)malloc((size_t)(pn > 0 ? pn : 1) * sizeof(double));
    int *f = (int *)calloc((size_t)n, sizeof(int));
    for (int j = 0; j < n; j++) for (int p = Kp[j]; p < Kp[j+1]; p++) {
        int a = ip[Ki[p]], b = ip[j];
        int col = a < b ? a : b, row = a < b ? b : a;
        int q = Pp[col] + f[col]++; Pi[q] = row; Pv[q] = Kx[p];
    }
    free(f);

    /* --- left-looking SpLdl with static 2x2 blocks --- */
    int **ci = (int **)calloc((size_t)n, sizeof(int *));
    double **cv = (double **)calloc((size_t)n, sizeof(double *));
    int *cn = (int *)calloc((size_t)n, sizeof(int));
    int *cc = (int *)calloc((size_t)n, sizeof(int));
    int **ri = (int **)calloc((size_t)n, sizeof(int *));
    double **rv = (double **)calloc((size_t)n, sizeof(double *));
    int *rn = (int *)calloc((size_t)n, sizeof(int));
    int *rc = (int *)calloc((size_t)n, sizeof(int));
    double *w = (double *)calloc((size_t)n, sizeof(double));
    double *w2 = (double *)calloc((size_t)n, sizeof(double));
    double *d = (double *)calloc((size_t)n, sizeof(double));
    double *Doff = (double *)calloc((size_t)n, sizeof(double));
    int *mark = (int *)calloc((size_t)n, sizeof(int));
    int *bseen = (int *)calloc((size_t)n, sizeof(int));
    int *touched = (int *)malloc((size_t)n * sizeof(int));
    int *touched2 = (int *)malloc((size_t)n * sizeof(int));
#define ENSURE(P,V,CAP,NEED) do { if ((CAP) < (int)(NEED)) { int nc=(CAP)?(CAP):4; while(nc<(int)(NEED)) nc*=2; \
    int *a=(int*)realloc(P,(size_t)nc*sizeof(int)); double *b=(double*)realloc(V,(size_t)nc*sizeof(double)); \
    if(!a||!b){free(a);free(b);FAIL();} (P)=a;(V)=b;(CAP)=nc; } } while(0)

#define FAIL() do { ok = 0; goto done; } while (0)
    int ok = 1;
    int stamp = 0;
    for (int c = 0; c < n && ok; ) {
        /* ---- updated column c ---- */
        stamp++;
        int nt = 0;
        for (int p = Pp[c]; p < Pp[c+1]; p++) { int i = Pi[p]; if (i < c) continue;
            if (mark[i] != stamp) { mark[i] = stamp; touched[nt++] = i; w[i] = 0.0; } w[i] += Pv[p]; }
        for (int q = 0; q < rn[c]; q++) {
            int m = ri[c][q];
            int isblk = piv2[m] || (m > 0 && piv2[m-1]);
            if (isblk) {
                int f = piv2[m] ? m : m - 1;
                if (bseen[f] == stamp) continue;
                bseen[f] = stamp;
                double lf = 0.0, lf1 = 0.0;
                for (int q2 = 0; q2 < rn[c]; q2++) { if (ri[c][q2] == f) lf = rv[c][q2]; else if (ri[c][q2] == f+1) lf1 = rv[c][q2]; }
                double a2 = d[f], b2 = Doff[f], c2 = d[f+1];
                double u1 = a2*lf + b2*lf1, u2 = b2*lf + c2*lf1;
                for (int p = 0; p < cn[f]; p++) { int i = ci[f][p]; if (i < c) continue;
                    if (mark[i] != stamp) { mark[i] = stamp; touched[nt++] = i; w[i] = 0.0; } w[i] -= cv[f][p]*u1; }
                for (int p = 0; p < cn[f+1]; p++) { int i = ci[f+1][p]; if (i < c) continue;
                    if (mark[i] != stamp) { mark[i] = stamp; touched[nt++] = i; w[i] = 0.0; } w[i] -= cv[f+1][p]*u2; }
            } else {
                double u = d[m]*rv[c][q];
                for (int p = 0; p < cn[m]; p++) { int i = ci[m][p]; if (i < c) continue;
                    if (mark[i] != stamp) { mark[i] = stamp; touched[nt++] = i; w[i] = 0.0; } w[i] -= cv[m][p]*u; }
            }
        }

        if (!piv2[c]) {
            double dj = w[c];
            if (!(fabs(dj) > 1e-300)) FAIL();
            d[c] = dj;
            /* column c of L */
            ENSURE(ci[c], cv[c], cc[c], nt + 1);
            ci[c][0] = c; cv[c][0] = 1.0; cn[c] = 1;
            for (int q = 0; q < nt; q++) { int i = touched[q]; double v = w[i]; w[i] = 0.0; if (i == c) continue; v /= dj; if (v == 0.0) continue;
                ci[c][cn[c]] = i; cv[c][cn[c]] = v; cn[c]++;
                ENSURE(ri[i], rv[i], rc[i], rn[i] + 1);
                ri[i][rn[i]] = c; rv[i][rn[i]] = v; rn[i]++; }
            c += 1;
        } else {
            /* ---- need updated column c+1 too (subtract prior blocks m < c) ---- */
            int k = c + 1;
            stamp++;
            int nt2 = 0;
            for (int p = Pp[k]; p < Pp[k+1]; p++) { int i = Pi[p]; if (i < k) continue;
                if (mark[i] != stamp) { mark[i] = stamp; touched2[nt2++] = i; w2[i] = 0.0; } w2[i] += Pv[p]; }
            for (int q = 0; q < rn[k]; q++) {
                int m = ri[k][q];
                if (m >= c) continue;                /* only blocks strictly before c */
                int isblk = piv2[m] || (m > 0 && piv2[m-1]);
                if (isblk) {
                    int f = piv2[m] ? m : m - 1;
                    if (bseen[f] == stamp) continue;
                    bseen[f] = stamp;
                    double lf = 0.0, lf1 = 0.0;
                    for (int q2 = 0; q2 < rn[k]; q2++) { if (ri[k][q2] == f) lf = rv[k][q2]; else if (ri[k][q2] == f+1) lf1 = rv[k][q2]; }
                    double a2 = d[f], b2 = Doff[f], c2 = d[f+1];
                    double u1 = a2*lf + b2*lf1, u2 = b2*lf + c2*lf1;
                    for (int p = 0; p < cn[f]; p++) { int i = ci[f][p]; if (i < k) continue;
                        if (mark[i] != stamp) { mark[i] = stamp; touched2[nt2++] = i; w2[i] = 0.0; } w2[i] -= cv[f][p]*u1; }
                    for (int p = 0; p < cn[f+1]; p++) { int i = ci[f+1][p]; if (i < k) continue;
                        if (mark[i] != stamp) { mark[i] = stamp; touched2[nt2++] = i; w2[i] = 0.0; } w2[i] -= cv[f+1][p]*u2; }
                } else {
                    double u = d[m]*rv[k][q];
                    for (int p = 0; p < cn[m]; p++) { int i = ci[m][p]; if (i < k) continue;
                        if (mark[i] != stamp) { mark[i] = stamp; touched2[nt2++] = i; w2[i] = 0.0; } w2[i] -= cv[m][p]*u; }
                }
            }
            double a = w[c], b = w[k], cdiag = w2[k];
            double det = a*cdiag - b*b;
            if (!(fabs(det) > 1e-300)) FAIL();
            d[c] = a; d[k] = cdiag; Doff[c] = b;
            /* columns c and k of L: below the pair only (i > k) */
            ENSURE(ci[c], cv[c], cc[c], nt + nt2 + 1);
            ENSURE(ci[k], cv[k], cc[k], nt + nt2 + 1);
            ci[c][0] = c; cv[c][0] = 1.0; cn[c] = 1;
            ci[k][0] = k; cv[k][0] = 1.0; cn[k] = 1;
            for (int i = k + 1; i < n; i++) {
                double wi = 0.0, w2i = 0.0;
                int found = 0;
                for (int q = 0; q < nt; q++) if (touched[q] == i) { wi = w[i]; found = 1; break; }
                int found2 = 0;
                for (int q = 0; q < nt2; q++) if (touched2[q] == i) { w2i = w2[i]; found2 = 1; break; }
                if (!found && !found2) continue;
                double lij = (wi*cdiag - w2i*b) / det;
                double lik = (w2i*a - wi*b) / det;
                if (lij != 0.0) { ci[c][cn[c]] = i; cv[c][cn[c]] = lij; cn[c]++;
                    ENSURE(ri[i], rv[i], rc[i], rn[i] + 1);
                    ri[i][rn[i]] = c; rv[i][rn[i]] = lij; rn[i]++; }
                if (lik != 0.0) { ci[k][cn[k]] = i; cv[k][cn[k]] = lik; cn[k]++;
                    ENSURE(ri[i], rv[i], rc[i], rn[i] + 1);
                    ri[i][rn[i]] = k; rv[i][rn[i]] = lik; rn[i]++; }
            }
            for (int q = 0; q < nt; q++) w[touched[q]] = 0.0;
            for (int q = 0; q < nt2; q++) w2[touched2[q]] = 0.0;
            c += 2;
        }
    }
done:
    if (!ok) {
        for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
        free(ci); free(cv); free(cn); free(cc); free(ri); free(rv); free(rn); free(rc);
        free(w); free(w2); free(d); free(Doff); free(mark); free(bseen); free(touched); free(touched2);
        free(Pp); free(Pi); free(Pv); free(perm); free(ip); free(piv2); free(pair);
        return NULL;
    }
    SpLdl *L = (SpLdl *)malloc(sizeof(SpLdl));
    int total = 0; for (int j = 0; j < n; j++) total += cn[j];
    L->n = n;
    L->Lp = (int *)malloc((size_t)(n+1)*sizeof(int));
    L->Li = (int *)malloc((size_t)(total>0?total:1)*sizeof(int));
    L->Lx = (double *)malloc((size_t)(total>0?total:1)*sizeof(double));
    L->D = (double *)malloc((size_t)n*sizeof(double));
    L->piv2 = (int *)malloc((size_t)n*sizeof(int));
    L->Doff = (double *)malloc((size_t)n*sizeof(double));
    L->perm = (int *)malloc((size_t)n*sizeof(int));
    int off = 0;
    for (int j = 0; j < n; j++) { L->Lp[j] = off; for (int p = 0; p < cn[j]; p++) { L->Li[off] = ci[j][p]; L->Lx[off] = cv[j][p]; off++; } }
    L->Lp[n] = off;
    for (int j = 0; j < n; j++) { L->D[j] = d[j]; L->piv2[j] = piv2[j]; L->Doff[j] = Doff[j]; L->perm[j] = perm[j]; }
    for (int q = 0; q < n; q++) { free(ci[q]); free(cv[q]); free(ri[q]); free(rv[q]); }
    free(ci); free(cv); free(cn); free(cc); free(ri); free(rv); free(rn); free(rc);
    free(w); free(w2); free(d); free(Doff); free(mark); free(bseen); free(touched); free(touched2);
    free(Pp); free(Pi); free(Pv); free(perm); free(ip); free(piv2); free(pair);
    return L;
#undef ENSURE
#undef FAIL
}

int spldl_solve(const SpLdl *L, double *rhs) {
    int n = L->n;
    double *buf = rhs;
    if (L->perm) { buf = (double *)malloc((size_t)n*sizeof(double)); for (int i = 0; i < n; i++) buf[i] = rhs[L->perm[i]]; }
    for (int k = 0; k < n; ) {
        if (L->piv2[k]) {
            double y0 = buf[k], y1 = buf[k+1];
            for (int p = L->Lp[k]; p < L->Lp[k+1]; p++) { int i = L->Li[p]; if (i > k+1) buf[i] -= L->Lx[p]*y0; }
            for (int p = L->Lp[k+1]; p < L->Lp[k+2]; p++) { int i = L->Li[p]; if (i > k+1) buf[i] -= L->Lx[p]*y1; }
            k += 2;
        } else {
            double yk = buf[k];
            for (int p = L->Lp[k]; p < L->Lp[k+1]; p++) { int i = L->Li[p]; if (i > k) buf[i] -= L->Lx[p]*yk; }
            k += 1;
        }
    }
    for (int k = 0; k < n; ) {
        if (L->piv2[k]) {
            double a = L->D[k], b = L->Doff[k], c = L->D[k+1];
            double det = a*c - b*b; if (!(fabs(det) > 0.0)) return -1;
            double inv = 1.0/det, y0 = buf[k], y1 = buf[k+1];
            buf[k] = (y0*c - y1*b)*inv; buf[k+1] = (y1*a - y0*b)*inv; k += 2;
        } else { if (L->D[k] == 0.0) return -1; buf[k] /= L->D[k]; k += 1; }
    }
    for (int k = n-1; k >= 0; ) {
        if (k > 0 && L->piv2[k-1]) {
            double s0 = buf[k-1], s1 = buf[k];
            for (int p = L->Lp[k-1]; p < L->Lp[k]; p++) { int i = L->Li[p]; if (i > k) s0 -= L->Lx[p]*buf[i]; }
            for (int p = L->Lp[k]; p < L->Lp[k+1]; p++) { int i = L->Li[p]; if (i > k) s1 -= L->Lx[p]*buf[i]; }
            buf[k-1] = s0; buf[k] = s1; k -= 2;
        } else {
            double s = buf[k];
            for (int p = L->Lp[k]; p < L->Lp[k+1]; p++) { int i = L->Li[p]; if (i > k) s -= L->Lx[p]*buf[i]; }
            buf[k] = s; k -= 1;
        }
    }
    if (L->perm) { for (int i = 0; i < n; i++) rhs[L->perm[i]] = buf[i]; free(buf); }
    return 0;
}

void spldl_free(SpLdl *L) { if (!L) return; free(L->Lp); free(L->Li); free(L->Lx); free(L->D); free(L->piv2); free(L->Doff); free(L->perm); free(L); }

/* ================= dense LDL^T with Bunch-Kaufman pivoting =================
 * For a symmetric indefinite K with zero-diagonal rows (the conic KKT Esoc
 * block): a 1x1 LDL has no pivot there, a 2x2 block does.  Symmetric
 * permutations, so the solve is forward/block-diagonal/backward plus the
 * permutation. */
SpBK *dmat_ldl_bk(int n, const double *A) {
    if (n <= 0 || !A) return NULL;
    SpBK *F = (SpBK *)calloc(1, sizeof(SpBK));
    if (!F) return NULL;
    F->n = n;
    F->perm = (int *)malloc((size_t)n * sizeof(int));
    F->piv2 = (int *)calloc((size_t)n, sizeof(int));
    F->LU = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!F->perm || !F->piv2 || !F->LU) { dmat_ldl_bk_free(F); return NULL; }
    memcpy(F->LU, A, (size_t)n * (size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) F->perm[k] = k;
    const double alpha = (1.0 + sqrt(17.0)) / 8.0;
    double *L = F->LU;
    int k = 0;
    while (k < n) {
        double mu = 0.0; int r = -1;
        for (int i = k + 1; i < n; i++) { double a = fabs(L[(size_t)i * n + k]); if (a > mu) { mu = a; r = i; } }
        if (fabs(L[(size_t)k * n + k]) >= alpha * mu || r < 0) {
            double d = L[(size_t)k * n + k];
            if (fabs(d) < 1e-300) { dmat_ldl_bk_free(F); return NULL; }
            for (int i = k + 1; i < n; i++) L[(size_t)i * n + k] /= d;
            for (int i = k + 1; i < n; i++) { double li = L[(size_t)i * n + k];
                for (int j = k + 1; j <= i; j++) L[(size_t)i * n + j] -= li * d * L[(size_t)j * n + k]; }
            F->piv2[k] = 0; k += 1;
        } else {
            if (r != k + 1) {   /* symmetric swap k+1 <-> r */
                for (int j = 0; j < n; j++) { double t = L[(size_t)(k+1)*n+j]; L[(size_t)(k+1)*n+j] = L[(size_t)r*n+j]; L[(size_t)r*n+j] = t; }
                for (int i = 0; i < n; i++) { double t = L[(size_t)i*n+(k+1)]; L[(size_t)i*n+(k+1)] = L[(size_t)i*n+r]; L[(size_t)i*n+r] = t; }
                int t = F->perm[k+1]; F->perm[k+1] = F->perm[r]; F->perm[r] = t;
            }
            double a = L[(size_t)k*n+k], b = L[(size_t)(k+1)*n+k], c = L[(size_t)(k+1)*n+(k+1)];
            double det = a * c - b * b;
            if (fabs(det) < 1e-300) { dmat_ldl_bk_free(F); return NULL; }
            double inv = 1.0 / det;
            for (int i = k + 2; i < n; i++) {
                double u1 = L[(size_t)i*n+k], u2 = L[(size_t)i*n+(k+1)];
                L[(size_t)i*n+k]     = (u1*c - u2*b) * inv;
                L[(size_t)i*n+(k+1)] = (u2*a - u1*b) * inv;
            }
            for (int i = k + 2; i < n; i++) {
                double li1 = L[(size_t)i*n+k], li2 = L[(size_t)i*n+(k+1)];
                for (int j = k + 2; j <= i; j++) {
                    double lj1 = L[(size_t)j*n+k], lj2 = L[(size_t)j*n+(k+1)];
                    L[(size_t)i*n+j] -= li1*(a*lj1 + b*lj2) + li2*(b*lj1 + c*lj2);
                }
            }
            F->piv2[k] = 1; F->piv2[k+1] = 0; k += 2;
        }
    }
    return F;
}

int dmat_ldl_bk_solve(const SpBK *F, double *rhs) {
    if (!F || !rhs) return -1;
    int n = F->n; const double *L = F->LU;
    double *b = (double *)malloc((size_t)n * sizeof(double));
    if (!b) return -1;
    for (int k = 0; k < n; k++) b[k] = rhs[F->perm[k]];
    /* forward L y = b: a 2x2 pivot's two columns have NO internal L entry, so
     * the block is advanced/consumed together (L[k+1][k] is D's off-diagonal). */
    for (int k = 0; k < n; ) {
        if (F->piv2[k]) {
            double y0 = b[k], y1 = b[k+1];
            for (int i = k + 2; i < n; i++) b[i] -= L[(size_t)i*n+k]*y0 + L[(size_t)i*n+(k+1)]*y1;
            k += 2;
        } else { double yk = b[k]; for (int i = k + 1; i < n; i++) b[i] -= L[(size_t)i*n+k]*yk; k += 1; }
    }
    for (int k = 0; k < n; ) {
        if (F->piv2[k]) {
            double a = L[(size_t)k*n+k], bb = L[(size_t)(k+1)*n+k], c = L[(size_t)(k+1)*n+(k+1)];
            double det = a * c - bb * bb; if (fabs(det) < 1e-300) { free(b); return -1; }
            double inv = 1.0 / det, y0 = b[k], y1 = b[k+1];
            b[k] = (y0*c - y1*bb) * inv; b[k+1] = (y1*a - y0*bb) * inv; k += 2;
        } else {
            if (L[(size_t)k*n+k] == 0.0) { free(b); return -1; }
            b[k] /= L[(size_t)k*n+k]; k += 1;
        }
    }
    /* backward L' x = z, same block-atomic treatment */
    for (int k = n - 1; k >= 0; ) {
        if (k > 0 && F->piv2[k-1]) {
            double s0 = b[k-1], s1 = b[k];
            for (int i = k + 1; i < n; i++) { s0 -= L[(size_t)i*n+(k-1)]*b[i]; s1 -= L[(size_t)i*n+k]*b[i]; }
            b[k-1] = s0; b[k] = s1; k -= 2;
        } else { double s = b[k]; for (int i = k + 1; i < n; i++) s -= L[(size_t)i*n+k]*b[i]; b[k] = s; k -= 1; }
    }
    for (int k = 0; k < n; k++) rhs[F->perm[k]] = b[k];
    free(b);
    return 0;
}

void dmat_ldl_bk_free(SpBK *F) {
    if (!F) return;
    free(F->perm); free(F->piv2); free(F->LU); free(F);
}
