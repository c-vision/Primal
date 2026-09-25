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

/* binary_quadratic.c - binary quadratic problem via its Shor SDP relaxation.
 *
 * Source: MOSEK/Tutorials, binary-quadratic/binquad.py.  The notebook solves
 *
 *   minimize  x'Qx + c'x + R      over x in {0,1}^n
 *
 * with a branch-and-bound whose lower bound is the Shor semidefinite
 * relaxation, obtained by lifting X = x x' to an (n+1)x(n+1) matrix
 *
 *   Z = [ X  x ]  >= 0,   diag(X) = x,   Z[n,n] = 1
 *       [ x' 1 ]
 *
 * and relaxing to  min R + c'x + <Q,X>.  This sample solves exactly that
 * relaxation with one (n+1)x(n+1) semidefinite bar (all terms go through
 * putbarcj/putbaraij) and checks it is a valid lower bound against a brute
 * force over the 2^n binary points.
 *
 * Deterministic instance n=4:  Q = tridiag(-2, 2, -2), c = 0, R = 0, whose
 * optimum is x = (1,1,1,1) with value -4.
 *
 * Usage: binary_quadratic   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"
#include "linalg.h"

#define N 4

static const double Q[N][N] = {
    { 2, -2,  0,  0}, {-2,  2, -2,  0},
    { 0, -2,  2, -2}, { 0,  0, -2,  2}};
static const double C[N] = {0, 0, 0, 0};

static double fval(const int *x) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) s += Q[i][j] * x[i] * x[j];
        s += C[i] * x[i];
    }
    return s;
}

/* Shor SDP relaxation; returns the optimal value and writes Z ((N+1)^2, row-major) */
static int shor(double *val_out, double *Z_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    int dim = N + 1;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, dim);              /* N diag=lastcol, 1 lastdiag=1 */

    {   /* objective symmat: Q on the top-left block, c/2 on the (i,n) entries */
        int si[(N + 1) * (N + 2) / 2], sj[(N + 1) * (N + 2) / 2];
        double sv[(N + 1) * (N + 2) / 2]; int k = 0;
        for (int i = 0; i < N; i++) { si[k] = i; sj[k] = i; sv[k] = Q[i][i]; k++; }
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) { si[k] = i; sj[k] = j; sv[k] = Q[i][j]; k++; }
        for (int i = 0; i < N; i++) { si[k] = i; sj[k] = N; sv[k] = 0.5 * C[i]; k++; }
        int idx;
        PRIMAL_appendsparsesymmat(t, dim, k, si, sj, sv, &idx);
        PRIMAL_putbarcj(t, 0, 1, &idx, (double[]){1.0});
    }
    for (int i = 0; i < N; i++) {           /* Z_ii - Z_iN = 0 */
        int m1, m2;
        int a = i, b = i; PRIMAL_appendsparsesymmat(t, dim, 1, &a, &b, (double[]){1.0}, &m1);
        int c = i, d = N; PRIMAL_appendsparsesymmat(t, dim, 1, &c, &d, (double[]){0.5}, &m2);
        PRIMAL_putbaraij(t, i, 0, 2, (int[]){m1, m2}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 0.0, 0.0);
    }
    {   int a = N, b = N, m;
        PRIMAL_appendsparsesymmat(t, dim, 1, &a, &b, (double[]){1.0}, &m);
        PRIMAL_putbaraij(t, N, 0, 1, &m, (double[]){1.0});
        PRIMAL_putconbound(t, N, PRIMAL_BK_FX, 1.0, 1.0);
    }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, val_out);
        if (Z_out) PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, Z_out);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double sdp = 0.0, Z[(N + 1) * (N + 1)];
    int ok = shor(&sdp, Z);

    /* exact optimum by brute force over 2^N */
    double best = 1e300; int bx[N] = {0};
    for (int mask = 0; mask < (1 << N); mask++) {
        int x[N];
        for (int i = 0; i < N; i++) x[i] = (mask >> i) & 1;
        double f = fval(x);
        if (f < best) { best = f; for (int i = 0; i < N; i++) bx[i] = x[i]; }
    }
    double ev[(N + 1)], evec[(N + 1) * (N + 1)];
    dmat_eig_jacobi(N + 1, Z, ev, evec);
    double mineig = ev[0];
    for (int i = 1; i <= N; i++) if (ev[i] < mineig) mineig = ev[i];

    int okall = ok && sdp <= best + 1e-6 && mineig > -1e-7;
    printf("binary_quadratic  n=%d\n", N);
    printf("  Shor SDP lower bound = %.8f   (Z PSD, min_eig=%.2e)\n", sdp, mineig);
    printf("  exact optimum        = %.8f   at x=(%d,%d,%d,%d)\n",
           best, bx[0], bx[1], bx[2], bx[3]);
    printf("  gap = %.8f\n", best - sdp);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
