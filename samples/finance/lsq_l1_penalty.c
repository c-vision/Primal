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

/* lsq_l1_penalty.c - positivity least squares with a weighted L1 penalty
 * (tschm/MosekRegression, mosek_tools.solver.lsq_pos_l1_penalty).
 *
 *   min  ||X w - rhs||_2^2  +  sum_i gamma_i |w_i - w0_i|
 *   s.t. e'w = 1,   w >= 0.
 *
 * gamma_i is a PER-ASSET cost multiplier (not a single lambda) and w0 a
 * reference portfolio, which is what distinguishes this model from the uniform
 * penalty of samples/finance/regression_regularized.c.  Conic form:
 *   squared residual  (1/2, v, Xw-rhs) in Qr      (one rotated cone)
 *   |gamma_i(w_i-w0_i)|  (t_i, p_i) in Q^2        (one cone per asset)
 *   objective min v + sum t_i.
 *
 * Independent checks:
 *   A. w0 = 0 and gamma_i = gamma uniform -> on the simplex w >= 0, e'w = 1 the
 *      L1 term is sum gamma*w_i = gamma*e'w = gamma exactly, so the model must
 *      reduce to the QP min ||Xw-rhs||^2 with objective (QP objective) + gamma
 *      and the same w.  Case A compares the two solves.
 *   B. M = 2 with a general gamma and w0: the simplex is one dimensional, so a
 *      fine scan of w_1 in [0,1] gives the optimum by brute force.
 *
 * Usage: lsq_l1_penalty   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* ---- conic model ---- */
static double l1_solve(int N, int M, const double *X, const double *b,
                       const double *gam, const double *w0, double *wout) {
    int W0 = 0, R0 = M, V = M + N, P0 = M + N + 1, T0 = M + N + 1 + M, HALF = M + N + 1 + 2 * M;
    int nv = HALF + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_appendcons(t, 1 + N + M);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < M; i++) PRIMAL_putvarbound(t, W0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, R0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, V, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < M; i++) PRIMAL_putvarbound(t, P0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < M; i++) PRIMAL_putvarbound(t, T0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, HALF, PRIMAL_BK_FX, 0.5, 0.5);
    /* e'w = 1 */
    { int sub[16]; double val[16]; for (int i = 0; i < M; i++) { sub[i] = W0 + i; val[i] = 1.0; }
      PRIMAL_putarow(t, 0, M, sub, val); PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0); }
    /* r - X w = -b */
    for (int i = 0; i < N; i++) {
        int sub[16]; double val[16]; int nn = 0;
        sub[nn] = R0 + i; val[nn] = 1.0; nn++;
        for (int j = 0; j < M; j++) if (X[i * M + j] != 0.0) { sub[nn] = W0 + j; val[nn] = -X[i * M + j]; nn++; }
        PRIMAL_putarow(t, 1 + i, nn, sub, val);
        PRIMAL_putconbound(t, 1 + i, PRIMAL_BK_FX, -b[i], -b[i]);
    }
    /* p_i - gamma_i w_i = -gamma_i w0_i */
    for (int i = 0; i < M; i++) {
        PRIMAL_putarow(t, 1 + N + i, 2, (int[]){P0 + i, W0 + i}, (double[]){1.0, -gam[i]});
        PRIMAL_putconbound(t, 1 + N + i, PRIMAL_BK_FX, -gam[i] * w0[i], -gam[i] * w0[i]);
    }
    /* (1/2, v, r) in Qr  ->  v >= ||r||^2 */
    { int mem[1 + 2 + 8]; mem[0] = HALF; mem[1] = V; for (int i = 0; i < N; i++) mem[2 + i] = R0 + i;
      PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, N + 2, mem); }
    /* (t_i, p_i) in Q^2  ->  t_i >= |p_i| */
    for (int i = 0; i < M; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 2, (int[]){T0 + i, P0 + i});
    PRIMAL_putcj(t, V, 1.0);
    for (int i = 0; i < M; i++) PRIMAL_putcj(t, T0 + i, 1.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) {
        double x[64]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < M; i++) wout[i] = x[W0 + i];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? obj : -1.0;
}

/* ---- QP reference: min ||Xw-rhs||^2 s.t. e'w=1, w>=0 ---- */
static double qp_solve(int N, int M, const double *X, const double *b, double *wout) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, M);
    PRIMAL_appendcons(t, 1);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < M; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
    { int sub[16]; double val[16]; for (int i = 0; i < M; i++) { sub[i] = i; val[i] = 1.0; }
      PRIMAL_putarow(t, 0, M, sub, val); PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0); }
    /* ||Xw-b||^2 = 0.5 w'(2 X'X) w - 2(X'b)'w + b'b. putqobjij takes only the
     * LOWER triangle (i >= j) and one value for the symmetric pair, so the loop
     * runs j <= i and the coefficient is the full 2 (X'X)_ij. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j <= i; j++) {
            double q = 0.0;
            for (int k = 0; k < N; k++) q += 2.0 * X[k * M + i] * X[k * M + j];
            PRIMAL_putqobjij(t, i, j, q);
        }
    }
    double cf = 0.0, c;
    for (int k = 0; k < N; k++) cf += b[k] * b[k];
    PRIMAL_putcfix(t, cf);
    for (int i = 0; i < M; i++) { c = 0.0; for (int k = 0; k < N; k++) c += -2.0 * b[k] * X[k * M + i]; PRIMAL_putcj(t, i, c); }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = -1.0;
    if (rc == PRIMAL_RES_OK) { double x[16]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < M; i++) wout[i] = x[i]; PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj); }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return obj;
}

int main(void) {
    int all = 1;
    printf("lsq_l1_penalty\n");
    /* ---- case A: w0=0, uniform gamma -> reduce to the QP + gamma ---- */
    {
        enum { N = 8, M = 3 };
        double X[N][M], b[N], w0[M] = {0, 0, 0}, gam[M], wl[M], wq[M];
        unsigned st = 987654321u;
        for (int i = 0; i < N; i++) {
            st = st * 1103515245u + 12345u; b[i] = (double)((st >> 16) & 0x3ff) / 1024.0;
            for (int j = 0; j < M; j++) {
                st = st * 1103515245u + 12345u;
                X[i][j] = 0.2 + (double)((st >> 16) & 0x3ff) / 1024.0;
            }
        }
        double gamma = 0.7; for (int i = 0; i < M; i++) gam[i] = gamma;
        double ol = l1_solve(N, M, (const double *)X, b, gam, w0, wl);
        double oq = qp_solve(N, M, (const double *)X, b, wq);
        double dw = 0; for (int i = 0; i < M; i++) dw = fmax(dw, fabs(wl[i] - wq[i]));
        int good = ol > 0 && oq > 0 && dw < 1e-6 && fabs(ol - (oq + gamma)) < 1e-6;
        printf("  A (w0=0, gamma=%.2f): L1 obj=%.8f  QP obj+gamma=%.8f  |dw|=%.2e  %s\n",
               gamma, ol, oq + gamma, dw, good ? "OK" : "FAIL");
        all &= good;
    }
    /* ---- case B: M=2, general gamma and w0 -> brute-force scan of w1 ---- */
    {
        enum { N = 5, M = 2 };
        double X[N][M], b[N], gam[M] = {0.3, 1.1}, w0[M] = {0.6, 0.4}, wl[M];
        X[0][0]=0.5; X[0][1]=1.0; b[0]=0.7;
        X[1][0]=1.0; X[1][1]=0.4; b[1]=0.5;
        X[2][0]=0.2; X[2][1]=0.9; b[2]=0.3;
        X[3][0]=0.8; X[3][1]=0.6; b[3]=0.9;
        X[4][0]=0.4; X[4][1]=0.4; b[4]=0.4;
        double ol = l1_solve(N, M, (const double *)X, b, gam, w0, wl);
        /* brute force over w1 in [0,1] */
        double best = INFINITY, bestw1 = 0;
        for (int k = 0; k <= 2000000; k++) {
            double w1 = (double)k / 2000000.0, w2 = 1.0 - w1;
            double r, f = 0;
            for (int i = 0; i < N; i++) { r = X[i][0]*w1 + X[i][1]*w2 - b[i]; f += r*r; }
            f += gam[0]*fabs(w1-w0[0]) + gam[1]*fabs(w2-w0[1]);
            if (f < best) { best = f; bestw1 = w1; }
        }
        int good = ol > 0 && fabs(ol - best) < 1e-5 && fabs(wl[0] - bestw1) < 1e-4;
        printf("  B (M=2, gamma=(%.1f,%.1f)): conic obj=%.8f  brute obj=%.8f  w1=%.6f (bf %.6f)  %s\n",
               gam[0], gam[1], ol, best, wl[0], bestw1, good ? "OK" : "FAIL");
        all &= good;
    }
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
