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

/* primal_svm.c - soft-margin primal SVM (MOSEK Cookbook 11.2, primal_svm.cc).
 *
 *   minimize   t + c * sum_i xi_i
 *   subject to y_i * ( <X_i, w> - b ) + xi_i >= 1     (each sample)
 *              xi_i >= 0
 *              (t, 1, w) in RQUAD   =>  2*t >= ||w||^2,   i.e. t >= ||w||^2 / 2.
 *
 * The MOSEK example draws X from a random normal distribution (std::mt19937(0)
 * + std::normal_distribution, which is not portable across standard libraries),
 * so it has no reproducible value; here the same technique is exercised on two
 * hand-derived instances:
 *
 *   A) separable diamond, X = (2,0),(0,2),(-2,0),(0,-2), y = +1,+1,-1,-1,
 *      c = 1.  Symmetry gives b = 0 and the margin constraints are
 *      |2*w_j| >= 1 -> w = (1/2,1/2), so t = (1/2)(1/4+1/4) = 1/4 and xi = 0:
 *      optimum 1/4, w = (1/2,1/2), b = 0.
 *   B) soft margin: two samples at the same feature with opposite labels,
 *      X = (0),(0), y = +1,-1, c = 1.  No w helps (w = 0), b = 0 and each
 *      slack saturates at 1: optimum t + c*(1+1) = 2, xi = (1,1).
 *
 * Usage: primal_svm   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve one SVM.  Returns 0 on success.  w has nim entries; xi has 0..nim-1. */
static int svm(int nd, int nim, const double X[][2], const double *y, double c,
               double *obj, double *w, double *b) {
    int T = 0, B = 1, W = 2, XI = 2 + nd, ONE = 2 + nd + nim, NV = 3 + nd + nim;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, nim);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMAL_putvarbound(t, T, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, B, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 0; j < nd; j++) PRIMAL_putvarbound(t, W + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nim; i++) PRIMAL_putvarbound(t, XI + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putcj(t, T, 1.0);
    for (int i = 0; i < nim; i++) PRIMAL_putcj(t, XI + i, c);
    /* (t, 1, w...) in RQUAD -> t >= ||w||^2 / 2 */
    {
        int mem[2 + 2]; mem[0] = T; mem[1] = ONE;
        for (int j = 0; j < nd; j++) mem[2 + j] = W + j;
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 2 + nd, mem);
    }
    for (int i = 0; i < nim; i++) {                  /* y_i(<X_i,w>-b) + xi_i >= 1 */
        int sub[5]; double val[5]; int nn = 0;
        for (int j = 0; j < nd; j++) if (X[i][j] != 0.0) { sub[nn] = W + j; val[nn] = y[i] * X[i][j]; nn++; }
        sub[nn] = B; val[nn] = -y[i]; nn++;
        sub[nn] = XI + i; val[nn] = 1.0; nn++;
        PRIMAL_putarow(t, i, nn, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_LO, 1.0, INFINITY);
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[16] = {0};
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, obj);
        *b = x[B];
        for (int j = 0; j < nd; j++) w[j] = x[W + j];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}

int main(void) {
    int bad = 0;
    /* A) separable diamond */
    {
        const double X[4][2] = {{2, 0}, {0, 2}, {-2, 0}, {0, -2}};
        const double y[4] = {1, 1, -1, -1};
        double obj = 0, w[2] = {0}, b = 0;
        int ok = svm(2, 4, X, y, 1.0, &obj, w, &b) == 0;
        int good = ok && fabs(obj - 0.25) < 1e-7 && fabs(w[0] - 0.5) < 1e-6 &&
                   fabs(w[1] - 0.5) < 1e-6 && fabs(b) < 1e-6;
        printf("SVM A (separable, c=1): obj=%.8f w=(%.4f,%.4f) b=%.2e  (atteso 0.25, w=(1/2,1/2), b=0)  %s\n",
               obj, w[0], w[1], b, good ? "OK" : "FAIL");
        bad |= !good;
    }
    /* B) soft margin (same feature, opposite labels) */
    {
        const double X[2][2] = {{0, 0}, {0, 0}};
        const double y[2] = {1, -1};
        double obj = 0, w[2] = {0}, b = 0;
        int ok = svm(1, 2, X, y, 1.0, &obj, w, &b) == 0;
        int good = ok && fabs(obj - 2.0) < 1e-7 && fabs(w[0]) < 1e-6 && fabs(b) < 1e-6;
        printf("SVM B (soft, c=1):      obj=%.8f w=(%.4f) b=%.2e      (atteso 2, w=0, b=0)          %s\n",
               obj, w[0], b, good ? "OK" : "FAIL");
        bad |= !good;
    }
    return bad;
}
