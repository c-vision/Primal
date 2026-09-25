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

/* omf_bl.c - Black-Litterman portfolio optimization of "Optimization Methods
 * in Finance" (Cornuejols & Tutuncu, 2007), Section 8.2 / Example 8.1.
 *
 * Uses the equilibrium returns and covariance of Section 8.1.1 (US stocks,
 * bonds, money market; Tables 8.1-8.3):
 *
 *   pi    = (0.1073, 0.0737, 0.0627)              (market equilibrium)
 *   Sigma = [[0.02778, 0.00387, 0.00021],
 *            [0.00387, 0.01112,-0.00020],
 *            [0.00021,-0.00020, 0.00115]]
 *
 * Two investor views (tau = 0.1):
 *   mu_M      = 0.02   strong view, omega_1 = 0.00001
 *   mu_S-mu_B = 0.05   weaker view, omega_2 = 0.001
 * i.e. P = [[0,0,1],[1,-1,0]], q = (0.02, 0.05), Omega = diag(1e-5, 1e-3).
 *
 * The posterior mean is (8.7):
 *   mu_bar = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
 *            [(tau Sigma)^-1 pi + P' Omega^-1 q]
 * which the book gives as mu_bar = (0.1177, 0.0751, 0.0234). The resulting
 * Markowitz QP (min x'Sigma x s.t. mu_bar'x >= R, sum x = 1, x >= 0) has the
 * efficient portfolios of Table 8.5:
 *   0.040  0.0012  0.08 0.17 0.75
 *   0.045  0.0015  0.11 0.21 0.68
 *   0.050  0.0020  0.15 0.24 0.61
 *   0.055  0.0025  0.18 0.28 0.54
 *   0.060  0.0032  0.22 0.31 0.47
 *
 * The sample computes mu_bar from the formula, checks it against the book,
 * then solves the QP for the five targets and checks Table 8.5.
 *
 * Usage: omf_bl   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static const double PI_0[3] = {0.1073, 0.0737, 0.0627};
static const double SIG[3][3] = {
    {0.02778, 0.00387, 0.00021},
    {0.00387, 0.01112, -0.00020},
    {0.00021, -0.00020, 0.00115}};
static const double MU_BOOK[3] = {0.1177, 0.0751, 0.0234};
static const double R[5]   = {0.040, 0.045, 0.050, 0.055, 0.060};
static const double VAR[5] = {0.0012, 0.0015, 0.0020, 0.0025, 0.0032};
static const double XS[5]  = {0.08, 0.11, 0.15, 0.18, 0.22};
static const double XB[5]  = {0.17, 0.21, 0.24, 0.28, 0.31};

static void invert3(const double a[3][3], double inv[3][3]) {
    double d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
             - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
             + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d;
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d;
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d;
}

static int solve(int k, const double mu[3]) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putqobj(t, 6, (int[]){0, 0, 0, 1, 1, 2}, (int[]){0, 1, 2, 1, 2, 2},
                   (double[]){2 * SIG[0][0], 2 * SIG[0][1], 2 * SIG[0][2],
                              2 * SIG[1][1], 2 * SIG[1][2], 2 * SIG[2][2]});
    PRIMAL_putcj(t, 0, 0.0); PRIMAL_putcj(t, 1, 0.0); PRIMAL_putcj(t, 2, 0.0);
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 2}, (double[]){mu[0], mu[1], mu[2]});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, R[k], INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double var, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &var);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(var - VAR[k]) < 1e-4 && fabs(x[0] - XS[k]) < 0.02 &&
             fabs(x[1] - XB[k]) < 0.02;
        printf("omf_bl R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) "
               "(book %.2f,%.2f,%.2f) %s\n", R[k], var, VAR[k], x[0], x[1], x[2],
               XS[k], XB[k], 1.0 - XS[k] - XB[k], ok ? "OK" : "FAIL");
    } else {
        printf("omf_bl R=%.3f rc=%d FAIL\n", R[k], (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    const double tau = 0.1;
    const double om[2] = {0.00001, 0.001};
    const double q[2] = {0.02, 0.05};
    const int P[2][3] = {{0, 0, 1}, {1, -1, 0}};

    double sig[3][3], sinv[3][3], A[3][3], Ai[3][3], b[3], mu[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) sig[i][j] = tau * SIG[i][j];
    invert3(sig, sinv);
    /* A = (tau Sigma)^-1 + P' Omega^-1 P ; b = (tau Sigma)^-1 pi + P' Omega^-1 q */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 2; k++) s += P[k][i] * (1.0 / om[k]) * P[k][j];
            A[i][j] = sinv[i][j] + s;
        }
        double s = 0.0;
        for (int k = 0; k < 2; k++) s += P[k][i] * (1.0 / om[k]) * q[k];
        double t2 = 0.0;
        for (int j = 0; j < 3; j++) t2 += sinv[i][j] * PI_0[j];
        b[i] = t2 + s;
    }
    invert3(A, Ai);
    for (int i = 0; i < 3; i++) {
        mu[i] = 0.0;
        for (int j = 0; j < 3; j++) mu[i] += Ai[i][j] * b[j];
    }
    printf("omf_bl mu_bar=(%.4f,%.4f,%.4f) (book %.4f,%.4f,%.4f) %s\n",
           mu[0], mu[1], mu[2], MU_BOOK[0], MU_BOOK[1], MU_BOOK[2],
           (fabs(mu[0] - MU_BOOK[0]) < 1e-3 && fabs(mu[1] - MU_BOOK[1]) < 1e-3 &&
            fabs(mu[2] - MU_BOOK[2]) < 1e-3) ? "OK" : "FAIL");

    int ok = 1;
    for (int k = 0; k < 5; k++) ok &= solve(k, mu);
    return ok ? 0 : 1;
}
