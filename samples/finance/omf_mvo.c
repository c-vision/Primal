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

/* omf_mvo.c - Markowitz mean-variance optimization (MVO) of "Optimization
 * Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 8.1.1 / Example
 * 8.1-8.2, using the historical data of Table 8.1 (US stocks, bonds, money
 * market, 1960-2003) and the geometric means / covariance matrix of Tables
 * 8.2-8.3.
 *
 *   minimize    x' Sigma x                       (portfolio variance)
 *   subject to  mu' x >= R                       (target return)
 *               xS + xB + xM = 1                 (budget)
 *               x >= 0                           (long only)
 *
 * with  mu    = (0.1073, 0.0737, 0.0627)         (geometric means)
 *       Sigma = [[0.02778, 0.00387, 0.00021],
 *                [0.00387, 0.01112,-0.00020],
 *                [0.00021,-0.00020, 0.00115]]
 *
 * The book solves it for R = 6.5% .. 10.5% in 0.5% steps and reports the
 * efficient portfolios of Table 8.3 (R, variance, stocks, bonds, MM):
 *   0.065  0.0010  0.03 0.10 0.87
 *   0.070  0.0014  0.13 0.12 0.75
 *   0.075  0.0026  0.24 0.14 0.62
 *   0.080  0.0044  0.35 0.16 0.49
 *   0.085  0.0070  0.45 0.18 0.37
 *
 * The sample solves the QP for those five targets and checks the book's
 * portfolios (rounded to 2 decimals) and variances.
 *
 * Usage: omf_mvo   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static const double MU[3] = {0.1073, 0.0737, 0.0627};
/* full covariance matrix Sigma */
static const double SIG[3][3] = {
    {0.02778, 0.00387, 0.00021},
    {0.00387, 0.01112, -0.00020},
    {0.00021, -0.00020, 0.00115}};

static const double R[5]    = {0.065, 0.070, 0.075, 0.080, 0.085};
static const double VAR[5]  = {0.0010, 0.0014, 0.0026, 0.0044, 0.0070};
static const double XS[5]   = {0.03, 0.13, 0.24, 0.35, 0.45};
static const double XB[5]   = {0.10, 0.12, 0.14, 0.16, 0.18};
static const double XM[5]   = {0.87, 0.75, 0.62, 0.49, 0.37};

static int solve(int k) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    /* objective x' Sigma x = 0.5 x' Q x with Q = 2 Sigma (upper triangle) */
    PRIMAL_putqobj(t, 6, (int[]){0, 0, 0, 1, 1, 2}, (int[]){0, 1, 2, 1, 2, 2},
                   (double[]){2 * SIG[0][0], 2 * SIG[0][1], 2 * SIG[0][2],
                              2 * SIG[1][1], 2 * SIG[1][2], 2 * SIG[2][2]});
    PRIMAL_putcj(t, 0, 0.0); PRIMAL_putcj(t, 1, 0.0); PRIMAL_putcj(t, 2, 0.0);
    /* budget: xS+xB+xM = 1 */
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* return: mu'x >= R */
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 2}, MU);
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, R[k], INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double var, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &var);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(var - VAR[k]) < 1e-4 &&
             fabs(x[0] - XS[k]) < 0.02 && fabs(x[1] - XB[k]) < 0.02 &&
             fabs(x[2] - XM[k]) < 0.02;
        printf("omf_mvo R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) "
               "(book %.2f,%.2f,%.2f) %s\n",
               R[k], var, VAR[k], x[0], x[1], x[2], XS[k], XB[k], XM[k],
               ok ? "OK" : "FAIL");
    } else {
        printf("omf_mvo R=%.3f rc=%d FAIL\n", R[k], (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    int ok = 1;
    for (int k = 0; k < 5; k++) ok &= solve(k);
    return ok ? 0 : 1;
}
