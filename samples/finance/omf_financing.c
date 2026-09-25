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

/* omf_financing.c - the short-term financing LP of "Optimization Methods in
 * Finance" (Cornuejols & Tutuncu, 2007), Section 3.1.1.
 *
 * A company faces monthly cash requirements (k$) of
 *   Jan..Jun:  150, 100, -200, 200, -50, -300.
 * It can draw on a line of credit (x_i, 1% per month, limit 100) and issue
 * commercial paper in Jan-Mar only (y_i, 2% for three months). Excess funds
 * earn 0.5% per month (z_i). Maximize the wealth v at the end of June.
 *
 * Model (x_i = credit-line balance, z_i = excess funds):
 *   max v
 *   s.t.  x1 + y1                                  - z1                =  150
 *         x2 + y2 - 1.01 x1        + 1.003 z1      - z2                =  100
 *         x3 + y3 - 1.01 x2        + 1.003 z2      - z3                = -200
 *         x4      - 1.02 y1 - 1.01 x3 + 1.003 z3  - z4                =  200
 *         x5      - 1.02 y2 - 1.01 x4 + 1.003 z4  - z5                =  -50
 *                 - 1.02 y3 - 1.01 x5 + 1.003 z5  - v                 = -300
 *         0 <= x_i <= 100,  y_i >= 0,  z_i >= 0.
 *
 * The book's sensitivity report (Section 3.3.1) gives shadow prices
 * u_Jan = -1.0373, u_Feb = -1.030, u_Mar = -1.020. (The primal solution is
 * not unique: the book reports x2 = 50.98, this solver finds another optimal
 * vertex with the same objective value and the same dual solution.)
 *
 * The sample solves the LP and checks the published shadow prices.
 *
 * Usage: omf_financing   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0..4 = x1..x5, 5..7 = y1..y3, 8..12 = z1..z5, 13 = v ; rows 0..5 */
    PRIMAL_maketask(env, 6, 14, &t);
    for (int i = 0; i < 5; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_RA, 0.0, 100.0);
    for (int i = 5; i < 13; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 13, PRIMAL_BK_FR, -INFINITY, INFINITY);   /* v */
    PRIMAL_putcj(t, 13, -1.0);                                      /* max v */

    const double rhs[6] = {150, 100, -200, 200, -50, -300};
    {
        int i0[] = {0, 5, 8};        double v0[] = {1.0, 1.0, -1.0};
        int i1[] = {1, 6, 0, 8, 9};  double v1[] = {1.0, 1.0, -1.01, 1.003, -1.0};
        int i2[] = {2, 7, 1, 9, 10}; double v2[] = {1.0, 1.0, -1.01, 1.003, -1.0};
        int i3[] = {3, 5, 2, 10, 11};double v3[] = {1.0, -1.02, -1.01, 1.003, -1.0};
        int i4[] = {4, 6, 3, 11, 12};double v4[] = {1.0, -1.02, -1.01, 1.003, -1.0};
        int i5[] = {7, 4, 12, 13};   double v5[] = {-1.02, -1.01, 1.003, -1.0};
        int n[6] = {3, 5, 5, 5, 5, 4};
        int *ii[6] = {i0, i1, i2, i3, i4, i5};
        double *vv[6] = {v0, v1, v2, v3, v4, v5};
        for (int k = 0; k < 6; k++) {
            PRIMAL_putarow(t, k, n[k], ii[k], vv[k]);
            PRIMAL_putconbound(t, k, PRIMAL_BK_FX, rhs[k], rhs[k]);
        }
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double obj, x[14], y[6];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        PRIMAL_gety(t, PRIMAL_SOL_ITR, y);
        double v = -obj;   /* the objective is max v, reported as min -v */
        /* book shadow prices Jan/Feb/Mar = -1.0373/-1.030/-1.020 */
        ok = fabs(fabs(y[0]) - 1.0373) < 1e-3 &&
             fabs(fabs(y[1]) - 1.030)  < 1e-3 &&
             fabs(fabs(y[2]) - 1.020)  < 1e-3;
        printf("omf_financing  v=%.4f  x=(%.2f,%.4f,%.2f,%.2f,%.2f)  "
               "uJan=%.4f uFeb=%.4f uMar=%.4f  %s\n",
               v, x[0], x[1], x[2], x[3], x[4],
               fabs(y[0]), fabs(y[1]), fabs(y[2]), ok ? "OK" : "FAIL");
    } else {
        printf("omf_financing  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
