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

/* multi_period.c - two-period portfolio with transaction costs.
 *
 *   max  sum_{t=0,1} mu_t' w_t - c * ( |w_0 - w_init|_1 + |w_1 - w_0|_1 )
 *   s.t. e'w_t = 1, w_t >= 0,
 * with the absolute values linearised by z_t >= ±(w_t - w_{t-1}), z_t >= 0.
 *
 * Hand case: mu = (0.5, 0.1) in both periods, w_init = (0.5, 0.5), c = 0.1.
 * Each period wants w = (1, 0) (the higher-return asset). The first move from
 * (0.5, 0.5) to (1, 0) costs 0.1 in turnover and gains 0.2 in return, so it is
 * taken; the second period is already there and pays nothing. Value:
 * (0.5 + 0.5) - 0.1 = 0.9.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double mu[2] = {0.5, 0.1};
    static const double w0init[2] = {0.5, 0.5};
    const double cost = 0.1;
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    /* vars: w0(0,1), w1(2,3), z0(4,5), z1(6,7) */
    PRIMAL_appendvars(t, 8);
    PRIMAL_appendcons(t, 10);
    for (int j = 0; j < 4; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int j = 4; j < 8; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = 0; j < 2; j++) {
        PRIMAL_putcj(t, j, mu[j]);
        PRIMAL_putcj(t, 2 + j, mu[j]);
        PRIMAL_putcj(t, 4 + j, -cost);
        PRIMAL_putcj(t, 6 + j, -cost);
    }
    /* e'w_0 = 1, e'w_1 = 1 */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putarow(t, 1, 2, (int[]){2, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);
    /* z0 >= w0 - w_init  ->  w0 - z0 <= w_init */
    for (int j = 0; j < 2; j++) {
        PRIMAL_putarow(t, 2 + j, 2, (int[]){j, 4 + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 2 + j, PRIMAL_BK_UP, -INFINITY, w0init[j]);
        PRIMAL_putarow(t, 4 + j, 2, (int[]){j, 4 + j}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, 4 + j, PRIMAL_BK_UP, -INFINITY, -w0init[j]);
    }
    /* z1 >= ±(w1 - w0) */
    for (int j = 0; j < 2; j++) {
        PRIMAL_putarow(t, 6 + j, 3, (int[]){2 + j, j, 6 + j},
                       (double[]){1.0, -1.0, -1.0});
        PRIMAL_putconbound(t, 6 + j, PRIMAL_BK_UP, -INFINITY, 0.0);
        PRIMAL_putarow(t, 8 + j, 3, (int[]){2 + j, j, 6 + j},
                       (double[]){-1.0, 1.0, -1.0});
        PRIMAL_putconbound(t, 8 + j, PRIMAL_BK_UP, -INFINITY, 0.0);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, x[8] = {0};
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    double turnover0 = fabs(x[0] - w0init[0]) + fabs(x[1] - w0init[1]);
    double turnover1 = fabs(x[2] - x[0]) + fabs(x[3] - x[1]);
    int ok = fabs(obj - 0.9) < 1e-7 &&
             fabs(x[0] - 1.0) < 1e-6 && fabs(x[1]) < 1e-6 &&
             fabs(x[2] - 1.0) < 1e-6 && fabs(x[3]) < 1e-6 &&
             fabs(turnover0 - 1.0) < 1e-6 && fabs(turnover1) < 1e-6;
    printf("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (atteso 0.9)\n",
           x[0], x[1], x[2], x[3], obj);
    printf("turnover = %.4f poi %.4f (atteso 1.0, 0.0)\n", turnover0, turnover1);
    printf("%s (multi-periodo: il secondo periodo e' gia' sul target)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
