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

/* index_tracking.c - two-period index tracking with turnover.
 *
 *   min  sum_t |w_t - b_t|_1 + c * ( |w_0 - w_init|_1 + |w_1 - w_0|_1 )
 *   s.t. e'w_t = 1, w_t >= 0,
 * with the absolute values linearised by z_t >= ±(w_t - b_t) and
 * v_0 >= ±(w_0 - w_init), v_1 >= ±(w_1 - w_0).
 *
 * Hand case: b_0 = (0.6, 0.4), b_1 = (0.3, 0.7), w_init = (0.5, 0.5), c = 0.1.
 * Because e'w = 1, each L1 term is 2*|first component|, so period 0 alone
 * minimises 2|w0-b0| + 0.2|w0-w_init| at w0 = b0 = 0.6 (0 + 0.02), and
 * period 1 minimises 2|w1-b1| + 0.2|w1-w0| at w1 = b1 = 0.3 (0 + 0.06).
 * Value: 0.02 + 0.06 = 0.08, w_0 = (0.6, 0.4), w_1 = (0.3, 0.7).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double b0[2] = {0.6, 0.4};
    static const double b1[2] = {0.3, 0.7};
    static const double winit[2] = {0.5, 0.5};
    const double c = 0.1;
    /* vars: w0(0,1), w1(2,3), z0(4,5), z1(6,7), v0(8,9), v1(10,11) */
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 12);
    PRIMAL_appendcons(t, 18);
    for (int j = 0; j < 12; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int j = 0; j < 2; j++) {
        PRIMAL_putcj(t, 4 + j, 1.0);   /* z0 */
        PRIMAL_putcj(t, 6 + j, 1.0);   /* z1 */
        PRIMAL_putcj(t, 8 + j, c);     /* v0 */
        PRIMAL_putcj(t, 10 + j, c);    /* v1 */
    }
    /* e'w_0 = 1, e'w_1 = 1 */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putarow(t, 1, 2, (int[]){2, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);
    int r = 2;
    for (int j = 0; j < 2; j++) {          /* z0 >= ±(w0 - b0) */
        PRIMAL_putarow(t, r++, 2, (int[]){j, 4 + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, b0[j]);
        PRIMAL_putarow(t, r++, 2, (int[]){j, 4 + j}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, -b0[j]);
    }
    for (int j = 0; j < 2; j++) {          /* z1 >= ±(w1 - b1) */
        PRIMAL_putarow(t, r++, 2, (int[]){2 + j, 6 + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, b1[j]);
        PRIMAL_putarow(t, r++, 2, (int[]){2 + j, 6 + j}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, -b1[j]);
    }
    for (int j = 0; j < 2; j++) {          /* v0 >= ±(w0 - w_init) */
        PRIMAL_putarow(t, r++, 2, (int[]){j, 8 + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, winit[j]);
        PRIMAL_putarow(t, r++, 2, (int[]){j, 8 + j}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, -winit[j]);
    }
    for (int j = 0; j < 2; j++) {          /* v1 >= ±(w1 - w0) */
        PRIMAL_putarow(t, r++, 3, (int[]){2 + j, j, 10 + j}, (double[]){1.0, -1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, 0.0);
        PRIMAL_putarow(t, r++, 3, (int[]){2 + j, j, 10 + j}, (double[]){-1.0, 1.0, -1.0});
        PRIMAL_putconbound(t, r - 1, PRIMAL_BK_UP, -INFINITY, 0.0);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, x[12] = {0};
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    double track0 = fabs(x[0] - b0[0]) + fabs(x[1] - b0[1]);
    double track1 = fabs(x[2] - b1[0]) + fabs(x[3] - b1[1]);
    int ok = fabs(obj - 0.08) < 1e-7 &&
             fabs(x[0] - 0.6) < 1e-6 && fabs(x[1] - 0.4) < 1e-6 &&
             fabs(x[2] - 0.3) < 1e-6 && fabs(x[3] - 0.7) < 1e-6 &&
             fabs(track0) < 1e-6 && fabs(track1) < 1e-6;
    printf("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (atteso 0.08)\n",
           x[0], x[1], x[2], x[3], obj);
    printf("tracking = %.4f poi %.4f (atteso 0.0, 0.0)\n", track0, track1);
    printf("%s (index tracking su due periodi con costo di turnover)\n",
           ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
