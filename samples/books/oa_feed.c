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

/* oa_feed.c - animal feed mix LP, "Optimization Algorithms: AI techniques for
 * design, planning, and control problems" (2024), Listing 2.6 and Appendix C.1
 * (model lines 1130-1162, output lines 1159-1162 / 13718-13721):
 *
 *   minimize   30.5 x1 + 10.0 x2 + 90 x3      (cents per kg)
 *   subject to 0.008 <= 0.001 x1 + 0.38 x2 + 0.002 x3 <= 0.012   (calcium)
 *              0.09 x1 + 0.50 x3 >= 0.22                          (protein)
 *              0.02 x1 + 0.08 x3 <= 0.05                          (fiber)
 *              x1 + x2 + x3 = 1
 *              0 <= x1, x2, x3 <= 1                               (fractions)
 *
 * Corn = x1, Limestone = x2, Soybean meal = x3.
 * The book's output: Corn = 65.0%, Limestone = 3.0%, Soybean meal = 32.0%,
 * total cost = 0.4916 $/kg (49.16 cents/kg).
 *
 * The sample solves the LP and checks solution and cost.
 *
 * Usage: oa_feed   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 4, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putcj(t, 0, 30.5); PRIMAL_putcj(t, 1, 10.0); PRIMAL_putcj(t, 2, 90.0);
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){0.001, 0.38, 0.002});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_RA, 0.008, 0.012);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 2}, (double[]){0.09, 0.50});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 0.22, INFINITY);
    PRIMAL_putarow(t, 2, 2, (int[]){0, 2}, (double[]){0.02, 0.08});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 0.05);
    PRIMAL_putarow(t, 3, 3, (int[]){0, 1, 2}, (double[]){1, 1, 1});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 0.65) < 0.01 && fabs(x[1] - 0.03) < 0.01 &&
             fabs(x[2] - 0.32) < 0.01 && fabs(z - 49.16) < 0.05;
        printf("oa_feed  corn=%.4g%% lime=%.4g%% soy=%.4g%% cost=%.4g c/kg (book 65/3/32, 49.16) %s\n",
               100.0 * x[0], 100.0 * x[1], 100.0 * x[2], z, ok ? "OK" : "FAIL");
    } else {
        printf("oa_feed  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
