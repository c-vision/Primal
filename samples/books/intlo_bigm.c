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

/* intlo_bigm.c - Big-M worked example, "Introduction to Linear Optimization"
 * (A. Nemirovski, 2024), Exercise 4.4(2) / section on the Big-M method
 * (problem at lines 12510-12518, answer at line 12564):
 *
 *   maximize   -2 x1 + x2 + x3 - 6 x4
 *   subject to  x1 + x2      = 2
 *               x3 + x4      = 2
 *               x1      + x3 = 2
 *               x1,x2,x3,x4 >= 0
 *
 * The book runs the Big-M primal simplex and finds the optimal solution
 *   x = [0, 2, 2, 0], optimal value 4.
 *
 * The sample solves the same LP with this library and checks the optimum.
 *
 * Usage: intlo_bigm   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 4, &t);
    for (int j = 0; j < 4; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(t, 0, -2.0); PRIMAL_putcj(t, 1, 1.0);
    PRIMAL_putcj(t, 2, 1.0);  PRIMAL_putcj(t, 3, -6.0);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 2.0, 2.0);
    PRIMAL_putarow(t, 1, 2, (int[]){2, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 2.0, 2.0);
    PRIMAL_putarow(t, 2, 2, (int[]){0, 2}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 2.0, 2.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[4];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 0.0) < 1e-6 && fabs(x[1] - 2.0) < 1e-6 &&
             fabs(x[2] - 2.0) < 1e-6 && fabs(x[3] - 0.0) < 1e-6 &&
             fabs(z - 4.0) < 1e-6;
        printf("intlo_bigm  x=(%.6g,%.6g,%.6g,%.6g) obj=%.6g (book [0,2,2,0], 4) %s\n",
               x[0], x[1], x[2], x[3], z, ok ? "OK" : "FAIL");
    } else {
        printf("intlo_bigm  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
