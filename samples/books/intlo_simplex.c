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

/* intlo_simplex.c - worked primal-simplex example, "Introduction to Linear
 * Optimization" (A. Nemirovski, 2024), Section 4.3.2 (lines 6240-6328):
 *
 *   maximize   5 x1 + 3 x2 + 6 x3
 *   subject to  x1 + x2 + 2 x3 <= 30
 *               4 x1 + x2 + 4 x3 <= 60
 *               2 x1 + x2 +  x3 <= 30
 *               x1, x2, x3 >= 0
 *
 * The book runs the tableau simplex to the optimum
 *   x* = (6, 12, 6), optimal value 102,
 * with reduced costs [0,0,0,-1.8,-0.4,-0.8] on the final tableau.
 *
 * The sample solves the same LP with this library and checks the value.
 *
 * Usage: intlo_simplex   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(t, 0, 5.0); PRIMAL_putcj(t, 1, 3.0); PRIMAL_putcj(t, 2, 6.0);
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 2.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 30.0);
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 2}, (double[]){4.0, 1.0, 4.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 60.0);
    PRIMAL_putarow(t, 2, 3, (int[]){0, 1, 2}, (double[]){2.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 30.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 6.0) < 1e-6 && fabs(x[1] - 12.0) < 1e-6 &&
             fabs(x[2] - 6.0) < 1e-6 && fabs(z - 102.0) < 1e-6;
        printf("intlo_simplex  x=(%.6g,%.6g,%.6g) obj=%.6g (book (6,12,6), 102) %s\n",
               x[0], x[1], x[2], z, ok ? "OK" : "FAIL");
    } else {
        printf("intlo_simplex  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
