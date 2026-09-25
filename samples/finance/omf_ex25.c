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

/* omf_ex25.c - Exercise 2.5 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 2 (duality):
 *
 *   minimize   2 x1 + 3 x2
 *   subject to x1 + x2 >= 5
 *              x1      >= 1
 *              x2      >= 2
 *
 * The book asks to prove x* = (3, 2) is optimal by showing every feasible
 * solution has objective value at least 12 (its dual is max 12y1 + 9y2 with
 * 2y1 + y2 <= -1, y1 + 2y2 <= -1, y <= 0).
 *
 * The sample solves it with this library and checks the book's optimum.
 *
 * Usage: omf_ex25   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 1, 2, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 1.0, INFINITY);   /* x1 >= 1 */
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 2.0, INFINITY);   /* x2 >= 2 */
    PRIMAL_putcj(t, 0, 2.0); PRIMAL_putcj(t, 1, 3.0);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_LO, 5.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 3.0) < 1e-6 && fabs(x[1] - 2.0) < 1e-6 &&
             fabs(z - 12.0) < 1e-6;
        printf("omf_ex25  x=(%.6g,%.6g) obj=%.6g (book x=(3,2), 12) %s\n",
               x[0], x[1], z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex25  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
