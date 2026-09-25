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

/* omf_ex217.c - Exercise 2.17 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Chapter 2 (the simplex method).
 *
 *   maximize  4 x1 + x2 - x3
 *   subject to  x1     + 3 x3 <= 6
 *               3 x1 + x2 + 3 x3 <= 9
 *               x1, x2, x3 >= 0
 *
 * The book solves it by the simplex method and states the optimum
 *   x1 = 3, x2 = x3 = 0     (objective 12).
 *
 * The sample solves the LP with this library and checks the book's optimum.
 *
 * Usage: omf_ex217   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, 0, -4.0); PRIMAL_putcj(t, 1, -1.0); PRIMAL_putcj(t, 2, 1.0);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 2}, (double[]){1.0, 3.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 6.0);
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 2}, (double[]){3.0, 1.0, 3.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 9.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(z + 12.0) < 1e-6 && fabs(x[0] - 3.0) < 1e-6 &&
             fabs(x[1]) < 1e-6 && fabs(x[2]) < 1e-6;
        printf("omf_ex217  x=(%.6g,%.6g,%.6g)  obj=%.6g (book x=(3,0,0), 12) %s\n",
               x[0], x[1], x[2], -z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex217  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
