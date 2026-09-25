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

/* omf_duality.c - first duality example, "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 2.2 (lines 508-574):
 *
 *   minimize   -x1 - x2
 *   subject to  2 x1 + x2 <= 12
 *                x1 + 2 x2 <= 9
 *                x1, x2 >= 0
 *
 * The book's standard form adds slacks x3, x4 and finds the optimal
 *   (x1, x2, x3, x4) = (5, 2, 0, 0), objective value -7,
 * which it certifies with the dual multipliers (1/3, 1/3).
 *
 * The sample solves the LP and checks the optimum.
 *
 * Usage: omf_duality   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2, 2, &t);
    for (int j = 0; j < 2; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, 0, -1.0); PRIMAL_putcj(t, 1, -1.0);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){2.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 12.0);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 1}, (double[]){1.0, 2.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 9.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 5.0) < 1e-6 && fabs(x[1] - 2.0) < 1e-6 &&
             fabs(z + 7.0) < 1e-6;
        printf("omf_duality  x=(%.6g,%.6g) obj=%.6g (book (5,2), -7) %s\n",
               x[0], x[1], z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_duality  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
