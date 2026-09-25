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

/* omf_capital.c - capital budgeting (0-1) problem, "Optimization Methods in
 * Finance" (Cornuejols & Tutuncu, 2007), Section 11.2 (lines 4627-4643):
 *
 *   maximize   8 x1 + 11 x2 + 6 x3 + 4 x4
 *   subject to 6.7 x1 + 10 x2 + 5.5 x3 + 3.4 x4 <= 19
 *              x1, x2, x3, x4 in {0, 1}
 *
 * The book reports the LP relaxation (x1 = 1, x2 = 0.89, x3 = 0, x4 = 1,
 * value 21790; rounding x2 down gives 12000) and the integer optimum
 *   x = (0, 1, 1, 1) with value 21000.
 *
 * The sample solves the MIP with this library and checks the integer optimum.
 *
 * Usage: omf_capital   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 1, 4, &t);
    for (int j = 0; j < 4; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT_BIN);
    }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(t, 0, 8.0); PRIMAL_putcj(t, 1, 11.0);
    PRIMAL_putcj(t, 2, 6.0); PRIMAL_putcj(t, 3, 4.0);
    PRIMAL_putarow(t, 0, 4, (int[]){0, 1, 2, 3}, (double[]){6.7, 10.0, 5.5, 3.4});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 19.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[4];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int j = 0; j < 4; j++) ok = ok && fabs(x[j] - floor(x[j] + 0.5)) < 1e-6;
        ok = ok && fabs(x[0]) < 1e-6 && fabs(x[1] - 1.0) < 1e-6 &&
             fabs(x[2] - 1.0) < 1e-6 && fabs(x[3] - 1.0) < 1e-6 &&
             fabs(z - 21.0) < 1e-6;
        printf("omf_capital  x=(%.6g,%.6g,%.6g,%.6g) Z=%.6g (=%.0f) (book (0,1,1,1), 21000) %s\n",
               x[0], x[1], x[2], x[3], z, 1000.0 * z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_capital  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
