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

/* omf_qp_grg.c - the QP solved by GRG in "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 5 (GRG / nonlinear programming):
 *
 *   minimize   f(x1,x2) = (x1 - 1/2)^2 + (x2 - 5/2)^2
 *   subject to x1 - x2 >= 0
 *              x1 >= 0, 0 <= x2 <= 2
 *
 * The book follows the generalized reduced gradient iterates and concludes the
 * optimal solution is x1 = 1.5, x2 = 1.5.
 *
 * Expanding, f = x1^2 + x2^2 - x1 - 5 x2 + 6.5, so the quadratic objective is
 * 0.5 x'Qx + c'x with Q = 2 I and c = (-1, -5); the solver reports
 * 0.5 x'Qx + c'x = -4.5 at the optimum, and f = that + 6.5 = 2.
 *
 * The sample solves the QP with this library and checks the book's optimum.
 *
 * Usage: omf_qp_grg   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 1, 2, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);          /* x1 */
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_RA, 0.0, 2.0);               /* 0 <= x2 <= 2 */
    PRIMAL_putcj(t, 0, -1.0); PRIMAL_putcj(t, 1, -5.0);
    /* Q = 2 I  (0.5 x'Qx = x1^2 + x2^2) */
    PRIMAL_putqobj(t, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
    /* x1 - x2 >= 0 */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double f = z + 6.5;   /* add back the constant of (x1-1/2)^2+(x2-5/2)^2 */
        ok = fabs(x[0] - 1.5) < 1e-6 && fabs(x[1] - 1.5) < 1e-6 &&
             fabs(f - 2.0) < 1e-6 && x[0] - x[1] >= -1e-6;
        printf("omf_qp_grg  x=(%.6g,%.6g)  f=%.6g (book x=(1.5,1.5), f=2) %s\n",
               x[0], x[1], f, ok ? "OK" : "FAIL");
    } else {
        printf("omf_qp_grg  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
