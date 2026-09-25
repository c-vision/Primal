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

/* omf_ex21.c - Example 2.1 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Chapter 2 (the running bond-portfolio LP).
 *
 * A manager has $100,000 to split between a corporate bond (4% yield,
 * maturity 3y, rating 2) and a government bond (3% yield, maturity 4y,
 * rating 1). The rest is cash (0% yield, not counted in the averages).
 * Maximize yield subject to average rating <= 1.5 and average maturity
 * <= 3.6 years. With x1, x2 in thousands of dollars:
 *
 *   maximize   Z = 4 x1 + 3 x2
 *   subject to x1 + x2       <= 100
 *              2 x1 + x2     <= 150      (rating)
 *              3 x1 + 4 x2   <= 360      (maturity)
 *              x1, x2 >= 0
 *
 * The book solves it by simplex: starting from (0,0) it moves to (75,0),
 * then to the optimum x1 = 50, x2 = 50 (the constraints x1+x2 <= 100 and
 * 2x1+x2 <= 150 are active), Z = 350. (The nearby point (62.5,25) belongs
 * to the dual-simplex example later in the chapter, a different problem.)
 *
 * The sample solves it with this library and checks the book's optimum.
 *
 * Usage: omf_ex21   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 2, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, 0, -4.0); PRIMAL_putcj(t, 1, -3.0);   /* maximize */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 100.0);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 1}, (double[]){2.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 150.0);
    PRIMAL_putarow(t, 2, 2, (int[]){0, 1}, (double[]){3.0, 4.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 360.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 50.0) < 1e-6 && fabs(x[1] - 50.0) < 1e-6 &&
             fabs(z + 350.0) < 1e-6;
        printf("omf_ex21  x=(%.6g,%.6g) Z=%.6g (book x=(50,50), Z=350) %s\n",
               x[0], x[1], -z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex21  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
