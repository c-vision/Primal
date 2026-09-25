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

/* pca_alloc.c - portfolio allocation LP, "Portfolio Construction and
 * Analytics" (2016), Section 6.3 (problem lines 2970-3037, solution line 3182):
 *
 * A university invests a $10,000,000 endowment in four mutual funds:
 *   Fund 1 growth (ret 20.69%, risk 4), Fund 2 index (5.87%, 2),
 *   Fund 3 corporate bond (10.52%, 2), Fund 4 money market (2.43%, 1).
 * Maximize expected return subject to: total = 10,000,000; amount in
 * Funds 1+3 <= 60% (6,000,000); average risk <= 2 (linearized as
 * 4x1 + 2x2 + 2x3 + x4 <= 20,000,000); each x_i <= 4,000,000 (40%); x >= 0.
 *
 * The book's spreadsheet optimum: return $931,800 with
 *   x = ($2,000,000, $0, $4,000,000, $4,000,000).
 *
 * The sample solves the LP and checks the return and the allocation.
 *
 * Usage: pca_alloc   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double mu[4] = {0.2069, 0.0587, 0.1052, 0.0243};
    static const double risk[4] = {4.0, 2.0, 2.0, 1.0};

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 4, &t);
    for (int j = 0; j < 4; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 4.0e6);
        PRIMAL_putcj(t, j, -mu[j]);   /* maximize return (c negated, minimize) */
    }
    PRIMAL_putarow(t, 0, 4, (int[]){0, 1, 2, 3}, (double[]){1, 1, 1, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0e7, 1.0e7);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 2}, (double[]){1, 1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 6.0e6);
    PRIMAL_putarow(t, 2, 4, (int[]){0, 1, 2, 3}, risk);
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 2.0e7);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[4];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(-z - 931800.0) < 1.0 &&
             fabs(x[0] - 2.0e6) < 1.0 && fabs(x[1]) < 1.0 &&
             fabs(x[2] - 4.0e6) < 1.0 && fabs(x[3] - 4.0e6) < 1.0;
        printf("pca_alloc  x=(%.6g,%.6g,%.6g,%.6g)e6 return=%.6g (book 2,0,4,4 -> 931800) %s\n",
               x[0] / 1e6, x[1] / 1e6, x[2] / 1e6, x[3] / 1e6, -z, ok ? "OK" : "FAIL");
    } else {
        printf("pca_alloc  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
