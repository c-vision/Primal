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

/* cardinality_portfolio.c - minimum variance with a cardinality constraint.
 *
 *   min w'Sigma w   s.t.  e'w = 1,  w >= 0,  sum_j z_j <= K,  w_j <= z_j,  z binary.
 * A mixed-integer quadratic program: the binary z_j says whether asset j is used
 * and at most K of them may be. The relaxations are QPs (the objective is
 * quadratic), and the branch & bound fixes z.
 *
 * Hand case: Sigma = I, N = 3, K = 2. Without the cardinality limit the minimum
 * is (1/3,1/3,1/3) with value 1/3; with K = 2 the best is to split equally over
 * two assets, (1/2,1/2,0), value 1/2 (any pair).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 6);   /* 0..2 w, 3..5 z */
    PRIMAL_appendcons(t, 5);
    for (int j = 0; j < 3; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, 1.0);
        PRIMAL_putvarbound(t, 3 + j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, 3 + j, PRIMAL_VAR_TYPE_INT_BIN);
        /* w_j - z_j <= 0 */
        PRIMAL_putarow(t, j, 2, (int[]){j, 3 + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, j, PRIMAL_BK_UP, -INFINITY, 0.0);
    }
    /* e'w = 1 */
    PRIMAL_putarow(t, 3, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 1.0, 1.0);
    /* sum z <= 2 */
    PRIMAL_putarow(t, 4, 3, (int[]){3, 4, 5}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 4, PRIMAL_BK_UP, -INFINITY, 2.0);
    /* objective: 1/2 w'Qw with Q = 2I -> w0^2+w1^2+w2^2 */
    for (int j = 0; j < 3; j++)
        PRIMAL_putqobj(t, 1, (int[]){j}, (int[]){j}, (double[]){2.0});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, x[8] = {0};
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    int used = 0;
    for (int j = 0; j < 3; j++) if (x[j] > 1e-6) used++;
    double quad = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    int ok = fabs(obj - 0.5) < 1e-6 && used <= 2 &&
             fabs(x[0] + x[1] + x[2] - 1.0) < 1e-6 && fabs(quad - obj) < 1e-6;
    printf("w = (%.4f, %.4f, %.4f), attivi = %d, obj = %.6f (atteso 0.5)\n",
           x[0], x[1], x[2], used, obj);
    printf("%s (MIQP: cardinalita' K=2 su 3 attivi)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
