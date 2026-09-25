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

/* transaction_cost.c - transaction-cost LP with free variables and a
 * degenerate (cost-free) column.
 *
 * Source: StackOverflow question 37586543, "MOSEK Markowitz portfolio
 * transaction costs" (MOSEK Fusion). n = 3, initial holdings
 * x0 = (-20,-50,-10), transaction costs t = (.01,.01,.01):
 *
 *   min t'z
 *   s.t. long1:        l - x0 >= 0
 *        buy:          z - (x - x0) >= 0
 *        sell:         z - (x0 - x) >= 0
 *        longeqshort:  e'x = 0
 *   x free, z free, l >= 0.
 *
 * Hand derivation: buy+sell say z >= |x - x0|, and e'x = 0 gives
 * e'(x - x0) = -e'x0 = 80, hence sum z >= 80 with equality iff every
 * x_i >= x0_i. The optimum is not a point but a FACE; the optimal value is
 * 0.01*80 = 0.8, and the vector the asker posted, x = (60,-50,-10), is one of
 * its points. `l` appears only in `l >= x0` and in no objective, so it is a
 * degenerate column the presolve is expected to remove.
 *
 * The right assertion is the VALUE plus membership of x in the optimal face,
 * not the posted vector: asserting (60,-50,-10) would guard the interior point
 * of one solver, not the model.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double x0[3] = {-20.0, -50.0, -10.0};
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 9);    /* 0..2 x, 3..5 z, 6..8 l */
    PRIMAL_appendcons(task, 10);   /* 0..2 long1, 3..5 buy, 6..8 sell, 9 longeqshort */
    for (int i = 0; i < 3; i++) {
        PRIMAL_putvarbound(task, i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(task, 3 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(task, 6 + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(task, 3 + i, 0.01);
    }
    for (int i = 0; i < 3; i++) {
        PRIMAL_putarow(task, i, 1, (int[]){6 + i}, (double[]){1.0});
        PRIMAL_putconbound(task, i, PRIMAL_BK_LO, x0[i], INFINITY);
        PRIMAL_putarow(task, 3 + i, 2, (int[]){3 + i, i}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(task, 3 + i, PRIMAL_BK_LO, -x0[i], INFINITY);
        PRIMAL_putarow(task, 6 + i, 2, (int[]){3 + i, i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(task, 6 + i, PRIMAL_BK_LO, x0[i], INFINITY);
    }
    PRIMAL_putarow(task, 9, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 9, PRIMAL_BK_FX, 0.0, 0.0);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, x[16] = {0}, pinf = 1.0;
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalinfeas(task, PRIMAL_SOL_ITR, &pinf);

    double sx = 0.0, viol = 0.0;
    for (int i = 0; i < 3; i++) {
        sx += x[i];
        if (x0[i] - x[i] > viol) viol = x0[i] - x[i];
        double need = fabs(x[i] - x0[i]);
        if (need - x[3 + i] > viol) viol = need - x[3 + i];
    }
    int on_face = (viol <= 1e-6) && (fabs(sx) <= 1e-6);
    int ok = fabs(obj - 0.8) < 1e-6 && on_face && pinf <= 1e-6;
    printf("x = (%.4f, %.4f, %.4f), obj = %.6f (atteso 0.8)\n", x[0], x[1], x[2], obj);
    printf("e'x = %.2e, x_i >= x0_i: %s, getprimalinfeas = %.2e\n",
           sx, on_face ? "si" : "NO", pinf);
    printf("%s (ottimo su una faccia: valore 0.8, non il vettore del post)\n",
           ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
