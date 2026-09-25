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

/* option_pricing.c - no-arbitrage price bounds as an LP.
 *
 * In a one-period model with states s = 0..S-1 and discount factor 1, a
 * probability vector q >= 0 with sum q = 1 that reproduces the traded prices
 *   E_q[payoff_k] = price_k
 * is a risk-neutral measure. Any option whose price is the expectation of its
 * payoff under SOME such q is arbitrage-free, so its lower and upper bounds are
 * the min and max of E_q[payoff] over the polytope of risk-neutral measures —
 * two LPs that share the constraints and differ only in the objective.
 *
 * Hand case: 3 states, stock S = (0.5, 1.0, 1.5) traded at 1.0, so the only
 * constraint is 0.5 q0 + 1.0 q1 + 1.5 q2 = 1 with sum q = 1, q >= 0. Subtract
 * to get q0 = q2, hence q2 <= 1/2. A call struck at 1.0 pays (0, 0, 0.5), so
 * its price is 0.5 q2, with bounds [0, 0.25].
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static const double STOCK[3] = {0.5, 1.0, 1.5};

static double solve(PRIMALenv_t env, int maximize) {
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 3);
    PRIMAL_appendcons(t, 2);
    for (int j = 0; j < 3; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    /* sum q = 1 */
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* E_q[S] = 1 (the traded stock) */
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 2},
                   (double[]){STOCK[0], STOCK[1], STOCK[2]});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);
    /* objective: the call payoff max(S-1, 0) = (0, 0, 0.5) */
    PRIMAL_putcj(t, 2, 0.5);
    if (maximize) PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_deletetask(&t);
    return obj;
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    double lo = solve(env, 0);
    double hi = solve(env, 1);
    int ok = fabs(lo - 0.0) < 1e-9 && fabs(hi - 0.25) < 1e-9;
    printf("prezzo call K=1: [%.6f, %.6f] (atteso [0, 0.25])\n", lo, hi);
    printf("%s (limiti di non-arbitraggio come due LP)\n", ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
