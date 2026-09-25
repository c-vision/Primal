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

/* risk_budgeting.c - risk budgeting: target risk contributions.
 *
 * Given a covariance Sigma and target risk budgets b (sum 1), find w > 0 with
 *   w_i (Sigma w)_i = b_i * (w'Sigma w)   for every i.
 * The convex formulation (Spinu 2013) is
 *   min (1/2) w'Sigma w - sum_i b_i ln(w_i),
 * whose stationarity is exactly the budget equation, and the solution is then
 * normalised. The logarithms are modelled with the exponential cone:
 *   s_i >= -ln(w_i)  <=>  w_i >= exp(-s_i)  <=>  (w_i, 1, -s_i) in K_exp.
 *
 * Hand case: Sigma = I, b = (0.8, 0.2). Stationarity gives w_i^2 = b_i, so
 * w is proportional to (sqrt(0.8), sqrt(0.2)); normalised that is (2/3, 1/3),
 * and the risk contributions are then exactly (0.8, 0.2).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double b[2] = {0.8, 0.2};
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    /* vars: w0, w1, s0, s1, v0, v1, one */
    PRIMAL_appendvars(t, 7);
    PRIMAL_appendcons(t, 2);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 4, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 5, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 6, PRIMAL_BK_FX, 1.0, 1.0);
    /* (w_i, one, v_i) in K_exp  =>  w_i >= exp(v_i) = exp(-s_i) */
    PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){0, 6, 4});
    PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){1, 6, 5});
    /* v_i + s_i = 0 */
    PRIMAL_putarow(t, 0, 2, (int[]){4, 2}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 1, 2, (int[]){5, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* objective: (1/2)(w0^2+w1^2) + b0 s0 + b1 s1 */
    PRIMAL_putqobj(t, 1, (int[]){0}, (int[]){0}, (double[]){2.0});
    PRIMAL_putqobj(t, 1, (int[]){1}, (int[]){1}, (double[]){2.0});
    PRIMAL_putcj(t, 2, b[0]);
    PRIMAL_putcj(t, 3, b[1]);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double x[8] = {0};
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    double w0 = x[0] / (x[0] + x[1]);
    double w1 = x[1] / (x[0] + x[1]);
    double rc0 = (w0 * w0) / (w0 * w0 + w1 * w1);
    double rc1 = (w1 * w1) / (w0 * w0 + w1 * w1);
    /* the conic IPM delivers ~1e-5 here; the budgets are the meaningful check */
    int ok = fabs(w0 - 2.0 / 3.0) < 1e-3 && fabs(w1 - 1.0 / 3.0) < 1e-3 &&
             fabs(rc0 - b[0]) < 1e-4 && fabs(rc1 - b[1]) < 1e-4;
    printf("w = (%.6f, %.6f) (atteso 0.6667, 0.3333)\n", w0, w1);
    printf("contributi di rischio = (%.6f, %.6f) (attesi 0.8, 0.2)\n", rc0, rc1);
    printf("%s (risk budgeting, log via cono esponenziale)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
