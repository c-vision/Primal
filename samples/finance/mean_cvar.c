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

/* mean_cvar.c - mean-CVaR portfolio.
 *
 *   max  mu'w - lambda * CVaR_beta(w)   s.t.  e'w = 1,  w >= 0,
 * with the Rockafellar-Uryasev CVaR linearised:
 *   CVaR_beta(w) = min_alpha alpha + 1/((1-beta)S) sum_s max(0, -r_s'w - alpha).
 * The whole thing is one LP in (w, alpha, z) with z_s >= -r_s'w - alpha, z_s >= 0.
 *
 * Hand case: 2 assets, 2 equiprobable scenarios, beta = 1/2 (so the CVaR weight
 * is 1), lambda = 1:
 *   mu = (0.2, 0.1),  r0 = (0.1, 0.3),  r1 = (0.4, -0.1).
 * For fixed w the CVaR is max(-r0'w, -r1'w), so the objective is
 *   mu'w - max(-r0'w, -r1'w),
 * which is the upper envelope of (mu+r0)'w and (mu+r1)'w; its maximum is at the
 * kink where r0'w = r1'w, i.e. w = (4/7, 3/7), value 12/35.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double mu[2] = {0.2, 0.1};
    static const double R0[2] = {0.1, 0.3};
    static const double R1[2] = {0.4, -0.1};
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    /* vars: w0, w1, alpha, z0, z1 */
    PRIMAL_appendvars(t, 5);
    PRIMAL_appendcons(t, 3);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 4, PRIMAL_BK_LO, 0.0, INFINITY);
    /* objective: max mu'w - alpha - z0 - z1 */
    PRIMAL_putcj(t, 0, mu[0]);
    PRIMAL_putcj(t, 1, mu[1]);
    PRIMAL_putcj(t, 2, -1.0);
    PRIMAL_putcj(t, 3, -1.0);
    PRIMAL_putcj(t, 4, -1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    /* z_s + r_s'w + alpha >= 0 */
    PRIMAL_putarow(t, 0, 4, (int[]){3, 0, 1, 2}, (double[]){1.0, R0[0], R0[1], 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putarow(t, 1, 4, (int[]){4, 0, 1, 2}, (double[]){1.0, R1[0], R1[1], 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    /* e'w = 1 */
    PRIMAL_putarow(t, 2, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, x[8] = {0};
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    double w0 = x[0], w1 = x[1];
    double cvar = fmax(-(R0[0] * w0 + R0[1] * w1), -(R1[0] * w0 + R1[1] * w1));
    int ok = fabs(w0 - 4.0 / 7.0) < 1e-6 && fabs(w1 - 3.0 / 7.0) < 1e-6 &&
             fabs(obj - 12.0 / 35.0) < 1e-6 &&
             fabs((mu[0] * w0 + mu[1] * w1) - cvar - obj) < 1e-6;
    printf("w = (%.6f, %.6f) (atteso 0.5714, 0.4286)\n", w0, w1);
    printf("obj = %.6f (atteso 12/35 = %.6f), CVaR = %.6f\n", obj, 12.0 / 35.0, cvar);
    printf("%s (mean-CVaR, ottimo al ginocchio r0'w = r1'w)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
