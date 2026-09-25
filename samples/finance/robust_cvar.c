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

/* robust_cvar.c - worst-case CVaR over a box of return uncertainty.
 *
 * CVaR at level beta (Rockafellar-Uryasev):
 *   min_alpha  alpha + 1/((1-beta)S) * sum_s max(0, -r_s'w - alpha),
 * an LP in (w, alpha, z) with z_s >= -r_s'w - alpha, z_s >= 0. If each scenario
 * return may deviate in a box r_s + delta, delta in [-d_s, d_s], and w >= 0,
 * the worst case is the low corner r_s - d_s, so the robust CVaR is the same LP
 * on the shifted returns (a property this sample checks against the nominal).
 *
 * Hand case: 2 equiprobable scenarios, 2 assets, beta = 1/2 (so the CVaR weight
 * is 1/((1-1/2)*2) = 1 and CVaR is the worse scenario's loss).
 *   nominal  r0 = (0.10, 0.20), r1 = (0.30, 0.00)
 *   box      d  = (0.05, 0.05)
 *   robust   r0-d = (0.05, 0.15), r1-d = (0.25, -0.05)
 * With e'w = 1, w >= 0, the binding losses equalise at w = (1/2, 1/2), where
 * both robust losses are -0.10, so the robust CVaR is -0.10. The nominal CVaR
 * at the same w is -0.15, so robustness costs 0.05 (a smaller gain).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* robust: subtract the box radius from each scenario return */
static double solve(PRIMALenv_t env, int robust, double *wout) {
    static const double R0[2] = {0.10, 0.20};
    static const double R1[2] = {0.30, 0.00};
    static const double D[2]  = {0.05, 0.05};
    double r0[2], r1[2];
    for (int j = 0; j < 2; j++) {
        r0[j] = R0[j] - (robust ? D[j] : 0.0);
        r1[j] = R1[j] - (robust ? D[j] : 0.0);
    }
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
    PRIMAL_putcj(t, 2, 1.0); PRIMAL_putcj(t, 3, 1.0); PRIMAL_putcj(t, 4, 1.0);
    /* z_s + r_s'w + alpha >= 0 */
    PRIMAL_putarow(t, 0, 4, (int[]){3, 0, 1, 2}, (double[]){1.0, r0[0], r0[1], 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putarow(t, 1, 4, (int[]){4, 0, 1, 2}, (double[]){1.0, r1[0], r1[1], 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    /* e'w = 1 */
    PRIMAL_putarow(t, 2, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = 1e30;
    if (rc == PRIMAL_RES_OK) {
        double x[8] = {0};
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        wout[0] = x[0]; wout[1] = x[1];
    }
    PRIMAL_deletetask(&t);
    return obj;
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    double wn[2], wr[2];
    double nom = solve(env, 0, wn);
    double rob = solve(env, 1, wr);
    int ok = fabs(rob - (-0.10)) < 1e-7 &&
             fabs(wr[0] - 0.5) < 1e-6 && fabs(wr[1] - 0.5) < 1e-6 &&
             fabs(nom - (-0.15)) < 1e-7 && rob >= nom - 1e-9;
    printf("nominal CVaR = %.6f (atteso -0.15), w = (%.4f, %.4f)\n", nom, wn[0], wn[1]);
    printf("robust  CVaR = %.6f (atteso -0.10), w = (%.4f, %.4f)\n", rob, wr[0], wr[1]);
    printf("%s (la robustezza costa: CVaR robusto >= nominale)\n", ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
