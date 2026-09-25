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

/* intlo_transport.c - transportation counterexample, "Introduction to Linear
 * Optimization" (A. Nemirovski, 2024), Exercise 5.3 (lines 12568-12576):
 *
 *   Opt = min { sum_{ij} c_ij x_ij : sum_j x_ij = a_i, sum_i x_ij = b_j,
 *                                    x_ij >= 0 }
 *
 * Instance (p = q = 2): c11 = c22 = 1, c12 = c21 = 100.
 *   (a) a = b = (1, 1): trivial solution x11 = x22 = 1, optimum 2.
 *   (b) supplies/demands decreased to a' = (1/2, 1), b' = (1, 1/2):
 *       the book proves the optimum is "at least 50" (it increases!), so
 *       decreasing supply+demand (keeping totals equal) can increase cost.
 *
 * The sample solves both LPs and checks (a) = 2 and (b) >= 50 (computed 51).
 *
 * Usage: intlo_transport   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* one transportation LP: min c11 x11 + c12 x12 + c21 x21 + c22 x22 */
static int solve(double a0, double a1, double b0, double b1, double *obj) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double c[4] = {1.0, 100.0, 100.0, 1.0};  /* 11,12,21,22 */
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 4, 4, &t);
    for (int j = 0; j < 4; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, j, c[j]);
    }
    /* row supplies */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, a0, a0);
    PRIMAL_putarow(t, 1, 2, (int[]){2, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, a1, a1);
    /* column demands */
    PRIMAL_putarow(t, 2, 2, (int[]){0, 2}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, b0, b0);
    PRIMAL_putarow(t, 3, 2, (int[]){1, 3}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, b1, b1);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, obj);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double z1 = -1.0, z2 = -1.0;
    int ok = solve(1.0, 1.0, 1.0, 1.0, &z1) &&
             solve(0.5, 1.0, 1.0, 0.5, &z2);
    ok = ok && fabs(z1 - 2.0) < 1e-6 && z2 >= 50.0 - 1e-6;
    printf("intlo_transport  z_initial=%.6g z_new=%.6g (book 2, >=50) %s\n",
           z1, z2, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
