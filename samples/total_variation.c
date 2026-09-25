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

/* total_variation.c - total-variation denoising (MOSEK Cookbook 11.3,
 * total_variation.cc).
 *
 *   minimize  sum over the grid of the TV terms t
 *   subject to  (t, du) in QUAD            (the total variation)
 *               (sigma, f - u) in QUAD     (the fidelity ball ||f-u|| <= sigma)
 *               u in [0,1]
 *
 * The MOSEK example uses a random 100x200 image (std::mt19937(0) +
 * std::normal_distribution, not portable) ~ 20k variables, with no known value.
 * The same technique is exercised here on the smallest non-trivial signal, a
 * 1-D two-point signal, which has a closed form:
 *
 *   f = (0,1), sigma = 1/2, u in [0,1]^2:
 *     minimize |u1-u0|  s.t.  u0^2 + (1-u1)^2 <= 1/4.
 *   With d = u1-u0 and u0 = a, the constraint is a^2 + (1-a-d)^2 <= 1/4, whose
 *   minimum over a is (1-d)^2/2; so (1-d)^2 <= 1/2 and the tightest d is
 *     1 - 1/sqrt(2) = 0.29289322, at a = 1/(2 sqrt2) = 0.35355339.
 *   (The line u0 = u1 is at distance 1/sqrt2 > 1/2 from f, so TV = 0 is not
 *   reachable.)
 *
 * Usage: total_variation   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const double f[2] = {0.0, 1.0}, sigma = 0.5;
    enum { U = 0, TT = 2, DX = 3, E0 = 4, E1 = 5, SG = 6, NV = 7 };

    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 3, NV, &t);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int j = 0; j < 2; j++) PRIMAL_putvarbound(t, U + j, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, DX, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, E0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, E1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, SG, PRIMAL_BK_FX, sigma, sigma);
    PRIMAL_putcj(t, TT, 1.0);
    /* dx = u1 - u0 ; e_j = f_j - u_j */
    PRIMAL_putarow(t, 0, 3, (int[]){DX, U, U + 1}, (double[]){1.0, 1.0, -1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 1, 2, (int[]){E0, U}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, f[0], f[0]);
    PRIMAL_putarow(t, 2, 2, (int[]){E1, U + 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, f[1], f[1]);
    /* TV cone (t, dx) and fidelity cone (sigma, e0, e1) */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 2, (int[]){TT, DX});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){SG, E0, E1});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, u[2] = {0};
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, u);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    const double tv = 1.0 - 1.0 / sqrt(2.0), a0 = 1.0 / (2.0 * sqrt(2.0));
    int good = ok && fabs(obj - tv) < 1e-7 && fabs(u[0] - a0) < 1e-6 &&
               fabs(u[1] - (1.0 - a0)) < 1e-6 &&
               u[0] * u[0] + (1.0 - u[1]) * (1.0 - u[1]) <= sigma * sigma + 1e-9;
    printf("total_variation: TV*=%.8f  u=(%.6f, %.6f)  (atteso 1-1/sqrt2, u=(1/(2sqrt2),1-1/(2sqrt2)))  %s\n",
           obj, u[0], u[1], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
