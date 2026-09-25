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

/* npo_minimax.c - minimax (L-infinity) regression as an LP, "Nonlinear
 * Parameter Optimization Using R Tools" (2014), Section 14.4 (lines 6970-7095):
 *
 *   x* = argmin_x  max_i | A x - y |_i
 *
 * with the same A 10x3 (columns 1, i, i^2 for i = 1..10) and perturbed response
 * y as in the L1 example (Section 14.3):
 *   y = (0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
 *        -45.193280, -60.690202, -77.243728, -96.406035, -117.241829).
 *
 * The book reports coeffs = (2.73966, -1.92954, -1.00881), mad = 0.726338.
 *
 * LP form: variables x (free), t (free); -(t) <= (Ax - y)_i <= t.
 *
 * Usage: npo_minimax   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double y[10] = {
        0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
        -45.193280, -60.690202, -77.243728, -96.406035, -117.241829};

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 20, 4, &t);   /* 3 x + t */
    for (int j = 0; j < 4; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, 3, 1.0);           /* min t */
    for (int i = 0; i < 10; i++) {
        double a1 = 1.0, a2 = (double)(i + 1), a3 = a2 * a2;
        /* (A x)_i - y_i <= t  =>  a_i.x - t <= y_i */
        PRIMAL_putarow(t, 2 * i, 4, (int[]){0, 1, 2, 3},
                       (double[]){a1, a2, a3, -1.0});
        PRIMAL_putconbound(t, 2 * i, PRIMAL_BK_UP, -INFINITY, y[i]);
        /* -(A x)_i + y_i <= t  =>  -a_i.x - t <= -y_i */
        PRIMAL_putarow(t, 2 * i + 1, 4, (int[]){0, 1, 2, 3},
                       (double[]){-a1, -a2, -a3, -1.0});
        PRIMAL_putconbound(t, 2 * i + 1, PRIMAL_BK_UP, -INFINITY, -y[i]);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, v[4];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        ok = fabs(v[0] - 2.73966) < 1e-4 && fabs(v[1] + 1.92954) < 1e-4 &&
             fabs(v[2] + 1.00881) < 1e-4 && fabs(z - 0.726338) < 1e-5;
        printf("npo_minimax  coeffs=(%.6g,%.6g,%.6g) mad=%.6g (book (2.73966,-1.92954,-1.00881), 0.726338) %s\n",
               v[0], v[1], v[2], z, ok ? "OK" : "FAIL");
    } else {
        printf("npo_minimax  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
