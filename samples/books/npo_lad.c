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

/* npo_lad.c - L1 (least absolute deviations) regression as an LP, "Nonlinear
 * Parameter Optimization Using R Tools" (2014), Section 14.3 (lines 5921-5973
 * and 6880-6929):
 *
 *   x* = argmin_x  sum_i | A x - y |_i
 *
 * with A 10x3, columns (1, i, i^2) for i = 1..10, and the perturbed response
 *   y = (0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
 *        -45.193280, -60.690202, -77.243728, -96.406035, -117.241829).
 *
 * The book solves it via an LP with z >= +/-(Ax - y) and reports
 *   coeffs = (3.489517, -2.174797, -0.989834),  sad = 3.2951.
 *
 * The sample builds the LP (x free, z >= 0) and checks both.
 *
 * Usage: npo_lad   (no arguments)
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
    PRIMAL_maketask(env, 20, 13, &t);   /* 3 x (free) + 10 z (>=0) */
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 3; j < 13; j++) { PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
                                   PRIMAL_putcj(t, j, 1.0); }
    for (int i = 0; i < 10; i++) {
        double a1 = 1.0, a2 = (double)(i + 1), a3 = a2 * a2;   /* row [1, i, i^2] */
        /* z_i - (A x)_i >= -y_i */
        PRIMAL_putarow(t, 2 * i, 4, (int[]){0, 1, 2, 3 + i},
                       (double[]){-a1, -a2, -a3, 1.0});
        PRIMAL_putconbound(t, 2 * i, PRIMAL_BK_LO, -y[i], INFINITY);
        /* z_i + (A x)_i >= y_i */
        PRIMAL_putarow(t, 2 * i + 1, 4, (int[]){0, 1, 2, 3 + i},
                       (double[]){a1, a2, a3, 1.0});
        PRIMAL_putconbound(t, 2 * i + 1, PRIMAL_BK_LO, y[i], INFINITY);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, v[13];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        ok = fabs(v[0] - 3.489517) < 1e-4 && fabs(v[1] + 2.174797) < 1e-4 &&
             fabs(v[2] + 0.989834) < 1e-4 && fabs(z - 3.2951) < 1e-4;
        printf("npo_lad  coeffs=(%.6g,%.6g,%.6g) sad=%.6g (book (3.489517,-2.174797,-0.989834), 3.2951) %s\n",
               v[0], v[1], v[2], z, ok ? "OK" : "FAIL");
    } else {
        printf("npo_lad  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
