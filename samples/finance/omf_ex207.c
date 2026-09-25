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

/* omf_ex207.c - relative robust portfolio, "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Exercise 20.7 / formulation (20.17)
 * (lines 7328-7363):
 *
 *   minimize   t
 *   subject to 5.662 - (6 x1 + 4 x2) <= t
 *              5.662 - (4 x1 + 6 x2) <= t
 *              5.0   - (5 x1 + 5 x2) <= t
 *              TE(x) <= 0.10
 *              x1 + x2 + x3 = 1,  x >= 0
 *
 * where TE(x) = sqrt( d' C d ), d = (x1-0.5, x2-0.5, x3),
 * C = [[0.1764, 0.09702, 0], [0.09702, 0.1089, 0], [0, 0, 0]].
 * The book states x* = (0.5, 0.5, 0) solves it with t* = 0.662.
 *
 * TE <= 0.10 is written as a rotated/quadratic cone: with the Cholesky
 * factor of the 2x2 block (L11 = 0.42, L21 = 0.2310, L22 = 0.235678),
 * y1 = L11 d1, y2 = L21 d1 + L22 d2, the constraint is
 * (0.10, y1, y2) in the quadratic cone.
 *
 * Usage: omf_ex207   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    const double L11 = 0.42, L21 = 0.231, L22 = 0.235678147;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0,1,2 = x1,x2,x3 ; 3 = t ; 4 = s(=0.10) ; 5,6 = y1,y2 */
    PRIMAL_maketask(env, 7, 7, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_FR, -INFINITY, INFINITY);       /* t */
    PRIMAL_putvarbound(t, 4, PRIMAL_BK_FX, 0.10, 0.10);                /* s */
    PRIMAL_putvarbound(t, 5, PRIMAL_BK_FR, -INFINITY, INFINITY);       /* y1 */
    PRIMAL_putvarbound(t, 6, PRIMAL_BK_FR, -INFINITY, INFINITY);       /* y2 */
    PRIMAL_putcj(t, 3, 1.0);                                           /* min t */

    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1, 1, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 3}, (double[]){6, 4, 1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 5.662, INFINITY);
    PRIMAL_putarow(t, 2, 3, (int[]){0, 1, 3}, (double[]){4, 6, 1});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_LO, 5.662, INFINITY);
    PRIMAL_putarow(t, 3, 3, (int[]){0, 1, 3}, (double[]){5, 5, 1});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_LO, 5.0, INFINITY);
    /* y1 = L11 (x1 - 0.5) */
    PRIMAL_putarow(t, 4, 2, (int[]){0, 5}, (double[]){-L11, 1});
    PRIMAL_putconbound(t, 4, PRIMAL_BK_FX, -0.5 * L11, -0.5 * L11);
    /* y2 = L21 (x1 - 0.5) + L22 (x2 - 0.5) */
    PRIMAL_putarow(t, 5, 3, (int[]){0, 1, 6}, (double[]){-L21, -L22, 1});
    PRIMAL_putconbound(t, 5, PRIMAL_BK_FX, -0.5 * (L21 + L22), -0.5 * (L21 + L22));

    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){4, 5, 6});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[7], te;
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        te = hypot(x[5], x[6]);
        ok = fabs(x[0] - 0.5) < 1e-4 && fabs(x[1] - 0.5) < 1e-4 &&
             fabs(x[2]) < 1e-4 && fabs(z - 0.662) < 1e-3 && te <= 0.10 + 1e-6;
        printf("omf_ex207  x=(%.6g,%.6g,%.6g) t=%.6g TE=%.6g (book (0.5,0.5,0), 0.662) %s\n",
               x[0], x[1], x[2], z, te, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex207  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
