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

/* boyd_qp_box.c - box-constrained QP, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Exercise 4.3 (lines 6004-6010):
 *
 *   minimize   (1/2) x'Px + q'x + r
 *   subject to -1 <= x_i <= 1,  i = 1,2,3
 *
 * with
 *   P = [ 13  12  -2 ; 12  17  6 ; -2  6  12 ],  q = (-22.0, -14.5, 13.0),
 *   r = 1.
 *
 * The book asks to prove x* = (1, 1/2, -1) is optimal.
 * The sample solves the QP with this library and checks that optimum
 * (objective 0.5 x*'Px* + q'x* + r = -21.625).
 *
 * Usage: boyd_qp_box   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 3, &t);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, -1.0, 1.0);
    PRIMAL_putcj(t, 0, -22.0); PRIMAL_putcj(t, 1, -14.5); PRIMAL_putcj(t, 2, 13.0);
    /* upper triangle of P with the full coefficient (mirrored internally) */
    PRIMAL_putqobj(t, 6, (int[]){0, 0, 0, 1, 1, 2}, (int[]){0, 1, 2, 1, 2, 2},
                   (double[]){13.0, 12.0, -2.0, 17.0, 6.0, 12.0});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 1.0) < 1e-5 && fabs(x[1] - 0.5) < 1e-5 &&
             fabs(x[2] + 1.0) < 1e-5;
        printf("boyd_qp_box  x=(%.6g,%.6g,%.6g) f=%.6g (book (1,1/2,-1), f=-21.625) %s\n",
               x[0], x[1], x[2], z + 1.0, ok ? "OK" : "FAIL");
    } else {
        printf("boyd_qp_box  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
