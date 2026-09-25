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

/* boyd_l1.c - parametrized l1-norm approximation, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Exercise 5.33 (lines 9327-9341):
 *
 *   minimize   || A x + b + eps d ||_1,   x in R^3
 *
 * with
 *   A = [ -2  7  1 ; -5 -1  3 ; -7  3 -5 ; -1  4 -4 ;  1  5  5 ;  2 -5 -1 ],
 *   b = ( -4, 3, 9, 0, -11, 5 ),  d = ( -10, -13, -27, -10, -7, 14 ).
 *
 * For eps = 0 the book asks to prove x* = 1 (all-ones) is optimal.
 * The sample solves the l1 problem as an LP (variables x free, z >= 0 with
 * z_i >= +/-(Ax+b)_i, minimize sum z) and checks x* = (1,1,1) (value 4).
 *
 * Usage: boyd_l1   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double A[6][3] = {
        {-2, 7, 1}, {-5, -1, 3}, {-7, 3, -5},
        {-1, 4, -4}, {1, 5, 5}, {2, -5, -1}};
    static const double b[6] = {-4, 3, 9, 0, -11, 5};

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 12, 9, &t);   /* 3 x + 6 z */
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 3; j < 9; j++) { PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
                                  PRIMAL_putcj(t, j, 1.0); }
    for (int i = 0; i < 6; i++) {
        /* z_i >= (A x + b)_i  <=>  z_i - a_i.x >= b_i */
        PRIMAL_putarow(t, 2 * i, 4, (int[]){0, 1, 2, 3 + i},
                       (double[]){-A[i][0], -A[i][1], -A[i][2], 1.0});
        PRIMAL_putconbound(t, 2 * i, PRIMAL_BK_LO, b[i], INFINITY);
        /* z_i >= -(A x + b)_i  <=>  z_i + a_i.x >= -b_i */
        PRIMAL_putarow(t, 2 * i + 1, 4, (int[]){0, 1, 2, 3 + i},
                       (double[]){A[i][0], A[i][1], A[i][2], 1.0});
        PRIMAL_putconbound(t, 2 * i + 1, PRIMAL_BK_LO, -b[i], INFINITY);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[9];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 1.0) < 1e-5 && fabs(x[1] - 1.0) < 1e-5 &&
             fabs(x[2] - 1.0) < 1e-5 && fabs(z - 4.0) < 1e-5;
        printf("boyd_l1  x=(%.6g,%.6g,%.6g) ||Ax+b||_1=%.6g (book (1,1,1), 4) %s\n",
               x[0], x[1], x[2], z, ok ? "OK" : "FAIL");
    } else {
        printf("boyd_l1  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
