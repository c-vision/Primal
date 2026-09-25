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

/* boyd_lp_unique.c - LP with unique optimum, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Exercise 5.28 (lines 9281-9289):
 *
 *   minimize   47 x1 + 93 x2 + 17 x3 - 93 x4
 *   subject to
 *      [ -1  -6   1   3 ] [x1]     [ -3 ]
 *      [ -1  -2   7   1 ] [x2]  <= [  5 ]
 *      [  0   3 -10  -1 ] [x3]     [ -8 ]
 *      [ -6 -11  -2  12 ] [x4]     [ -7 ]
 *      [  1   6  -1  -3 ]           [  4 ]
 *
 * The book states the optimal solution is unique and equal to
 *   x* = (1, 1, 1, 1).
 *
 * The sample solves the LP with this library and checks the optimum (the
 * objective at x* is 64).
 *
 * Usage: boyd_lp_unique   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double A[5][4] = {
        {-1, -6,   1,   3}, {-1, -2,   7,   1}, { 0,   3, -10,  -1},
        {-6, -11, -2,  12}, { 1,   6,  -1,  -3}};
    static const double b[5] = {-3, 5, -8, -7, 4};
    static const double c[4] = {47, 93, 17, -93};

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 5, 4, &t);
    for (int j = 0; j < 4; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putcj(t, j, c[j]);
    }
    for (int i = 0; i < 5; i++) {
        PRIMAL_putarow(t, i, 4, (int[]){0, 1, 2, 3}, A[i]);
        PRIMAL_putconbound(t, i, PRIMAL_BK_UP, -INFINITY, b[i]);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[4];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = 1;
        for (int j = 0; j < 4; j++) ok = ok && fabs(x[j] - 1.0) < 1e-5;
        printf("boyd_lp_unique  x=(%.6g,%.6g,%.6g,%.6g) obj=%.6g (book (1,1,1,1)) %s\n",
               x[0], x[1], x[2], x[3], z, ok ? "OK" : "FAIL");
    } else {
        printf("boyd_lp_unique  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
