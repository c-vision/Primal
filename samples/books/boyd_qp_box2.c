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

/* boyd_qp_box2.c - barrier-method example, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Exercise 11.1 (lines 19175-19179):
 *
 *   minimize   x^2 + 1
 *   subject to 2 <= x <= 4
 *
 * The book states the feasible set is [2, 4] and the optimal point x* = 2.
 *
 * The sample solves the (trivial) QP with this library and checks x* = 2
 * (objective x^2 = 4 plus the constant 1 gives 5).
 *
 * Usage: boyd_qp_box2   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 1, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_RA, 2.0, 4.0);
    PRIMAL_putcj(t, 0, 0.0);
    PRIMAL_putqobj(t, 1, (int[]){0}, (int[]){0}, (double[]){2.0});  /* 0.5*2*x^2 = x^2 */

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[1];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 2.0) < 1e-5;
        printf("boyd_qp_box2  x=%.6g f=%.6g (book x*=2, f=5) %s\n",
               x[0], z + 1.0, ok ? "OK" : "FAIL");
    } else {
        printf("boyd_qp_box2  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
