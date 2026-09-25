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

/* omf_bb.c - branch-and-bound examples from "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 11.3.2 ("Branch and bound - An
 * example"). Two pure integer linear programs on the same feasible region:
 *
 *   -x1 + x2 <= 2,  8 x1 + 2 x2 <= 19,  x1, x2 >= 0 integer.
 *
 * (A) max x1 + x2    -> the book's optimum x = (1, 3), objective 4.
 * (B) max 3 x1 + x2  -> the book's optimum x = (2, 1), objective 7.
 *
 * The sample solves both as MILPs with this library and checks the book's
 * optima. (The book obtains them by hand branch and bound; here the solver's
 * own B&B must land on the same integral points.)
 *
 * Usage: omf_bb   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static int solve(const char *name, double c1, double c2,
                 double want_obj, double wx1, double wx2) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2, 2, &t);
    for (int j = 0; j < 2; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT);
    }
    PRIMAL_putcj(t, 0, -c1); PRIMAL_putcj(t, 1, -c2);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){-1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 2.0);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 1}, (double[]){8.0, 2.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 19.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(z + want_obj) < 1e-6 && fabs(x[0] - wx1) < 1e-6 && fabs(x[1] - wx2) < 1e-6;
        printf("omf_bb %-3s x=(%.6g,%.6g) obj=%.6g (book x=(%.4g,%.4g), %.4g) %s\n",
               name, x[0], x[1], -z, wx1, wx2, want_obj, ok ? "OK" : "FAIL");
    } else {
        printf("omf_bb %-3s rc=%d FAIL\n", name, (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    int a = solve("(A)", 1.0, 1.0, 4.0, 1.0, 3.0);
    int b = solve("(B)", 3.0, 1.0, 7.0, 2.0, 1.0);
    return (a && b) ? 0 : 1;
}
