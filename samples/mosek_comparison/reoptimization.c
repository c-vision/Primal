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
 *
 * Port of a MOSEK example (see the comment below), rewritten against
 * the PrimalSolver (PRIMAL_*) API.  The MOSEK examples are Copyright (c)
 * MOSEK ApS; this port re-implements the same optimization problem and is
 * distributed under the Apache License, Version 2.0.  PrimalSolver is not
 * affiliated with, or endorsed by, MOSEK.
 */

/* reoptimization.c — porting dell'esempio "reoptimization.jl" della Julia
 * MOSEK API: risolvere, modificare un bound, ri-ottimizzare.
 *
 * Sequenza (verificata a mano):
 *   1. max x0 + x1 s.t. x0 + x1 <= 4, x >= 0            -> obj 4
 *   2. il bound della riga passa a 10                    -> obj 10
 *   3. la riga passa a 2                                 -> obj 2
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});

    double steps[3] = {4.0, 10.0, 2.0};
    double expect[3] = {4.0, 10.0, 2.0};
    int ok = 1;
    for (int k = 0; k < 3; k++) {
        PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, steps[k]);
        PRIMALrescodee rc = PRIMAL_optimize(task);
        if (rc != PRIMAL_RES_OK) { printf("optimize %d rc=%d\n", k, rc); return 1; }
        double po;
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
        printf("passo %d (rhs=%g): obj = %.4f (atteso %.4f)\n",
               k + 1, steps[k], po, expect[k]);
        if (fabs(po - expect[k]) > 1e-6) ok = 0;
    }
    printf("%s (reoptimization: 3 risolve successive)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
