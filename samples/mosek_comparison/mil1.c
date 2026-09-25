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

/* mil1.c — porting dell'esempio "mil1.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): piccolo MIP intero puro.
 *
 * max 5x0 + 4x1 + 3x2
 * s.t. 2x0 + 3x1 + x2 <= 5
 *      4x0 + x1 + 2x2 <= 11
 *      3x0 + 4x1 + 2x2 <= 8
 *      x0, x1, x2 interi >= 0
 *
 * Soluzione attesa: x = (2, 0, 1), obj = 13
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 3);
    PRIMAL_appendcons(task, 3);

    double c[3] = {5.0, 4.0, 3.0};
    for (int j = 0; j < 3; j++) PRIMAL_putcj(task, j, c[j]);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){2.0, 3.0, 1.0});
    PRIMAL_putarow(task, 1, 3, (int[]){0, 1, 2}, (double[]){4.0, 1.0, 2.0});
    PRIMAL_putarow(task, 2, 3, (int[]){0, 1, 2}, (double[]){3.0, 4.0, 2.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 5.0);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 11.0);
    PRIMAL_putconbound(task, 2, PRIMAL_BK_UP, -INFINITY, 8.0);

    for (int j = 0; j < 3; j++) {
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvartype(task, j, PRIMAL_VAR_TYPE_INT);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("x = (%.0f, %.0f, %.0f), obj = %.4f\n", xx[0], xx[1], xx[2], obj);

    int ok = sta == PRIMAL_SOL_STA_INTEGER_OPTIMAL &&
             fabs(xx[0] - 2) < 1e-6 && fabs(xx[1] - 0) < 1e-6 &&
             fabs(xx[2] - 1) < 1e-6 && fabs(obj - 13.0) < 1e-6;
    printf("%s (atteso x=(2,0,1), obj=13)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}