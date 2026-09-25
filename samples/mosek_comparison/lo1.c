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

/* lo1.c — LP semplice (equivale all'esempio Python Optimizer API "lo1")
 *
 * Problema: min 2x + 3y  s.t. x+y >= 1, x-y <= 0, x,y >= 0
 * Soluzione attesa: x = 0.5, y = 0.5, obj = 2.5
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    int numvar = 2, numcon = 2;
    PRIMAL_appendvars(task, numvar);
    PRIMAL_appendcons(task, numcon);

    /* Bounds: x, y >= 0 */
    for (int j = 0; j < numvar; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);

    /* Obiettivo: min 2x + 3y */
    PRIMAL_putcj(task, 0, 2.0);
    PRIMAL_putcj(task, 1, 3.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* Vincolo 1: x + y >= 1 */
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 1.0, INFINITY);

    /* Vincolo 2: x - y <= 0 */
    PRIMAL_putarow(task, 1, 2, (int[]){0, 1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 0.0);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_BAS, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_BAS, &obj);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_BAS, &sta);
    printf("solsta = %d\n", (int)sta);
    printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[0], xx[1], obj);

    int ok = fabs(xx[0] - 0.5) < 1e-6 && fabs(xx[1] - 0.5) < 1e-6 &&
             fabs(obj - 2.5) < 1e-6 && sta == PRIMAL_SOL_STA_OPTIMAL;
    printf("%s\n", ok ? "OK (atteso x=0.5, y=0.5, obj=2.5)" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}