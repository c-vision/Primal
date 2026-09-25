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

/* qo1.c — QP (equivale all'esempio Python Optimizer API "qo1")
 *
 * Problema: min x^2 + y^2  s.t. x+y >= 1, x,y >= 0
 * Soluzione attesa: x = 0.5, y = 0.5, obj = 0.5
 *
 * Nota: putqobj riceve il triangolo inferiore/0.5·q per entry — il clone
 * usa la stessa convenzione di MOSEK (metà dei termini fuori diagonale;
 * la Q va data con coefficiente pieno: Q = diag(2) => 0.5*x'Qx = x^2+y^2).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    int numvar = 2, numcon = 1;
    PRIMAL_appendvars(task, numvar);
    PRIMAL_appendcons(task, numcon);

    for (int j = 0; j < numvar; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);

    /* Obiettivo quadratico: min x^2 + y^2 */
    PRIMAL_putqobj(task, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});  /* Q = diag(2): obiettivo 0.5*x'Qx = x^2+y^2 */
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* Vincolo: x + y >= 1 */
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 1.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[0], xx[1], obj);

    int ok = fabs(xx[0] - 0.5) < 1e-6 && fabs(xx[1] - 0.5) < 1e-6 &&
             fabs(obj - 0.5) < 1e-6;
    printf("%s\n", ok ? "OK (atteso x=0.5, y=0.5, obj=0.5)" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}