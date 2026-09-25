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

/* solvelinear.c — porting dell'esempio "solvelinear.jl" della Julia API
 * MOSEK: risolvere un sistema lineare A x = b come problema di ottimizzazione
 * (min ||Ax - b|| su un LP: qui min t s.t. -t <= (Ax-b)_i <= t).
 *
 * Sistema 3x3 (verificato a mano):
 *   A = [2 1 -1; -3 -1 2; -2 1 2],  b = (8, -11, -3)
 * Soluzione: x = (2, 3, -1) (sostituzione diretta).
 * Il LP: min t con |Ax-b| <= t ha ottimo t=0 e x=(2,3,-1).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: x0, x1, x2, t */
    PRIMAL_appendvars(task, 4);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, 3, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 3, 1.0);    /* min t */

    /* righe: (Ax)_i - t <= b_i  e  -(Ax)_i - t <= -b_i */
    PRIMAL_appendcons(task, 6);
    double A[3][3] = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    double b[3] = {8, -11, -3};
    for (int i = 0; i < 3; i++) {
        int sub[4] = {0, 1, 2, 3};
        double v[4] = {A[i][0], A[i][1], A[i][2], -1.0};
        PRIMAL_putarow(task, i, 4, sub, v);
        PRIMAL_putconbound(task, i, PRIMAL_BK_UP, -INFINITY, b[i]);
        double v2[4] = {-A[i][0], -A[i][1], -A[i][2], -1.0};
        PRIMAL_putarow(task, 3 + i, 4, sub, v2);
        PRIMAL_putconbound(task, 3 + i, PRIMAL_BK_UP, -INFINITY, -b[i]);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[4], po;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    printf("x = (%.4f, %.4f, %.4f), t = %.6f (atteso (2, 3, -1), t=0)\n",
           xx[0], xx[1], xx[2], xx[3]);

    int ok = fabs(xx[0] - 2.0) < 1e-6 &&
             fabs(xx[1] - 3.0) < 1e-6 &&
             fabs(xx[2] + 1.0) < 1e-6 &&
             fabs(po) < 1e-6;
    printf("%s (sistema lineare risolto: residuo ~ 0)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
