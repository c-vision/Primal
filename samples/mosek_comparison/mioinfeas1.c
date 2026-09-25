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

/* mioinfeas1.c — porting dell'esempio "mioinfeas1.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): un MIP infeasible.
 *
 * min 0  s.t. x >= 5,  x <= 2,  x intero >= 0
 *
 * Atteso (tabella 7.3 del reference, problemi interi): prosta = PRIM_INFEAS,
 * solsta = UNKNOWN. Il reference non pubblica mai un certificato di
 * infeasibilita' come solsta su un MIP: quel numero e' riservato a un vettore
 * di Farkas, che un branch-and-bound non produce. rc resta ERR_INFEASIBLE.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 1);
    PRIMAL_appendcons(task, 2);
    PRIMAL_putarow(task, 0, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putarow(task, 1, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 5.0, INFINITY);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 2.0);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvartype(task, 0, PRIMAL_VAR_TYPE_INT);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    PRIMALsolstae sta; PRIMALprostae pro;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    PRIMAL_getprosta(task, PRIMAL_SOL_ITR, &pro);
    printf("rc = %d, prosta = %d, solsta = %d\n", (int)rc, (int)pro, (int)sta);

    int ok = rc == PRIMAL_RES_ERR_INFEASIBLE
          && pro == PRIMAL_PRO_STA_PRIM_INFEAS
          && sta == PRIMAL_SOL_STA_UNKNOWN;
    printf("%s (atteso MIP infeasible: prosta PRIM_INFEAS + solsta UNKNOWN)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}