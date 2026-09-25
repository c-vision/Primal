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

/* pinfeas.c — porting dell'esempio "pinfeas.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): un LP primariamente infeasible.
 * Il solver deve rilevare l'infeasibility e produrre il certificato.
 *
 * min 0  s.t. x0 + x1 >= 4,  x0 + x1 <= 2,  x >= 0
 * (le due righe si escludono a vicenda: nessun punto ammissibile)
 *
 * Atteso: rc = PRIMAL_RES_ERR_INFEASIBLE, solsta = PRIM_INFEAS_CER
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
    PRIMAL_appendcons(task, 2);

    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putarow(task, 1, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 4.0, INFINITY);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 2.0);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("rc = %d (ERR_INFEASIBLE = %d), solsta = %d (PRIM_INFEAS_CER = %d)\n",
           (int)rc, (int)PRIMAL_RES_ERR_INFEASIBLE, (int)sta, (int)PRIMAL_SOL_STA_PRIM_INFEAS_CER);

    int ok = rc == PRIMAL_RES_ERR_INFEASIBLE && sta == PRIMAL_SOL_STA_PRIM_INFEAS_CER;
    printf("%s (atteso infeasible con certificato)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}