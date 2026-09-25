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

/* ceo1.c — cono esponenziale (variante dell'esempio "ceo1" della
 * documentazione MOSEK Optimizer API)
 *
 * Problema: min x0  s.t. (x0, x1, x2) in EXP, x1 = 1, -1 <= x2 <= -0.5
 *           cioe' x0 >= x1 * exp(x2/x1) = exp(x2)
 * Soluzione attesa: x0 = exp(-1) = 0.367879 in x2 = -1
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

    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_RA, -1.0, -0.5);

    /* min x0 */
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, (int[]){0, 1, 2});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.6f, %.6f, %.6f), obj = %.6f\n", xx[0], xx[1], xx[2], obj);

    int ok = fabs(obj - exp(-1.0)) < 1e-6 && fabs(xx[2] - (-1.0)) < 1e-6;
    printf("%s (atteso obj = e^-1 = 0.367879 in x2 = -1)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}