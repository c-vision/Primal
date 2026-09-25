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

/* pow1.c — cono di potenza (equivalente all'esempio "pgo"/pow della
 * documentazione MOSEK: min x + y s.t. x^0.5 * y^0.5 >= 1, x,y >= 0)
 *
 * Forma cono PPOW (t,u,v): t^a * u^(1-a) >= |v|, a = 0.5.
 * Con (t,u,v) = (x, 1, y): x^0.5 >= y  =>  x >= y^2.
 * Problema equivalente: min x + y s.t. x >= y^2, y >= 0  => y=1, x=1.
 * Il valore y=1 tocca il bordo del dominio PEXP/PPOW (w = v/u = 1 <= cap).
 * Soluzione attesa: obj = 2 in (x, y) = (1, 1).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: t = x, u = y, v = 1 (fissa) */
    PRIMAL_appendvars(task, 3);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_FX, 1.0, 1.0);

    /* min x + y */
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMAL_appendcone(task, PRIMAL_CT_PPOW, 0.5, 3, (int[]){0, 1, 2});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[0], xx[1], obj);

    int ok = fabs(obj - 2.0) < 1e-4 && fabs(xx[0] - 1.0) < 1e-3 &&
             fabs(xx[1] - 1.0) < 1e-3;
    printf("%s (atteso obj = 2.0 in x = y = 1)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}