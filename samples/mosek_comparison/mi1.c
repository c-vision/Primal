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

/* mi1.c — MIP (equivale all'esempio Julia Optimizer API, putacolslice)
 *
 * Problema: max x1 + 0.64*x2
 *           s.t. 50*x1 + 31*x2 <= 250
 *                 3*x1 - 2*x2 >= -4
 *                 x1, x2 interi >= 0
 * Soluzione attesa: x1 = 5, x2 = 0, obj = 5.0000
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendcons(task, 2);
    PRIMAL_appendvars(task, 2);

    /* bkx = LO, blx = 0, bux = Inf */
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);

    /* c = [1.0, 0.64] (putclist) */
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 0.64);

    /* A per colonne (putacolslice):
     * col 1: 50 in row1, 3 in row2; col 2: 31 in row1, -2 in row2 */
    PRIMAL_putacol(task, 0, 2, (int[]){0, 1}, (double[]){50.0, 3.0});
    PRIMAL_putacol(task, 1, 2, (int[]){0, 1}, (double[]){31.0, -2.0});

    /* bkc = [UP, LO], blc = [-Inf, -4], buc = [250, Inf] */
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 250.0);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_LO, -4.0, INFINITY);

    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    /* putvartypelist: entrambe intere */
    PRIMAL_putvartype(task, 0, PRIMAL_VAR_TYPE_INT);
    PRIMAL_putvartype(task, 1, PRIMAL_VAR_TYPE_INT);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("solsta = %d\n", (int)sta);
    printf("x1 = %.0f, x2 = %.0f, obj = %.4f\n", xx[0], xx[1], obj);

    int ok = fabs(xx[0] - 5.0) < 1e-6 && fabs(xx[1] - 0.0) < 1e-6 &&
             fabs(obj - 5.0) < 1e-6 && sta == PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    printf("%s\n", ok ? "OK (atteso x1=5, x2=0, obj=5.0000)" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}