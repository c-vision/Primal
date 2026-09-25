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

/* simple.c — porting dell'esempio "simple.jl" della MOSEK Julia API:
 * il LP piu' semplice possibile con verifica dell'intera catena API.
 *
 *   max x0 + x1  s.t.  x0 + x1 = 1,  0 <= x0, x1 <= 1
 * Ottimo: qualsiasi punto sulla riga con obj=1 (soluzione degenere:
 * verifica del pobj, non del punto).
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
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], y[1], po, dobj;
    PRIMALsolstae sta;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_gety(task, PRIMAL_SOL_ITR, y);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(task, PRIMAL_SOL_ITR, &dobj);
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("x = (%.4f, %.4f), y = (%.4f)\n", xx[0], xx[1], y[0]);
    printf("primal obj = %.4f, dual obj = %.4f, sta = %d\n", po, dobj, sta);

    int ok = sta == PRIMAL_SOL_STA_OPTIMAL &&
             fabs(po - 1.0) < 1e-9 &&
             fabs(dobj - 1.0) < 1e-6 &&          /* strong duality */
             fabs(xx[0] + xx[1] - 1.0) < 1e-6 &&  /* fattibilita' */
             fabs(xx[0] - (1.0 - xx[1])) < 1e-6;
    printf("%s (atteso obj=1, pobj=dobj, fattibile)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
