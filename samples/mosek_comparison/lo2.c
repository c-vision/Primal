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

/* lo2.c — porting dell'esempio "lo2.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): LP con 4 variabili e 3 vincoli
 * (uno di uguaglianza, uno upper-bounded, uno lower-bounded).
 *
 * max 3x0 + x1 + 5x2 + x3
 * s.t. 3x0 + x1 + 2x2       = 30
 *      2x0 + x1 + 3x2 + x3 <= 25
 *       x0 + 2x1 +  x2 + 2x3 >= 15
 *      0 <= x0, 0 <= x1 <= 10, x2, x3 >= 0
 *
 * Verifica: primal = dual (no duality gap) e cross-check BAS vs ITR.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 4);
    PRIMAL_appendcons(task, 3);

    double c[4] = {3.0, 1.0, 5.0, 1.0};
    for (int j = 0; j < 4; j++) PRIMAL_putcj(task, j, c[j]);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    /* A per righe */
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){3.0, 1.0, 2.0});
    PRIMAL_putarow(task, 1, 4, (int[]){0, 1, 2, 3}, (double[]){2.0, 1.0, 3.0, 1.0});
    PRIMAL_putarow(task, 2, 4, (int[]){0, 1, 2, 3}, (double[]){1.0, 2.0, 1.0, 2.0});

    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 30.0, 30.0);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 25.0);
    PRIMAL_putconbound(task, 2, PRIMAL_BK_LO, 15.0, INFINITY);

    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_RA, 0.0, 10.0);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 3, PRIMAL_BK_LO, 0.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[4], po, dobj, xb[4];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(task, PRIMAL_SOL_ITR, &dobj);
    PRIMAL_getxx(task, PRIMAL_SOL_BAS, xb);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("x = (%.6f, %.6f, %.6f, %.6f)\n", xx[0], xx[1], xx[2], xx[3]);
    printf("pobj = %.6f, dobj = %.6f, solsta = %d\n", po, dobj, (int)sta);

    /* primal feasibility */
    int ok = sta == PRIMAL_SOL_STA_OPTIMAL;
    double r0 = 3 * xx[0] + xx[1] + 2 * xx[2];
    double r1 = 2 * xx[0] + xx[1] + 3 * xx[2] + xx[3];
    double r2 = xx[0] + 2 * xx[1] + xx[2] + 2 * xx[3];
    if (fabs(r0 - 30.0) > 1e-6 || r1 > 25.0 + 1e-6 || r2 < 15.0 - 1e-6) ok = 0;
    /* duality gap nullo + BAS/ITR coerenti */
    if (fabs(po - dobj) > 1e-6 * (1 + fabs(po))) ok = 0;
    if (fabs(po - (c[0]*xx[0]+c[1]*xx[1]+c[2]*xx[2]+c[3]*xx[3])) > 1e-6) ok = 0;
    for (int j = 0; j < 4; j++) if (fabs(xx[j] - xb[j]) > 1e-6) ok = 0;

    printf("%s (ottimo certificato da pobj = dobj)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}