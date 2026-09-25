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

/* mico1.c — porting dell'esempio "mico1.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): mixed-integer conic optimization
 * (MIP + coni quadratrici).
 *
 * Problema (verificato a mano):
 *   max x0
 *   s.t. (t, x0, x1) in QUAD   (t >= ||(x0, x1)||),  t = 1
 *        x0 + x1 >= 0.5,  x0 intera >= 0, x1 >= 0
 * Il cono dà x0^2 + x1^2 <= 1; x0 intero -> x0 in {0, 1}. Con x0=1:
 * x1 = 0 e x0+x1 = 1 >= 0.5 ok -> obj = 1. Ottimo: x = (1, 1, 0), obj 1.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: t, x0, x1 */
    PRIMAL_appendvars(task, 3);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);       /* t = 1 */
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvartype(task, 1, PRIMAL_VAR_TYPE_INT);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    /* riga: x0 + x1 >= 0.5 */
    PRIMAL_putarow(task, 0, 2, (int[]){1, 2}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 0.5, INFINITY);

    /* cono: (t, x0, x1) in QUAD */
    PRIMAL_appendcone(task, PRIMAL_CT_QUAD, 0.0, 3, (int[]){0, 1, 2});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f), obj = %.4f\n", xx[1], xx[2], obj);

    /* verifica per enumerazione: x0 intero in {0,1}, cono + riga */
    double best = -1e30;
    for (int a = 0; a <= 1; a++) {
        double xmax = sqrt(1.0 - (double)a * a);   /* cono: x1 <= sqrt(1-a^2) */
        double need = 0.5 - a;                     /* riga: x1 >= 0.5-a */
        if (need > xmax + 1e-12) continue;          /* x1 non puo' coprire la riga */
        double o = (double)a;
        if (o > best) best = o;
    }
    printf("enumerazione: best = %.4f\n", best);

    int ok = fabs(obj - best) < 1e-6 &&
             fabs(xx[1] - floor(xx[1] + 0.5)) < 1e-6;
    printf("%s (MIP+cono: ottimo per enumerazione)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
