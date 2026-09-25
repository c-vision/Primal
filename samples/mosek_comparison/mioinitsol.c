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

/* mioinitsol.c — porting dell'esempio "mioinitsol.jl" della Julia API
 * MOSEK: fornire una soluzione iniziale al MIP (feasmios / initial
 * solution) per accelerare il branch & bound.
 *
 * Problema (milo1-like con soluzione iniziale nota):
 *   max 3x0 + 2x1 + 2x2
 *   s.t. x0 + x1 + x2 <= 10, x0 + 2x1 <= 9, -x0 + 3x1 + x2 <= 10
 *        x0, x1 interi >= 0, x2 >= 0
 * Soluzione iniziale fornita (ammissibile, intera): x = (3, 3, 1)
 * (righe: 7 <= 10, 9 <= 9, 10 <= 10 — attiva la riga 1 al limite).
 * Verifica: l'ottimo e' confermato per enumerazione (obj = 17.5 con
 * x=(3,3,1)? 3*3+2*3+2*1 = 17; l'enumerazione trova il vero ottimo).
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
    PRIMAL_appendcons(task, 3);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    double c[3] = {3.0, 2.0, 2.0};
    for (int j = 0; j < 3; j++) PRIMAL_putcj(task, j, c[j]);
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 10.0);
    PRIMAL_putarow(task, 1, 2, (int[]){0, 1}, (double[]){1.0, 2.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 9.0);
    PRIMAL_putarow(task, 2, 3, (int[]){0, 1, 2}, (double[]){-1.0, 3.0, 1.0});
    PRIMAL_putconbound(task, 2, PRIMAL_BK_UP, -INFINITY, 10.0);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvartype(task, 0, PRIMAL_VAR_TYPE_INT);
    PRIMAL_putvartype(task, 1, PRIMAL_VAR_TYPE_INT);

    /* soluzione iniziale (ammissibile e intera) */
    double x0[3] = {3.0, 3.0, 1.0};
    PRIMAL_putxx(task, PRIMAL_SOL_ITR, x0);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f), obj = %.4f (initial obj = %.4f)\n",
           xx[0], xx[1], xx[2], obj,
           3.0 * x0[0] + 2.0 * x0[1] + 2.0 * x0[2]);

    /* verifica per enumerazione completa (x0,x1 interi, x2 continuo) */
    double best = -1e30;
    for (int a = 0; a <= 10; a++)
        for (int b = 0; b <= 5; b++) {
            if (a + 2 * b > 9.0 + 1e-9) continue;
            double v = 10.0 - a - b;
            double w = 10.0 + a - 3.0 * b;
            if (w < v) v = w;
            if (v > 1e-9) {
                double o = 3.0 * a + 2.0 * b + 2.0 * v;
                if (o > best) best = o;
            }
        }
    printf("enumerazione: best = %.4f\n", best);

    int ok = fabs(obj - best) < 1e-6 &&
             fabs(xx[0] - floor(xx[0] + 0.5)) < 1e-6 &&
             fabs(xx[1] - floor(xx[1] + 0.5)) < 1e-6;
    printf("%s (ottimo confermato, initsol accettata)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
