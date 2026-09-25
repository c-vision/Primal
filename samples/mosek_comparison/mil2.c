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

/* mil2.c — porting dell'esempio "mil2.jl" (C API docs.mosek.com):
 * variabili semi-continue e semi-intere.
 *
 * Problema (verificato a mano): min x0 + 2*x1 s.t. x0 + x1 >= 3 con
 *   x0 semi-cont inua [2,10]  (x0 = 0 oppure 2 <= x0 <= 10)
 *   x1 semi-intera   [2.5,10] (x1 = 0 oppure 2.5 <= x1 <= 10 intera)
 * Enumerazione: x0=0 -> x1 >= 3 intero -> min x1=3, cost 6;
 *               x1=0 -> x0 >= 3, min x0=3, cost 3;  <-- migliore
 *               entrambi attivi -> x0>=2, x1>=2.5(->3), cost >= 2+6=8
 * Ottimo: x=(3,0), obj=3.
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
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_RA, 2.0, 10.0);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_RA, 2.5, 10.0);
    PRIMAL_putvartype(task, 0, PRIMAL_VAR_TYPE_SEMI_CONT);
    PRIMAL_putvartype(task, 1, PRIMAL_VAR_TYPE_SEMI_INT);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 2.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 3.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f), obj = %.4f\n", xx[0], xx[1], obj);

    /* enumerazione esaustiva dei casi semi (attivo/disattivo) */
    double best = INFINITY;
    for (int a = 0; a <= 1; a++)          /* x0 attivo? */
        for (int b = 0; b <= 1; b++) {    /* x1 attivo? */
            double v1 = b ? 3.0 : 0.0;
            /* minimizza x0+2x1 con x0+x1>=3 nei domini */
            double x0 = a ? fmax(2.0, 3.0 - v1) : 0.0;
            double x1 = b ? ceil(fmax(2.5, 3.0 - (a ? 2.0 : 0.0))) : 0.0;
            if (!a && !b) continue;                       /* 0+0 < 3 */
            if (!a && x1 < 3.0 - 1e-9 && b) continue;     /* x1 copre da sola? gestito */
            double cov = (a ? x0 : 0.0) + (b ? x1 : 0.0);
            if (!a && !b) continue;
            if (cov < 3.0 - 1e-9) {
                /* il membro attivo deve coprire da solo o col compagno */
                if (a && !b) x0 = 3.0;
                if (!a && b) x1 = 3.0;
                if (a && b) continue;   /* entrambi attivi: copertura minima 2+3=5 ok */
                cov = (a ? x0 : 0.0) + (b ? x1 : 0.0);
            }
            double c = (a ? x0 : 0.0) + 2.0 * (b ? x1 : 0.0);
            if (cov >= 3.0 - 1e-9 && c < best) best = c;
        }
    printf("enumerazione: best = %.4f\n", best);

    int ok = fabs(obj - 3.0) < 1e-6 && fabs(obj - best) < 1e-6 &&
             fabs(xx[0] - 3.0) < 1e-6 && fabs(xx[1]) < 1e-6;
    printf("%s (atteso x=(3,0), obj=3)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
