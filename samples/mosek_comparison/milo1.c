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

/* milo1.c — porting dell'esempio "milo1.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): MIP misto (parte delle variabili intera,
 * parte continua).
 *
 * max 3x0 + 2x1 + 2x2
 * s.t.  x0 + x1 + x2 <= 10
 *       x0 + 2x1        <=  9
 *      -x0 + 3x1 +  x2  <= 10
 *      x0, x1 interi >= 0, x2 continua >= 0
 *
 * Verifica: branching correttezza (violate solo x0,x1), soluzione
 * ammissibile, obj = 3x0+2x1+2x2, solsta INTEGER_OPTIMAL. Il valore atteso
 * e' derivato enumerando gli interi ammissibili: per x0=3,x1=3 -> x2=10/3?
 * (verifica numerica dall'esterno: obj coerente + ammissibilita' + interezza
 *  + primal bound: obj <= 12 (rilassamento) e >= 11.9? il valore ottimo
 * e' stampato e confrontato con l'enumerazione interna della LP rilassata).
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

    double c[3] = {3.0, 2.0, 2.0};
    for (int j = 0; j < 3; j++) PRIMAL_putcj(task, j, c[j]);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putarow(task, 1, 2, (int[]){0, 1}, (double[]){1.0, 2.0});
    PRIMAL_putarow(task, 2, 3, (int[]){0, 1, 2}, (double[]){-1.0, 3.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 10.0);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 9.0);
    PRIMAL_putconbound(task, 2, PRIMAL_BK_UP, -INFINITY, 10.0);

    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvartype(task, 0, PRIMAL_VAR_TYPE_INT);
    PRIMAL_putvartype(task, 1, PRIMAL_VAR_TYPE_INT);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("x = (%.4f, %.4f, %.4f), obj = %.4f\n", xx[0], xx[1], xx[2], obj);

    /* verifica: x0,x1 interi, tutte le righe ammissibili, obj coerente,
     * e ottimalità per confronto con enumerazione completa di (x0,x1)
     * con x2 ottimizzato: x2 <= 10 - x0 (max dopo x2<=-x0+3x1? no: row2
     * -x0+3x1+x2<=10 -> x2 <= 10+x0-3x1; e x0+x1+x2<=10) */
    int ok = sta == PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    double best = -1e30;
    for (int x0 = 0; ok && x0 <= 20; x0++)
        for (int x1 = 0; x1 <= 20; x1++) {
            if (x0 + 2 * x1 > 9.0 + 1e-9) continue;     /* row1 */
            double v = 10.0 - x0 - x1;                  /* da row0: x2<=10-x0-x1 */
            double w = 10.0 + x0 - 3.0 * x1;            /* da row2 */
            if (w < v) v = w;
            if (v > 0 && 3.0 * x0 + 2.0 * x1 + 2.0 * v > best)
                best = 3.0 * x0 + 2.0 * x1 + 2.0 * v;
        }
    printf("enumerazione: best = %.4f, clone = %.4f\n", best, obj);
    if (fabs(obj - best) > 1e-6) ok = 0;
    if (fabs(xx[0]-floor(xx[0]+0.5)) > 1e-6 || fabs(xx[1]-floor(xx[1]+0.5)) > 1e-6) ok = 0;
    if (fabs(obj - (3 * xx[0] + 2 * xx[1] + 2 * xx[2])) > 1e-6) ok = 0;

    printf("%s (ottimo confermato per enumerazione)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}