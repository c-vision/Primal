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

/* concurrent1.c — porting dell'esempio "concurrent1.jl" della Julia API
 * MOSEK: optimizer concorrente — piu' algoritmi in competizione sullo
 * stesso problema, si prende il migliore.
 * Il clone e' single-process: i tre optimizer (FREE, INTPNT, PRIMAL_SIMPLEX,
 * DUAL_SIMPLEX) girano in SEQUENZA sullo stesso problema e si confrontano
 * i pobj (deviazione documentata: nessun vero parallelismo).
 *
 * Problema: lo1-like: max 3x0+x1+5x2+x3 con righe EQ/LO/UP (T1 della
 * suite, pobj noto 250/3 ~ 83.333).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static void build(PRIMALtask_t t) {
    PRIMAL_appendvars(t, 4);
    PRIMAL_appendcons(t, 3);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    double c[4] = {3, 1, 5, 1};
    for (int j = 0; j < 4; j++) PRIMAL_putcj(t, j, c[j]);
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 30, 30);
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 15, 0);
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, 0, 25);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0, 0);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_RA, 0, 10);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_LO, 0, 0);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_LO, 0, 0);
    PRIMAL_putacol(t, 0, 2, (int[]){0, 1}, (double[]){3, 2});
    PRIMAL_putacol(t, 1, 3, (int[]){0, 1, 2}, (double[]){1, 1, 2});
    PRIMAL_putacol(t, 2, 2, (int[]){0, 1}, (double[]){2, 3});
    PRIMAL_putacol(t, 3, 2, (int[]){1, 2}, (double[]){1, 3});
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    const char *names[4] = {"FREE", "INTPNT", "PRIMAL_SIMPLEX", "DUAL_SIMPLEX"};
    int opt[4] = {PRIMAL_OPTIMIZER_FREE, PRIMAL_OPTIMIZER_INTPNT,
                  PRIMAL_OPTIMIZER_PRIMAL_SIMPLEX, PRIMAL_OPTIMIZER_DUAL_SIMPLEX};
    double best = -INFINITY;
    int bopt = -1;
    double vals[4];
    int all_ok = 1;

    for (int k = 0; k < 4; k++) {
        PRIMALtask_t task;
        PRIMAL_maketask(env, 0, 0, &task);
        build(task);
        PRIMAL_putintparam(task, PRIMAL_IPAR_OPTIMIZER, opt[k]);
        PRIMALrescodee rc = PRIMAL_optimize(task);
        double po = -INFINITY;
        if (rc == PRIMAL_RES_OK) {
            PRIMALsolstae sta;
            PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
            if (sta == PRIMAL_SOL_STA_OPTIMAL) {
                PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
                printf("%-16s pobj = %.6f\n", names[k], po);
            } else all_ok = 0;
        } else all_ok = 0;
        vals[k] = po;
        if (po > best) { best = po; bopt = k; }
        PRIMAL_deletetask(&task);
    }
    printf("migliore: %s con pobj = %.6f (atteso 83.333333 = 250/3)\n",
           bopt >= 0 ? names[bopt] : "?", best);

    int ok = all_ok && fabs(best - 250.0 / 3.0) < 1e-6;
    for (int k = 0; k < 4; k++)
        if (fabs(vals[k] - best) > 1e-6) ok = 0;   /* tutti d'accordo */
    printf("%s (4 optimizer, stesso ottimo — concorrenza simulata)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
