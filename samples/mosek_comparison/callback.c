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

/* callback.c — porting dell'esempio "callback.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): progress callback durante l'ottimizzazione.
 *
 * Registra PRIMAL_setprogresscb + PRIMAL_setinfoconnname e risolve un piccolo
 * portfolio QP; verifica che il callback riceva >= 1 notifica di progresso
 * e che la soluzione sia quella attesa (min variance portfolio, verifica
 * del valore con la formula analitica su 2 asset correlati).
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "primal.h"

static int n_calls = 0;
static char last_info[256] = "";

static void progcb(void *handle, const char *info) {
    (void)handle;
    n_calls++;
    strncpy(last_info, info, sizeof last_info - 1);
    last_info[sizeof last_info - 1] = '\0';
    printf("  [progress] %s\n", info);
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* min 0.5*(x0^2 + x1^2) s.t. x0 + x1 = 1 (varianza minima, 2 asset
     * scorrelati): ottimo x* = (0.5, 0.5), var = 0.25 */
    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    for (int j = 0; j < 2; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putqobj(task, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMAL_setprogresscb(task, progcb, NULL);
    PRIMAL_setinfoconnname(task, "callback-demo");
    n_calls = 0;

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.6f, %.6f), obj = %.6f, callback calls = %d\n",
           xx[0], xx[1], obj, n_calls);
    printf("last info: %s\n", last_info);

    int ok = fabs(xx[0] - 0.5) < 1e-6 && fabs(xx[1] - 0.5) < 1e-6 &&
             fabs(obj - 0.25) < 1e-6 && n_calls >= 1;
    printf("%s (atteso x=(0.5,0.5), obj=0.25, >=1 chiamate)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
