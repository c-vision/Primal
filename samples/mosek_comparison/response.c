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

/* response.c — porting dell'esempio "response.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): cattura l'output di log dell'ottimizzatore
 * tramite la callback di stream PRIMAL_linkfunctotaskstream(PRIMAL_STREAM_LOG).
 *
 * Modello LP identico a lo1: min -2x0-3x1 s.c. x0+x1<=4, x0+3x1<=6,
 * ottimo x=(3,1), obj=-9. Il test verifica che la callback riceva davvero
 * messaggi di log e che il risultato sia corretto.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "primal.h"

static int msg_count = 0;
static char first_msg[64] = "";

static void logcb(void *handle, const char *msg) {
    (void)handle;
    if (msg_count == 0 && msg && strlen(msg) < sizeof(first_msg))
        strcpy(first_msg, msg);
    msg_count++;
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 2);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMAL_putcj(task, 0, -2.0);
    PRIMAL_putcj(task, 1, -3.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 4.0);
    PRIMAL_putarow(task, 1, 2, (int[]){0, 1}, (double[]){1.0, 3.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 6.0);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_RA, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_RA, 0.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_linkfunctotaskstream(task, PRIMAL_STREAM_LOG, NULL, logcb);
    if (rc != PRIMAL_RES_OK) { printf("link rc=%d\n", rc); return 1; }

    rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double x[2], po;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    printf("x = (%.6f, %.6f), obj = %.6f\n", x[0], x[1], po);
    printf("log ricevuto: %d messaggi (primo: \"%s\")\n", msg_count, first_msg);

    int ok = msg_count > 0 &&
             fabs(x[0] - 3.0) < 1e-6 && fabs(x[1] - 1.0) < 1e-6 &&
             fabs(po + 9.0) < 1e-6;
    printf("%s (callback stream + ottimo)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}