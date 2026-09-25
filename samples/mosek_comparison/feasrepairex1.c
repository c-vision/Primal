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

/* feasrepairex1.c — porting dell'esempio "feasrepairex1.jl" della Julia API
 * MOSEK: feasibility repair — un LP infeasibile viene "riparato" trovando
 * il punto con la minima violazione totale (elastic relaxation, PRIMAL_feasrepair).
 *
 * Problema infeasibile (verificato a mano):
 *   min x0  s.t.  x0 >= 2,  x0 <= 1   (nessun x ammissibile)
 * Riparazione minima: la violazione totale minima e' 1 (muovere un bound di
 * 1): la somma delle slack elastiche ottimali = 1. Il punto riparato
 * soddisfa la definizione con violazione totale 1.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 1);
    PRIMAL_appendcons(task, 2);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putarow(task, 0, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 2.0, INFINITY);   /* x0 >= 2 */
    PRIMAL_putarow(task, 1, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 1.0);  /* x0 <= 1 */

    /* 1. optimize: infeasibile */
    PRIMALrescodee rc = PRIMAL_optimize(task);
    printf("optimize: rc=%d (atteso 1002 infeasible)\n", rc);
    int infeas = (rc == PRIMAL_RES_ERR_INFEASIBLE);

    /* 2. feasrepair: punto a violazione minima */
    rc = PRIMAL_feasrepair(task);
    if (rc != PRIMAL_RES_OK) { printf("feasrepair rc=%d\n", rc); return 1; }
    double x[1];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    /* violazione totale del punto riparato: (2 - x0)+ + (x0 - 1)+ = 1 */
    double viol = 0.0;
    if (x[0] < 2.0) viol += 2.0 - x[0];
    if (x[0] > 1.0) viol += x[0] - 1.0;
    printf("feasrepair: x0 = %.4f, violazione totale = %.4f (attesa = 1)\n",
           x[0], viol);

    int ok = infeas && fabs(viol - 1.0) < 1e-6;
    printf("%s (repair a violazione minima)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
