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

/* sdo1.c — SDP (variante dell'esempio "sdo1" della documentazione MOSEK)
 *
 * Problema: min tr(X)  s.t. <J,X> = 1 (J = matrice di tutti 1, 2x2), X >= 0
 * Analisi: X = [[a,b],[b,c]] PSD, a+c+2b = 1  =>  tr = s = a+c,
 *          ac >= (1-s)^2/4 e ac <= s^2/4  =>  s >= 1/2.
 * Soluzione attesa: tr(X) = 0.5 con X = [[0.25,0.25],[0.25,0.25]]
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili bar: una X 2x2 */
    int dim = 2;
    PRIMAL_appendbarvars(task, 1, &dim);

    /* simmat: J = [[1,1],[1,1]] = E00 + E11 + 2*E01 (entry (0,1) conta due
     * volte nel prodotto interno: si fornisce solo un triangolo) */
    int mJ;
    PRIMAL_appendsparsesymmat(task, 2, 3, (int[]){0, 1, 0}, (int[]){0, 1, 1},
                           (double[]){1.0, 1.0, 1.0}, &mJ);

    /* vincolo: <J,X> = 1  <=>  ⟨[[1,1],[1,1]],X⟩ = 1 con entry (0,1) doppia */
    PRIMAL_appendcons(task, 1);
    PRIMAL_putbaraij(task, 0, 0, 1, (int[]){mJ}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    /* obiettivo: min tr(X) = <I,X> */
    int mI;
    PRIMAL_appendsparsesymmat(task, 2, 2, (int[]){0, 1}, (int[]){0, 1},
                           (double[]){1.0, 1.0}, &mI);
    PRIMAL_putbarcj(task, 0, 1, (int[]){mI}, (double[]){1.0});
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double obj;
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("tr(X) = %.6f\n", obj);

    int ok = fabs(obj - 0.5) < 1e-6;
    printf("%s (atteso tr(X) = 0.5, X = J/2 rank-1)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}