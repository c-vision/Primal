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

/* sdo2.c — porting dell'esempio "sdo2.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): SDP con DUE variabili bar (matrici PSD)
 * e una costante di obiettivo.
 *
 *   min  <I, X1> + <I, X2> + 1.0            (cfix = 1)
 *   s.t. <J, X1> + <I, X2> = 1              (J = matrice di tutti 1)
 *        X1, X2 >= 0 (PSD)
 *
 * Analisi: min tr(X1) + tr(X2). La variabile X1 paga <J,X1> con tr(X1) >=
 * <J,X1>/2 = 1/2 (rank-1, X1 = J/4), mentre X2 = 0.
 * Atteso: obj = 0.5 + 1 = 1.5, X1 = J/4, X2 = 0.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendcons(task, 1);

    int mI, mJ;
    PRIMAL_appendsparsesymmat(task, 2, 2, (int[]){0, 1}, (int[]){0, 1},
                           (double[]){1.0, 1.0}, &mI);
    PRIMAL_appendsparsesymmat(task, 2, 3, (int[]){0, 1, 0}, (int[]){0, 1, 1},
                           (double[]){1.0, 1.0, 1.0}, &mJ);

    int dims[2] = {2, 2};
    PRIMAL_appendbarvars(task, 2, dims);

    PRIMAL_putcfix(task, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* <I, X1> + <I, X2> nell'obiettivo */
    PRIMAL_putbarcj(task, 0, 1, (int[]){mI}, (double[]){1.0});
    PRIMAL_putbarcj(task, 1, 1, (int[]){mI}, (double[]){1.0});
    /* <J, X1> + <I, X2> = 1 */
    PRIMAL_putbaraij(task, 0, 0, 1, (int[]){mJ}, (double[]){1.0});
    PRIMAL_putbaraij(task, 0, 1, 1, (int[]){mI}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double X1[4], X2[4], po;
    PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, X1);
    PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 1, X2);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    printf("X1 = [%.6f %.6f; %.6f %.6f]\n", X1[0], X1[1], X1[2], X1[3]);
    printf("X2 = [%.6f %.6f; %.6f %.6f]\n", X2[0], X2[1], X2[2], X2[3]);
    printf("obj = %.6f\n", po);

    /* verifica: obj = 1.5, X1 = J/4, X2 = 0 */
    int ok = fabs(po - 1.5) < 1e-4;
    if (fabs(X1[0] - 0.25) > 1e-4 || fabs(X1[1] - 0.25) > 1e-4 ||
        fabs(X1[2] - 0.25) > 1e-4 || fabs(X1[3] - 0.25) > 1e-4) ok = 0;
    if (fabs(X2[0]) > 1e-4 || fabs(X2[1]) > 1e-4 ||
        fabs(X2[2]) > 1e-4 || fabs(X2[3]) > 1e-4) ok = 0;

    printf("%s (atteso obj = 1.5, X1 = J/4, X2 = 0)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}