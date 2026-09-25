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

/* sdo_lmi.c — porting dell'esempio "sdo_lmi.jl" della MOSEK Julia API:
 * linear matrix inequality (LMI) con variabili scalari.
 *
 *   max x0 + x1  s.t.  F(x) = F0 + x0*F1 + x1*F2 PSD (LMI 3x3)
 *   con F0 = I3, e le F1/F2 con un solo termine ciascuna:
 *     F1 = -E00,  F2 = -E11  (scala le diagonali giu')
 *   piu' x0 + x1 <= 0.5.
 * Verifica a mano: F(x) = diag(1 - x0, 1 - x1, 1) PSD: 1-x0 >= 0,
 * 1-x1 >= 0. max x0+x1 <= 0.5: la LMI e' lasca (x <= 1): il vincolo lineare
 * domina -> ottimo x0+x1 = 0.5, obj = 0.5. La LMI rende invece attivo il
 * caso con x0+x1 <= 0.5 sostituito... per rendere la LMI ATTIVA: bound
 * x0+x1 <= 1.6: la LMI blocca a x0=x1=1 (1-x0 >= 0): max x0+x1 = 2 > 1.6 ->
 * il lineare domina ancora. Con vincolo x0+x1 <= 2.5: la LMI attiva a
 * x0=x1=1, obj=2.
 * Modello (bar 3x3 X PSD): X = F0 + x0 F1 + x1 F2 elemento per elemento
 * (righe FX sulle entrate diagonal) e le offdiagonal di X a 0.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define D 3

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili scalari x0, x1 */
    PRIMAL_appendvars(task, 2);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    /* bar X 3x3 PSD */
    int dim = D;
    PRIMAL_appendbarvars(task, 1, &dim);

    /* matrix store: E_ii per le diagonali */
    int mE[D];
    for (int i = 0; i < D; i++)
        PRIMAL_appendsparsesymmat(task, D, 1,
                               (int[]){i}, (int[]){i}, (double[]){1.0}, &mE[i]);

    /* righe: X_ii = F0_ii + x_i * F_i_ii (per i<2, con F_i = -E_ii):
     *   X_00 = 1 - x0,  X_11 = 1 - x1,  X_22 = 1
     *   X_01 = 0, X_02 = 0, X_12 = 0
     * e il vincolo lineare x0 + x1 <= 2.5
     * Totale righe: 3 (diag) + 3 (offdiag) + 1 (lineare) = 7 */
    PRIMAL_appendcons(task, 7);
    int r = 0;
    /* X_00 = 1 - x0 */
    PRIMAL_putbaraij(task, r, 0, 1, (int[]){mE[0]}, (double[]){1.0});
    PRIMAL_putarow(task, r, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, r, PRIMAL_BK_FX, 1.0, 1.0);
    r++;
    /* X_11 = 1 - x1 */
    PRIMAL_putbaraij(task, r, 0, 1, (int[]){mE[1]}, (double[]){1.0});
    PRIMAL_putarow(task, r, 1, (int[]){1}, (double[]){1.0});
    PRIMAL_putconbound(task, r, PRIMAL_BK_FX, 1.0, 1.0);
    r++;
    /* X_22 = 1 */
    PRIMAL_putbaraij(task, r, 0, 1, (int[]){mE[2]}, (double[]){1.0});
    PRIMAL_putconbound(task, r, PRIMAL_BK_FX, 1.0, 1.0);
    r++;
    /* off-diagonali a 0: X_01 = X_02 = X_12 = 0 (la simmetria e' implicita:
     * la matrice del matrix store con entrata (i,j) conta 2X_ij) */
    int mO[3];
    PRIMAL_appendsparsesymmat(task, D, 1, (int[]){0}, (int[]){1}, (double[]){1.0}, &mO[0]);
    PRIMAL_appendsparsesymmat(task, D, 1, (int[]){0}, (int[]){2}, (double[]){1.0}, &mO[1]);
    PRIMAL_appendsparsesymmat(task, D, 1, (int[]){1}, (int[]){2}, (double[]){1.0}, &mO[2]);
    for (int k = 0; k < 3; k++) {
        PRIMAL_putbaraij(task, r, 0, 1, (int[]){mO[k]}, (double[]){1.0});
        PRIMAL_putconbound(task, r, PRIMAL_BK_FX, 0.0, 0.0);
        r++;
    }
    /* x0 + x1 <= 2.5 */
    PRIMAL_putarow(task, r, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, r, PRIMAL_BK_UP, -INFINITY, 2.5);
    r++;

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[2], po, X[9];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, X);
    printf("x = (%.4f, %.4f), obj = %.4f (atteso (1,1), 2)\n", xx[0], xx[1], po);
    printf("X = diag(%.4f, %.4f, %.4f) (atteso (0,0,1))\n", X[0], X[4], X[8]);

    int ok = fabs(xx[0] - 1.0) < 1e-4 && fabs(xx[1] - 1.0) < 1e-4 &&
             fabs(po - 2.0) < 1e-4 &&
             fabs(X[0]) < 1e-4 && fabs(X[4]) < 1e-4 && fabs(X[8] - 1.0) < 1e-4;
    printf("%s (LMI attiva: x=(1,1), X=diag(0,0,1) PSD al bordo)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
