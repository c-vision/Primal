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

/* parallel.c — porting dell'esempio "parallel.jl" della MOSEK Julia API:
 * piu' task indipendenti risolti (nel clone: in sequenza — deviazione
 * documentata: single-thread; l'esempio verifica l'isolamento dei task).
 *
 * 3 task indipendenti con ottimi noti (verificati a mano):
 *   A: max x s.t. x <= 4        -> 4
 *   B: min x s.t. x >= 2        -> 2
 *   C: max x0+x1 s.t. x0+x1 <= 7 -> 7
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    /* task A: max x s.t. x <= 4 */
    PRIMALtask_t A;
    PRIMAL_maketask(env, 0, 0, &A);
    PRIMAL_appendvars(A, 1);
    PRIMAL_appendcons(A, 1);
    PRIMAL_putcj(A, 0, 1.0);
    PRIMAL_putobjsense(A, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putarow(A, 0, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(A, 0, PRIMAL_BK_UP, -INFINITY, 4.0);

    /* task B: min x s.t. x >= 2 */
    PRIMALtask_t B;
    PRIMAL_maketask(env, 0, 0, &B);
    PRIMAL_appendvars(B, 1);
    PRIMAL_appendcons(B, 1);
    PRIMAL_putcj(B, 0, 1.0);
    PRIMAL_putarow(B, 0, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(B, 0, PRIMAL_BK_LO, 2.0, INFINITY);

    /* task C: max x0+x1 s.t. x0+x1 <= 7 */
    PRIMALtask_t C;
    PRIMAL_maketask(env, 0, 0, &C);
    PRIMAL_appendvars(C, 2);
    PRIMAL_appendcons(C, 1);
    PRIMAL_putobjsense(C, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(C, 0, 1.0);
    PRIMAL_putcj(C, 1, 1.0);
    PRIMAL_putarow(C, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(C, 0, PRIMAL_BK_UP, -INFINITY, 7.0);

    PRIMALrescodee rA = PRIMAL_optimize(A);
    PRIMALrescodee rB = PRIMAL_optimize(B);
    PRIMALrescodee rC = PRIMAL_optimize(C);
    double pA, pB, pC;
    PRIMAL_getprimalobj(A, PRIMAL_SOL_ITR, &pA);
    PRIMAL_getprimalobj(B, PRIMAL_SOL_ITR, &pB);
    PRIMAL_getprimalobj(C, PRIMAL_SOL_ITR, &pC);
    printf("A: rc=%d pobj=%.4f (atteso 4)\n", rA, pA);
    printf("B: rc=%d pobj=%.4f (atteso 2)\n", rB, pB);
    printf("C: rc=%d pobj=%.4f (atteso 7)\n", rC, pC);

    int ok = (rA == PRIMAL_RES_OK && fabs(pA - 4.0) < 1e-9) &&
             (rB == PRIMAL_RES_OK && fabs(pB - 2.0) < 1e-9) &&
             (rC == PRIMAL_RES_OK && fabs(pC - 7.0) < 1e-9);
    printf("%s (3 task indipendenti, isolamento verificato)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&A);
    PRIMAL_deletetask(&B);
    PRIMAL_deletetask(&C);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
