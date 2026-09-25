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

/* sensitivity.c — porting dell'esempio "sensitivity.jl" della Julia API
 * MOSEK: analisi post-ottima (cost sensitivity) su un piccolo LP.
 *
 * Problema (base dell'esempio T65, verificato a mano):
 *   min x0 + x1   s.t.  x0 + 2*x1 >= 4,  x0, x1 >= 0
 * Ottimo: x* = (0, 2), obj = 2. Cost sensitivity:
 *   - x0 (nonbasic al lower): resta a 0 finche' c0 >= 1/2
 *     (sotto 1/2 conviene spostare peso su x0) -> lcost = 0.5, ucost = inf
 *   - x1 (basic): la soluzione resta ottimale per c1 qualsiasi nel range
 *     degenere [1,1] (il clone non espone la base: range puntiforme)
 * Verifica esterna: re-solve con c0 = 0.4 (fuori range -> x0 > 0) e
 * c0 = 0.6 (dentro range -> x0 resta 0).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static double solve_with_c0(double c0, double *x0, double *x1) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, c0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 2.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 4.0, INFINITY);
    PRIMALrescodee rc = PRIMAL_optimize(task);
    double po = -1e30;
    if (rc == PRIMAL_RES_OK) {
        double x[2];
        PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
        *x0 = x[0]; *x1 = x[1];
    }
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return po;
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 2.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 4.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double x[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("soluzione base: x = (%.4f, %.4f), obj = %.4f\n", x[0], x[1], obj);

    double lc0, uc0, lc1, uc1;
    PRIMAL_costsensitivity(task, 0, &lc0, &uc0);
    PRIMAL_costsensitivity(task, 1, &lc1, &uc1);
    printf("c0 in [%.4f, %s]\n", lc0, uc0 == INFINITY ? "+inf" : "...");
    printf("c1 in [%.4f, %.4f] (basic: range degenere)\n", lc1, uc1);

    /* verifica esterna */
    double xa, xb, ya, yb;
    double p_in = solve_with_c0(0.6, &xa, &ya);
    double p_out = solve_with_c0(0.4, &xb, &yb);
    printf("c0=0.6 (in range):  x = (%.4f, %.4f), obj = %.4f\n", xa, ya, p_in);
    printf("c0=0.4 (out range): x = (%.4f, %.4f), obj = %.4f\n", xb, yb, p_out);

    int ok = fabs(x[0]) < 1e-9 && fabs(x[1] - 2.0) < 1e-9 &&
             fabs(lc0 - 0.5) < 1e-9 && uc0 == INFINITY &&
             fabs(xa) < 1e-9 &&            /* in range: x0 resta 0 */
             fabs(xb) > 1e-9 &&            /* out range: x0 > 0 */
             fabs(p_out - (0.4 * 4.0)) < 1e-6;  /* tutto su x0: 1.6 */
    printf("%s (range c0 confermato da re-solve)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
