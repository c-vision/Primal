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

/* sos1.c — porting dell'esempio "sos1.jl" (C API docs.mosek.com):
 * SOS type-1 constraint.
 *
 * Problema (verificato a mano): min x0 + x1 + x2
 *   s.t. x0 + 2*x1 + 3*x2 >= 4,   SOS1 {x0, x1, x2} (pesi 1,2,3),
 *        0 <= x <= 1
 * SOS1: al piu' UNA variabile non nulla. La piu' "efficiente" per coprire
 * la riga e' x2 (3 unita' di copertura per unita' di costo): x2 = 2.5/3
 * = 0.8333, obj = 0.8333 (il continuo entro [0,1] domina).
 * Ottimo: x=(0,0,5/6), obj=5/6.
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
    PRIMAL_appendcons(task, 1);
    for (int j = 0; j < 3; j++) {
        PRIMAL_putvarbound(task, j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putcj(task, j, 1.0);
    }
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 2.0, 3.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 2.5, INFINITY);
    PRIMAL_appendsos1(task, 3, (int[]){0, 1, 2}, (double[]){1.0, 2.0, 3.0});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f), obj = %.4f\n", xx[0], xx[1], xx[2], obj);

    /* verifica SOS1 (al piu' un non nullo) via enumerazione dei 3 membri:
     * solo il membro k attivo con x_k = 2.5/coverage_k se <= 1 */
    double best = INFINITY;
    double cov[3] = {1.0, 2.0, 3.0};
    for (int k = 0; k < 3; k++) {
        double need = 2.5 / cov[k];
        if (need <= 1.0 + 1e-9 && need < best) best = need;
    }
    int nact = 0;
    for (int j = 0; j < 3; j++) if (xx[j] > 1e-6) nact++;
    int ok = fabs(obj - best) < 1e-6 && nact <= 1 &&
             fabs(obj - (xx[0] + xx[1] + xx[2])) < 1e-6;
    printf("enumerazione: best = %.4f, non-nulli = %d\n", best, nact);
    printf("%s (atteso obj=5/6, un solo membro non nullo)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
