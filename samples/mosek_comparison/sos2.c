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

/* sos2.c — porting dell'esempio "sos2.jl" (C API docs.mosek.com):
 * SOS type-2 constraint (al piu' due membri non nulli ADIACENTI nell'ordine
 * dei pesi — la base della piecewise-linear approximation).
 *
 * Problema (verificato a mano): approssimare la funzione piecewise-linear
 * f(t) su breakpoint t=(0,1,2) con valori f=(0,1,0.5) tramite pesi SOS2:
 *   t = sum_i w_i * t_i,  y = sum_i w_i * f_i,  sum w = 1, w >= 0
 * con SOS2{w0,w1,w2} (pesi = i breakpoint). SOS2 garantisce che al piu' due
 * w adiacenti sono attive: y interpola linearmente f in t.
 * Test: minimizza (y - y_target)^2 via... per restare LP: fissiamo t=1.5
 * e calcoliamo y: tra i segmenti [1,2]: y = 1 + 0.5*(1.5-1) = 0.75.
 * Verifica: con t fisso a 1.5, la soluzione SOS2 deve dare y=0.75.
 * Modella: variabili w0,w1,w2 >= 0, w0+w1+w2=1 (EQ), 0*w0+1*w1+2*w2=1.5 (EQ),
 * y = 0*w0+1*w1+0.5*w2 (definita via riga EQ su y), obj = 0 (feasibility).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: w0, w1, w2, y */
    PRIMAL_appendvars(task, 4);
    PRIMAL_appendcons(task, 3);
    for (int j = 0; j < 4; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 3, PRIMAL_BK_FR, -INFINITY, INFINITY);   /* y libero */
    /* w0 + w1 + w2 = 1 */
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* 0*w0 + 1*w1 + 2*w2 = 1.5 (t = 1.5) */
    PRIMAL_putarow(task, 1, 3, (int[]){0, 1, 2}, (double[]){0.0, 1.0, 2.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_FX, 1.5, 1.5);
    /* y = 0*w0 + 1*w1 + 0.5*w2 */
    PRIMAL_putarow(task, 2, 4, (int[]){0, 1, 2, 3}, (double[]){0.0, 1.0, 0.5, -1.0});
    PRIMAL_putconbound(task, 2, PRIMAL_BK_FX, 0.0, 0.0);
    /* SOS2 sui pesi, ordine dei breakpoint */
    PRIMAL_appendsos2(task, 3, (int[]){0, 1, 2}, (double[]){0.0, 1.0, 2.0});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[4];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    printf("w = (%.4f, %.4f, %.4f), y = %.4f (atteso 0.75)\n",
           xx[0], xx[1], xx[2], xx[3]);

    int ok = fabs(xx[3] - 0.75) < 1e-6 &&
             fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-6 &&
             fabs(xx[1] + 2.0 * xx[2] - 1.5) < 1e-6;
    /* SOS2: w0 e w2 non entrambi positivi (non adiacenti) */
    if (xx[0] > 1e-6 && xx[2] > 1e-6) ok = 0;
    printf("%s (piecewise-linear SOS2: y(1.5)=0.75, adiacenza rispettata)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
