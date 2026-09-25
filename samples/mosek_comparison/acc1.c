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

/* acc1.c — porting dell'esempio "acc1.jl" della MOSEK Julia API:
 * affine conic constraints (ACC) via appendaccseq.
 *
 * Problema (verificato a mano):
 *   min  x1
 *   s.t. x0 = 3 (riga FX)
 *        ACC: (v0, v1) in QUAD con v0 = 2*x0 + x1, v1 = x0 - 1
 *   => v0 = 6 + x1 >= |v1| = |2| = 2  =>  x1 >= -4
 * Ottimo: x = (3, -4), obj = -4.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FX, 3.0, 3.0);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putarow(task, 0, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 3.0, 3.0);

    /* ACC: (v0, v1) in QUAD, v0 = 2*x0 + x1, v1 = x0 - 1 (via AFE + dominio) */
    PRIMAL_appendafes(task, 2);
    PRIMAL_putafefentry(task, 0, 0, 2.0);
    PRIMAL_putafefentry(task, 0, 1, 1.0);
    PRIMAL_putafefentry(task, 1, 0, 1.0);
    PRIMAL_putafeg(task, 1, -1.0);
    PRIMALint64t dom;
    if (PRIMAL_appendquadraticconedomain(task, 2, &dom) != PRIMAL_RES_OK) {
        printf("domain rc\n"); return 1;
    }
    PRIMALint64t afeidx[2] = {0, 1};
    double b[2] = {0.0, 0.0};
    PRIMALrescodee rc = PRIMAL_appendacc(task, dom, 2, afeidx, b);
    if (rc != PRIMAL_RES_OK) { printf("appendacc rc=%d\n", rc); return 1; }

    rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    int nv;
    PRIMAL_getnumvar(task, &nv);   /* le ausiliarie ACC aggiungono variabili */
    double xx[16], obj;
    if (nv > 16) nv = 16;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f), obj = %.4f (numvar=%d incl. ausiliarie ACC)\n",
           xx[0], xx[1], obj, nv);

    /* verifica esterna: v0 >= |v1| */
    double v0 = 2.0 * xx[0] + xx[1];
    double v1 = xx[0] - 1.0;
    int ok = fabs(xx[0] - 3.0) < 1e-6 &&
             fabs(xx[1] + 4.0) < 1e-4 &&
             fabs(obj + 4.0) < 1e-4 &&
             v0 >= fabs(v1) - 1e-6;
    printf("v0 = %.4f, |v1| = %.4f (ACC soddisfatto)\n", v0, fabs(v1));
    printf("%s (atteso x=(3,-4), obj=-4)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
