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

/* cqo1.c — porting dell'esempio "cqo1.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi/cqo1.html): problema conico piccolo con
 * un cono quadratico e uno ruotato.
 *
 * NOTA: l'esempio Julia usa l'API ACC (appendafes/putafefentrylist/appendacc);
 * qui gli ACC sono coni su variabili dirette, quindi si usano le appendcone
 * equivalenti (l'API ACC del riferimento e' comunque disponibile).
 *
 *   min  x3 + x4 + x5
 *   s.t. x0 + x1 + 2*x2 = 1
 *        (x3, x0, x1) in QUAD_3      -> x3 >= sqrt(x0^2 + x1^2)
 *        (x4, x5, x2) in RQUAD_3     -> 2*x4*x5 >= x2^2
 *        x0,x1,x2 >= 0; x3,x4,x5 libere
 *
 * Ottimo noto (dalla docs MOSEK): obj = 1/sqrt(2) ≈ 0.7071.
 * La somma x3+x4+x5 e' costante = 1/sqrt(2) per ogni punto ottimo
 * (sqrt(x0^2+x1^2) >= (x0+x1)/sqrt2 e la parte RQUAD completa il resto),
 * quindi si verifica sia l'obiettivo sia l'adesione ai coni.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 6);
    PRIMAL_appendcons(task, 1);

    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMAL_putcj(task, 3, 1.0);
    PRIMAL_putcj(task, 4, 1.0);
    PRIMAL_putcj(task, 5, 1.0);

    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int j = 3; j < 6; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);

    /* x0 + x1 + 2*x2 = 1 */
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 2.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    /* QUAD_3: (x3, x0, x1) */
    PRIMAL_appendcone(task, PRIMAL_CT_QUAD, 0.0, 3, (int[]){3, 0, 1});
    /* RQUAD_3: (x4, x5, x2) */
    PRIMAL_appendcone(task, PRIMAL_CT_RQUAD, 0.0, 3, (int[]){4, 5, 2});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double x[6], po;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    printf("x = (%.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n",
           x[0], x[1], x[2], x[3], x[4], x[5]);
    printf("obj = %.6f (atteso 1/sqrt(2) = %.6f)\n", po, 1.0 / sqrt(2.0));

    /* verifica: uguaglianza, adesione ai coni, obiettivo */
    int ok = fabs(x[0] + x[1] + 2 * x[2] - 1.0) < 1e-6 &&
             fabs(po - 1.0 / sqrt(2.0)) < 1e-6 &&
             x[3] * x[3] >= x[0] * x[0] + x[1] * x[1] - 1e-8 &&
             2 * x[4] * x[5] >= x[2] * x[2] - 1e-8 &&
             fabs(x[3] + x[4] + x[5] - po) < 1e-9;
    printf("%s (uguaglianza + coni QUAD/RQUAD + ottimo)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}