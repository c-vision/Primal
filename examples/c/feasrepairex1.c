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

/* feasrepairex1.c — port of the MOSEK Julia API "feasrepairex1.jl" example:
 * feasibility repair — an infeasible LP is "repaired" by finding
 * the point with minimum total violation (elastic relaxation, PRIMAL_feasrepair).
 *
 * Infeasible problem (verified by hand):
 *   min x0  s.t.  x0 >= 2,  x0 <= 1   (no feasible x)
 * Minimum repair: the minimum total violation is 1 (moving one bound by
 * 1): the sum of the optimal elastic slacks = 1. The repaired point
 * satisfies the definition with total violation 1.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Optimize (infeasible) then feasrepair and check the minimum violation. */
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

    /* 1. optimize: infeasible */
    PRIMALrescodee rc = PRIMAL_optimize(task);
    printf("optimize: rc=%d (expected 1002 infeasible)\n", rc);
    int infeas = (rc == PRIMAL_RES_ERR_INFEASIBLE);

    /* 2. feasrepair: point of minimum violation */
    rc = PRIMAL_feasrepair(task);
    if (rc != PRIMAL_RES_OK) { printf("feasrepair rc=%d\n", rc); return 1; }
    double x[1];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    /* total violation of the repaired point: (2 - x0)+ + (x0 - 1)+ = 1 */
    double viol = 0.0;
    if (x[0] < 2.0) viol += 2.0 - x[0];
    if (x[0] > 1.0) viol += x[0] - 1.0;
    printf("feasrepair: x0 = %.4f, total violation = %.4f (expected = 1)\n",
           x[0], viol);

    int ok = infeas && fabs(viol - 1.0) < 1e-6;
    printf("%s (repair to minimum violation)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
