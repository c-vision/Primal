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

/* solutionquality.c — port of the MOSEK Julia API "solutionquality.jl"
 * example: measuring solution quality (primal/dual violations) via
 * PRIMAL_getprimalinfeas / PRIMAL_getdualinfeas.
 *
 * Hand check:
 *   1. optimal LP: min x0+x1 s.t. x0+x1 >= 1, x >= 0 -> both
 *      violations ~ 0 (exact solution)
 *   2. BROKEN point injected via putxx + solvebasis? no: a
 *      second task is built whose "solution" is deliberately violated: the
 *      primal violation is measured directly (x outside its bounds): to do it
 *      without solving, we use the already-solved task and corrupt x via
 *      a second optimize with tight bounds... simple approach:
 *      check that after optimize the violations are < 1e-6, and that the
 *      getters reject a task with no solution (ERR_ARG).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve an LP and check the violation getters, including the no-solution refusal. */
int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    /* 1. optimal LP */
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putcj(task, 1, 1.0);
    PRIMAL_putarow(task, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 1.0, INFINITY);
    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double pinf, dinf;
    PRIMAL_getprimalinfeas(task, PRIMAL_SOL_ITR, &pinf);
    PRIMAL_getdualinfeas(task, PRIMAL_SOL_ITR, &dinf);
    printf("primal violation = %.3e, dual = %.3e (expected ~ 0)\n", pinf, dinf);

    /* 2. without a solution: refused */
    PRIMALtask_t t2;
    PRIMAL_maketask(env, 0, 0, &t2);
    PRIMAL_appendvars(t2, 1);
    double dummy;
    int rejected = (PRIMAL_getprimalinfeas(t2, PRIMAL_SOL_ITR, &dummy) == PRIMAL_RES_ERR_ARG);

    int ok = (pinf < 1e-6) && (dinf < 1e-6) && rejected;
    printf("%s (quality measured + protected getters)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t2);
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
