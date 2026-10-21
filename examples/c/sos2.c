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

/* sos2.c — port of the "sos2.jl" example (C API docs.mosek.com):
 * SOS type-2 constraint (at most two nonzero members ADJACENT in the weight
 * order — the basis of piecewise-linear approximation).
 *
 * Problem (verified by hand): approximate the piecewise-linear function
 * f(t) on breakpoints t=(0,1,2) with values f=(0,1,0.5) via SOS2 weights:
 *   t = sum_i w_i * t_i,  y = sum_i w_i * f_i,  sum w = 1, w >= 0
 * with SOS2{w0,w1,w2} (weights = breakpoint index). SOS2 guarantees that at
 * most two adjacent w are active: y interpolates f linearly in t.
 * Test: minimize (y - y_target)^2 via... to stay LP: we fix t=1.5
 * and compute y: within segment [1,2]: y = 1 + 0.5*(1.5-1) = 0.75.
 * Check: with t fixed at 1.5, the SOS2 solution must give y=0.75.
 * Model: variables w0,w1,w2 >= 0, w0+w1+w2=1 (EQ), 0*w0+1*w1+2*w2=1.5 (EQ),
 * y = 0*w0+1*w1+0.5*w2 (defined by an EQ row on y), obj = 0 (feasibility).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve the SOS2 piecewise-linear interpolation and check y(1.5)=0.75. */
int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variables: w0, w1, w2, y */
    PRIMAL_appendvars(task, 4);
    PRIMAL_appendcons(task, 3);
    for (int j = 0; j < 4; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, 3, PRIMAL_BK_FR, -INFINITY, INFINITY);   /* y free */
    /* w0 + w1 + w2 = 1 */
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* 0*w0 + 1*w1 + 2*w2 = 1.5 (t = 1.5) */
    PRIMAL_putarow(task, 1, 3, (int[]){0, 1, 2}, (double[]){0.0, 1.0, 2.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_FX, 1.5, 1.5);
    /* y = 0*w0 + 1*w1 + 0.5*w2 */
    PRIMAL_putarow(task, 2, 4, (int[]){0, 1, 2, 3}, (double[]){0.0, 1.0, 0.5, -1.0});
    PRIMAL_putconbound(task, 2, PRIMAL_BK_FX, 0.0, 0.0);
    /* SOS2 on the weights, breakpoint order */
    PRIMAL_appendsos2(task, 3, (int[]){0, 1, 2}, (double[]){0.0, 1.0, 2.0});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[4];
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    printf("w = (%.4f, %.4f, %.4f), y = %.4f (expected 0.75)\n",
           xx[0], xx[1], xx[2], xx[3]);

    int ok = fabs(xx[3] - 0.75) < 1e-6 &&
             fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-6 &&
             fabs(xx[1] + 2.0 * xx[2] - 1.5) < 1e-6;
    /* SOS2: w0 and w2 not both positive (not adjacent) */
    if (xx[0] > 1e-6 && xx[2] > 1e-6) ok = 0;
    printf("%s (piecewise-linear SOS2: y(1.5)=0.75, adjacency respected)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
