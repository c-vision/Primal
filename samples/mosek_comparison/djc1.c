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

/* djc1.c — port of the MOSEK Julia API "djc1.jl" example:
 * disjunctive constraints (DJC) with the reference API
 * (appenddjcs + putdjc on AFEs and domains).
 *
 * Problem (verified by hand):
 *   min x0   s.t. 0 <= x0 <= 10 and the disjunction:
 *        [ x0 <= 2 ]  OR  [ x0 >= 6 AND x0 <= 7 ]
 *   The second disjunct imposes the interval [6,7]; the first x0 <= 2:
 *   min x0 = 0 (via the first).
 * Second part: max x0 with the same disjunction -> x0 = 7 (the second).
 *
 * DJC encoding: a single AFE f = x0; term 0 is the RMINUS domain
 * (f - 2 <= 0); term 1 is RMINUS (f - 7 <= 0) AND RPLUS (f - 6 >= 0).
 * The reference convention is F x + g - b, so the b values are the constants
 * 2, 7, 6.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve the disjunction for min or max of x0; returns the objective value. */
static double solve(int maximize, double *x0) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_RA, 0.0, 10.0);
    PRIMAL_putcj(task, 0, 1.0);
    if (maximize) PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    /* one AFE f = x0 and two (one-dimensional) linear domains */
    PRIMAL_appendafes(task, 1);
    PRIMAL_putafefentry(task, 0, 0, 1.0);
    PRIMALint64t dminus, dplus;
    PRIMAL_appendrminusdomain(task, 1, &dminus);
    PRIMAL_appendrplusdomain(task, 1, &dplus);
    PRIMAL_appenddjcs(task, 1);
    PRIMALint64t termsize[2] = {1, 2};
    PRIMALint64t domlist[3] = {dminus, dminus, dplus};
    PRIMALint64t afelist[3] = {0, 0, 0};
    double b[3] = {2.0, 7.0, 6.0};
    PRIMALrescodee rc = PRIMAL_putdjc(task, 0, 3, domlist, 3, afelist, b, 2, termsize);
    double po = -1e30;
    if (rc == PRIMAL_RES_OK) {
        rc = PRIMAL_optimize(task);
        if (rc == PRIMAL_RES_OK) {
            int nv;
            PRIMAL_getnumvar(task, &nv);   /* the DJC binaries add variables */
            double x[16];
            if (nv > 16) nv = 16;
            PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
            PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
            *x0 = x[0];
        }
    }
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return po;
}

/* Solve both directions and check min x0 = 0 and max x0 = 7. */
int main(void) {
    double xmin, xmax;
    double pmin = solve(0, &xmin);
    double pmax = solve(1, &xmax);
    printf("min: x0 = %.4f (atteso 0), obj = %.4f\n", xmin, pmin);
    printf("max: x0 = %.4f (atteso 7), obj = %.4f\n", xmax, pmax);

    int ok = fabs(xmin) < 1e-6 && fabs(pmin) < 1e-6 &&
             fabs(xmax - 7.0) < 1e-6 && fabs(pmax - 7.0) < 1e-6;
    printf("%s (disgiunzione: x<=2 OR 6<=x<=7)\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
