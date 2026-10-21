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

/* gp1.c — port of the MOSEK Julia API "gp1.jl" example:
 * geometric programming via log-log transform and exponential cones.
 *
 * GP (classic notation):  min x0 * x1^-1
 *   s.t. x0 + x1 <= 1,  x0 >= 0.5,  x1 >= 0.5
 * In pure GP form: posynomial. Log transform: x_i = e^{y_i}:
 *   min e^{y0 - y1}
 *   s.t. log(e^{y0} + e^{y1}) <= log(1) = 0   (log-sum-exp <= 0)
 *        y0 >= log(0.5), y1 >= log(0.5)
 * Hand check: from x0+x1<=1 with x0,x1>=0.5: the only feasible
 * point is x0=x1=0.5 -> GP objective = 0.5/0.5 = 1 -> y0=y1=log(0.5),
 * min e^{y0-y1} = e^0 = 1.
 * Conic model: t >= e^{y0-y1} via PEXP(u=1, v=y0-y1): (t, 1, y0-y1)
 * in PEXP; logsumexp with PEXP: per term (u_k, 1, y_k) with
 * sum u_k <= 1 (softplus: log(sum e^yk) <= log 1 = 0 iff there is a
 * decomposition u_k with sum u_k <= log... standard: log-sum-exp <= 0
 * <=> exists u: sum u <= 1, u_k >= e^{y_k}). The optimum is verified
 * at the point (0.5, 0.5).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve the gp1 geometric program and check x=(0.5,0.5), obj=1. */
int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variables: y0, y1 (log of the x), t (objective), one, u0, u1 (lse) */
    enum { Y0 = 0, Y1, T, ONE, U0, U1 };
    PRIMAL_appendvars(task, 6);
    PRIMAL_putvarbound(task, Y0, PRIMAL_BK_LO, log(0.5), INFINITY);
    PRIMAL_putvarbound(task, Y1, PRIMAL_BK_LO, log(0.5), INFINITY);
    PRIMAL_putvarbound(task, T, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putvarbound(task, U0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, U1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, T, 1.0);   /* min t */

    /* rows:
     * r0: u0 + u1 <= 1                        (lse budget)
     * r1: PEXP member v is a VARIABLE:
     *     cones over the auxiliaries: (u_k, one, y_k) in PEXP  [u_k >= e^{y_k}]
     * (t, one, v) in PEXP with r1: v - y0 + y1 = 0  (v = y0 - y1)
     */
    PRIMAL_appendcons(task, 2);
    {   /* r0: u0 + u1 <= e^c = 1 (log-sum-exp <= log 1) */
        int sub[2] = {U0, U1};
        double v[2] = {1.0, 1.0};
        PRIMAL_putarow(task, 0, 2, sub, v);
        PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 1.0);
    }
    {   /* r1: v - y0 + y1 = 0 — v is an extra variable */
        PRIMAL_appendvars(task, 1);
        int nv;
        PRIMAL_getnumvar(task, &nv);
        int V = nv - 1;
        PRIMAL_putvarbound(task, V, PRIMAL_BK_FR, -INFINITY, INFINITY);
        int sub[3] = {V, Y0, Y1};
        double v[3] = {1.0, -1.0, 1.0};
        PRIMAL_putarow(task, 1, 3, sub, v);
        PRIMAL_putconbound(task, 1, PRIMAL_BK_FX, 0.0, 0.0);
        /* cone: (t, one, v) in PEXP */
        PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, (int[]){T, ONE, V});
    }
    /* cones: (u0, one, y0), (u1, one, y1) in PEXP */
    PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, (int[]){U0, ONE, Y0});
    PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, (int[]){U1, ONE, Y1});

    PRIMAL_putintparam(task, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 5000);
    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[8], po;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    double x0 = exp(xx[Y0]), x1 = exp(xx[Y1]);
    printf("y = (%.4f, %.4f) -> x = (%.4f, %.4f), t = %.6f\n",
           xx[Y0], xx[Y1], x0, x1, xx[T]);
    printf("GP objective = x0/x1 = %.6f (expected 1), conic pobj = %.6f\n",
           x0 / x1, po);

    /* check: GP feasibility and analytic optimum */
    int ok = fabs(x0 + x1 - 1.0) < 1e-3 &&
             x0 >= 0.5 - 1e-6 && x1 >= 0.5 - 1e-6 &&
             fabs(po - 1.0) < 1e-3 &&
             fabs(x0 / x1 - 1.0) < 1e-3;
    printf("%s (GP log-log: x=(0.5,0.5), obj=1)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
