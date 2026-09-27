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

/* portfolio_3.c — port of the MOSEK Julia API "portfolio_3_impact.jl"
 * example: portfolio optimization with (linear) transaction costs
 * and market-impact constraints — MIP version: maximum budget
 * of distinct assets (cardinality handled here as an integer selection
 * constraint, without cones: the official portfolio_3 uses absolute
 * constraints |x_j - x0_j| <= u_j with semi-continuous u_j; here the MIP
 * part is the semi-continuity of u_j: costs are paid only if the asset is
 * traded).
 *
 *   max  r'x - gamma*x'Sx - f' u
 *   s.t. sum(x) = 1,  0 <= x <= 1
 *        u_j >= |x_j - x0_j|  (2 linear rows per j)
 *        u_j semi-continuous [0.0001, 1]: the cost f_j u_j is active only if
 *        the asset is traded (u=0 or u >= 0.0001)
 *
 * Data: r, GT, gamma from the MOSEK tutorial (portfolio_1), f = 0.01 uniform,
 * x0 = uniform initial portfolio (1/3, 1/3, 1/3).
 * Check: feasibility (budget, semi-continuity), comparison with the
 * version without costs (portfolio_1: pobj_3 <= pobj_1).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve the market-impact MIQP and verify budget and semi-continuity. */
int main(void) {
    const int n = 3;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    const double GT[9] = {0.1667, 0.0, 0.0,
                          0.0247, 0.1592, 0.0,
                          0.0197, 0.0158, 0.1327};
    const double gamma = 0.05;
    const double f = 0.01;             /* linear transaction cost */
    const double x0[3] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

    double S[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            S[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) S[i * 3 + j] += GT[i * 3 + k] * GT[j * 3 + k];
        }

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variables: x (3), u (3) — total 6 */
    PRIMAL_appendvars(task, 2 * n);
    /* rows: budget (1), u_j >= x_j - x0_j, u_j >= x0_j - x_j (2n) */
    PRIMAL_appendcons(task, 1 + 2 * n);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(task, j, r[j]);
        PRIMAL_putcj(task, n + j, -f);
        PRIMAL_putvarbound(task, j, PRIMAL_BK_RA, 0.0, 1.0);
        /* u_j semi-continuous: 0 or [0.0001, 1] */
        PRIMAL_putvarbound(task, n + j, PRIMAL_BK_RA, 0.0001, 1.0);
        PRIMAL_putvartype(task, n + j, PRIMAL_VAR_TYPE_SEMI_CONT);
    }

    /* budget: sum x = 1 */
    {
        int sub[3] = {0, 1, 2};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 0, 3, sub, v);
    }
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);

    /* |x_j - x0_j| <= u_j */
    for (int j = 0; j < n; j++) {
        /* row: x_j - u_j <= x0_j */
        int sub[2] = {j, n + j};
        double v[2] = {1.0, -1.0};
        PRIMAL_putarow(task, 1 + j, 2, sub, v);
        PRIMAL_putconbound(task, 1 + j, PRIMAL_BK_UP, -INFINITY, x0[j]);
        /* row: -x_j - u_j <= -x0_j */
        double v2[2] = {-1.0, -1.0};
        PRIMAL_putarow(task, 1 + n + j, 2, sub, v2);
        PRIMAL_putconbound(task, 1 + n + j, PRIMAL_BK_UP, -INFINITY, -x0[j]);
    }

    /* obj = r'x - f'u - gamma x'Sx (Q = -2*gamma*S) */
    {
        int sub[6] = {0, 0, 0, 1, 1, 2};
        int subj[6] = {0, 1, 2, 1, 2, 2};
        double v[6] = {
            -2.0 * gamma * S[0 * 3 + 0], -2.0 * gamma * S[0 * 3 + 1],
            -2.0 * gamma * S[0 * 3 + 2], -2.0 * gamma * S[1 * 3 + 1],
            -2.0 * gamma * S[1 * 3 + 2], -2.0 * gamma * S[2 * 3 + 2]};
        PRIMAL_putqobj(task, 6, sub, subj, v);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[6], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f)\n", xx[0], xx[1], xx[2]);
    printf("u = (%.4f, %.4f, %.4f)\n", xx[3], xx[4], xx[5]);
    printf("obj = %.6f\n", obj);

    /* check: budget, |x-x0|<=u, semi-continuity of u */
    int ok = fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-5;
    for (int j = 0; j < n; j++) {
        if (fabs(xx[j] - x0[j]) > xx[n + j] + 1e-6) ok = 0;
        if (xx[n + j] > 1e-6 && xx[n + j] < 0.0001 - 1e-9) ok = 0;
    }
    /* without costs the portfolio dominates: pobj_portfolio1 ~ 0.119066
     * (computed by the portfolio_1 sample with the same data) */
    printf("%s (bilancio+semi-continuita' verificati, obj <= 0.1191: %s)\n",
           ok ? "OK" : "FAIL", obj <= 0.1191 ? "si" : "NO");
    if (obj > 0.1191) ok = 0;

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
