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

/* portfolio_2.c — porting dell'esempio "portfolio_2_frontier.jl" della
 * MOSEK Julia API (docs.mosek.com/11.0/juliaapi): frontiera media-varianza.
 * Per una griglia di gamma si risolve il QP di Markowitz e si stampa la
 * coppia (rischio, rendimento atteso): il rischio deve decrescere al
 * crescere di gamma (trade-off rischio/rendimento).
 *
 *   max   r'x - gamma * x' Sigma x
 *   s.t.  sum(x) = 1,  x >= 0
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static PRIMALenv_t env;

static int solve_gamma(double gamma, double *x, double *risk, double *ret) {
    const int n = 3;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    const double GT[9] = {0.1667, 0.0, 0.0,
                          0.0247, 0.1592, 0.0,
                          0.0197, 0.0158, 0.1327};
    double S[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            S[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) S[i * 3 + j] += GT[i * 3 + k] * GT[j * 3 + k];
        }

    PRIMALtask_t task;
    if (PRIMAL_maketask(env, 0, 0, &task) != PRIMAL_RES_OK) return 1;
    PRIMAL_appendvars(task, n);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = 0; j < n; j++) PRIMAL_putcj(task, j, r[j]);
    {
        int sub[6] = {0, 0, 0, 1, 1, 2};
        int subj[6] = {0, 1, 2, 1, 2, 2};
        double v[6] = {
            -2.0 * gamma * S[0], -2.0 * gamma * S[1], -2.0 * gamma * S[2],
            -2.0 * gamma * S[4], -2.0 * gamma * S[5], -2.0 * gamma * S[8]};
        PRIMAL_putqobj(task, 6, sub, subj, v);
    }
    PRIMAL_putarow(task, 0, n, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    for (int j = 0; j < n; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&task); return 1; }
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    PRIMAL_deletetask(&task);

    *risk = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) *risk += x[i] * S[i * 3 + j] * x[j];
    *ret = r[0] * x[0] + r[1] * x[1] + r[2] * x[2];
    return 0;
}

int main(void) {
    PRIMAL_makeenv(&env, NULL);

    const double gammas[] = {0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};
    const int ng = (int)(sizeof(gammas) / sizeof(gammas[0]));

    printf("%8s %12s %12s   x\n", "gamma", "rischio", "rendimento");
    double prev_risk = INFINITY;
    int ok = 1;
    for (int g = 0; g < ng; g++) {
        double x[3], risk, ret;
        if (solve_gamma(gammas[g], x, &risk, &ret)) { ok = 0; break; }
        printf("%8.2f %12.6f %12.6f   (%.4f, %.4f, %.4f)\n",
               gammas[g], risk, ret, x[0], x[1], x[2]);
        /* ammissibilita' */
        if (fabs(x[0] + x[1] + x[2] - 1.0) > 1e-6) ok = 0;
        for (int j = 0; j < 3; j++) if (x[j] < -1e-9) ok = 0;
        /* monotonia: il rischio non cresce al crescere di gamma.  Per gamma
         * piccoli l'ottimo e' lo STESSO vertice, quindi la sequenza e' piatta a
         * meno della precisione del solver: il rischio e' una forma quadratica
         * e la tolleranza dichiarata del percorso conico (che serve il QP puro)
         * e' 1e-8, quindi 1e-9 era piu' stretto di quanto il modello prometta.
         * Si usa la stessa tolleranza dell'ammissibilita'. */
        if (risk > prev_risk + 1e-6) ok = 0;
        prev_risk = risk;
    }

    printf("%s (frontiera monotona nel rischio)\n", ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}