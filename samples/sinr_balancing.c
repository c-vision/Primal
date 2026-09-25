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
 */

/* sinr_balancing.c - max-min SINR power control (uplink), by bisection on t.
 *
 * Source: MOSEK Modeling Cookbook / MOSEK Tutorials, "SINR".
 *
 * With per-user power p_i, channel gains G (G_ij = gain from user j to
 * receiver i), noise sigma_i^2 and a total power budget P:
 *
 *   SINR_i = G_ii p_i / ( sum_{j!=i} G_ij p_j + sigma_i^2 )
 *   max t  s.t.  SINR_i >= t  for all i,  sum_i p_i <= P,  p >= 0.
 *
 * The constraint is bilinear in (p, t), so it is NOT a single cone. For a
 * FIXED t it is LINEAR in p:
 *
 *   p_i - t * sum_{j!=i} G_ij p_j >= t * sigma_i^2,   sum_i p_i <= P,  p >= 0
 *
 * and the optimal t is the largest t for which that LP is feasible: the
 * classic generalized-eigenvalue / bisection scheme. Each feasibility check
 * is one LP solved by this library (no cone needed).
 *
 * Symmetric instance (hand-derived optimum t* = 10/21):
 *   G = [[1, 0.1], [0.1, 1]],  sigma^2 = (1, 1),  P = 1.
 *   By symmetry p_0 = p_1 = 1/2 (the budget is tight), and
 *   1/2 >= t * (0.1 * 1/2 + 1) = t * 21/20, so t* = (1/2)/(21/20) = 10/21.
 *
 * Usage: sinr_balancing   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define N 2
static const double G[N][N] = { {1.0, 0.1}, {0.1, 1.0} };
static const double sigma2[N] = {1.0, 1.0};
static const double P = 1.0;

/* feasibility LP for a fixed t: returns 1 if feasible */
static int feasible(double t, double *p_out) {
    PRIMALenv_t env; PRIMALtask_t task;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, N + 1, N, &task);
    for (int i = 0; i < N; i++) PRIMAL_putvarbound(task, i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < N; i++) PRIMAL_putcj(task, i, 0.0);   /* feasibility only */

    int row = 0;
    for (int i = 0; i < N; i++) {          /* p_i - t*sum_{j!=i} G_ij p_j >= t*sigma_i^2 */
        int sub[N]; double val[N];
        for (int j = 0; j < N; j++) {
            sub[j] = j;
            val[j] = (j == i) ? 1.0 : -t * G[i][j];
        }
        PRIMAL_putarow(task, row, N, sub, val);
        PRIMAL_putconbound(task, row, PRIMAL_BK_LO, t * sigma2[i], INFINITY);
        row++;
    }
    {   /* sum_i p_i <= P */
        int sub[N]; double val[N];
        for (int i = 0; i < N; i++) { sub[i] = i; val[i] = 1.0; }
        PRIMAL_putarow(task, row, N, sub, val);
        PRIMAL_putconbound(task, row, PRIMAL_BK_UP, -INFINITY, P);
        row++;
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok && p_out) PRIMAL_getxx(task, PRIMAL_SOL_ITR, p_out);
    PRIMAL_deletetask(&task); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    /* bisection on t: [0, hi]; hi is an obvious upper bound (SINR with p=P). */
    double lo = 0.0, hi = 100.0;
    for (int it = 0; it < 80; it++) {
        double mid = 0.5 * (lo + hi);
        if (feasible(mid, NULL)) lo = mid; else hi = mid;
    }
    double p[N] = {0, 0};
    feasible(lo, p);

    double want = 10.0 / 21.0;               /* 0.476190... */
    double gmin = 1e30, budget = 0.0;
    for (int i = 0; i < N; i++) {
        double inter = sigma2[i];
        for (int j = 0; j < N; j++) if (j != i) inter += G[i][j] * p[j];
        double s = G[i][i] * p[i] / inter;
        if (s < gmin) gmin = s;
        budget += p[i];
    }
    int ok = fabs(lo - want) < 1e-6 && fabs(gmin - lo) < 1e-6 && budget <= P + 1e-7;
    printf("sinr_balancing  t*=%.9f (atteso 10/21=%.9f)  min_SINR=%.9f  p=(%.6f,%.6f) %s\n",
           lo, want, gmin, p[0], p[1], ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
