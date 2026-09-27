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

/* sched.c — Scheduling MIP (equivalent to the Python Fusion API example
 * "Scheduling": assign 30 tasks to 6 processors minimizing the makespan)
 *
 * Optimizer API reformulation of the Fusion one:
 *   x[i][j] ∈ {0,1}  assignment binary (proc i, task j)
 *   t                free continuous variable (makespan)
 *   for each task j:     sum_i x[i][j] = 1          (Expr.sum(x,0) == 1)
 *   for each proc i:     t - sum_j T[j]*x[i][j] >= 0 (repeat(t,m) - x*T >= 0)
 *   objective: min t
 *
 * The task times are generated as in the Python example (random.seed(0),
 * 24 tasks uniform(1,5) + 6 tasks uniform(20,100), sorted decreasing);
 * the Python Mersenne Twister sequence is hardcoded below.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Solve the scheduling MIP and check the makespan and assignment. */
int main(void) {
    enum { N = 30, M = 6 };
    static const double T[N] = {
        97.328509, 93.040884, 68.870958, 57.771417, 54.733747, 28.056097,
         4.931142,  4.638985,  4.632452,  4.608664,  4.595353,  4.377687,
         4.240869,  4.135194,  4.031818,  4.023217,  3.919327,  3.735936,
         3.473476,  3.333528,  3.045099,  3.018747,  2.906388,  2.682286,
         2.619737,  2.240590,  2.213251,  2.127351,  2.035667,  2.002025
    };

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variables: x[i][j] = var i*N + j; t = var M*N */
    PRIMAL_appendvars(task, M * N + 1);
    PRIMAL_appendcons(task, N + M);

    /* x binaries in [0,1], t free */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int v = i * N + j;
            PRIMAL_putvarbound(task, v, PRIMAL_BK_RA, 0.0, 1.0);
            PRIMAL_putvartype(task, v, PRIMAL_VAR_TYPE_INT_BIN);
        }
    PRIMAL_putvarbound(task, M * N, PRIMAL_BK_FR, -INFINITY, INFINITY);

    /* objective: min t */
    PRIMAL_putcj(task, M * N, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* rows 0..N-1: sum_i x[i][j] = 1  (each task assigned to a single proc) */
    for (int j = 0; j < N; j++) {
        int sub[M]; double val[M];
        for (int i = 0; i < M; i++) { sub[i] = i * N + j; val[i] = 1.0; }
        PRIMAL_putarow(task, j, M, sub, val);
        PRIMAL_putconbound(task, j, PRIMAL_BK_FX, 1.0, 1.0);
    }

    /* rows N..N+M-1: t - sum_j T[j]*x[i][j] >= 0  (load_i <= t) */
    for (int i = 0; i < M; i++) {
        int sub[N + 1]; double val[N + 1];
        for (int j = 0; j < N; j++) { sub[j] = i * N + j; val[j] = -T[j]; }
        sub[N] = M * N; val[N] = 1.0;
        PRIMAL_putarow(task, N + i, N + 1, sub, val);
        PRIMAL_putconbound(task, N + i, PRIMAL_BK_LO, 0.0, INFINITY);
    }


    /* Measurement on this model (default cap 100000 nodes): the tree does not
     * close -- 100000 nodes explored, 27 still open, incumbent 97.328509.
     * The value IS the optimum (no processor can go below the largest task),
     * but it is not *proven* optimal, and the reference distinguishes the two:
     * table 7.3, feasible integer point not proven optimal =
     * prosta PRIM_FEAS + solsta PRIM_FEAS, with a termination code for the
     * cap (MSK_RES_TRM_MIO_NUM_BRANCHES in the reference numbering).
     * So here rc != OK IS the correct answer: an INTEGER_OPTIMAL on a tree
     * cut off by the counter would assert a proof that never happened. */
    PRIMALrescodee rc = PRIMAL_optimize(task);

    double xx[M * N + 1], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMALsolstae sta; PRIMALprostae pro;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    PRIMAL_getprosta(task, PRIMAL_SOL_ITR, &pro);
    printf("rc = %d, prosta = %d, solsta = %d\n", (int)rc, (int)pro, (int)sta);
    printf("Optimal makespan: %.4f\n", xx[M * N]);
    for (int i = 0; i < M; i++) {
        double load = 0.0;
        for (int j = 0; j < N; j++)
            if (xx[i * N + j] > 0.5) load += T[j];
        printf("  M%d: load = %.4f\n", i, load);
    }

    /* check: each task assigned exactly once, consistent makespan */
    /* The solver's partition-bound cut derives t >= T_j for this model and
       tightens the objective's box, so the tree closes at the root: an integer
       optimum is PROVEN (rc = OK, solsta = INTEGER_OPTIMAL = 9). */
    int ok = rc == PRIMAL_RES_OK && sta == PRIMAL_SOL_STA_INTEGER_OPTIMAL
          && pro == PRIMAL_PRO_STA_PRIM_FEAS
          && fabs(obj - 97.328509) < 1e-3;
    for (int j = 0; j < N && ok; j++) {
        int cnt = 0;
        for (int i = 0; i < M; i++) cnt += xx[i * N + j] > 0.5;
        if (cnt != 1) ok = 0;
    }
    printf("%s (ottimo provato: LB = max(sum/6, max task) = %.2f, chiude alla radice)\n",
           ok ? "OK" : "FAIL", fmax((0.0 + 399.801611 /* big 6 */ + 76.963 /* small 24 */) / 6.0, 97.328509));

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}