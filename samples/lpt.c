/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 ... (see LICENSE).
 */

/* lpt.c - multi-processor scheduling by an LPT-warm-started MIP
 * (MOSEK Cookbook 11.4, lpt.cc).
 *
 * Assign n tasks to m identical machines so that the makespan is minimal:
 *     minimize   t
 *     s.t.       sum_i x_ij = 1                       (each task on one machine)
 *                t >= sum_j T_j x_ij   for each i     (t >= every machine load)
 *                x_ij in {0,1},  t >= 0.
 * The Cookbook feeds the answer of the **Longest Processing Time** heuristic
 * (sort the tasks descending, put each on the currently least-loaded machine)
 * as a MIP initial solution and solves with a 1% relative gap.
 *
 * Hand instance: m = 2, tasks {3,3,2,2,2}.  The total is 12, so any makespan
 * is at least 6, and the partition {3,3} / {2,2,2} attains it: optimum 6.
 * LPT gives 7 (3->M0, 3->M1, 2->M0, 2->M1, 2->M0), so the warm start is NOT
 * optimal and the MIP has to improve on it -- which is the point of feeding it.
 *
 * Usage: lpt   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define M 2
#define NT 5
static const int TASK[NT] = {3, 3, 2, 2, 2};   /* already sorted descending (LPT) */

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, M * NT + 1);
    const int TT = M * NT;                  /* makespan variable */
    PRIMAL_appendcons(t, NT + M);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < NT; j++) {
            PRIMAL_putvarbound(t, i * NT + j, PRIMAL_BK_RA, 0.0, 1.0);
            PRIMAL_putvartype(t, i * NT + j, PRIMAL_VAR_TYPE_INT_BIN);
        }
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, TT, 1.0);
    int row = 0;
    for (int j = 0; j < NT; j++) {          /* each task on exactly one machine */
        int sub[M]; double v[M];
        for (int i = 0; i < M; i++) { sub[i] = i * NT + j; v[i] = 1.0; }
        PRIMAL_putarow(t, row, M, sub, v);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    for (int i = 0; i < M; i++) {           /* t >= load of machine i */
        int sub[NT + 1]; double v[NT + 1];
        sub[0] = TT; v[0] = 1.0;
        for (int j = 0; j < NT; j++) { sub[j + 1] = i * NT + j; v[j + 1] = -TASK[j]; }
        PRIMAL_putarow(t, row, NT + 1, sub, v);
        PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY); row++;
    }

    /* LPT initial solution: least-loaded machine for each (descending) task. */
    int load[M] = {0};
    double warm[M * NT + 1] = {0};
    for (int j = 0; j < NT; j++) {
        int i = 0;
        for (int k = 1; k < M; k++) if (load[k] < load[i]) i = k;
        warm[i * NT + j] = 1.0;
        load[i] += TASK[j];
    }
    int lpt_makespan = load[0];
    for (int i = 1; i < M; i++) if (load[i] > lpt_makespan) lpt_makespan = load[i];
    warm[TT] = (double)lpt_makespan;
    PRIMAL_putxx(t, PRIMAL_SOL_ITR, warm);
    PRIMAL_putdouparam(t, PRIMAL_DPAR_MIP_TOL_REL_GAP, 1e-4);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, x[M * NT + 1] = {0};
    if (ok) { PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj); PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); }
    int good = ok && fabs(obj - 6.0) < 1e-6 && fabs(x[TT] - 6.0) < 1e-6 &&
               lpt_makespan == 7;   /* the MIP improved on the LPT warm start */
    printf("lpt: makespan=%.6f (LPT iniziale=%d)  atteso 6 (LPT=7 subottimo)  %s\n",
           x[TT], lpt_makespan, good ? "OK" : "FAIL");
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return good ? 0 : 1;
}
