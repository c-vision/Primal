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

/* knapsack.c - 0/1 knapsack MILP (scaling test).
 *
 * max sum_i p_i x_i   s.t. sum_i w_i x_i <= C,  x_i in {0,1}
 *
 * Verified against an exact O(nC) dynamic program.
 * Usage: knapsack [n] [C]   (default 120 3000)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 24681357u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 120;
    int C = argc > 2 ? atoi(argv[2]) : 3000;
    if (n < 1 || C < 1) return 2;
    int *w = malloc((size_t)n * sizeof(int));
    int *p = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        w[i] = 1 + (int)(rnd() * 100.0);
        p[i] = 1 + (int)(rnd() * 100.0);
    }

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, n);
    PRIMAL_appendcons(t, 1);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int i = 0; i < n; i++) {
        PRIMAL_putcj(t, i, (double)p[i]);
        PRIMAL_putvarbound(t, i, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, i, PRIMAL_VAR_TYPE_INT_BIN);
    }
    {
        int *sub = malloc((size_t)n * sizeof(int));
        double *val = malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) { sub[i] = i; val[i] = (double)w[i]; }
        PRIMAL_putarow(t, 0, n, sub, val);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, (double)C);
        free(sub); free(val);
    }

    struct timeval a, b;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b, NULL);
    double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);

    double obj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        double *x = malloc((size_t)n * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        int cap = 0;
        double val = 0.0;
        for (int i = 0; i < n; i++) {
            int xi = (int)floor(x[i] + 0.5);
            if (fabs(x[i] - xi) > 1e-6) ok = 0;
            cap += xi * w[i];
            val += (double)xi * p[i];
        }
        if (cap > C) ok = 0;
        if (fabs(val - obj) > 1e-6) ok = 0;
        /* exact DP reference */
        int *dp = malloc((size_t)(C + 1) * sizeof(int));
        for (int c = 0; c <= C; c++) dp[c] = 0;
        for (int i = 0; i < n; i++)
            for (int c = C; c >= w[i]; c--) {
                int cand = dp[c - w[i]] + p[i];
                if (cand > dp[c]) dp[c] = cand;
            }
        double dpbest = (double)dp[C];
        if (fabs(obj - dpbest) > 1e-6) ok = 0;
        printf("knapsack   n=%d C=%d vars=%d obj=%.0f dp=%.0f t=%.4fs %s\n",
               n, C, n, obj, dpbest, sec, ok ? "OK" : "FAIL");
        free(dp); free(x);
    } else {
        printf("knapsack   n=%d C=%d rc=%d %s\n", n, C, (int)rc, "FAIL");
    }

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(w); free(p);
    return ok ? 0 : 1;
}
