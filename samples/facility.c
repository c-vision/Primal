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

/* facility.c - capacitated facility location MILP (scaling test).
 *
 * min sum_j f_j y_j + sum_ij c_ij x_ij
 *   s.t. sum_j x_ij = d_i          (each client served)
 *        sum_i x_ij <= q_j y_j     (capacity if open)
 *        y_j in {0,1}, x_ij >= 0
 *
 * Verified: demands met, capacities respected, y integral, and the MIP
 * objective is >= the LP relaxation bound (both computed here).
 * Usage: facility [m] [n]   (default 15 40)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 55555u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static void build(PRIMALtask_t t, int m, int n, int integral,
                  const double *f, const double *q, const double *d,
                  const double *c) {
    /* vars: y_0..y_{m-1}, x_ij at m + i*n + j */
    PRIMAL_appendvars(t, m + m * n);
    PRIMAL_appendcons(t, n + m);
    for (int j = 0; j < m; j++) {
        PRIMAL_putcj(t, j, f[j]);
        PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 1.0);
        if (integral) PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT_BIN);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            int k = m + i * m + j;
            PRIMAL_putcj(t, k, c[i * m + j]);
            PRIMAL_putvarbound(t, k, PRIMAL_BK_LO, 0.0, INFINITY);
        }
    for (int i = 0; i < n; i++) {          /* sum_j x_ij = d_i */
        int *sub = malloc((size_t)m * sizeof(int));
        double *val = malloc((size_t)m * sizeof(double));
        for (int j = 0; j < m; j++) { sub[j] = m + i * m + j; val[j] = 1.0; }
        PRIMAL_putarow(t, i, m, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, d[i], d[i]);
        free(sub); free(val);
    }
    for (int j = 0; j < m; j++) {          /* sum_i x_ij - q_j y_j <= 0 */
        int *sub = malloc((size_t)(n + 1) * sizeof(int));
        double *val = malloc((size_t)(n + 1) * sizeof(double));
        for (int i = 0; i < n; i++) { sub[i] = m + i * m + j; val[i] = 1.0; }
        sub[n] = j; val[n] = -q[j];
        PRIMAL_putarow(t, n + j, n + 1, sub, val);
        PRIMAL_putconbound(t, n + j, PRIMAL_BK_UP, -INFINITY, 0.0);
        free(sub); free(val);
    }
}

int main(int argc, char **argv) {
    int m = argc > 1 ? atoi(argv[1]) : 15;
    int n = argc > 2 ? atoi(argv[2]) : 40;
    if (m < 1 || n < 1) return 2;
    double *f = malloc((size_t)m * sizeof(double));
    double *q = malloc((size_t)m * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    double *c = malloc((size_t)m * n * sizeof(double));
    double totald = 0.0;
    for (int j = 0; j < m; j++) { f[j] = 100.0 + rnd() * 200.0; q[j] = 40.0 + rnd() * 40.0; }
    for (int i = 0; i < n; i++) { d[i] = 1.0 + rnd() * 8.0; totald += d[i]; }
    for (int k = 0; k < m * n; k++) c[k] = 1.0 + rnd() * 20.0;
    /* scale capacities so total capacity is ample */
    double totalq = 0.0;
    for (int j = 0; j < m; j++) totalq += q[j];
    for (int j = 0; j < m; j++) q[j] *= 1.5 * totald / totalq;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    build(t, m, n, 1, f, q, d, c);
    struct timeval a, b;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b, NULL);
    double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);

    double obj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        double *x = malloc((size_t)(m + m * n) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double maxdem = 0.0, maxcap = 0.0, maxfrac = 0.0;
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < m; j++) s += x[m + i * m + j];
            if (fabs(s - d[i]) > maxdem) maxdem = fabs(s - d[i]);
        }
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += x[m + i * m + j];
            double cap = q[j] * x[j];
            if (s - cap > maxcap) maxcap = s - cap;
            double fr = fabs(x[j] - floor(x[j] + 0.5));
            if (fr > maxfrac) maxfrac = fr;
        }
        ok = maxdem < 1e-5 && maxcap < 1e-5 && maxfrac < 1e-6;
        free(x);
    }
    /* LP relaxation lower bound */
    double lp = 0.0;
    {
        PRIMALtask_t r; PRIMAL_maketask(env, 0, 0, &r);
        build(r, m, n, 0, f, q, d, c);
        if (PRIMAL_optimize(r) == PRIMAL_RES_OK)
            PRIMAL_getprimalobj(r, PRIMAL_SOL_ITR, &lp);
        PRIMAL_deletetask(&r);
    }
    if (obj + 1e-6 < lp) ok = 0;

    printf("facility   m=%d n=%d vars=%d cons=%d obj=%.4f lp=%.4f t=%.4fs %s\n",
           m, n, m + m * n, n + m, obj, lp, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(f); free(q); free(d); free(c);
    return ok ? 0 : 1;
}
