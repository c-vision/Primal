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

/* cvar_portfolio.c - CVaR portfolio optimization (Rockafellar-Uryasev) as an LP.
 *
 *   min  t + 1/(alpha*S) sum_s u_s
 *   s.t. u_s >= -(r_s'x) - t,  u_s >= 0
 *        e'x = 1,  x >= 0
 *
 * Verified: the LP objective equals the empirical CVaR of x* computed
 * directly from the scenario losses (mean of the worst alpha-fraction).
 * Usage: cvar_portfolio [n] [S] [alpha]   (default 40 200 0.10)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 606060u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int cmp_desc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 40;
    int S = argc > 2 ? atoi(argv[2]) : 200;
    double alpha = argc > 3 ? atof(argv[3]) : 0.10;
    if (n < 1 || S < 1 || alpha <= 0.0 || alpha >= 1.0) return 2;
    double *R = malloc((size_t)S * n * sizeof(double));
    for (int s = 0; s < S; s++)
        for (int i = 0; i < n; i++) R[s * n + i] = (rnd() - 0.5) * 0.2 + 0.01;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (n), t (1), u (S) */
    int X0 = 0, TV = n, U0 = n + 1;
    PRIMAL_appendvars(t, n + 1 + S);
    PRIMAL_appendcons(t, S + 1);
    for (int i = 0; i < n; i++) {
        PRIMAL_putvarbound(t, X0 + i, PRIMAL_BK_LO, 0.0, 1.0);
    }
    PRIMAL_putvarbound(t, TV, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, TV, 1.0);
    for (int s = 0; s < S; s++) {
        PRIMAL_putvarbound(t, U0 + s, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, U0 + s, 1.0 / (alpha * S));
        /* u_s + r_s'x + t >= 0 */
        int *sub = malloc((size_t)(n + 2) * sizeof(int));
        double *val = malloc((size_t)(n + 2) * sizeof(double));
        int nz = 0;
        for (int i = 0; i < n; i++) { sub[nz] = X0 + i; val[nz] = R[s * n + i]; nz++; }
        sub[nz] = TV; val[nz] = 1.0; nz++;
        sub[nz] = U0 + s; val[nz] = 1.0; nz++;
        PRIMAL_putarow(t, s, nz, sub, val);
        PRIMAL_putconbound(t, s, PRIMAL_BK_LO, 0.0, INFINITY);
        free(sub); free(val);
    }
    {   /* budget */
        int *sub = malloc((size_t)n * sizeof(int));
        double *val = malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) { sub[i] = X0 + i; val[i] = 1.0; }
        PRIMAL_putarow(t, S, n, sub, val);
        PRIMAL_putconbound(t, S, PRIMAL_BK_FX, 1.0, 1.0);
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
        double *x = malloc((size_t)(n + 1 + S) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 0.0, minx = 0.0;
        for (int i = 0; i < n; i++) { budget += x[i]; if (x[i] < minx) minx = x[i]; }
        /* empirical CVaR of x*: mean of the worst alpha*S losses */
        double *loss = malloc((size_t)S * sizeof(double));
        for (int s = 0; s < S; s++) {
            double l = 0.0;
            for (int i = 0; i < n; i++) l -= R[s * n + i] * x[i];
            loss[s] = l;
        }
        qsort(loss, (size_t)S, sizeof(double), cmp_desc);
        int ktail = (int)ceil(alpha * S);
        double cvar = 0.0;
        for (int s = 0; s < ktail; s++) cvar += loss[s];
        cvar /= (double)ktail;
        ok = fabs(budget - 1.0) < 1e-6 && minx > -1e-6 &&
             fabs(obj - cvar) < 1e-4 * (1.0 + fabs(cvar));
        printf("cvar       n=%d S=%d alpha=%.2f obj=%.6f empirical=%.6f t=%.4fs %s\n",
               n, S, alpha, obj, cvar, sec, ok ? "OK" : "FAIL");
        free(x); free(loss);
    } else {
        printf("cvar       n=%d S=%d rc=%d FAIL\n", n, S, (int)rc);
    }

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(R);
    return ok ? 0 : 1;
}
