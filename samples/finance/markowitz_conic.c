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

/* markowitz_conic.c - Markowitz portfolio as a second-order cone program.
 *
 *   max mu'x   s.t.  ||G'x||_2 <= sqrt(gamma)   [(sqrt(gamma), G'x) in Q]
 *                    e'x = 1,  x >= 0
 * with the factor risk model  Sigma = F F' + diag(D),  G = [F, sqrt(D)].
 *
 * Verified: budget, nonnegativity, risk <= sqrt(gamma); when gamma is large
 * enough that the risk constraint is inactive the optimum is the single
 * best-mean asset (objective = max mu_i).
 * Usage: markowitz_conic [n] [k] [gamma]   (default 60 4 0.05)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 42424242u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 60;
    int k = argc > 2 ? atoi(argv[2]) : 4;
    double gamma = argc > 3 ? atof(argv[3]) : 0.05;
    if (n < 1 || k < 1 || gamma <= 0.0) return 2;
    double *mu = malloc((size_t)n * sizeof(double));
    double *F = malloc((size_t)n * k * sizeof(double));
    double *D = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        mu[i] = 0.02 + rnd() * 0.15;
        D[i] = 0.001 + rnd() * 0.004;
        for (int j = 0; j < k; j++) F[i * k + j] = (rnd() - 0.5) * 0.2;
    }
    /* G = [F | diag(sqrt(D))]: n x (k+n). g[j] = column j of G'  */
    int m2 = k + n;
    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (n), z (m2), tcone (1) */
    int X0 = 0, Z0 = n, TC = n + m2;
    PRIMAL_appendvars(t, n + m2 + 1);
    PRIMAL_appendcons(t, m2);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int i = 0; i < n; i++) {
        PRIMAL_putcj(t, X0 + i, mu[i]);
        PRIMAL_putvarbound(t, X0 + i, PRIMAL_BK_LO, 0.0, 1.0);
    }
    for (int j = 0; j < m2; j++) PRIMAL_putvarbound(t, Z0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, TC, PRIMAL_BK_FX, sqrt(gamma), sqrt(gamma));
    for (int j = 0; j < m2; j++) {          /* z_j - (G'x)_j = 0 */
        int *sub = malloc((size_t)(n + 1) * sizeof(int));
        double *val = malloc((size_t)(n + 1) * sizeof(double));
        int nz = 0;
        for (int i = 0; i < n; i++) {
            double g = (j < k) ? F[i * k + j]
                               : ((j - k == i) ? sqrt(D[i]) : 0.0);
            if (g != 0.0) { sub[nz] = X0 + i; val[nz] = -g; nz++; }
        }
        sub[nz] = Z0 + j; val[nz] = 1.0; nz++;
        PRIMAL_putarow(t, j, nz, sub, val);
        PRIMAL_putconbound(t, j, PRIMAL_BK_FX, 0.0, 0.0);
        free(sub); free(val);
    }
    {   /* (tcone, z) in QUAD */
        int *mem = malloc((size_t)(m2 + 1) * sizeof(int));
        mem[0] = TC;
        for (int j = 0; j < m2; j++) mem[1 + j] = Z0 + j;
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, m2 + 1, mem);
        free(mem);
    }
    {   /* budget */
        int *sub = malloc((size_t)n * sizeof(int));
        double *val = malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) { sub[i] = X0 + i; val[i] = 1.0; }
        PRIMAL_appendcons(t, 1);
        PRIMAL_putarow(t, m2, n, sub, val);
        PRIMAL_putconbound(t, m2, PRIMAL_BK_FX, 1.0, 1.0);
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
        double *x = malloc((size_t)(n + m2 + 1) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 0.0, minx = 0.0, risk2 = 0.0, maxmu = mu[0];
        for (int i = 0; i < n; i++) {
            budget += x[i];
            if (x[i] < minx) minx = x[i];
            if (mu[i] > maxmu) maxmu = mu[i];
            risk2 += D[i] * x[i] * x[i];
        }
        for (int a2 = 0; a2 < k; a2++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += F[i * k + a2] * x[i];
            risk2 += s * s;
        }
        ok = fabs(budget - 1.0) < 1e-6 && minx > -1e-6 &&
             risk2 <= gamma * (1.0 + 1e-4);
        /* when the risk budget is not binding the optimum is the best asset */
        if (ok && risk2 < gamma * (1.0 - 1e-3))
            ok = fabs(obj - maxmu) < 1e-5 * (1.0 + maxmu);
        free(x);
    }
    printf("markowitz  n=%d k=%d gamma=%.4f obj=%.6f t=%.4fs %s\n",
           n, k, gamma, obj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(mu); free(F); free(D);
    return ok ? 0 : 1;
}
