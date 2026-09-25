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

/* portfolio_large.c - large Markowitz QP with a factor risk model (scaling).
 *
 * max r'x - gamma * x'Sx   s.t. sum x = 1, x >= 0
 * with S = F F' + diag(D)  (factor model), so x'Sx is quadratic.
 *
 * Verified: budget, nonnegativity, and strong duality pobj ~ dobj.
 * Usage: portfolio_large [n] [k]   (default 100 5)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 31415926u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100;
    int k = argc > 2 ? atoi(argv[2]) : 5;
    if (n < 1 || k < 1) return 2;
    const double gamma = 0.5;
    double *r = malloc((size_t)n * sizeof(double));
    double *F = malloc((size_t)n * k * sizeof(double));
    double *D = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        r[i] = rnd() * 0.2 - 0.05;
        D[i] = 0.01 + rnd() * 0.02;
        for (int j = 0; j < k; j++) F[i * k + j] = rnd() - 0.5;
    }
    /* objective (as written, MAXIMIZE): r'x - gamma x'Sx = r'x + 0.5 x'Qx
     * with Q = -2*gamma*S (NSD), matching the clone's convention. */
    double *Q = calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int l = 0; l < k; l++) s += F[i * k + l] * F[j * k + l];
            if (i == j) s += D[i];
            Q[i * n + j] = -2.0 * gamma * s;
        }

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, n);
    PRIMAL_appendcons(t, 1);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int i = 0; i < n; i++) {
        PRIMAL_putcj(t, i, r[i]);
        PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, 1.0);
    }
    {
        int nz = 0;
        int *qi = malloc((size_t)n * (n + 1) / 2 * sizeof(int));
        int *qj = malloc((size_t)n * (n + 1) / 2 * sizeof(int));
        double *qv = malloc((size_t)n * (n + 1) / 2 * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                double v = Q[i * n + j];
                if (v != 0.0) { qi[nz] = i; qj[nz] = j; qv[nz] = v; nz++; }
            }
        PRIMAL_putqobj(t, nz, qi, qj, qv);
        free(qi); free(qj); free(qv);
    }
    {
        int *sub = malloc((size_t)n * sizeof(int));
        double *val = malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) { sub[i] = i; val[i] = 1.0; }
        PRIMAL_putarow(t, 0, n, sub, val);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
        free(sub); free(val);
    }

    struct timeval a, b;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b, NULL);
    double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);

    double obj = 0.0, dobj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj);
        double *x = malloc((size_t)n * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 0.0, minx = 0.0;
        for (int i = 0; i < n; i++) { budget += x[i]; if (x[i] < minx) minx = x[i]; }
        ok = fabs(budget - 1.0) < 1e-6 && minx > -1e-6 &&
             fabs(obj - dobj) < 1e-4 * (1.0 + fabs(obj));
        free(x);
    }
    printf("portfolio  n=%d k=%d vars=%d nnzQ=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           n, k, n, n * (n + 1) / 2, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(r); free(F); free(D); free(Q);
    return ok ? 0 : 1;
}
