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

/* assignment.c - n x n assignment problem (LP, scaling test).
 *
 * min sum_ij c_ij x_ij
 *   s.t. sum_j x_ij = 1, sum_i x_ij = 1, x >= 0
 *
 * The assignment polytope is integral, so the LP optimum is a permutation.
 * Verified: row/column sums = 1, integrality, strong duality.
 * Usage: assignment [n]   (default 80)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 987654321u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 80;
    if (n < 1) return 2;
    int nvar = n * n;
    double *c = malloc((size_t)nvar * sizeof(double));
    for (int k = 0; k < nvar; k++) c[k] = rnd() * 100.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nvar);
    PRIMAL_appendcons(t, 2 * n);
    for (int k = 0; k < nvar; k++) {
        PRIMAL_putcj(t, k, c[k]);
        PRIMAL_putvarbound(t, k, PRIMAL_BK_LO, 0.0, 1.0);
    }
    for (int i = 0; i < n; i++) {
        int *sub = malloc((size_t)n * sizeof(int));
        double *val = malloc((size_t)n * sizeof(double));
        for (int j = 0; j < n; j++) { sub[j] = i * n + j; val[j] = 1.0; }
        PRIMAL_putarow(t, i, n, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
        for (int j = 0; j < n; j++) { sub[j] = j * n + i; val[j] = 1.0; }
        PRIMAL_putarow(t, n + i, n, sub, val);
        PRIMAL_putconbound(t, n + i, PRIMAL_BK_FX, 1.0, 1.0);
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
        double *x = malloc((size_t)nvar * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double maxrow = 0.0, maxcol = 0.0, maxfrac = 0.0;
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                double v = x[i * n + j];
                s += v;
                double fr = fabs(v - floor(v + 0.5));
                if (fr > maxfrac) maxfrac = fr;
            }
            if (fabs(s - 1.0) > maxrow) maxrow = fabs(s - 1.0);
        }
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += x[i * n + j];
            if (fabs(s - 1.0) > maxcol) maxcol = fabs(s - 1.0);
        }
        ok = maxrow < 1e-6 && maxcol < 1e-6 && maxfrac < 1e-6 &&
             fabs(obj - dobj) < 1e-5 * (1.0 + fabs(obj));
        free(x);
    }
    printf("assignment n=%d vars=%d cons=%d nnz=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           n, nvar, 2 * n, 2 * nvar, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(c);
    return ok ? 0 : 1;
}
