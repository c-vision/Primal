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

/* transport.c - large transportation LP (scaling test).
 *
 * min sum_ij c_ij x_ij
 *   s.t. sum_j x_ij = supply_i   (M rows)
 *        sum_i x_ij = demand_j   (N rows)
 *        x_ij >= 0
 *
 * Usage: transport [M] [N]   (default 60 60)
 * Feasible by construction (supplies/demands come from a reference flow).
 * Verified: row/column sums, nonnegativity, strong duality pobj ~ dobj.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 12345u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 60;
    int N = argc > 2 ? atoi(argv[2]) : 60;
    if (M < 1 || N < 1) return 2;
    int nvar = M * N;
    double *c = malloc((size_t)nvar * sizeof(double));
    double *sup = calloc((size_t)M, sizeof(double));
    double *dem = calloc((size_t)N, sizeof(double));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int k = i * N + j;
            double x0 = rnd() * 10.0;
            c[k] = rnd() * 5.0 + 0.1;
            sup[i] += x0;
            dem[j] += x0;
        }

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nvar);
    PRIMAL_appendcons(t, M + N);
    for (int k = 0; k < nvar; k++) {
        PRIMAL_putcj(t, k, c[k]);
        PRIMAL_putvarbound(t, k, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    for (int i = 0; i < M; i++) {
        int *sub = malloc((size_t)N * sizeof(int));
        double *val = malloc((size_t)N * sizeof(double));
        for (int j = 0; j < N; j++) { sub[j] = i * N + j; val[j] = 1.0; }
        PRIMAL_putarow(t, i, N, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, sup[i], sup[i]);
        free(sub); free(val);
    }
    for (int j = 0; j < N; j++) {
        int *sub = malloc((size_t)M * sizeof(int));
        double *val = malloc((size_t)M * sizeof(double));
        for (int i = 0; i < M; i++) { sub[i] = i * N + j; val[i] = 1.0; }
        PRIMAL_putarow(t, M + j, M, sub, val);
        PRIMAL_putconbound(t, M + j, PRIMAL_BK_FX, dem[j], dem[j]);
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
        double maxsup = 0.0, maxdem = 0.0, minx = 0.0;
        for (int i = 0; i < M; i++) {
            double s = 0.0;
            for (int j = 0; j < N; j++) s += x[i * N + j];
            if (fabs(s - sup[i]) > maxsup) maxsup = fabs(s - sup[i]);
        }
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int i = 0; i < M; i++) s += x[i * N + j];
            if (fabs(s - dem[j]) > maxdem) maxdem = fabs(s - dem[j]);
        }
        for (int k = 0; k < nvar; k++) if (x[k] < minx) minx = x[k];
        ok = maxsup < 1e-5 && maxdem < 1e-5 && minx > -1e-6 &&
             fabs(obj - dobj) < 1e-5 * (1.0 + fabs(obj));
        free(x);
    }
    printf("transport  M=%d N=%d vars=%d cons=%d nnz=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           M, N, nvar, M + N, 2 * nvar, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(c); free(sup); free(dem);
    return ok ? 0 : 1;
}
