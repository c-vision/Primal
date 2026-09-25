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

/* maxcut_sdp.c - max-cut SDP relaxation on a complete bipartite graph.
 *
 *   min (1/4) sum_{(i,j) in E} X_ij   s.t.  X_ii = 1, X >= 0
 *
 * For a bipartite graph the relaxation is exact, so the optimum equals
 * -|E|/4 (all edges cut).  Reference: -p*q/4 for K_{p,q}.
 * Usage: maxcut_sdp [p] [q]   (default 6 6)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

int main(int argc, char **argv) {
    int p = argc > 1 ? atoi(argv[1]) : 4;
    int q = argc > 2 ? atoi(argv[2]) : 4;
    if (p < 1 || q < 1) return 2;
    int n = p + q;
    int E = p * q;
    double expected = -0.25 * (double)E;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 1);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);   /* dummy scalar */
    int dim = n;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, n);                            /* X_ii = 1 */

    /* objective: coefficient 1/4 on each edge X_ij (i<p<=j):
     * <E_ij,X> = 2 X_ij, so provide val = (1/4)/2 = 1/8 */
    {
        int *si = malloc((size_t)E * sizeof(int));
        int *sj = malloc((size_t)E * sizeof(int));
        double *sv = malloc((size_t)E * sizeof(double));
        int nz = 0;
        for (int i = 0; i < p; i++)
            for (int j = p; j < n; j++) { si[nz] = i; sj[nz] = j; sv[nz] = 0.125; nz++; }
        int mC;
        PRIMAL_appendsparsesymmat(t, n, nz, si, sj, sv, &mC);
        PRIMAL_putbarcj(t, 0, 1, (int[]){mC}, (double[]){1.0});
        free(si); free(sj); free(sv);
    }
    for (int i = 0; i < n; i++) {
        int mE;
        PRIMAL_appendsparsesymmat(t, n, 1, (int[]){i}, (int[]){i}, (double[]){1.0}, &mE);
        PRIMAL_putbaraij(t, i, 0, 1, (int[]){mE}, (double[]){1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    /* valid inequalities |X_ij| <= 1 on the edges (implied by PSD + diag=1):
     * they tighten the outer LP relaxation substantially.  <E_ij,X> = 2 X_ij. */
    {
        int row = n;
        PRIMAL_appendcons(t, E);
        for (int i = 0; i < p; i++)
            for (int j = p; j < n; j++) {
                int mO;
                PRIMAL_appendsparsesymmat(t, n, 1, (int[]){i}, (int[]){j}, (double[]){1.0}, &mO);
                PRIMAL_putbaraij(t, row, 0, 1, (int[]){mO}, (double[]){1.0});
                PRIMAL_putconbound(t, row, PRIMAL_BK_RA, -2.0, 2.0);
                row++;
            }
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
        ok = fabs(obj - expected) < 1e-4 * (1.0 + fabs(expected));
    }
    printf("maxcut_sdp p=%d q=%d n=%d edges=%d obj=%.6f ref=%.6f rc=%d t=%.4fs %s\n",
           p, q, n, E, obj, expected, (int)rc, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
