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

/* qp_sparse.c - large sparse convex QP (scaling test for the sparse
 * normal-equations QP interior point).
 *
 *   min 1/2 x'Qx + c'x   s.t.  A x = b,  x >= 0
 *
 * Q is a sparse PSD matrix (1-D graph Laplacian + 0.5*I: tridiagonal, so PD and
 * ~3n nonzeros), A has m rows, and b = A*x_ref for a planted feasible x_ref>=0,
 * so the problem is feasible by construction and strictly convex (unique opt).
 * For large n with sparse Q and moderate m this routes to the sparse QP IPM:
 * factor M = Q + Z/X with a sparse Cholesky, W = M^-1 A', K = A W + delta I
 * (m x m); the dense augmented LU would be O((n+m)^3) per iteration and is
 * prohibitive at these sizes. See README "Performance".
 * Verified: primal feasibility (Ax=b, x>=0) and strong duality (pobj ~ dobj).
 * Usage: qp_sparse [n] [m]   (default 2000 8)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 13579u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2000;
    int m = argc > 2 ? atoi(argv[2]) : 8;
    if (n < 2 || m < 1) return 2;

    /* Q = 1-D Laplacian + 0.5 I (tridiagonal, PD); lower-triangle triplets */
    int nq = 2 * n - 1;
    int *qi = (int *)malloc((size_t)nq * sizeof(int));
    int *qj = (int *)malloc((size_t)nq * sizeof(int));
    double *qv = (double *)malloc((size_t)nq * sizeof(double));
    int w = 0;
    for (int j = 0; j < n; j++) { qi[w] = j; qj[w] = j; qv[w] = ((j == 0 || j == n - 1) ? 1.0 : 2.0) + 0.5; w++; }
    for (int j = 0; j < n - 1; j++) { qi[w] = j + 1; qj[w] = j; qv[w] = -1.0; w++; }

    double *xref = (double *)malloc((size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * sizeof(double));
    for (int j = 0; j < n; j++) {
        xref[j] = (rnd() < 0.7) ? (rnd() * 5.0 + 0.1) : 0.0;   /* ~30% at the bound */
        c[j] = rnd() * 4.0 - 2.0;
    }
    double *A = (double *)malloc((size_t)m * (size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)m * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) A[(size_t)i * n + j] = (i == 0) ? 1.0 : (rnd() * 2.0 - 1.0);
    for (int i = 0; i < m; i++) { b[i] = 0.0; for (int j = 0; j < n; j++) b[i] += A[(size_t)i * n + j] * xref[j]; }

    int *sub = (int *)malloc((size_t)n * sizeof(int));
    double *val = (double *)malloc((size_t)n * sizeof(double));
    for (int j = 0; j < n; j++) sub[j] = j;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, n); PRIMAL_appendcons(t, m);
    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(t, j, c[j]);
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMAL_putqobj(t, nq, qi, qj, qv);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) val[j] = A[(size_t)i * n + j];
        PRIMAL_putarow(t, i, n, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, b[i], b[i]);
    }

    struct timeval a, bb;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&bb, NULL);
    double sec = (bb.tv_sec - a.tv_sec) + 1e-6 * (bb.tv_usec - a.tv_usec);

    double obj = 0.0, dobj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj);
        double *x = (double *)malloc((size_t)n * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double maxviol = 0.0, minx = 0.0;
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) s += A[(size_t)i * n + j] * x[j];
            double e = fabs(s - b[i]); if (e > maxviol) maxviol = e;
        }
        for (int j = 0; j < n; j++) if (x[j] < minx) minx = x[j];
        double bn = 1.0;
        for (int i = 0; i < m; i++) { double e = fabs(b[i]); if (e > bn) bn = e; }
        ok = maxviol < 1e-4 * bn && minx > -1e-6 &&
             fabs(obj - dobj) < 1e-4 * (1.0 + fabs(obj));
        free(x);
    }
    printf("qp_sparse  n=%d m=%d nnzQ=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           n, m, nq, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(qi); free(qj); free(qv); free(xref); free(c); free(A); free(b); free(sub); free(val);
    return ok ? 0 : 1;
}
