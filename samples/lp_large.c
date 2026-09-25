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

/* lp_large.c - large generic sparse LP (scaling test for the sparse
 * normal-equations interior point).
 *
 *   min c'x   s.t.  A x = b,  x >= 0
 *
 * A is a random sparse matrix (each column carries `d` nonzeros). The instance
 * is built with a PLANTED optimal solution x* derived from the KKT conditions,
 * so the exact optimum obj* = c'x* is known independently of the solver:
 *   pick x* >= 0 with support S, a free y*, and z*_j = 0 (j in S) / z*_j > 0
 *   (j not in S); then b = A x* and c = A'y* + z* make (x*,y*,z*) a KKT point,
 *   hence x* is optimal (and obj* = y*'b, since z*'x* = 0).
 *
 * At large sizes the LP is routed automatically to the sparse Mehrotra IPM
 * (normal equations A*Theta*A' + delta*I with a left-looking sparse Cholesky);
 * see README "Performance". Verified: obj == c'x*, primal feasibility
 * (Ax = b, x >= 0) and strong duality (pobj ~ dobj).
 * Usage: lp_large [rows] [cols] [density]   (default 300 6000 8)
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
    int m = argc > 1 ? atoi(argv[1]) : 300;   /* rows    */
    int n = argc > 2 ? atoi(argv[2]) : 6000;  /* cols    */
    int d = argc > 3 ? atoi(argv[3]) : 8;     /* nnz/col */
    if (m < 1 || n < 1 || d < 1 || d > m) return 2;

    /* ---- sparse A in column form: d distinct rows per column ---- */
    int nnz = n * d;
    int *arow = (int *)malloc((size_t)nnz * sizeof(int));
    double *aval = (double *)malloc((size_t)nnz * sizeof(double));
    char *used = (char *)calloc((size_t)m, sizeof(char));
    if (!arow || !aval || !used) return 2;
    for (int j = 0; j < n; j++) {
        for (int q = 0; q < m; q++) used[q] = 0;
        for (int k = 0; k < d; k++) {
            int i;
            do { i = (int)(rnd() * m); if (i >= m) i = m - 1; } while (used[i]);
            used[i] = 1;
            arow[j * d + k] = i;
            aval[j * d + k] = 2.0 * rnd() - 1.0;   /* in [-1,1) */
        }
    }
    free(used);

    /* ---- planted optimum: x*, y*, z* ---- */
    double *xs = (double *)malloc((size_t)n * sizeof(double));  /* x* */
    double *ys = (double *)malloc((size_t)m * sizeof(double));  /* y* */
    double *zs = (double *)malloc((size_t)n * sizeof(double));  /* z* */
    for (int j = 0; j < n; j++)
        xs[j] = (rnd() < 0.6) ? (rnd() * 10.0 + 0.1) : 0.0;   /* ~60% positive */
    for (int i = 0; i < m; i++) ys[i] = 2.0 * rnd() - 1.0;
    for (int j = 0; j < n; j++)
        zs[j] = (xs[j] > 0.0) ? 0.0 : (rnd() * 5.0 + 0.1);

    /* b = A x* */
    double *b = (double *)calloc((size_t)m, sizeof(double));
    for (int j = 0; j < n; j++)
        if (xs[j] != 0.0)
            for (int k = 0; k < d; k++) b[arow[j*d+k]] += aval[j*d+k] * xs[j];

    /* c = A'y* + z* ;  obj* = c'x* */
    double *c = (double *)malloc((size_t)n * sizeof(double));
    for (int j = 0; j < n; j++) {
        double s = zs[j];
        for (int k = 0; k < d; k++) s += aval[j*d+k] * ys[arow[j*d+k]];
        c[j] = s;
    }
    double objstar = 0.0;
    for (int j = 0; j < n; j++) objstar += c[j] * xs[j];

    /* ---- row CSR for putarow ---- */
    int *rcnt = (int *)calloc((size_t)m, sizeof(int));
    for (int e = 0; e < nnz; e++) rcnt[arow[e]]++;
    int *rstart = (int *)malloc((size_t)(m + 1) * sizeof(int));
    rstart[0] = 0;
    for (int i = 0; i < m; i++) rstart[i+1] = rstart[i] + rcnt[i];
    int *rsub = (int *)malloc((size_t)nnz * sizeof(int));
    double *rval = (double *)malloc((size_t)nnz * sizeof(double));
    int *fill = (int *)calloc((size_t)m, sizeof(int));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < d; k++) {
            int i = arow[j*d+k];
            int pos = rstart[i] + fill[i]++;
            rsub[pos] = j; rval[pos] = aval[j*d+k];
        }
    free(fill);

    /* ---- task ---- */
    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, n);
    PRIMAL_appendcons(t, m);
    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(t, j, c[j]);
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    for (int i = 0; i < m; i++) {
        PRIMAL_putarow(t, i, rcnt[i], &rsub[rstart[i]], &rval[rstart[i]]);
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
        double *Ax = (double *)calloc((size_t)m, sizeof(double));
        for (int j = 0; j < n; j++)
            if (x[j] != 0.0)
                for (int k = 0; k < d; k++) Ax[arow[j*d+k]] += aval[j*d+k] * x[j];
        double maxviol = 0.0, minx = 0.0;
        for (int i = 0; i < m; i++) { double e = fabs(Ax[i] - b[i]); if (e > maxviol) maxviol = e; }
        for (int j = 0; j < n; j++) if (x[j] < minx) minx = x[j];
        free(Ax);
        double bn = 1.0;
        for (int i = 0; i < m; i++) { double e = fabs(b[i]); if (e > bn) bn = e; }
        ok = maxviol < 1e-4 * bn && minx > -1e-6 &&
             fabs(obj - objstar) < 1e-4 * (1.0 + fabs(objstar)) &&
             fabs(obj - dobj) < 1e-4 * (1.0 + fabs(obj));
        free(x);
    }
    printf("lp_large  rows=%d cols=%d nnz=%d obj=%.6f obj*=%.6f dobj=%.6f t=%.4fs %s\n",
           m, n, nnz, obj, objstar, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(arow); free(aval); free(xs); free(ys); free(zs);
    free(b); free(c); free(rcnt); free(rstart); free(rsub); free(rval);
    return ok ? 0 : 1;
}
