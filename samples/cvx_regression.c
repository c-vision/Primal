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

/* cvx_regression.c - L1 (least absolute deviations) regression as an LP
 * (scaling test: many rows).
 *
 *   min ||A x - b||_1   <=>   min sum_i t_i
 *     s.t.  (A x - b)_i <= t_i,  -(A x - b)_i <= t_i,  t_i >= 0
 *
 * Verified: feasibility and strong duality pobj ~ dobj.
 * Usage: cvx_regression [m] [d]   (default 400 30)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 14142135u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int m = argc > 1 ? atoi(argv[1]) : 400;
    int d = argc > 2 ? atoi(argv[2]) : 30;
    if (m < 1 || d < 1) return 2;
    double *A = malloc((size_t)m * d * sizeof(double));
    double *b = malloc((size_t)m * sizeof(double));
    for (int k = 0; k < m * d; k++) A[k] = rnd() * 2.0 - 1.0;
    for (int i = 0; i < m; i++) b[i] = rnd() * 2.0 - 1.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (d) free, t (m) >= 0 */
    PRIMAL_appendvars(t, d + m);
    PRIMAL_appendcons(t, 2 * m);
    for (int j = 0; j < d; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < m; i++) {
        PRIMAL_putvarbound(t, d + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, d + i, 1.0);
    }
    for (int i = 0; i < m; i++) {
        /* row i: A_i x - t_i <= b_i */
        int *sub = malloc((size_t)(d + 1) * sizeof(int));
        double *val = malloc((size_t)(d + 1) * sizeof(double));
        for (int j = 0; j < d; j++) { sub[j] = j; val[j] = A[i * d + j]; }
        sub[d] = d + i; val[d] = -1.0;
        PRIMAL_putarow(t, i, d + 1, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_UP, -INFINITY, b[i]);
        /* row m+i: -A_i x - t_i <= -b_i */
        for (int j = 0; j < d; j++) { sub[j] = j; val[j] = -A[i * d + j]; }
        sub[d] = d + i; val[d] = -1.0;
        PRIMAL_putarow(t, m + i, d + 1, sub, val);
        PRIMAL_putconbound(t, m + i, PRIMAL_BK_UP, -INFINITY, -b[i]);
        free(sub); free(val);
    }

    struct timeval a, b2;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b2, NULL);
    double sec = (b2.tv_sec - a.tv_sec) + 1e-6 * (b2.tv_usec - a.tv_usec);

    double obj = 0.0, dobj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj);
        double *x = malloc((size_t)(d + m) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double maxviol = 0.0, mint = 0.0;
        for (int i = 0; i < m; i++) {
            double r = -b[i];
            for (int j = 0; j < d; j++) r += A[i * d + j] * x[j];
            if (fabs(r) - x[d + i] > maxviol) maxviol = fabs(r) - x[d + i];
            if (x[d + i] < mint) mint = x[d + i];
        }
        ok = maxviol < 1e-5 && mint > -1e-6 &&
             fabs(obj - dobj) < 1e-5 * (1.0 + fabs(obj));
        free(x);
    }
    printf("cvx_reg    m=%d d=%d vars=%d cons=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           m, d, d + m, 2 * m, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(A); free(b);
    return ok ? 0 : 1;
}
