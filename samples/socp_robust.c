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

/* socp_robust.c - least-squares regression as a second-order cone program
 * (scaling test for the conic path).
 *
 *   min ||A x - b||_2   <=>   min t  s.t. (t, r) in QUAD, r = A x - b
 *
 * Optimum t* = ||A x* - b|| with x* solving the normal equations A'A x = A'b,
 * computed here independently (dense LU) as a reference.
 * Usage: socp_robust [m] [d]   (default 300 40)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"
#include "linalg.h"

static unsigned st = 16180339u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int m = argc > 1 ? atoi(argv[1]) : 300;
    int d = argc > 2 ? atoi(argv[2]) : 40;
    if (m < 1 || d < 1) return 2;
    double *A = malloc((size_t)m * d * sizeof(double));
    double *b = malloc((size_t)m * sizeof(double));
    for (int k = 0; k < m * d; k++) A[k] = rnd() * 2.0 - 1.0;
    for (int i = 0; i < m; i++) b[i] = rnd() * 2.0 - 1.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (d), r (m), t (1) */
    int X0 = 0, R0 = d, T = d + m;
    PRIMAL_appendvars(t, d + m + 1);
    for (int j = 0; j < d; j++) PRIMAL_putvarbound(t, X0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < m; i++) PRIMAL_putvarbound(t, R0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, T, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, T, 1.0);
    PRIMAL_appendcons(t, m);
    for (int i = 0; i < m; i++) {
        int *sub = malloc((size_t)(d + 1) * sizeof(int));
        double *val = malloc((size_t)(d + 1) * sizeof(double));
        for (int j = 0; j < d; j++) { sub[j] = X0 + j; val[j] = -A[i * d + j]; }
        sub[d] = R0 + i; val[d] = 1.0;
        PRIMAL_putarow(t, i, d + 1, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -b[i], -b[i]);
        free(sub); free(val);
    }
    {
        int *mem = malloc((size_t)(m + 1) * sizeof(int));
        mem[0] = T;
        for (int i = 0; i < m; i++) mem[1 + i] = R0 + i;
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, m + 1, mem);
        free(mem);
    }

    struct timeval a, b2;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b2, NULL);
    double sec = (b2.tv_sec - a.tv_sec) + 1e-6 * (b2.tv_usec - a.tv_usec);

    double obj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        /* reference: solve A'A x = A'b with dense LU */
        DMat *M = dmat_new(d, d);
        double *rhs = calloc((size_t)d, sizeof(double));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                double s = 0.0;
                for (int k = 0; k < m; k++) s += A[k * d + i] * A[k * d + j];
                M->v[i][j] = s;
            }
        for (int i = 0; i < d; i++) {
            double s = 0.0;
            for (int k = 0; k < m; k++) s += A[k * d + i] * b[k];
            rhs[i] = s;
        }
        double tref = 0.0;
        if (dmat_solve_lu(M, rhs, d) == 0) {
            double ss = 0.0;
            for (int k = 0; k < m; k++) {
                double r = -b[k];
                for (int j = 0; j < d; j++) r += A[k * d + j] * rhs[j];
                ss += r * r;
            }
            tref = sqrt(ss);
        }
        ok = fabs(obj - tref) < 1e-4 * (1.0 + fabs(tref));
        printf("socp_ls    m=%d d=%d cone=%d obj=%.6f ref=%.6f t=%.4fs %s\n",
               m, d, m + 1, obj, tref, sec, ok ? "OK" : "FAIL");
        dmat_free(M); free(rhs);
    } else {
        printf("socp_ls    m=%d d=%d rc=%d FAIL\n", m, d, (int)rc);
    }

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(A); free(b);
    return ok ? 0 : 1;
}
