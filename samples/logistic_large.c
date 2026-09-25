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

/* logistic_large.c - sum-of-exponentials regression via exponential cones
 * (scaling test for the conic path with many PEXP cones).
 *
 *   min sum_i v_i + 0.5*gamma*||x||^2
 *   s.t. v_i >= exp(a_i'x + c_i)      (PEXP cone per i)
 *
 * The quadratic term is absorbed exactly into a RQUAD cone (as in the
 * logistic benchmark), keeping the whole model in the conic path.
 * Verified against gradient descent on f(x) = sum_i exp(a_i'x+c_i)
 * + 0.5*gamma*||x||^2.
 * Usage: logistic_large [n] [d]   (default 40 8)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 11235813u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int N, D;
static double *A, *c;
static const double GAMMA = 0.1;

static double fval(const double *x) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double u = c[i];
        for (int j = 0; j < D; j++) u += A[i * D + j] * x[j];
        s += exp(u);
    }
    for (int j = 0; j < D; j++) s += 0.5 * GAMMA * x[j] * x[j];
    return s;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 40;
    D = argc > 2 ? atoi(argv[2]) : 8;
    if (N < 1 || D < 1) return 2;
    A = malloc((size_t)N * D * sizeof(double));
    c = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N * D; k++) A[k] = (rnd() * 2.0 - 1.0) * 0.5;
    for (int i = 0; i < N; i++) c[i] = rnd() * 2.0 - 1.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (D), v (N), u (N), one, w  */
    int X0 = 0, V0 = D, U0 = D + N, ONE = D + 2 * N, W = ONE + 1;
    PRIMAL_appendvars(t, D + 2 * N + 2);
    for (int j = 0; j < D; j++) PRIMAL_putvarbound(t, X0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < N; i++) {
        PRIMAL_putvarbound(t, V0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, V0 + i, 1.0);
        PRIMAL_putvarbound(t, U0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    }
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putvarbound(t, W, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, W, GAMMA);
    PRIMAL_appendcons(t, N);
    for (int i = 0; i < N; i++) {
        int *sub = malloc((size_t)(D + 1) * sizeof(int));
        double *val = malloc((size_t)(D + 1) * sizeof(double));
        for (int j = 0; j < D; j++) { sub[j] = X0 + j; val[j] = A[i * D + j]; }
        sub[D] = U0 + i; val[D] = -1.0;
        PRIMAL_putarow(t, i, D + 1, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -c[i], -c[i]);
        free(sub); free(val);
    }
    for (int i = 0; i < N; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){V0 + i, ONE, U0 + i});
    {   /* RQUAD (W, ONE, x_0..x_{D-1}): W >= 0.5*||x||^2 */
        int *mem = malloc((size_t)(D + 2) * sizeof(int));
        mem[0] = W; mem[1] = ONE;
        for (int j = 0; j < D; j++) mem[2 + j] = X0 + j;
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, D + 2, mem);
        free(mem);
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
        /* gradient descent reference */
        double *xg = calloc((size_t)D, sizeof(double));
        double fg = fval(xg), step = 0.5;
        for (int it = 0; it < 20000; it++) {
            double *g = calloc((size_t)D, sizeof(double));
            for (int j = 0; j < D; j++) {
                double h = 1e-6;
                double *xp = malloc((size_t)D * sizeof(double));
                double *xm = malloc((size_t)D * sizeof(double));
                for (int l = 0; l < D; l++) { xp[l] = xg[l]; xm[l] = xg[l]; }
                xp[j] += h; xm[j] -= h;
                g[j] = (fval(xp) - fval(xm)) / (2 * h);
                free(xp); free(xm);
            }
            double *xn = malloc((size_t)D * sizeof(double));
            for (int j = 0; j < D; j++) xn[j] = xg[j] - step * g[j];
            double fn = fval(xn);
            if (fn < fg) { for (int j = 0; j < D; j++) xg[j] = xn[j]; fg = fn; step *= 1.05; }
            else if (fn < fg + 1e-12) step *= 0.7;
            else step *= 0.5;
            free(xn); free(g);
            if (step < 1e-12) break;
        }
        ok = fabs(obj - fg) < 1e-3 * (1.0 + fabs(fg));
        printf("logistic   n=%d d=%d cones=%d obj=%.6f grad=%.6f t=%.4fs %s\n",
               N, D, N, obj, fg, sec, ok ? "OK" : "FAIL");
        free(xg);
    } else {
        printf("logistic   n=%d d=%d rc=%d FAIL\n", N, D, (int)rc);
    }

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(A); free(c);
    return ok ? 0 : 1;
}
