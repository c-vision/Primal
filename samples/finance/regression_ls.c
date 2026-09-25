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

/* regression_ls.c - least-squares regression as a second-order cone program.
 * (Section 2 of Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for
 *  Portfolio Optimisation using MOSEK", arXiv:1310.3397 [q-fin.PM],
 *  https://arxiv.org/abs/1310.3397 , DOI: 10.48550/arXiv.1310.3397 ,
 *  code: https://github.com/tschm/MosekRegression .)
 *
 *   min ||X w - y||_2   <=>   min v   s.t. (v, Xw - y) in Q^{n+1}
 * (rotated form: (1/2, v, Xw - y) in Qr^{n+2} for the squared residual).
 *
 * Two variants: unconstrained (verified against the normal equations
 * X'X w = X'y solved with dense LU) and the constrained portfolio form
 * sum(w)=1, w>=0 (Section 2.6, "Constrained regression").
 * Usage: regression_ls [n] [m]   (default 60 6)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"
#include "linalg.h"

static unsigned st = 13103397u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int N, M;
static double *X, *y;

/* build: vars w (M), r (N), v (1); rows r_i - (Xw)_i = -y_i; cone (v, r) */
static PRIMALtask_t build(int constrained, PRIMALenv_t env) {
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    int W0 = 0, R0 = M, V = M + N;
    PRIMAL_appendvars(t, M + N + 1);
    PRIMAL_appendcons(t, N);
    for (int j = 0; j < M; j++)
        PRIMAL_putvarbound(t, W0 + j, constrained ? PRIMAL_BK_LO : PRIMAL_BK_FR,
                          constrained ? 0.0 : -INFINITY, INFINITY);
    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, R0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, V, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, V, 1.0);
    for (int i = 0; i < N; i++) {
        int *sub = malloc((size_t)(M + 1) * sizeof(int));
        double *val = malloc((size_t)(M + 1) * sizeof(double));
        int nz = 0;
        for (int j = 0; j < M; j++) {
            double xij = X[i * M + j];
            if (xij != 0.0) { sub[nz] = W0 + j; val[nz] = -xij; nz++; }
        }
        sub[nz] = R0 + i; val[nz] = 1.0; nz++;
        PRIMAL_putarow(t, i, nz, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -y[i], -y[i]);
        free(sub); free(val);
    }
    {
        int *mem = malloc((size_t)(N + 1) * sizeof(int));
        mem[0] = V;
        for (int i = 0; i < N; i++) mem[1 + i] = R0 + i;
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, N + 1, mem);
        free(mem);
    }
    if (constrained) {
        int *sub = malloc((size_t)M * sizeof(int));
        double *val = malloc((size_t)M * sizeof(double));
        for (int j = 0; j < M; j++) { sub[j] = W0 + j; val[j] = 1.0; }
        PRIMAL_appendcons(t, 1);
        PRIMAL_putarow(t, N, M, sub, val);
        PRIMAL_putconbound(t, N, PRIMAL_BK_FX, 1.0, 1.0);
        free(sub); free(val);
    }
    return t;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 60;
    M = argc > 2 ? atoi(argv[2]) : 6;
    if (N < 1 || M < 1) return 2;
    X = malloc((size_t)N * M * sizeof(double));
    y = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N * M; k++) X[k] = (rnd() - 0.5) * 2.0;
    for (int i = 0; i < N; i++) y[i] = (rnd() - 0.5) * 2.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    struct timeval a, b;
    int ok = 1;

    /* unconstrained LS */
    PRIMALtask_t t = build(0, env);
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b, NULL);
    double sec1 = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
    double v1 = 0.0, w1[16];
    if (rc == PRIMAL_RES_OK) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &v1);
        double *x = malloc((size_t)(M + N + 1) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int j = 0; j < M; j++) w1[j] = x[j];
        free(x);
    } else ok = 0;
    PRIMAL_deletetask(&t);
    /* normal-equations reference */
    DMat *G = dmat_new(M, M);
    double *rhs = calloc((size_t)M, sizeof(double));
    for (int j = 0; j < M; j++)
        for (int k = 0; k < M; k++) {
            double s = 0.0;
            for (int i = 0; i < N; i++) s += X[i * M + j] * X[i * M + k];
            G->v[j][k] = s;
        }
    for (int j = 0; j < M; j++) {
        double s = 0.0;
        for (int i = 0; i < N; i++) s += X[i * M + j] * y[i];
        rhs[j] = s;
    }
    double wref[16];
    int have_ref = (dmat_solve_lu(G, rhs, M) == 0);
    if (have_ref) for (int j = 0; j < M; j++) wref[j] = rhs[j];
    if (have_ref) {
        double wdiff = 0.0;
        for (int j = 0; j < M; j++) if (fabs(w1[j] - wref[j]) > wdiff) wdiff = fabs(w1[j] - wref[j]);
        if (wdiff > 1e-6) ok = 0;
    }
    dmat_free(G); free(rhs);

    /* constrained (portfolio) LS */
    PRIMALtask_t t2 = build(1, env);
    gettimeofday(&a, NULL);
    PRIMALrescodee rc2 = PRIMAL_optimize(t2);
    gettimeofday(&b, NULL);
    double sec2 = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
    double v2 = 0.0, budget = 0.0, minw = 0.0;
    if (rc2 == PRIMAL_RES_OK) {
        PRIMAL_getprimalobj(t2, PRIMAL_SOL_ITR, &v2);
        double *x = malloc((size_t)(M + N + 1) * sizeof(double));
        PRIMAL_getxx(t2, PRIMAL_SOL_ITR, x);
        for (int j = 0; j < M; j++) { budget += x[j]; if (x[j] < minw) minw = x[j]; }
        free(x);
    } else ok = 0;
    PRIMAL_deletetask(&t2);
    if (ok && (fabs(budget - 1.0) > 1e-6 || minw < -1e-6)) ok = 0;
    if (ok && v2 < v1 - 1e-6) ok = 0;   /* constraints cannot lower the residual */

    printf("regr_ls    n=%d m=%d uncons=%.6f cons=%.6f t=%.4f/%.4fs %s\n",
           N, M, v1, v2, sec1, sec2, ok ? "OK" : "FAIL");

    PRIMAL_deleteenv(&env);
    free(X); free(y);
    return ok ? 0 : 1;
}
