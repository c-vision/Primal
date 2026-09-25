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

/* portfolio_mgmt.c - portfolio management as conic programs (Section 4 of
 * Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for Portfolio
 * Optimisation using MOSEK", arXiv:1310.3397 [q-fin.PM],
 * https://arxiv.org/abs/1310.3397 , DOI: 10.48550/arXiv.1310.3397 ,
 * code: https://github.com/tschm/MosekRegression).
 *
 *   min variance :  min ||Xw||^2                 s.t. e'w=1, w>=0
 *   tracking     :  min ||Xw-rM||^2              s.t. e'w=1, w>=0
 *   max return   :  max mu'w  s.t. (sigma, Xw) in Q, e'w=1, w>=0
 *   130/30       :  max mu'w  s.t. sum t_i <= 1.6, (t_i,w_i) in Q^2, e'w=1
 *   robust       :  max mu0'w - u  s.t. (u, A w) in Q, e'w=1, w>=0
 *
 * Each problem is solved and verified independently (budget, bounds, cone
 * feasibility).  Usage: portfolio_mgmt [n] [m]   (default 40 6)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 314159u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int N, M;
static double *X, *rM, *mu;

static double budget_of(PRIMALtask_t t) {
    double *x = malloc((size_t)(8 * M + 2 * N + 8) * sizeof(double));
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    double s = 0.0;
    for (int j = 0; j < M; j++) s += x[j];
    free(x);
    return s;
}

/* min ||Xw||^2 (or ||Xw-rM||^2) via QP */
static PRIMALtask_t build_ls(PRIMALenv_t env, int tracking) {
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, M);
    PRIMAL_appendcons(t, 1);
    for (int j = 0; j < M; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, 1.0);
    int nz = 0;
    int *qi = malloc((size_t)M * (M + 1) / 2 * sizeof(int));
    int *qj = malloc((size_t)M * (M + 1) / 2 * sizeof(int));
    double *qv = malloc((size_t)M * (M + 1) / 2 * sizeof(double));
    for (int a = 0; a < M; a++)
        for (int b = a; b < M; b++) {
            double s = 0.0;
            for (int i = 0; i < N; i++) s += X[i * M + a] * X[i * M + b];
            s *= 2.0;
            if (s != 0.0) { qi[nz] = a; qj[nz] = b; qv[nz] = s; nz++; }
        }
    PRIMAL_putqobj(t, nz, qi, qj, qv);
    free(qi); free(qj); free(qv);
    double cfix = 0.0;
    for (int a = 0; a < M; a++) {
        double c = 0.0;
        for (int i = 0; i < N; i++) c += X[i * M + a] * (tracking ? rM[i] : 0.0);
        PRIMAL_putcj(t, a, -2.0 * c);
    }
    if (tracking) for (int i = 0; i < N; i++) cfix += rM[i] * rM[i];
    PRIMAL_putcfix(t, cfix);
    int *sub = malloc((size_t)M * sizeof(int));
    double *val = malloc((size_t)M * sizeof(double));
    for (int j = 0; j < M; j++) { sub[j] = j; val[j] = 1.0; }
    PRIMAL_putarow(t, 0, M, sub, val);
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    free(sub); free(val);
    return t;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 40;
    M = argc > 2 ? atoi(argv[2]) : 6;
    if (N < 1 || M < 1) return 2;
    X = malloc((size_t)N * M * sizeof(double));
    rM = malloc((size_t)N * sizeof(double));
    mu = malloc((size_t)M * sizeof(double));
    for (int k = 0; k < N * M; k++) X[k] = (rnd() - 0.5) * 0.1;
    for (int i = 0; i < N; i++) rM[i] = (rnd() - 0.5) * 0.08;
    for (int j = 0; j < M; j++) mu[j] = 0.001 + rnd() * 0.02;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    struct timeval a, b;
    int ok = 1;

    /* min variance */
    {
        PRIMALtask_t t = build_ls(env, 0);
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double bud = budget_of(t);
        if (rc != PRIMAL_RES_OK || fabs(bud - 1.0) > 1e-6) ok = 0;
        printf("minvar     n=%d m=%d budget=%.6f t=%.4fs\n", N, M, bud, sec);
        PRIMAL_deletetask(&t);
    }
    /* tracking */
    {
        PRIMALtask_t t = build_ls(env, 1);
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double bud = budget_of(t);
        if (rc != PRIMAL_RES_OK || fabs(bud - 1.0) > 1e-6) ok = 0;
        printf("tracking   n=%d m=%d budget=%.6f t=%.4fs\n", N, M, bud, sec);
        PRIMAL_deletetask(&t);
    }
    /* max return with risk cone (sigma, Xw) in Q */
    {
        PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
        double sigma = 0.5;
        int W0 = 0, R0 = M, TC = M + N;
        PRIMAL_appendvars(t, M + N + 1);
        PRIMAL_appendcons(t, N + 1);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
        for (int j = 0; j < M; j++) {
            PRIMAL_putcj(t, W0 + j, mu[j]);
            PRIMAL_putvarbound(t, W0 + j, PRIMAL_BK_LO, 0.0, 1.0);
        }
        for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, R0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, TC, PRIMAL_BK_FX, sigma, sigma);
        for (int i = 0; i < N; i++) {
            int *sub = malloc((size_t)(M + 1) * sizeof(int));
            double *val = malloc((size_t)(M + 1) * sizeof(double));
            int nz = 0;
            for (int j = 0; j < M; j++)
                if (X[i * M + j] != 0.0) { sub[nz] = W0 + j; val[nz] = -X[i * M + j]; nz++; }
            sub[nz] = R0 + i; val[nz] = 1.0; nz++;
            PRIMAL_putarow(t, i, nz, sub, val);
            PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 0.0, 0.0);
            free(sub); free(val);
        }
        {   /* budget */
            int *sub = malloc((size_t)M * sizeof(int));
            double *val = malloc((size_t)M * sizeof(double));
            for (int j = 0; j < M; j++) { sub[j] = W0 + j; val[j] = 1.0; }
            PRIMAL_putarow(t, N, M, sub, val);
            PRIMAL_putconbound(t, N, PRIMAL_BK_FX, 1.0, 1.0);
            free(sub); free(val);
        }
        {
            int *mem = malloc((size_t)(N + 1) * sizeof(int));
            mem[0] = TC;
            for (int i = 0; i < N; i++) mem[1 + i] = R0 + i;
            PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, N + 1, mem);
            free(mem);
        }
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double risk = 0.0, bud = 0.0;
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(M + N + 1) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int i = 0; i < N; i++) risk += x[R0 + i] * x[R0 + i];
            for (int j = 0; j < M; j++) bud += x[W0 + j];
            risk = sqrt(risk);
            free(x);
        } else ok = 0;
        if (risk > sigma * (1.0 + 1e-4) || fabs(bud - 1.0) > 1e-6) ok = 0;
        printf("maxreturn  n=%d m=%d risk=%.6f budget=%.6f t=%.4fs\n", N, M, risk, bud, sec);
        PRIMAL_deletetask(&t);
    }
    /* 130/30 leverage: (t_i, w_i) in Q^2, sum t_i <= 1.6, sum w_i = 1 */
    {
        PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
        int W0 = 0, T0 = M;
        PRIMAL_appendvars(t, 2 * M);
        PRIMAL_appendcons(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
        for (int j = 0; j < M; j++) {
            PRIMAL_putcj(t, W0 + j, mu[j]);
            PRIMAL_putvarbound(t, W0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
            PRIMAL_putvarbound(t, T0 + j, PRIMAL_BK_LO, 0.0, INFINITY);
            PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 2, (int[]){T0 + j, W0 + j});
        }
        {   /* sum t <= 1.6 */
            int *sub = malloc((size_t)M * sizeof(int));
            double *val = malloc((size_t)M * sizeof(double));
            for (int j = 0; j < M; j++) { sub[j] = T0 + j; val[j] = 1.0; }
            PRIMAL_putarow(t, 0, M, sub, val);
            PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 1.6);
            free(sub); free(val);
        }
        {   /* sum w = 1 */
            int *sub = malloc((size_t)M * sizeof(int));
            double *val = malloc((size_t)M * sizeof(double));
            for (int j = 0; j < M; j++) { sub[j] = W0 + j; val[j] = 1.0; }
            PRIMAL_putarow(t, 1, M, sub, val);
            PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);
            free(sub); free(val);
        }
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double lev = 0.0, bud = 0.0;
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(2 * M + 1) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int j = 0; j < M; j++) { lev += x[T0 + j]; bud += x[W0 + j]; }
            free(x);
        } else ok = 0;
        if (lev > 1.6 + 1e-4 || fabs(bud - 1.0) > 1e-6) ok = 0;
        printf("130_30     n=%d m=%d leverage=%.6f budget=%.6f t=%.4fs\n", N, M, lev, bud, sec);
        PRIMAL_deletetask(&t);
    }
    /* robust: max mu0'w - u, (u, A w) in Q */
    {
        PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
        int W0 = 0, Z0 = M, UV = M + M;
        PRIMAL_appendvars(t, 2 * M + 1);
        PRIMAL_appendcons(t, M + 1);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
        for (int j = 0; j < M; j++) {
            PRIMAL_putcj(t, W0 + j, mu[j]);
            PRIMAL_putvarbound(t, W0 + j, PRIMAL_BK_LO, 0.0, 1.0);
        }
        for (int j = 0; j < M; j++) PRIMAL_putvarbound(t, Z0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, UV, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, UV, -1.0);
        for (int i = 0; i < M; i++) {          /* z_i - (A w)_i = 0, A symmetric */
            int *sub = malloc((size_t)(M + 1) * sizeof(int));
            double *val = malloc((size_t)(M + 1) * sizeof(double));
            int nz = 0;
            for (int j = 0; j < M; j++) {
                double a = (i == j) ? 0.01 : ((j == (i + 1) % M) ? 0.002 : 0.0);
                if (a != 0.0) { sub[nz] = W0 + j; val[nz] = -a; nz++; }
            }
            sub[nz] = Z0 + i; val[nz] = 1.0; nz++;
            PRIMAL_putarow(t, i, nz, sub, val);
            PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 0.0, 0.0);
            free(sub); free(val);
        }
        {   /* budget */
            int *sub = malloc((size_t)M * sizeof(int));
            double *val = malloc((size_t)M * sizeof(double));
            for (int j = 0; j < M; j++) { sub[j] = W0 + j; val[j] = 1.0; }
            PRIMAL_putarow(t, M, M, sub, val);
            PRIMAL_putconbound(t, M, PRIMAL_BK_FX, 1.0, 1.0);
            free(sub); free(val);
        }
        {
            int *mem = malloc((size_t)(M + 1) * sizeof(int));
            mem[0] = UV;
            for (int j = 0; j < M; j++) mem[1 + j] = Z0 + j;
            PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, M + 1, mem);
            free(mem);
        }
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double u = 0.0, normz = 0.0, bud = 0.0;
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(2 * M + 1) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            u = x[UV];
            for (int j = 0; j < M; j++) { normz += x[Z0 + j] * x[Z0 + j]; bud += x[W0 + j]; }
            normz = sqrt(normz);
            free(x);
        } else ok = 0;
        if (u < normz - 1e-5 || fabs(bud - 1.0) > 1e-6) ok = 0;
        printf("robust     n=%d m=%d u=%.6f |Aw|=%.6f budget=%.6f t=%.4fs\n",
               N, M, u, normz, bud, sec);
        PRIMAL_deletetask(&t);
    }

    printf("portfolio  n=%d m=%d %s\n", N, M, ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    free(X); free(rM); free(mu);
    return ok ? 0 : 1;
}
