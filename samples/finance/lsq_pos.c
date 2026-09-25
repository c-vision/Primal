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

/* lsq_pos.c - the exact `lsq_pos` form of the MosekRegression repository.
 *
 * Source: https://github.com/tschm/MosekRegression (the code the finance paper
 * arXiv:1310.3397 names), `src/mosek_tools/solver.py`, `lsq_pos`:
 *
 *   min ||X w - rhs||_2   (NOT squared: (v, X w - rhs) in Q^{N+1})
 *   s.t.  e'w = 1,   0 <= w <= 1.
 *
 * The upper bound w <= 1 is what the paper's Section 2.6 form and our
 * `regression_ls`/`portfolio_mgmt` do NOT have (they impose w >= 0 only): it is
 * redundant on the simplex e'w = 1, w >= 0, but the model is stated with it and
 * a faithful clone must carry it.
 *
 * The data is synthetic and deterministic (the sample's own LCG), so the
 * optimal value is not a number from the paper: the independent check is a
 * second solve of the QP-equivalent `min ||Xw-rhs||^2` (quadratic objective),
 * whose optimum must be `v*^2`. Both routes use the same public API.
 * Usage: lsq_pos [n] [m]   (default 40 5)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

static unsigned st = 13103397u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int N, M;
static double *X, *rhs;

/* min v s.t. (v, Xw-rhs) in Q, e'w = 1, 0 <= w <= 1 */
static double solve_socp(PRIMALenv_t env, double *wout) {
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    int W0 = 0, R0 = M, V = M + N;
    PRIMAL_appendvars(t, M + N + 1);
    PRIMAL_appendcons(t, N + 1);
    for (int j = 0; j < M; j++)
        PRIMAL_putvarbound(t, W0 + j, PRIMAL_BK_RA, 0.0, 1.0);
    for (int i = 0; i < N; i++)
        PRIMAL_putvarbound(t, R0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, V, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, V, 1.0);
    for (int i = 0; i < N; i++) {
        int sub[64]; double val[64]; int nz = 0;
        for (int j = 0; j < M; j++) {
            double xij = X[i * M + j];
            if (xij != 0.0) { sub[nz] = W0 + j; val[nz] = -xij; nz++; }
        }
        sub[nz] = R0 + i; val[nz] = 1.0; nz++;
        PRIMAL_putarow(t, i, nz, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -rhs[i], -rhs[i]);
    }
    for (int j = 0; j < M; j++) PRIMAL_putcj(t, W0 + j, 0.0);
    {   /* e'w = 1 */
        int sub[64]; double val[64];
        for (int j = 0; j < M; j++) { sub[j] = W0 + j; val[j] = 1.0; }
        PRIMAL_putarow(t, N, M, sub, val);
        PRIMAL_putconbound(t, N, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* (v, r) in Q */
        int mem[64];
        mem[0] = V;
        for (int i = 0; i < N; i++) mem[1 + i] = R0 + i;
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, N + 1, mem);
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double v = -1.0;
    if (rc == PRIMAL_RES_OK) {
        double *x = malloc((size_t)(M + N + 1) * sizeof(double));
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &v);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int j = 0; j < M; j++) wout[j] = x[j];
        free(x);
    }
    PRIMAL_deletetask(&t);
    return v;
}

/* min ||Xw-rhs||^2 = w'(X'X)w - 2 rhs'X w + const, as 1/2 w'Qw + c'w with
 * Q = 2 X'X (full symmetric, MOSEK's convention) and c = -2 X'rhs. */
static double solve_qp(PRIMALenv_t env, double *wout) {
    PRIMALtask_t t;
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, M);
    PRIMAL_appendcons(t, 1);
    for (int j = 0; j < M; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 1.0);
    for (int j = 0; j < M; j++) {
        double s = 0.0;
        for (int i = 0; i < N; i++) s += X[i * M + j] * rhs[i];
        PRIMAL_putcj(t, j, -2.0 * s);
    }
    for (int j = 0; j < M; j++)
        for (int k = j; k < M; k++) {
            double s = 0.0;
            for (int i = 0; i < N; i++) s += X[i * M + j] * X[i * M + k];
            PRIMAL_putqobj(t, 1, (int[]){j}, (int[]){k}, (double[]){2.0 * s});
        }
    {   /* e'w = 1 */
        int sub[64]; double val[64];
        for (int j = 0; j < M; j++) { sub[j] = j; val[j] = 1.0; }
        PRIMAL_putarow(t, 0, M, sub, val);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double q = 0.0;
    if (rc == PRIMAL_RES_OK) {
        double *x = malloc((size_t)M * sizeof(double));
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &q);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int j = 0; j < M; j++) wout[j] = x[j];
        free(x);
    }
    PRIMAL_deletetask(&t);
    return q;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 40;
    M = argc > 2 ? atoi(argv[2]) : 5;
    if (N < 1 || M < 1 || M > 60) return 2;
    X = malloc((size_t)N * M * sizeof(double));
    rhs = malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N * M; k++) X[k] = (rnd() - 0.5) * 2.0;
    for (int i = 0; i < N; i++) rhs[i] = (rnd() - 0.5) * 2.0;

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    double ws[64], wq[64];
    double v = solve_socp(env, ws);
    double q = solve_qp(env, wq);

    double budget = 0.0, lo = 0.0, hi = 0.0;
    for (int j = 0; j < M; j++) {
        budget += ws[j];
        if (ws[j] < lo) lo = ws[j];
        if (ws[j] > hi) hi = ws[j];
    }
    double rhs2 = 0.0;
    for (int i = 0; i < N; i++) rhs2 += rhs[i] * rhs[i];
    /* v*^2 == pobj_qp + ||rhs||^2 */
    double v2_from_qp = q + rhs2;
    int ok = v >= 0.0 && fabs(budget - 1.0) <= 1e-6 && lo >= -1e-6 && hi <= 1.0 + 1e-6 &&
             fabs(v * v - v2_from_qp) <= 1e-5 * (1.0 + fabs(v2_from_qp));

    printf("lsq_pos  n=%d m=%d  v*=%.8f  sqrt(QP)=%.8f  e'w=%.2e  w in [%.3f,%.3f] %s\n",
           N, M, v, sqrt(fabs(v2_from_qp)), budget, lo, hi, ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    free(X); free(rhs);
    return ok ? 0 : 1;
}
