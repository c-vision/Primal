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

/* regression_regularized.c - regularised regression (Section 3 of
 * Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for Portfolio
 * Optimisation using MOSEK", arXiv:1310.3397).
 *
 * =====================================================================
 * PAPER: "Regression techniques for Portfolio Optimisation using MOSEK"
 *   Thomas Schmelzer, Raphael Hauser, Erling D. Andersen, Joachim Dahl
 *   arXiv:1310.3397 [q-fin.PM], 12 Oct 2013
 *   https://arxiv.org/abs/1310.3397   DOI: 10.48550/arXiv.1310.3397
 *   Code/data: https://github.com/tschm/MosekRegression
 *
 * CORE IDEA. Regression is reformulated as a second-order cone program,
 * which adds the flexibility (bounds, constraints, penalties) that the
 * normal equations cannot express.
 *
 *   Quadratic cone:         Q^n  = { x : x_1 >= sqrt(x_2^2 + ... + x_n^2) }
 *   Rotated quadratic cone: Qr^n = { x : 2 x_1 x_2 >= x_3^2 + ... + x_n^2,
 *                                    x_1, x_2 >= 0 }
 *
 * Key equivalences (Section 2):
 *   |x| <= t                 <=>  (t, x) in Q^2
 *   ||A x - b||_2 <= t       <=>  (t, A x - b) in Q^{n+1}
 *   ||A x - b||_2^2 <= t     <=>  (1/2, t, A x - b) in Qr^{n+2}
 *
 * Least squares (Sec. 2.4):
 *   min ||X w - y||_2   <=>  min v  s.t. (v, X w - y) in Q^{n+1}
 *   min ||X w - y||_2^2 <=>  min v  s.t. (1/2, v, X w - y) in Qr^{n+2}
 *
 * Regularisation (Sec. 3) - in portfolio terms a trading-cost model on the
 * trade Dw = w - w0:
 *   ridge  (L2 / Tikhonov): min ||Xw-y||^2 + lambda ||G(w-w0)||^2
 *           -> add (1/2, u, G(w-w0)) in Qr
 *   sparse (L1 / LASSO):    min ||Xw-y||^2 + lambda sum_i |[G(w-w0)]_i|
 *           -> add (t_i, [G(w-w0)]_i) in Q^2, one per coordinate
 *   3/2:                    min ||Xw-y||^2 + lambda sum_i |[G(w-w0)]_i|^{3/2}
 *           -> Qr for the residual, Q^2 for the absolute value and rotated
 *              cones for the power term: 3m+1 cones in total
 *   (ridge + sparse = elastic net).
 *
 * Portfolio management (Sec. 4), with X = n historic return vectors and
 * Xw the portfolio return time series:
 *   tracking error:  min ||Xw - rM||^2  s.t. sum w = 1, w >= 0
 *   min variance:    same with rM = 0
 *   max return:      max mu'w  s.t. ||Xw||^2 <= sigma_max^2, sum w = 1, w >= 0
 *   leverage:        sum |w_i| via (t_i, w_i) in Q^2; 130/30 => sum t_i <= 1.6,
 *                    market neutral => sum w = 0 and sum t_i = 2
 *   robust:          max w'mu0 - ||A w||_2  s.t. ||Xw||^2 <= sigma_max^2,
 *                    sum w = 1, w >= 0, for an ellipsoidal uncertainty set
 *                    E = { mu = A u + mu0 : ||u||_2 <= 1 }
 *
 * Return prediction (Sec. 5): regress r_n on its own lagged history.
 * =====================================================================
 *
 *   ridge :  min ||Xw-y||^2 + lambda ||w-w0||^2
 *   sparse:  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|        (LASSO)
 *   3/2   :  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|^{3/2}
 *
 * The squared residual is passed as a quadratic objective (identical to the
 * paper's rotated-cone form ||Xw-y||^2, but numerically more robust in this
 * dense engine); the penalty terms use the conic forms from the paper:
 *   |p_i|       <=> (t_i, p_i) in Q^2
 *   |p_i|^{3/2} <=> (t_i, 1, p_i) in POW^{2/3,1/3}
 * Ridge is verified against the closed form (X'X + lambda I)w = X'y + lambda w0;
 * sparse/3-2 against their cone feasibility.
 * Usage: regression_regularized [n] [m] [lambda]   (default 40 5 0.5)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"
#include "linalg.h"

static unsigned st = 27182818u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

static int N, M;
static double LAM;
static double *X, *y, *w0;

enum { REG_RIDGE = 0, REG_SPARSE = 1, REG_POW32 = 2 };

static void put_qobj_residual(PRIMALtask_t t) {
    /* Q = 2 X'X (upper triangle), c = -2 X'y, cfix = y'y */
    int *qi = malloc((size_t)M * (M + 1) / 2 * sizeof(int));
    int *qj = malloc((size_t)M * (M + 1) / 2 * sizeof(int));
    double *qv = malloc((size_t)M * (M + 1) / 2 * sizeof(double));
    int nz = 0;
    for (int a = 0; a < M; a++)
        for (int b = a; b < M; b++) {
            double s = 0.0;
            for (int i = 0; i < N; i++) s += X[i * M + a] * X[i * M + b];
            s *= 2.0;
            if (s != 0.0) { qi[nz] = a; qj[nz] = b; qv[nz] = s; nz++; }
        }
    PRIMAL_putqobj(t, nz, qi, qj, qv);
    free(qi); free(qj); free(qv);
    for (int a = 0; a < M; a++) {
        double s = 0.0;
        for (int i = 0; i < N; i++) s += X[i * M + a] * y[i];
        PRIMAL_putcj(t, a, -2.0 * s);
    }
    double yy = 0.0;
    for (int i = 0; i < N; i++) yy += y[i] * y[i];
    PRIMAL_putcfix(t, yy);
}

static PRIMALtask_t build(int kind, PRIMALenv_t env) {
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    if (kind == REG_RIDGE) {
        /* pure QP: Q = 2(X'X + lam I), c = -2(X'y + lam w0), cfix */
        PRIMAL_appendvars(t, M);
        for (int j = 0; j < M; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
        put_qobj_residual(t);
        int *qi = malloc((size_t)M * sizeof(int));
        int *qj = malloc((size_t)M * sizeof(int));
        double *qv = malloc((size_t)M * sizeof(double));
        for (int j = 0; j < M; j++) { qi[j] = j; qj[j] = j; qv[j] = 2.0 * LAM; }
        PRIMAL_putqobj(t, M, qi, qj, qv);   /* accumulate */
        free(qi); free(qj); free(qv);
        for (int j = 0; j < M; j++) {
            double cj; PRIMAL_getcj(t, j, &cj);
            PRIMAL_putcj(t, j, cj - 2.0 * LAM * w0[j]);
        }
        double cf; PRIMAL_getcfix(t, &cf);
        double w02 = 0.0;
        for (int j = 0; j < M; j++) w02 += w0[j] * w0[j];
        PRIMAL_putcfix(t, cf + LAM * w02);
        return t;
    }
    /* sparse / 3-2: QP residual + conic penalty on p = w - w0 */
    int W0 = 0, P0 = M, T0 = 2 * M, ONE = 3 * M;
    int nv = (kind == REG_SPARSE) ? 3 * M : 3 * M + 1;
    PRIMAL_appendvars(t, nv);
    PRIMAL_appendcons(t, M);
    for (int j = 0; j < M; j++) PRIMAL_putvarbound(t, W0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 0; j < M; j++) PRIMAL_putvarbound(t, P0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 0; j < M; j++) {
        PRIMAL_putvarbound(t, T0 + j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, T0 + j, LAM);
    }
    put_qobj_residual(t);
    for (int j = 0; j < M; j++) {          /* p_j - w_j = -w0_j */
        int sub[2] = {W0 + j, P0 + j};
        double val[2] = {-1.0, 1.0};
        PRIMAL_putarow(t, j, 2, sub, val);
        PRIMAL_putconbound(t, j, PRIMAL_BK_FX, -w0[j], -w0[j]);
    }
    if (kind == REG_SPARSE) {
        for (int j = 0; j < M; j++)
            PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 2, (int[]){T0 + j, P0 + j});
    } else {
        PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
        for (int j = 0; j < M; j++)
            PRIMAL_appendcone(t, PRIMAL_CT_PPOW, 2.0 / 3.0, 3, (int[]){T0 + j, ONE, P0 + j});
    }
    return t;
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 30;
    M = argc > 2 ? atoi(argv[2]) : 4;
    LAM = argc > 3 ? atof(argv[3]) : 0.5;
    if (N < 1 || M < 1 || LAM < 0.0) return 2;
    X = malloc((size_t)N * M * sizeof(double));
    y = malloc((size_t)N * sizeof(double));
    w0 = calloc((size_t)M, sizeof(double));
    for (int k = 0; k < N * M; k++) X[k] = (rnd() - 0.5) * 2.0;
    for (int i = 0; i < N; i++) y[i] = (rnd() - 0.5) * 2.0;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    struct timeval a, b;
    int ok = 1;
    double wref[16];

    /* ridge: verify against (X'X + lam I) w = X'y + lam w0 */
    {
        PRIMALtask_t t = build(REG_RIDGE, env);
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double w[16];
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(M + 1) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int j = 0; j < M; j++) w[j] = x[j];
            free(x);
        } else ok = 0;
        PRIMAL_deletetask(&t);
        DMat *G = dmat_new(M, M);
        double *rhs = calloc((size_t)M, sizeof(double));
        for (int j = 0; j < M; j++)
            for (int k = 0; k < M; k++) {
                double s = (j == k) ? LAM : 0.0;
                for (int i = 0; i < N; i++) s += X[i * M + j] * X[i * M + k];
                G->v[j][k] = s;
            }
        for (int j = 0; j < M; j++) {
            double s = LAM * w0[j];
            for (int i = 0; i < N; i++) s += X[i * M + j] * y[i];
            rhs[j] = s;
        }
        if (dmat_solve_lu(G, rhs, M) == 0) {
            for (int j = 0; j < M; j++) wref[j] = rhs[j];
            double d = 0.0;
            for (int j = 0; j < M; j++) if (fabs(w[j] - wref[j]) > d) d = fabs(w[j] - wref[j]);
            if (d > 1e-5) ok = 0;
            printf("ridge      n=%d m=%d lam=%.2f wdiff=%.2e t=%.4fs\n", N, M, LAM, d, sec);
        }
        dmat_free(G); free(rhs);
    }

    /* sparse: verify (t_j, p_j) in Q^2 */
    {
        PRIMALtask_t t = build(REG_SPARSE, env);
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double worst = 0.0;
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(3 * M + 1) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int j = 0; j < M; j++) {
                double slack = fabs(x[M + j]) - x[2 * M + j];
                if (slack > worst) worst = slack;
            }
            free(x);
        } else ok = 0;
        PRIMAL_deletetask(&t);
        if (worst > 1e-5) ok = 0;
        printf("sparse     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs\n",
               N, M, LAM, (int)rc, worst, sec);
    }

    /* 3/2: verify (t_j, 1, p_j) in POW^{2/3,1/3}, i.e. t_j >= |p_j|^{3/2} */
    {
        PRIMALtask_t t = build(REG_POW32, env);
        gettimeofday(&a, NULL);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        gettimeofday(&b, NULL);
        double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);
        double worst = 0.0;
        if (rc == PRIMAL_RES_OK) {
            double *x = malloc((size_t)(3 * M + 2) * sizeof(double));
            PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int j = 0; j < M; j++) {
                double slack = pow(fabs(x[M + j]), 1.5) - x[2 * M + j];
                if (slack > worst) worst = slack;
            }
            free(x);
        } else ok = 0;
        PRIMAL_deletetask(&t);
        if (worst > 1e-4) ok = 0;
        printf("pow3_2     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs\n",
               N, M, LAM, (int)rc, worst, sec);
    }

    printf("regr_reg   n=%d m=%d lam=%.2f %s\n", N, M, LAM, ok ? "OK" : "FAIL");
    PRIMAL_deleteenv(&env);
    free(X); free(y); free(w0);
    return ok ? 0 : 1;
}
