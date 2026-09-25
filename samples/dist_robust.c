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

/* dist_robust.c - distributionally robust portfolio (Wasserstein DRO), LP.
 *
 * Source: MOSEK/Tutorials, dist-robust-portfolio (the "program in 9" model).
 * Given N training scenarios and a Wasserstein radius eps, minimize the
 * worst-case (within the ball) expected loss:
 *
 *   minimize  eps * lambda + (1/N) sum_i s_i
 *   subject to  b_k * t + a_k * (data_i . x) <= s_i     for all i, k in {0,1}
 *               | a_k * x_j | <= lambda                   for all j, k
 *               sum_j x_j = 1,  x >= 0
 *
 * a linear program (a_k, b_k are the two pieces of the loss function; here
 * a = (-1,-51), b = (10,-40) as in the notebook).  It is checked against a
 * brute force over the weight simplex with t minimized on a grid.
 *
 * Usage: dist_robust   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define M 2
#define NS 3
#define AK0 (-1.0)
#define AK1 (-51.0)
#define BK0 (10.0)
#define BK1 (-40.0)

static const double DATA[NS][M] = {{0.02, 0.01}, {-0.01, 0.03}, {0.04, -0.02}};
static const double EPS = 0.05;

static double worst_objective(const double *x, double t) {
    double lambda = 0.0, ssum = 0.0;
    for (int j = 0; j < M; j++) {
        double v0 = fabs(AK0 * x[j]), v1 = fabs(AK1 * x[j]);
        if (v0 > lambda) lambda = v0;
        if (v1 > lambda) lambda = v1;
    }
    for (int i = 0; i < NS; i++) {
        double d = 0.0;
        for (int j = 0; j < M; j++) d += DATA[i][j] * x[j];
        double s0 = BK0 * t + AK0 * d, s1 = BK1 * t + AK1 * d;
        ssum += (s0 > s1 ? s0 : s1);
    }
    return EPS * lambda + ssum / NS;
}

int main(void) {
    PRIMALenv_t env; PRIMALtask_t task;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &task);
    enum { X = 0, S = M, LAM = M + NS, TT = M + NS + 1, NV = M + NS + 2 };
    PRIMAL_appendvars(task, NV);
    PRIMAL_appendcons(task, 2 * NS + 4 * M + 1);
    for (int j = 0; j < M; j++) PRIMAL_putvarbound(task, X + j, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < NS; i++) PRIMAL_putvarbound(task, S + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, LAM, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, TT, PRIMAL_BK_FR, -INFINITY, INFINITY);

    int row = 0;
    for (int i = 0; i < NS; i++)                     /* C1 for both k */
        for (int k = 0; k < 2; k++) {
            double bk = k == 0 ? BK0 : BK1, ak = k == 0 ? AK0 : AK1;
            int sub[M + 2]; double val[M + 2]; int nn = 0;
            sub[nn] = TT; val[nn] = bk; nn++;
            for (int j = 0; j < M; j++) { sub[nn] = X + j; val[nn] = ak * DATA[i][j]; nn++; }
            sub[nn] = S + i; val[nn] = -1.0; nn++;
            PRIMAL_putarow(task, row, nn, sub, val);
            PRIMAL_putconbound(task, row, PRIMAL_BK_UP, -INFINITY, 0.0);
            row++;
        }
    for (int j = 0; j < M; j++)                      /* C2 pos/neg */
        for (int k = 0; k < 2; k++) {
            double ak = k == 0 ? AK0 : AK1;
            PRIMAL_putarow(task, row, 2, (int[]){X + j, LAM}, (double[]){ak, -1.0});
            PRIMAL_putconbound(task, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;
            PRIMAL_putarow(task, row, 2, (int[]){X + j, LAM}, (double[]){-ak, -1.0});
            PRIMAL_putconbound(task, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;
        }
    {   int sub[M]; double val[M];
        for (int j = 0; j < M; j++) { sub[j] = X + j; val[j] = 1.0; }
        PRIMAL_putarow(task, row, M, sub, val);
        PRIMAL_putconbound(task, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    PRIMAL_putcj(task, LAM, EPS);
    for (int i = 0; i < NS; i++) PRIMAL_putcj(task, S + i, 1.0 / NS);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, x[M] = {0};
    if (ok) {
        double xx[NV];
        PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
        for (int j = 0; j < M; j++) x[j] = xx[X + j];
        obj = EPS * xx[LAM];
        for (int i = 0; i < NS; i++) obj += xx[S + i] / NS;
    }
    PRIMAL_deletetask(&task); PRIMAL_deleteenv(&env);

    /* brute force: x = (w, 1-w), t on a grid */
    double best = 1e300;
    for (int a = 0; a <= 4000; a++) {
        double w = (double)a / 4000.0, xr[M] = {w, 1.0 - w};
        for (int b = -2000; b <= 2000; b++) {
            double t = b * 0.01;
            double f = worst_objective(xr, t);
            if (f < best) best = f;
        }
    }
    int okall = ok && fabs(obj - best) < 2e-3 * (1.0 + fabs(best)) && fabs(x[0] + x[1] - 1.0) < 1e-7;
    printf("dist_robust  m=%d N=%d eps=%.2f\n", M, NS, EPS);
    printf("  x = (%.8f, %.8f)   obiettivo LP = %.8f\n", x[0], x[1], obj);
    printf("  brute force = %.8f\n", best);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
