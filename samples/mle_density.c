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

/* mle_density.c - MLE of a log-concave density via exponential cones.
 *
 * Source: MOSEK/Tutorials, mle-convex-density-function.  On a grid
 * y_0<..<y_{n-1} with spacings dy_i = y_{i+1}-y_i, estimate the values
 * x_i = g(y_i) of a concave function g = log f (log-concave density) by
 * maximizing the weighted log-likelihood:
 *
 *   minimize    sum_i weight_i * u_i
 *   subject to  u_i >= -log(x_i)                       (PEXP cone per i)
 *               x concave:  -dy_{i+1} x_i + (dy_i+dy_{i+1}) x_{i+1}
 *                           - dy_i x_{i+2} <= 0        for all inner i
 *               sum_i ((x_i+x_{i+1})/2) dy_i = 1        (density integrates to 1)
 *
 * The cone is this library's PRIMAL_CT_PEXP: (x_i, 1, -u_i) in K_exp, i.e.
 * x_i >= exp(-u_i)  <=>  u_i >= -log(x_i).
 *
 * Hand-derived instance: n=3, dy = (1/2,1/2,1/2), weights (1,2,3).  The
 * normalization 0.25 x_0 + 0.5 x_1 + 0.25 x_2 = 1 has trapezoidal weights, so
 * the KKT stationarity -w_i/x_i + lambda c_i = 0 (with c = (1/4,1/2,1/4))
 * gives x proportional to (4,4,12); the concavity constraint is inactive
 * (4 <= 8) and lambda = 6, hence x = (2/3, 2/3, 2) and
 *   objective = -3 log(4/3) = -0.8630462174.
 *
 * Usage: mle_density   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define ND 3

int main(void) {
    const double dy[ND] = {0.5, 0.5, 0.5};
    const double w[ND] = {1.0, 2.0, 3.0};
    enum { X = 0, U = ND, NNEG = 2 * ND, ONE = 3 * ND, NV = 3 * ND + 1 };
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, (ND - 2) + 1 + ND);   /* concavity + normalization + links */
    for (int i = 0; i < ND; i++) PRIMAL_putvarbound(t, X + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < ND; i++) PRIMAL_putvarbound(t, U + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < ND; i++) PRIMAL_putvarbound(t, NNEG + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    int row = 0;
    for (int i = 0; i + 2 < ND; i++) {          /* concavity of x */
        PRIMAL_putarow(t, row, 3, (int[]){X + i, X + i + 1, X + i + 2},
                       (double[]){-dy[i + 1], dy[i] + dy[i + 1], -dy[i]});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;
    }
    {   /* normalization: sum ((x_i+x_{i+1})/2) dy_i = 1 */
        int sub[ND]; double val[ND];
        for (int i = 0; i < ND; i++) { sub[i] = X + i; val[i] = 0.0; }
        for (int i = 0; i + 1 < ND; i++) { val[i] += 0.5 * dy[i]; val[i + 1] += 0.5 * dy[i]; }
        PRIMAL_putarow(t, row, ND, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    for (int i = 0; i < ND; i++) {              /* -u_i - N_i = 0 */
        PRIMAL_putarow(t, row, 2, (int[]){NNEG + i, U + i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
    }
    for (int i = 0; i < ND; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){X + i, ONE, NNEG + i});
    for (int i = 0; i < ND; i++) PRIMAL_putcj(t, U + i, w[i]);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, xv[ND] = {0};
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < ND; i++) xv[i] = x[X + i];
        for (int i = 0; i < ND; i++) obj += w[i] * x[U + i];
    }
    double want = -3.0 * log(4.0 / 3.0);
    int okall = ok && fabs(xv[0] - 2.0 / 3.0) < 2e-3 && fabs(xv[1] - 2.0 / 3.0) < 2e-3 &&
                fabs(xv[2] - 2.0) < 2e-3 && fabs(obj - want) < 1e-6;
    printf("mle_density  n=%d\n", ND);
    printf("  x = (%.8f, %.8f, %.8f)   (atteso 2/3, 2/3, 2)\n", xv[0], xv[1], xv[2]);
    printf("  obiettivo = %.10f   (atteso %.10f)\n", obj, want);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
