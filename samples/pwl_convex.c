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

/* pwl_convex.c - least-squares fit by a convex piecewise-linear function.
 *
 * Source: MOSEK/Tutorials, pwl-convex-approximation.  Given data points
 * X_i in R^d and values Y_i, fit convex PWL function values t_i at the knots
 * together with subgradients s_i:
 *
 *   minimize  m
 *   subject to  m >= sum_i (t_i - Y_i)^2          (rotated cone)
 *               t_i >= t_j + < s_j, X_i - X_j >    for all i,j  (convexity)
 *
 * The squared error is the notebook's rotated cone  (1/2, m, t - Y) in RQCone,
 * i.e. 2*(1/2)*m >= ||t - Y||^2; the convexity rows are the tangent-plane
 * majorization.  This library's PRIMAL_CT_RQUAD takes the same form.
 *
 * Hand-derived instance (d=1): X=(0,1,2), Y=(0,1,0).  Symmetry gives
 * t0 = t2 = a, t1 = b with the convexity constraint b <= a active, so
 * minimize 2a^2 + (a-1)^2 -> a = 1/3, hence t = (1/3,1/3,1/3) and
 * m = ||t - Y||^2 = 2/3.
 *
 * Usage: pwl_convex   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NP 3

static const double XD[NP] = {0.0, 1.0, 2.0};
static const double YD[NP] = {0.0, 1.0, 0.0};

int main(void) {
    enum { M = 0, T = 1, S = 1 + NP, AD = 1 + 2 * NP, HALF = 1 + 3 * NP, NV = 1 + 3 * NP + 1 };
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, NP + NP * (NP - 1));     /* auxDiff + off-diagonal convexity */
    PRIMAL_putvarbound(t, M, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < NP; i++) {
        PRIMAL_putvarbound(t, T + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, S + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, AD + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    }
    PRIMAL_putvarbound(t, HALF, PRIMAL_BK_FX, 0.5, 0.5);
    int row = 0;
    for (int i = 0; i < NP; i++) {                /* auxDiff_i = t_i - Y_i */
        PRIMAL_putarow(t, row, 2, (int[]){AD + i, T + i}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, -YD[i], -YD[i]);
        row++;
    }
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < NP; j++) {            /* t_j - t_i + s_j(X_i - X_j) <= 0 */
            if (i == j) continue;                 /* the i=j row is the trivial 0<=0 */
            PRIMAL_putarow(t, row, 3, (int[]){T + j, T + i, S + j},
                           (double[]){1.0, -1.0, XD[i] - XD[j]});
            PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0);
            row++;
        }
    {   /* RQCone: 2*HALF*M >= sum AD_i^2 */
        int mem[2 + NP] = {HALF, M, AD, AD + 1, AD + 2};
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 2 + NP, mem);
    }
    PRIMAL_putcj(t, M, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double m = 0.0, tv[NP] = {0};
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        m = x[M];
        for (int i = 0; i < NP; i++) tv[i] = x[T + i];
    }
    double sse = 0.0;
    for (int i = 0; i < NP; i++) sse += (tv[i] - YD[i]) * (tv[i] - YD[i]);

    int okall = ok && fabs(m - 2.0 / 3.0) < 1e-7 &&
                fabs(tv[0] - 1.0 / 3.0) < 1e-6 && fabs(tv[1] - 1.0 / 3.0) < 1e-6 &&
                fabs(tv[2] - 1.0 / 3.0) < 1e-6 && fabs(m - sse) < 1e-7;
    printf("pwl_convex  n=%d\n", NP);
    printf("  t = (%.8f, %.8f, %.8f)   (atteso 1/3,1/3,1/3)\n", tv[0], tv[1], tv[2]);
    printf("  m = SSE = %.8f   (atteso 2/3=%.8f)   sse diretto=%.8f\n", m, 2.0 / 3.0, sse);
    printf("%s\n", okall ? "OK" : "FAIL");
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return okall ? 0 : 1;
}
