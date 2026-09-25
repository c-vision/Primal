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

/* hard_uncertain.c - robust optimization with a hard uncertain inequality,
 * solved EXACTLY by evaluating the log-sum-exp at the box vertices (MOSEK
 * Tutorials, approx-uncertain-ineq).
 *
 * For x in R^n, matrices B_i (L x n), i = 1..p, and the uncertainty box
 * [-1,1]^L with vertices z_k, the notebook's "exact" model is
 *
 *   minimize  sum(x)
 *   subject to  sum_i exp( (B_i x)^T z_k - sum(x) ) <= 1   for every vertex z_k,
 *
 * each term an exponential cone (PRIMAL_CT_PEXP): h1_i >= exp(h3_i), h2 = 1,
 * sum_i h1_i <= 1.
 *
 * Instance L=1 (two vertices +/-1), n=1 (scalar x), p=2, B_0 = (1/2),
 * B_1 = (1/5).  The feasible set is x >= x*, where x* is the largest root of
 * exp(-x/2) + exp(-4x/5) = 1 (the +1 vertex binds); checked against a dense
 * scan of x.
 *
 * Usage: hard_uncertain   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define L 1
#define NP 2
#define NVX 1
#define NVERT (1 << L)
#define NH (NP * NVERT)
static const double B[NP][L][NVX] = {{{0.5}}, {{0.2}}};
static const double VERT[L][(1 << L)] = {{-1.0, 1.0}};

int main(void) {
    int nvert = NVERT;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { X = 0, H1 = 1, H3 = 1 + NH, ONE = 1 + 2 * NH, NV = ONE + 1 };
    int nH = NH;
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, nH + nvert);
    PRIMAL_putvarbound(t, X, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int q = 0; q < nH; q++) PRIMAL_putvarbound(t, H1 + q, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int q = 0; q < nH; q++) PRIMAL_putvarbound(t, H3 + q, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    int row = 0;
    for (int i = 0; i < NP; i++)
        for (int k = 0; k < nvert; k++) {          /* h3_ik - (B_i z_k - 1) x = 0 */
            double coef = B[i][0][0] * VERT[0][k] - 1.0;
            PRIMAL_putarow(t, row, 2, (int[]){H3 + i * nvert + k, X}, (double[]){1.0, -coef});
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
        }
    for (int k = 0; k < nvert; k++) {              /* sum_i h1_ik <= 1 */
        int sub[NP]; double val[NP];
        for (int i = 0; i < NP; i++) { sub[i] = H1 + i * nvert + k; val[i] = 1.0; }
        PRIMAL_putarow(t, row, NP, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 1.0); row++;
    }
    for (int q = 0; q < nH; q++)
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){H1 + q, ONE, H3 + q});
    PRIMAL_putcj(t, X, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double x = 0.0;
    if (ok) { double v[NV] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, v); x = v[X]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* reference: the smallest x where every vertex constraint holds */
    double xstar = 0.0; int found = 0;
    for (int q = -400000; q <= 400000; q++) {
        double xx = q * 1e-5;
        int feas = 1;
        for (int k = 0; k < nvert && feas; k++) {
            double s = 0.0;
            for (int i = 0; i < NP; i++) s += exp(B[i][0][0] * VERT[0][k] * xx - xx);
            if (s > 1.0) feas = 0;
        }
        if (feas) { xstar = xx; found = 1; break; }
    }
    int okall = ok && found && fabs(x - xstar) < 1e-4 * (1.0 + fabs(xstar));
    printf("hard_uncertain  L=%d n=%d p=%d\n", L, NVX, NP);
    printf("  x* = %.8f   [scan %.8f]\n", x, xstar);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
