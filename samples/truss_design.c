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

/* truss_design.c - minimum-volume truss by semidefinite programming.
 *
 * Source: MOSEK Modeling Cookbook / MOSEK Tutorials, "truss topology design".
 *
 * Two symmetric bars A-C and B-C (A=(-1,1), B=(1,1) fixed, C=(0,0) free and
 * loaded with F=(0,-1)); bar cross-sections t_1, t_2 >= 0, Young modulus E=1,
 * length L=sqrt(2). Minimize the volume L(t_1+t_2) subject to a compliance
 * bound f'K(t)^{-1}f <= 1. The Schur complement turns it into an LMI:
 *
 *   min  L (t_1 + t_2)
 *   s.t. X = [[K(t), f], [f', 1]] ⪰ 0,   K(t) = sum_i t_i (E/L_i) d_i d_i'
 *        t >= 0
 *
 * with d_i the bar direction. Hand derivation:
 *   d_1 = (1,-1)/L, d_2 = (-1,-1)/L, and (E/L) = 1/L = 1/sqrt(2), so
 *   K(t) = (1/(2 sqrt 2)) [[t1+t2, t2-t1], [t2-t1, t1+t2]].
 *   The compliance is f'K^{-1}f = sqrt(2)/(2 t1 t2) * (t1+t2); the bound
 *   <= 1 gives sqrt(2)(t1+t2) <= 2 t1 t2. By symmetry t1 = t2 = t:
 *   2 sqrt(2) t <= 2 t^2  =>  t >= sqrt(2), and the volume is
 *   L(2 t) = sqrt(2) * 2 sqrt(2) = 4.
 *
 * The sample cross-checks the optimum against the closed form 4 and against
 * the compliance constraint recomputed from the published t.
 *
 * Usage: truss_design   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NDOF 2
#define DIM  (NDOF + 1)          /* [[K, f], [f', 1]] */
static const double L = 1.4142135623730951;   /* sqrt(2) */
static const double K0 = 1.0 / (2.0 * 1.4142135623730951);   /* 1/(2 sqrt2) */

/* add the LMI entry `coef * X[i][j]` to row `row` of the (DIM x DIM) bar block */
static void barentry(PRIMALtask_t t, int row, int i, int j, double coef) {
    int m;
    PRIMAL_appendsparsesymmat(t, DIM, 1, (int[]){i}, (int[]){j}, (double[]){1.0}, &m);
    PRIMAL_putbaraij(t, row, 0, 1, &m, &coef);
}

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 6, 2, &t);
    /* scalar vars: 0 = t1, 1 = t2 (bar cross-sections) */
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, 0, L);       /* min L (t1 + t2) */
    PRIMAL_putcj(t, 1, L);

    int dim = DIM;
    PRIMAL_appendbarvars(t, 1, &dim);

    /* row 0: X[0][0] - K0 (t1 + t2) = 0 */
    barentry(t, 0, 0, 0, 1.0);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){-K0, -K0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    /* row 1: X[1][1] - K0 (t1 + t2) = 0 */
    barentry(t, 1, 1, 1, 1.0);
    PRIMAL_putarow(t, 1, 2, (int[]){0, 1}, (double[]){-K0, -K0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* row 2: X[0][1] - K0 (t2 - t1) = 0  (<E_01,X> = 2 X_01, so coef 1/2) */
    barentry(t, 2, 0, 1, 0.5);
    PRIMAL_putarow(t, 2, 2, (int[]){0, 1}, (double[]){K0, -K0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.0, 0.0);
    /* row 3: X[0][2] = 0   (f_x = 0) */
    barentry(t, 3, 0, 2, 0.5);
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    /* row 4: X[1][2] = -1  (f_y = -1) */
    barentry(t, 4, 1, 2, 0.5);
    PRIMAL_putconbound(t, 4, PRIMAL_BK_FX, -1.0, -1.0);
    /* row 5: X[2][2] = 1 */
    barentry(t, 5, 2, 2, 1.0);
    PRIMAL_putconbound(t, 5, PRIMAL_BK_FX, 1.0, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0;
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        double x[2];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double t1 = x[0], t2 = x[1];
        /* compliance recomputed from t: sqrt(2)/(2 t1 t2) * (t1+t2) */
        double comp = (t1 > 0 && t2 > 0) ? (L / (2.0 * t1 * t2)) * (t1 + t2) : 1e30;
        ok = fabs(obj - 4.0) < 1e-5 && comp <= 1.0 + 1e-6;
        printf("truss_design  t=(%.6f,%.6f) volume=%.6f (atteso 4) compliance=%.6f %s\n",
               t1, t2, obj, comp, ok ? "OK" : "FAIL");
    } else {
        printf("truss_design  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
