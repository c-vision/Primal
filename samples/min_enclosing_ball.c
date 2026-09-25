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

/* min_enclosing_ball.c - smallest ball enclosing a finite point set.
 *
 * Source: MOSEK/Tutorials, minimum-ellipsoid/minimum-ellipsoid.py (the
 * primal model of the notebook).  Given points p_i in R^n, find the center
 * p0 and radius r0 of the smallest enclosing ball:
 *
 *   minimize  r0
 *   subject to  r0 >= || p0 - p_i ||_2 ,   i = 1..k
 *               r0 >= 0,  p0 free,
 *
 * i.e. one quadratic cone  [r0, p0 - p_i] in Q^(n+1)  per point -- this
 * library's PRIMAL_CT_QUAD.  The dual (maximize sum_i p_i'y_i over
 * sum y_i = 0, ||[z_i,y_i]|| <= z_i) is equivalent but not needed here.
 *
 * Hand-derived instances (all exact):
 *   2 points at distance d          -> r* = d/2
 *   equilateral triangle, side a    -> r* = a/sqrt(3)  (the circumradius)
 *
 * Usage: min_enclosing_ball   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define KMAX 3
#define ND 2

/* smallest enclosing ball of the k points in P (row-major k x 2); returns
 * the radius and writes the center.  Variables: px, py, r, then a per-point
 * difference (dx_i, dy_i). */
static double min_ball(int k, const double P[KMAX][ND], double c_out[ND], int *ok_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { PX = 0, PY = 1, R = 2, D = 3 };
    int nv = 3 + 2 * k;
    PRIMAL_appendvars(t, nv);
    PRIMAL_appendcons(t, 2 * k);
    PRIMAL_putvarbound(t, PX, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, PY, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, R, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < k; i++) {
        PRIMAL_putvarbound(t, D + 2 * i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, D + 2 * i + 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
        /* dx_i - px = -P[i][0] ; dy_i - py = -P[i][1] */
        PRIMAL_putarow(t, 2 * i, 2, (int[]){D + 2 * i, PX}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 2 * i, PRIMAL_BK_FX, -P[i][0], -P[i][0]);
        PRIMAL_putarow(t, 2 * i + 1, 2, (int[]){D + 2 * i + 1, PY}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 2 * i + 1, PRIMAL_BK_FX, -P[i][1], -P[i][1]);
    }
    for (int i = 0; i < k; i++)                       /* r >= || p0 - p_i || */
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){R, D + 2 * i, D + 2 * i + 1});
    PRIMAL_putcj(t, R, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double r = 0.0;
    if (ok) {
        double x[16];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        r = x[R];
        c_out[0] = x[PX]; c_out[1] = x[PY];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = ok;
    return r;
}

int main(void) {
    double P2[KMAX][ND] = {{0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};
    double P3[KMAX][ND] = {{0.0, 0.0}, {2.0, 0.0}, {1.0, 1.7320508075688772}}; /* side 2 */
    double c[ND]; int ok2 = 0, ok3 = 0;
    double r2 = min_ball(2, P2, c, &ok2);
    double r3 = min_ball(3, P3, c, &ok3);
    double want2 = 1.0;                       /* d/2, d = 2        */
    double want3 = 2.0 / sqrt(3.0);           /* a/sqrt(3), a = 2  */

    int ok = ok2 && ok3 && fabs(r2 - want2) < 1e-7 && fabs(r3 - want3) < 1e-7;
    printf("min_enclosing_ball\n");
    printf("  2 punti (d=2):     r*=%.10f   atteso d/2=%.10f\n", r2, want2);
    printf("  triangolo (a=2):   r*=%.10f   atteso a/sqrt(3)=%.10f\n", r3, want3);
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
