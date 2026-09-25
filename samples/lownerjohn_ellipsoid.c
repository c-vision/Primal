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

/* lownerjohn_ellipsoid.c - Lowner-John inner ellipsoidal approximation
 * (MOSEK Cookbook 11.6, lownerjohn_ellipsoid.cc).
 *
 * The max-volume ellipsoid { C u + d : ||u|| <= 1 } inscribed in the polytope
 * { x : A x <= b } solves
 *     maximize   t
 *     s.t.       t <= det(C)^(1/n)            (det_rootn)
 *                || C a_i || <= b_i - a_i'd    (one QUAD cone per face)
 *                C >= 0.
 *
 * det_rootn for n = 2 uses the PSD block Y = [[C, Z],[Z', diag(Z)]] (4x4), Z
 * lower triangular (Z01 = 0), and the geometric mean t <= sqrt(Z00*Z11), written
 * as the rotated cone 2*Z00*(Z11/2) >= t^2.
 *
 * Hand instance: the unit square [0,1]^2 (A = +-e0, +-e1; b = 1,0,1,0).  By
 * symmetry the max inscribed ellipse is the circle C = (1/2) I, d = (1/2,1/2),
 * so det(C) = 1/4 and t = 1/2.
 *
 * The variables carry the bounds the model really has (C00,C11,Z00,Z11,A,B,CC,
 * EE,T >= 0, d in [0,1]). The equivalent all-FREE model is covered by T245:
 * shared cone members need separate dual contributions, whose sum is the
 * variable's reduced cost (T246 also checks unequal contributions).
 *
 * Usage: lownerjohn_ellipsoid   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static int unit_sym(PRIMALtask_t t, int n, int i, int j) {
    int id;
    PRIMAL_appendsparsesymmat(t, n, 1, (int[]){i}, (int[]){j}, (double[]){1.0}, &id);
    return id;
}

int main(void) {
    enum { T = 0, D0, D1, X00, X01, X11, Z00, Z11, UU, AA, BB, CC, EE, NV };
    enum { R = 14 };
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, R, NV, &t);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    int d4 = 4;
    PRIMAL_appendbarvars(t, 1, &d4);
    PRIMAL_putvarbound(t, T, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, T, 1.0);
    PRIMAL_putvarbound(t, D0, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putvarbound(t, D1, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putvarbound(t, X00, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, X01, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, X11, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, Z00, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, Z11, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, UU, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int v = AA; v <= EE; v++) PRIMAL_putvarbound(t, v, PRIMAL_BK_LO, 0.0, INFINITY);
    int E00 = unit_sym(t, 4, 0, 0), E01 = unit_sym(t, 4, 0, 1), E11 = unit_sym(t, 4, 1, 1),
        E02 = unit_sym(t, 4, 0, 2), E13 = unit_sym(t, 4, 1, 3), E03 = unit_sym(t, 4, 0, 3),
        E22 = unit_sym(t, 4, 2, 2), E33 = unit_sym(t, 4, 3, 3), E23 = unit_sym(t, 4, 2, 3);
    /* Y entries = scalar variables (off-diagonal counted twice: coeff 1/2) */
    PRIMAL_putarow(t, 0, 1, (int[]){X00}, (double[]){-1.0});
    PRIMAL_putbaraij(t, 0, 0, 1, &E00, (double[]){1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 1, 1, (int[]){X01}, (double[]){-1.0});
    PRIMAL_putbaraij(t, 1, 0, 1, &E01, (double[]){0.5});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 2, 1, (int[]){X11}, (double[]){-1.0});
    PRIMAL_putbaraij(t, 2, 0, 1, &E11, (double[]){1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 3, 1, (int[]){Z00}, (double[]){-1.0});
    PRIMAL_putbaraij(t, 3, 0, 1, &E02, (double[]){0.5});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 4, 1, (int[]){Z11}, (double[]){-1.0});
    PRIMAL_putbaraij(t, 4, 0, 1, &E13, (double[]){0.5});
    PRIMAL_putconbound(t, 4, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putbaraij(t, 5, 0, 1, &E03, (double[]){0.5});
    PRIMAL_putconbound(t, 5, PRIMAL_BK_FX, 0.0, 0.0);       /* Z01 = 0 */
    PRIMAL_putbaraij(t, 6, 0, 1, &E22, (double[]){1.0});
    PRIMAL_putbaraij(t, 6, 0, 1, &E02, (double[]){-0.5});
    PRIMAL_putconbound(t, 6, PRIMAL_BK_FX, 0.0, 0.0);       /* Y22 = Z00 */
    PRIMAL_putbaraij(t, 7, 0, 1, &E33, (double[]){1.0});
    PRIMAL_putbaraij(t, 7, 0, 1, &E13, (double[]){-0.5});
    PRIMAL_putconbound(t, 7, PRIMAL_BK_FX, 0.0, 0.0);       /* Y33 = Z11 */
    PRIMAL_putbaraij(t, 8, 0, 1, &E23, (double[]){0.5});
    PRIMAL_putconbound(t, 8, PRIMAL_BK_FX, 0.0, 0.0);       /* Y23 = 0 */
    PRIMAL_putarow(t, 9, 1, (int[]){UU}, (double[]){1.0});
    PRIMAL_putbaraij(t, 9, 0, 1, &E13, (double[]){-0.25});
    PRIMAL_putconbound(t, 9, PRIMAL_BK_FX, 0.0, 0.0);       /* U = Z11/2 */
    PRIMAL_putarow(t, 10, 2, (int[]){AA, D0}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 10, PRIMAL_BK_FX, 1.0, 1.0);      /* A = 1-d0 */
    PRIMAL_putarow(t, 11, 2, (int[]){BB, D0}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 11, PRIMAL_BK_FX, 0.0, 0.0);      /* B = d0   */
    PRIMAL_putarow(t, 12, 2, (int[]){CC, D1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, 12, PRIMAL_BK_FX, 1.0, 1.0);      /* C = 1-d1 */
    PRIMAL_putarow(t, 13, 2, (int[]){EE, D1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 13, PRIMAL_BK_FX, 0.0, 0.0);      /* E = d1   */
    PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 3, (int[]){Z00, UU, T});   /* Z00*Z11 >= T^2 */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){AA, X00, X01});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){BB, X00, X01});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){CC, X01, X11});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){EE, X01, X11});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, x[16] = {0};
    if (ok) { PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj); PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    int good = ok && fabs(obj - 0.5) < 1e-5 && fabs(x[X00] - 0.5) < 1e-4 &&
               fabs(x[X11] - 0.5) < 1e-4 && fabs(x[X01]) < 1e-4 &&
               fabs(x[D0] - 0.5) < 1e-4 && fabs(x[D1] - 0.5) < 1e-4;
    printf("lownerjohn: rc=%d t*=%.8f  C=(%.5f,%.5f,%.5f) d=(%.5f,%.5f)  (atteso 0.5, C=0.5I, d=(0.5,0.5))  %s\n",
           (int)rc, obj, x[X00], x[X01], x[X11], x[D0], x[D1], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
