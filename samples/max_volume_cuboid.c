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

/* max_volume_cuboid.c - the largest axis-aligned cuboid inscribed in a
 * polyhedron {x : a_i'x <= b_i}.
 *
 * Source: MOSEK Modeling Cookbook, "Geometric programming" / max-volume cuboid.
 * The problem is a geometric program (maximize a product of the half-widths):
 *
 *   max  prod_j (2 r_j)
 *   s.t. a_i' c + |a_i|' r <= b_i   (cuboid [c-r, c+r] inside the polyhedron)
 *        r >= 0
 *
 * With n = 2 the product is a geometric mean, which the 3D power cone carries:
 *   (r_0, r_1, t) in POW^{1/2}   <=>   sqrt(r_0 r_1) >= t,
 * so maximizing t maximizes r_0 r_1 and the cuboid volume is (2 t)^2 = 4 t^2.
 *
 * Hand-derived optima (n = 2):
 *   unit square [0,1]^2                      -> volume 1.00  (t = 0.50)
 *   right triangle {x>=0, y>=0, x+y<=1}      -> volume 0.25  (t = 0.25)
 *
 * Both are derived by hand from the KKT of the GP: for the square the largest
 * rectangle is the square itself; for the triangle the optimal rectangle is
 * c = r = (1/4, 1/4) (the faces x>=0 and y>=0 are tight, and x+y+r_x+r_y = 1).
 *
 * Usage: max_volume_cuboid [square|triangle]   (default square)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* each face is a0*x0 + a1*x1 <= b */
static const double SQ[4][3]  = { {-1, 0, 0}, {0, -1, 0}, {1, 0, 1}, {0, 1, 1} };
static const double TRI[3][3] = { {-1, 0, 0}, {0, -1, 0}, {1, 1, 1} };

static int solve(const char *name, const double F[][3], int nf, double want) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, nf, 5, &t);
    /* vars: 0=c0  1=c1  2=r0  3=r1  4=t */
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 4, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, 4, -1.0);   /* maximize t */

    for (int i = 0; i < nf; i++) {
        int sub[4]; double val[4]; int nz = 0;
        if (F[i][0] != 0.0) { sub[nz] = 0; val[nz] = F[i][0]; nz++; }
        if (F[i][1] != 0.0) { sub[nz] = 1; val[nz] = F[i][1]; nz++; }
        if (F[i][0] != 0.0) { sub[nz] = 2; val[nz] = fabs(F[i][0]); nz++; }
        if (F[i][1] != 0.0) { sub[nz] = 3; val[nz] = fabs(F[i][1]); nz++; }
        PRIMAL_putarow(t, i, nz, sub, val);
        PRIMAL_putconbound(t, i, PRIMAL_BK_UP, -INFINITY, F[i][2]);
    }
    /* (r0, r1, t) in POW^{1/2}: sqrt(r0*r1) >= |t| */
    PRIMAL_appendcone(t, PRIMAL_CT_PPOW, 0.5, 3, (int[]){2, 3, 4});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[5];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double vol = 4.0 * x[4] * x[4];
        ok = fabs(vol - want) < 1e-5;
        printf("%-8s c=(%.4f,%.4f) r=(%.4f,%.4f) volume=%.6f (atteso %.4f) %s\n",
               name, x[0], x[1], x[2], x[3], vol, want, ok ? "OK" : "FAIL");
    } else {
        printf("%-8s rc=%d FAIL\n", name, (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(int argc, char **argv) {
    const char *which = (argc > 1) ? argv[1] : "square";
    int ok;
    if (which[0] == 't') ok = solve("triangle", TRI, 3, 0.25);
    else                 ok = solve("square",   SQ,  4, 1.00);
    return ok ? 0 : 1;
}
