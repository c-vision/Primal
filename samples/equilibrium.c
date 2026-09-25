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

/* equilibrium.c - equilibrium of masses hanging on strings (SOCP).
 *
 * Source: MOSEK/Tutorials, equilibrium/equilibrium.py.  Masses at positions
 * x_i in R^2 are joined by strings of given maximum lengths; the system
 * settles in the minimum-energy (lowest) configuration:
 *
 *   minimize  g * sum_i w_i x_i,y
 *   subject to  || x_i - x_j ||_2 <= l_ij   for every string (i,j)
 *               x_i = f_i                   for the pinned masses
 *
 * The string constraint is the notebook's quadratic cone
 * (l_ij, x_i - x_j) in Q^3.  This sample pins two masses at (-1,0) and (1,0)
 * and hangs one free mass from two strings of length l = 2; by symmetry the
 * mass settles directly below the midpoint at y = -sqrt(l^2 - 1) = -sqrt(3).
 *
 * Usage: equilibrium   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define LSTR 2.0

/* one free mass at (x0, x1) joined to two fixed anchors by strings of length
 * LSTR; minimize x1. */
static int solve(double *x0_out, double *y_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { X0 = 0, X1, D0 = 2, D1 = 4, L = 6, NV = 7 };
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, 4);
    for (int k = 0; k < 6; k++) PRIMAL_putvarbound(t, k, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, L, PRIMAL_BK_FX, LSTR, LSTR);
    /* D0 = (x0,x1) - (-1,0) ; D1 = (x0,x1) - (1,0) */
    PRIMAL_putarow(t, 0, 2, (int[]){D0, X0}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putarow(t, 1, 2, (int[]){D0 + 1, X1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 2, 2, (int[]){D1, X0}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, -1.0, -1.0);
    PRIMAL_putarow(t, 3, 2, (int[]){D1 + 1, X1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){L, D0, D0 + 1});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){L, D1, D1 + 1});
    PRIMAL_putcj(t, X1, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) { double x[NV]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); *x0_out = x[X0]; *y_out = x[X1]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double x0 = 0, y = 0;
    int ok = solve(&x0, &y);
    double want = -sqrt(LSTR * LSTR - 1.0);   /* -sqrt(3) */
    int okall = ok && fabs(y - want) < 1e-7 && fabs(x0) < 1e-6;
    printf("equilibrium  l=%.1f\n", LSTR);
    printf("  massa libera = (%.8f, %.8f)   (atteso (0, -sqrt(3)=%.8f))\n", x0, y, want);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
