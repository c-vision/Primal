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

/* f_sparc.c - subcarrier and power allocation as a mixed-integer exp-cone
 * program (MOSEK Tutorials, f-sparc).
 *
 *   maximize  sum_{i,j} z_ij
 *   subject to  sum_j x_ij <= 1,  p_tilde_ij <= x_ij,  x binary
 *               SIGMA * t + sum p_tilde = 1,   t in [1/P, 1/SIGMA]
 *               sum_i z_ij >= t * d_j
 *               (t + p_tilde/n, t, z * log(2)/BW) in K_exp   per (i,j)
 *
 * The last cone is this library's PRIMAL_CT_PEXP: x1 >= x2 exp(x3/x2), i.e.
 *   t + p/n >= t * 2^( z / (BW t) ).
 *
 * Instance I=1, J=1, n=1, d=0.2 (BW=1.25, P=36, SIGMA=10): the power identity
 * gives p = 1 - SIGMA t and p > 0 forces x = 1, so the model reduces to
 *   maximize  BW t log2(1 + (1 - SIGMA t)/(n t))   over t in [1/P, 1/SIGMA],
 * checked against a dense scan of t (optimum z ~ 0.2206 at t ~ 0.058).
 *
 * Usage: f_sparc   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define SIGMA 10.0
#define PMAX 36.0
#define BW 1.25
#define NOISE 1.0
#define DEMAND 0.2

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { Z = 0, X = 1, PP = 2, TT = 3, TP = 4, ZC = 5, NV = 6 };
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, 5);
    PRIMAL_putvarbound(t, Z, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, X, PRIMAL_BK_RA, 0.0, 1.0);
    PRIMAL_putvartype(t, X, PRIMAL_VAR_TYPE_INT_BIN);
    PRIMAL_putvarbound(t, PP, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_RA, 1.0 / PMAX, 1.0 / SIGMA);
    PRIMAL_putvarbound(t, TP, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, ZC, PRIMAL_BK_FR, -INFINITY, INFINITY);
    /* SIGMA*t + p = 1 */
    PRIMAL_putarow(t, 0, 2, (int[]){TT, PP}, (double[]){SIGMA, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* p - x <= 0 */
    PRIMAL_putarow(t, 1, 2, (int[]){PP, X}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 0.0);
    /* t*d - z <= 0 */
    PRIMAL_putarow(t, 2, 2, (int[]){TT, Z}, (double[]){DEMAND, -1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 0.0);
    /* tp - p/n - t = 0 */
    PRIMAL_putarow(t, 3, 3, (int[]){TP, PP, TT}, (double[]){1.0, -1.0 / NOISE, -1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    /* zc - z*log2/BW = 0 */
    PRIMAL_putarow(t, 4, 2, (int[]){ZC, Z}, (double[]){1.0, -log(2.0) / BW});
    PRIMAL_putconbound(t, 4, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){TP, TT, ZC});
    PRIMAL_putcj(t, Z, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double z = 0.0, tv = 0.0;
    if (ok) { double x[NV] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); z = x[Z]; tv = x[TT]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* reference: dense scan of t (p = 1 - SIGMA t, x = 1 since p > 0) */
    double best = -1e300, bt = 0;
    for (int q = 0; q <= 400000; q++) {
        double tt = 1.0 / PMAX + (1.0 / SIGMA - 1.0 / PMAX) * q / 400000.0;
        double p = 1.0 - SIGMA * tt;
        if (p < 0.0) break;
        double zz = BW * tt * log2(1.0 + p / (NOISE * tt));
        if (zz < DEMAND * tt) continue;               /* demand constraint */
        if (zz > best) { best = zz; bt = tt; }
    }
    int okall = ok && fabs(z - best) < 1e-4 * (1.0 + fabs(best));
    printf("f_sparc  I=1 J=1  n=%.1f d=%.1f\n", NOISE, DEMAND);
    printf("  z* = %.8f  t = %.6f   [scan %.8f at t=%.6f]\n", z, tv, best, bt);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
