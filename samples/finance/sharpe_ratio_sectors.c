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

/* sharpe_ratio_sectors.c - max-Sharpe portfolio, long-only, with per-asset and
 * per-sector diversification constraints.
 *
 * Source: matimacazaga/portfolio_optimization_with_mosek (sharpe_ratio.ipynb).
 * The unconstrained core is the Charnes-Cooper reformulation of
 *     max (mu'w - rf) / sqrt(w'Sigma w)   s.t. e'w = 1
 * already covered by samples/finance/sharpe_ratio.c; this repo adds LONG-ONLY
 * (y >= 0) and the DIVERSIFICATION constraints, which are linear in (y, z):
 *     all assets : y_i <=/>= w_i * z
 *     sector     : sum_{i in sector} y_i <=/>= w * z
 * As a minimum-norm problem with the normalisation mu'y - rf z = 1:
 *     minimize  s        s.t.  s >= || G y ||,  mu'y - rf z = 1,  e'y = z,
 *                              y >= 0, z >= 0,  + the boxes/sector caps,
 * and w = y / z, Sharpe = 1 / s.
 *
 * Case: 4 assets, Sigma = I (G = I), mu = (3,1,3,1), rf = 0, two sectors
 * {0,1} and {2,3}.  With no cap the max Sharpe is ||mu|| = sqrt(20) at
 * w = mu / e'mu = (3,1,3,1)/8.  The sector-{0,1} cap:
 *   - at 0.5 it is what the optimum already does (w'_{0,1} = 0.5), so the value
 *     stays EXACTLY sqrt(20) -- a hand value;
 *   - at 0.4 it binds, the value must DROP below sqrt(20); the returned weights
 *     are checked feasible against every original constraint.
 *
 * Usage: sharpe_ratio_sectors   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NA 4

/* solve with sector-{0,1} cap = cap (only that diversification constraint);
 * returns the Sharpe (or -1) and writes w. */
static double solve_case(double cap, double wout[NA], double *sout, int *ok_out) {
    static const double mu[NA] = {3.0, 1.0, 3.0, 1.0};
    int Y0 = 0, Z = NA, S = NA + 1;                     /* y(4), z, s */
    int nv = S + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < NA; i++) PRIMAL_putvarbound(t, Y0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, Z, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, S, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_appendcons(t, 3);                            /* mu'y = 1 ; e'y - z = 0 ; cap */
    /* mu'y = 1 */
    PRIMAL_putarow(t, 0, NA, (int[]){Y0, Y0+1, Y0+2, Y0+3}, (double[]){mu[0], mu[1], mu[2], mu[3]});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* e'y - z = 0 */
    PRIMAL_putarow(t, 1, NA + 1, (int[]){Y0, Y0+1, Y0+2, Y0+3, Z}, (double[]){1,1,1,1,-1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* sector cap: y0 + y1 - cap*z <= 0 */
    PRIMAL_putarow(t, 2, 3, (int[]){Y0, Y0+1, Z}, (double[]){1.0, 1.0, -cap});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 0.0);
    /* s >= ||y|| (G = I): (s, y0..y3) in Q_5 */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, NA + 1, (int[]){S, Y0, Y0+1, Y0+2, Y0+3});
    PRIMAL_putcj(t, S, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = -1.0, w[NA] = {0};
    if (rc == PRIMAL_RES_OK) {
        double x[16] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double z = x[Z];
        for (int i = 0; i < NA; i++) wout[i] = z > 0 ? x[Y0+i]/z : 0.0;
        obj = x[S];                                     /* ||y|| = 1/Sharpe */
        *sout = x[S];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = (rc == PRIMAL_RES_OK);
    (void)w;
    return obj > 0 ? 1.0 / obj : -1.0;                  /* Sharpe = 1/s */
}

int main(void) {
    printf("sharpe_ratio_sectors (4 asset, 2 settori, Sigma=I)\n");
    int all = 1;
    double w[NA], s = 0; int ok = 0;

    /* case A: cap 0.5 -- the optimum already satisfies it, Sharpe = sqrt(20) */
    double sh = solve_case(0.5, w, &s, &ok);
    double sw = w[0]+w[1]+w[2]+w[3];
    int good = ok && fabs(sh - sqrt(20.0)) < 1e-4 && fabs(w[0]+w[1]-0.5) < 1e-4 &&
                  fabs(w[0]-0.375) < 1e-4 && fabs(w[1]-0.125) < 1e-4 &&
                  fabs(w[2]-0.375) < 1e-4 && fabs(w[3]-0.125) < 1e-4 && fabs(sw-1.0) < 1e-6;
    printf("  A cap=0.50: Sharpe=%.6f (atteso %.6f)  w=(%.4f,%.4f,%.4f,%.4f)  %s\n",
           sh, sqrt(20.0), w[0], w[1], w[2], w[3], good ? "OK" : "FAIL");
    all &= good;

    /* case B: cap 0.4 binds -> Sharpe < sqrt(20), weights feasible */
    double sh2 = solve_case(0.4, w, &s, &ok);
    double sum = w[0]+w[1]+w[2]+w[3], s1 = w[0]+w[1];
    good = ok && sh2 > 0 && sh2 < sqrt(20.0) - 1e-6 &&
           w[0] >= -1e-9 && w[1] >= -1e-9 && w[2] >= -1e-9 && w[3] >= -1e-9 &&
           fabs(sum - 1.0) < 1e-6 && s1 <= 0.4 + 1e-6;
    printf("  B cap=0.40: Sharpe=%.6f (< %.6f)  w=(%.4f,%.4f,%.4f,%.4f) sum=%.6f s1=%.6f  %s\n",
           sh2, sqrt(20.0), w[0], w[1], w[2], w[3], sum, s1, good ? "OK" : "FAIL");
    all &= good;

    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
