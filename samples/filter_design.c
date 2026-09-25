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

/* filter_design.c - FIR filter design as a trigonometric-polynomial SDP.
 *
 * Source: MOSEK/Tutorials, filterdesign.  A real filter response
 *
 *   H(w) = x_0 + 2 x_1 cos(w) + ... + 2 x_n cos(n w)
 *
 * is nonnegative on an interval iff its coefficient vector lies in a cone
 * K^n_[a,b] characterized by PSD Toeplitz matrices X:
 *   [0,pi] :  x_i = <T_i^{n+1}, X>,                    X >= 0
 *   [0,a]  :  x_i = <T_i^{n+1},X1> + <T_{i+1}^n,X2> + <T_{i-1}^n,X2>
 *                   - 2 cos(a) <T_i^n, X2>,            X1,X2 >= 0
 *   [a,pi] :  x_i = <T_i^{n+1},X1> - <T_{i+1}^n,X2> - <T_{i-1}^n,X2>
 *                   + 2 cos(a) <T_i^n, X2>,            X1,X2 >= 0
 * with <T_i^m, Y> = sum_k Y_{k,k+|i|} (the i-th diagonal sum).
 *
 * The design minimizes the stopband level t subject to  H >= 0 (global),
 * 1-d <= H <= 1+d (passband [0,wp]) and H <= t (stopband [ws,pi]); the
 * epigraph/hypograph are handled by flipping the sign of (x_0 - level, x_1..).
 *
 * The returned response is checked on a dense grid.  DECLARED DEVIATION: the
 * specifications are verified numerically (sampling), not by a second solver.
 *
 * Usage: filter_design   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NF 3                       /* polynomial degree n */

/* add coef * <T_i^dim, X_bar> to row (one symmat per diagonal entry) */
static void addT(PRIMALtask_t t, int row, int bar, int dim, int i, double coef) {
    int ii = i < 0 ? -i : i;
    if (ii >= dim) return;
    for (int k = 0; k + ii < dim; k++) {
        int a = k, b = k + ii;
        double sv = (ii == 0) ? 1.0 : 0.5;     /* <m,X> = X_{k,k+ii} */
        int m;
        PRIMAL_appendsparsesymmat(t, dim, 1, &a, &b, &sv, &m);
        double v = coef;
        PRIMAL_putbaraij(t, row, bar, 1, &m, &v);
    }
}

int main(void) {
    const double DELTA = 0.05, WP = M_PI / 4.0, WS = M_PI / 4.0 + M_PI / 8.0;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    /* scalar vars: x[0..NF], t */
    enum { X = 0, TT = NF + 1, NV = NF + 2 };
    PRIMAL_appendvars(t, NV);
    for (int i = 0; i <= NF; i++) PRIMAL_putvarbound(t, X + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_LO, 0.0, INFINITY);
    /* bars: 0 global X0(NF+1); 1,2 passband upper; 3,4 passband lower; 5,6 stopband */
    int d1 = NF + 1, d2 = NF;
    PRIMAL_appendbarvars(t, 1, &d1);   /* 0 */
    PRIMAL_appendbarvars(t, 1, &d1);   /* 1 */
    PRIMAL_appendbarvars(t, 1, &d2);   /* 2 */
    PRIMAL_appendbarvars(t, 1, &d1);   /* 3 */
    PRIMAL_appendbarvars(t, 1, &d2);   /* 4 */
    PRIMAL_appendbarvars(t, 1, &d1);   /* 5 */
    PRIMAL_appendbarvars(t, 1, &d2);   /* 6 */
    PRIMAL_appendcons(t, 4 * (NF + 1));
    int row = 0;
    /* global nonnegativity: <T_i^{n+1},X0> - x_i = 0 */
    for (int i = 0; i <= NF; i++) {
        addT(t, row, 0, NF + 1, i, 1.0);
        PRIMAL_putarow(t, row, 1, (int[]){X + i}, (double[]){-1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
    }
    /* passband upper (level 1+delta): sum_0_a(...) + (x_0 or x_i) = rhs */
    for (int i = 0; i <= NF; i++) {
        addT(t, row, 1, NF + 1, i, 1.0);
        addT(t, row, 2, NF, i + 1, 1.0);
        addT(t, row, 2, NF, i - 1, 1.0);
        addT(t, row, 2, NF, i, -2.0 * cos(WP));
        double rhs = (i == 0) ? (1.0 + DELTA) : 0.0;
        PRIMAL_putarow(t, row, 1, (int[]){X + i}, (double[]){1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, rhs, rhs); row++;
    }
    /* passband lower (level 1-delta): sum_0_a(...) - (x_0 or x_i) = rhs */
    for (int i = 0; i <= NF; i++) {
        addT(t, row, 3, NF + 1, i, 1.0);
        addT(t, row, 4, NF, i + 1, 1.0);
        addT(t, row, 4, NF, i - 1, 1.0);
        addT(t, row, 4, NF, i, -2.0 * cos(WP));
        double rhs = (i == 0) ? -(1.0 - DELTA) : 0.0;
        PRIMAL_putarow(t, row, 1, (int[]){X + i}, (double[]){-1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, rhs, rhs); row++;
    }
    /* stopband (level t): sum_a_pi(...) + (x_0 - t or x_i) = 0 */
    for (int i = 0; i <= NF; i++) {
        addT(t, row, 5, NF + 1, i, 1.0);
        addT(t, row, 6, NF, i + 1, -1.0);
        addT(t, row, 6, NF, i - 1, -1.0);
        addT(t, row, 6, NF, i, 2.0 * cos(WS));
        if (i == 0) { PRIMAL_putarow(t, row, 2, (int[]){X, TT}, (double[]){1.0, -1.0}); }
        else        { PRIMAL_putarow(t, row, 1, (int[]){X + i}, (double[]){1.0}); }
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
    }
    PRIMAL_putcj(t, TT, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double x[NV] = {0}, topt = 0.0;
    if (ok) {
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        topt = x[TT];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* verify the specs on a dense grid */
    int neg = 0, pa_hi = 0, pa_lo = 0, sb = 0;
    for (int q = 0; q <= 4000; q++) {
        double w = M_PI * q / 4000.0;
        double H = x[X];
        for (int i = 1; i <= NF; i++) H += 2.0 * x[X + i] * cos(i * w);
        if (H < -1e-5) neg = 1;
        if (w <= WP + 1e-9) { if (H > 1.0 + DELTA + 1e-4) pa_hi = 1; if (H < 1.0 - DELTA - 1e-4) pa_lo = 1; }
        if (w >= WS - 1e-9) { if (H > topt + 1e-4) sb = 1; }
    }
    int okall = ok && !neg && !pa_hi && !pa_lo && !sb && topt < 1.0;
    printf("filter_design  n=%d  wp=pi/4  ws=3pi/8  delta=%.2f\n", NF, DELTA);
    printf("  t* (stopband) = %.6f   H>=0:%d  passband ok:%d  stopband ok:%d\n",
           topt, !neg, !(pa_hi || pa_lo), !sb);
    printf("  x = (");
    for (int i = 0; i <= NF; i++) printf("%s%.5f", i ? ", " : "", x[X + i]);
    printf(")\n");
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
