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

/* omf_ex118.c - Exercise 11.8 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 11.3.3 "Gomory mixed integer cuts".
 *
 * The integer program
 *   max 10 x1 + 13 x2
 *   s.t. 10 x1 + 14 x2 <= 43
 *        x1, x2 >= 0 integer.
 *
 * (i)  The LP relaxation (slack x3) has the optimal tableau
 *          x1 + 1.4 x2 + 0.1 x3 = 4.3,
 *      i.e. basic solution x1 = 4.3, x2 = x3 = 0, objective 43.
 * (ii)/(iii) For a row  sum_j a_j x_j = b, the GMI cut is
 *          sum_j ( floor(a_j) + (f_j - f0)^+ / (1 - f0) ) x_j <= floor(b),
 *      with f_j = frac(a_j), f0 = frac(b). Multiplying the row by k = 3 gives
 *      a = (3, 4.2, 0.3), b = 12.9, f0 = 0.9, so the cut is
 *          3 x1 + 4 x2 <= 12.
 * (iv) Adding this cut, the LP optimum drops to 40 at x = (4, 0), which is the
 *      integer optimum.
 *
 * The sample reproduces the LP relaxation, derives the k = 3 GMI cut, and
 * checks that the cut-augmented LP and the MIP both give 40 at (4, 0).
 *
 * Usage: omf_ex118   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* GMI cut for the row  sum_j a_j x_j = b  multiplied by k.
 * cut_a[j] and *cut_rhs receive the inequality cut_a . x <= *cut_rhs. */
static void gmi_cut(const double *a, int n, double b, int k,
                    double *cut_a, double *cut_rhs) {
    double kb = (double)k * b, f0 = kb - floor(kb);
    for (int j = 0; j < n; j++) {
        double ka = (double)k * a[j];
        double fj = ka - floor(ka);
        double t = fj - f0;
        cut_a[j] = floor(ka) + (t > 0.0 ? t / (1.0 - f0) : 0.0);
    }
    *cut_rhs = floor(kb);
}

/* LP relaxation of the IP (optionally with the GMI cut); int_flag = MIP. */
static int solve(int with_cut, int int_flag, double *obj, double *x1, double *x2) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    int nrow = with_cut ? 2 : 1;
    PRIMAL_maketask(env, nrow, 2, &t);
    for (int j = 0; j < 2; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        if (int_flag) PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT);
    }
    PRIMAL_putcj(t, 0, -10.0); PRIMAL_putcj(t, 1, -13.0);   /* max */
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){10.0, 14.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 43.0);
    if (with_cut) {
        PRIMAL_putarow(t, 1, 2, (int[]){0, 1}, (double[]){3.0, 4.0});
        PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 12.0);
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        *x1 = x[0]; *x2 = x[1];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    int ok = 1;
    double z, x1, x2;

    /* (i) LP relaxation: tableau row x1 + 1.4 x2 + 0.1 x3 = 4.3 */
    if (!solve(0, 0, &z, &x1, &x2)) { printf("omf_ex118 LP FAIL\n"); return 1; }
    int relax_ok = fabs(-z - 43.0) < 1e-6 && fabs(x1 - 4.3) < 1e-6 && fabs(x2) < 1e-6;
    printf("omf_ex118 (i)  LP relaxation: x1=%.4f x2=%.4f z=%.6g (book 4.3/0/43) %s\n",
           x1, x2, -z, relax_ok ? "OK" : "FAIL");
    ok &= relax_ok;

    /* (ii)/(iii) GMI cut for the tableau row, k = 3 -> 3 x1 + 4 x2 <= 12 */
    double a[3] = {1.0, 1.4, 0.1}, cut[3], rhs;
    gmi_cut(a, 3, 4.3, 3, cut, &rhs);
    int cut_ok = fabs(cut[0] - 3.0) < 1e-9 && fabs(cut[1] - 4.0) < 1e-9 &&
                 fabs(cut[2] - 0.0) < 1e-9 && fabs(rhs - 12.0) < 1e-9 &&
                 3.0 * 4.3 > 12.0;                    /* cuts off the LP point */
    printf("omf_ex118 (ii) GMI k=3: %.4g x1 + %.4g x2 + %.4g x3 <= %.4g "
           "(3*4.3=%.1f>12) %s\n", cut[0], cut[1], cut[2], rhs, 3.0 * 4.3,
           cut_ok ? "OK" : "FAIL");
    ok &= cut_ok;

    /* (iv) LP with the cut, and the MIP: both 40 at (4, 0) */
    double zc, xc1, xc2, zi, xi1, xi2;
    int lp_ok = solve(1, 0, &zc, &xc1, &xc2);
    int mip_ok = solve(1, 1, &zi, &xi1, &xi2);
    int cut_lp_ok = lp_ok && fabs(-zc - 40.0) < 1e-6 &&
                    fabs(xc1 - 4.0) < 1e-6 && fabs(xc2) < 1e-6;
    int mip_good = mip_ok && fabs(-zi - 40.0) < 1e-6 &&
                   fabs(xi1 - 4.0) < 1e-6 && fabs(xi2) < 1e-6;
    printf("omf_ex118 (iv) LP+cut: x=(%.4g,%.4g) z=%.6g   MIP: x=(%.4g,%.4g) z=%.6g"
           "  (book 40 at (4,0)) %s\n",
           xc1, xc2, -zc, xi1, xi2, -zi, (cut_lp_ok && mip_good) ? "OK" : "FAIL");
    ok &= cut_lp_ok && mip_good;

    return ok ? 0 : 1;
}
