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

/* omf_dedication.c - the dedication (cash-flow matching) bond portfolio of
 * "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 3.2
 * and Exercise 3.11.
 *
 * A bank must fund a municipality's liability stream over 8 years by buying,
 * today, a least-cost portfolio of 10 risk-free non-collapsible bonds (0%
 * reinvestment rate). Liabilities (Year 1..8):
 *
 *   12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000
 *
 * Bonds (face $100, annual coupon; Price / Coupon / Maturity year):
 *   1: 102 / 5.0 / 1     6: 104 / 9.0 / 5
 *   2:  99 / 3.5 / 2     7: 100 / 6.0 / 5
 *   3: 101 / 5.0 / 2     8: 101 / 8.0 / 6
 *   4:  98 / 3.5 / 3     9: 102 / 9.0 / 7
 *   5:  98 / 4.0 / 4    10:  94 / 7.0 / 8
 *
 * Model (as in the book):
 *   min   z(0) + sum_i P(i) x(i)
 *   s.t.  sum_{i: M(i) >= t} C(i) x(i) + sum_{i: M(i) = t} 100 x(i)
 *           + z(t-1) - z(t) = L(t),  t = 1..8
 *         x(i) >= 0, z(t) >= 0.
 *
 * Exercise 3.11 states the optimum costs $93,944 with holdings
 *   62 Bond1, 125 Bond3, 152 Bond4, 157 Bond5, 123 Bond6, 124 Bond8,
 *   104 Bond9, 93 Bond10   (none of Bond 2 or Bond 7).
 * The book's solution table also gives shadow prices: Year 1 = 0.971428571,
 * Year 2 = 0.915646259, Year 3 = 0.883045779 (i.e. r_3 = 4.23%), and the
 * Section 3.3.2 sensitivity table gives the reduced costs of the nonbasic
 * variables: Bond 2 = 0.830612245, Bond 7 = 8.786840002, z0 = 0.028571429,
 * z8 = 0.524288903. Exercise 3.17 reads Bond 7's reduced cost (8.786) as the
 * price cut that would make Bond 7 enter the optimal portfolio, i.e. a more
 * realistic price just above $91.
 *
 * The sample solves the LP with this library and checks the book's cost,
 * holdings, shadow prices and reduced costs.
 *
 * Usage: omf_dedication   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define T 8
#define N 10

static const double L[T + 1] = {0, 12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000};
static const double P[N] = {102, 99, 101, 98, 98, 104, 100, 101, 102, 94};
static const double C[N] = {5.0, 3.5, 5.0, 3.5, 4.0, 9.0, 6.0, 8.0, 9.0, 7.0};
static const int    M[N] = {1, 2, 2, 3, 4, 5, 5, 6, 7, 8};
/* book optimum holdings (Exercise 3.11) */
static const double XBOOK[N] = {62, 0, 125, 152, 157, 123, 0, 124, 104, 93};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0..9 = x(i), 10..18 = z(0..8) ; rows: 0..7 = year 1..8 */
    PRIMAL_maketask(env, T, N + T + 1, &t);

    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int j = 0; j <= T; j++) PRIMAL_putvarbound(t, N + j, PRIMAL_BK_LO, 0.0, INFINITY);

    PRIMAL_putcj(t, N, 1.0);                      /* z(0) in the objective */
    for (int i = 0; i < N; i++) PRIMAL_putcj(t, i, P[i]);

    for (int yr = 1; yr <= T; yr++) {
        int idx[N + 2]; double val[N + 2]; int n = 0;
        for (int i = 0; i < N; i++) {
            double a = 0.0;
            if (M[i] >= yr) a += C[i];
            if (M[i] == yr) a += 100.0;
            if (a != 0.0) { idx[n] = i; val[n] = a; n++; }
        }
        idx[n] = N + (yr - 1); val[n] = 1.0; n++;   /* + z(t-1) */
        idx[n] = N + yr;       val[n] = -1.0; n++;  /* - z(t)   */
        PRIMAL_putarow(t, yr - 1, n, idx, val);
        PRIMAL_putconbound(t, yr - 1, PRIMAL_BK_FX, L[yr], L[yr]);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[N + T + 1];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double cost_book = 0.0, hold_err = 0.0;
        for (int i = 0; i < N; i++) {
            cost_book += P[i] * XBOOK[i];
            double d = fabs(x[i] - XBOOK[i]);
            if (d > hold_err) hold_err = d;
        }
        /* shadow prices (dual of the year constraints) */
        double sp[T] = {0};
        PRIMAL_gety(t, PRIMAL_SOL_ITR, sp);
        /* the book prints the shadow prices positive; our dual sign convention
         * gives them negative (same magnitude, equality constraints). */
        double sp_book[3] = {0.971428571, 0.915646259, 0.883045779};
        int sp_ok = fabs(fabs(sp[0]) - sp_book[0]) < 1e-5 &&
                    fabs(fabs(sp[1]) - sp_book[1]) < 1e-5 &&
                    fabs(fabs(sp[2]) - sp_book[2]) < 1e-5;

        /* reduced costs from the Section 3.3.2 sensitivity table; the min form
         * makes them nonnegative. Exercise 3.17 uses Bond 7's value (8.786). */
        double rc[N + T + 1] = {0};
        PRIMAL_getreducedcosts(t, PRIMAL_SOL_ITR, 0, N + T + 1, rc);
        double rc_book[4] = {0.830612245, 8.786840002, 0.028571429, 0.524288903};
        int rc_ok = fabs(rc[1]   - rc_book[0]) < 1e-5 &&   /* Bond 2  */
                    fabs(rc[6]   - rc_book[1]) < 1e-5 &&   /* Bond 7  */
                    fabs(rc[N]   - rc_book[2]) < 1e-5 &&   /* z0      */
                    fabs(rc[N+8] - rc_book[3]) < 1e-5;     /* z8      */

        ok = fabs(z - 93944.0) < 1.0 && hold_err < 1.0 && sp_ok && rc_ok;
        printf("omf_dedication  cost=%.4f (book 93944)  max|x-xbook|=%.4f  %s\n",
               z, hold_err, ok ? "OK" : "FAIL");
        printf("  shadow:  sp1=%.6f sp2=%.6f sp3=%.6f  %s\n",
               fabs(sp[0]), fabs(sp[1]), fabs(sp[2]), sp_ok ? "OK" : "FAIL");
        printf("  reduced: B2=%.6f B7=%.6f z0=%.6f z8=%.6f  %s\n",
               rc[1], rc[6], rc[N], rc[N+8], rc_ok ? "OK" : "FAIL");
        printf("  holdings: ");
        for (int i = 0; i < N; i++) printf("B%d=%.4g ", i + 1, x[i]);
        printf("\n  (bond-only cost = %.2f, z0 = %.4f, book bond-only = %.2f)\n",
               z - x[N], x[N], cost_book);
    } else {
        printf("omf_dedication  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
