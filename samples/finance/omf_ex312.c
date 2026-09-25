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

/* omf_ex312.c - Exercise 3.12 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 3.2 (dedication / cash-flow matching).
 *
 * A small pension fund must fund a 9-year liability stream (in million $):
 *
 *   24, 26, 28, 28, 26, 29, 32, 33, 34
 *
 * by buying, today, a least-cost portfolio of 16 risk-free non-collapsible
 * bonds (face $100, annual coupon), assuming a 2% reinvestment rate on cash
 * carried forward. Bonds (Price / Coupon / Maturity year):
 *
 *   1: 102.44 / 5.625 / 1     9: 110.29 / 6.875 / 6
 *   2:  99.95 / 4.75  / 2    10: 108.85 / 6.5   / 6
 *   3: 100.02 / 4.25  / 2    11: 109.95 / 6.625 / 7
 *   4: 102.66 / 5.25  / 3    12: 107.36 / 6.125 / 7
 *   5:  87.90 / 0.0   / 3    13: 104.62 / 5.625 / 8
 *   6:  85.43 / 0.0   / 4    14:  99.07 / 4.75  / 8
 *   7:  83.42 / 0.0   / 5    15: 103.78 / 5.5   / 9
 *   8: 103.82 / 5.75  / 5    16:  64.66 / 0.0   / 9
 *
 * Model (as in the book, with reinvestment rate r = 2%):
 *   min   z(0) + sum_i P(i) x(i)
 *   s.t.  sum_{i: M(i) >= t} C(i) x(i) + sum_{i: M(i) = t} 100 x(i)
 *           + (1+r) z(t-1) - z(t) = L(t),  t = 1..9
 *         x(i) >= 0, z(t) >= 0.
 *
 * The book gives no numerical optimum for this exercise, so the sample checks
 * feasibility instead: it recomputes the cash flows from the solution and
 * verifies that every liability is met with a nonnegative surplus, and that
 * the reported objective equals z(0) + sum P(i) x(i).
 *
 * Usage: omf_ex312   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define T 9
#define N 16
#define R 0.02

static const double L[T + 1] = {0, 24, 26, 28, 28, 26, 29, 32, 33, 34};
static const double P[N] = {102.44, 99.95, 100.02, 102.66, 87.90, 85.43, 83.42,
                            103.82, 110.29, 108.85, 109.95, 107.36, 104.62,
                            99.07, 103.78, 64.66};
static const double C[N] = {5.625, 4.75, 4.25, 5.25, 0.0, 0.0, 0.0, 5.75,
                            6.875, 6.5, 6.625, 6.125, 5.625, 4.75, 5.5, 0.0};
static const int    M[N] = {1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0..15 = x(i), 16..25 = z(0..9) ; rows: 0..8 = year 1..9 */
    PRIMAL_maketask(env, T, N + T + 1, &t);

    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int j = 0; j <= T; j++) PRIMAL_putvarbound(t, N + j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, N, 1.0);
    for (int i = 0; i < N; i++) PRIMAL_putcj(t, i, P[i]);

    for (int yr = 1; yr <= T; yr++) {
        int idx[N + 2]; double val[N + 2]; int n = 0;
        for (int i = 0; i < N; i++) {
            double a = 0.0;
            if (M[i] >= yr) a += C[i];
            if (M[i] == yr) a += 100.0;
            if (a != 0.0) { idx[n] = i; val[n] = a; n++; }
        }
        idx[n] = N + (yr - 1); val[n] = 1.0 + R; n++;   /* +(1+r) z(t-1) */
        idx[n] = N + yr;       val[n] = -1.0; n++;      /* - z(t)        */
        PRIMAL_putarow(t, yr - 1, n, idx, val);
        PRIMAL_putconbound(t, yr - 1, PRIMAL_BK_FX, L[yr], L[yr]);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[N + T + 1];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);

        /* independent feasibility recomputation of the cash-flow recurrence */
        double cost = x[N], minz = 0.0, maxviol = 0.0, carry = 0.0;
        for (int i = 0; i < N; i++) cost += P[i] * x[i];
        for (int yr = 1; yr <= T; yr++) {
            double inflow = 0.0;
            for (int i = 0; i < N; i++) {
                if (M[i] >= yr) inflow += C[i] * x[i];
                if (M[i] == yr) inflow += 100.0 * x[i];
            }
            carry = (1.0 + R) * carry + inflow - L[yr];
            if (carry < minz) minz = carry;
            double zt = x[N + yr];
            if (fabs(zt - carry) > 1e-6) {
                double d = fabs(zt - carry); if (d > maxviol) maxviol = d;
            }
        }
        ok = fabs(cost - z) < 1e-6 && minz > -1e-6 && maxviol < 1e-6;
        printf("omf_ex312  cost=%.4f  min surplus=%.3e  max |z-recurrence|=%.2e  %s\n",
               z, minz, maxviol, ok ? "OK" : "FAIL");
        printf("  x = ");
        for (int i = 0; i < N; i++) printf("%.4g ", x[i]);
        printf("\n");
    } else {
        printf("omf_ex312  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
