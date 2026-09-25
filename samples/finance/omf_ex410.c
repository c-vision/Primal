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

/* omf_ex410.c - Exercise 4.10 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Chapter 4, "Linear programming models: asset
 * / liability management".
 *
 * ---------------------------------------------------------------------------
 * PROBLEM
 * ---------------------------------------------------------------------------
 * You have $20 000 to invest. Stock XYZ sells at $20/share today. A European
 * call option to buy 100 shares of XYZ at $15 exactly six months from today
 * sells for $1000; you may also sell such options (raising funds immediately).
 * A six-month riskless zero-coupon bond with $100 face value sells for $90.
 * You limit the call options bought or sold to at most 50. Three equally
 * likely scenarios for XYZ in six months: $20 (same), $40 (up), $12 (down).
 *
 * Variables: B = bonds, S = shares, C = calls (>0 bought, <0 sold).
 * Expected per-unit profits: bond 10, stock (20+0-8)/3 = 4, call 0.
 *
 * (i)  maximize 10 B + 4 S
 *      s.t.  90 B + 20 S + 1000 C <= 20000        (budget)
 *            -50 <= C <= 50                       (call limit)
 *            B >= 0, S >= 0
 *      BOOK optimum: B = 0, S = 3500, C = -50, expected profit 14000.
 *
 * (ii)/(iii) require a profit >= 2000 in EVERY scenario:
 *      maximize (P1 + P2 + P3)/3
 *      s.t.  90 B + 20 S + 1000 C <= 20000
 *            P1 = 10 B + 20 S + 1500 C
 *            P2 = 10 B - 500 C
 *            P3 = 10 B - 8 S - 1000 C
 *            P1, P2, P3 >= 2000,  -50 <= C <= 50,  B, S >= 0
 *      BOOK optimum: 2800 shares, sell 36 calls, expected profit 11200.
 *
 * (iv) maximize the riskless profit Z, i.e. the profit guaranteed in every
 *      scenario: P_i >= Z for all i, with the same budget and |C| <= 50.
 *      BOOK optimum: riskless profit Z = 7272 (about 2273 shares, selling
 *      25.45 calls), expected profit 9091.
 *
 * The sample solves these LPs with this library and checks the book's answers
 * (objective and portfolio), plus that the budget and the scenario profits are
 * what the book states.
 *
 * Usage: omf_ex410   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* (i) maximize 10 B + 4 S, budget 90 B + 20 S + 1000 C <= 20000, |C| <= 50 */
static int part_i(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 1, 3, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);          /* B */
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);          /* S */
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_RA, -50.0, 50.0);            /* C */
    PRIMAL_putcj(t, 0, -10.0); PRIMAL_putcj(t, 1, -4.0);            /* max 10B+4S */
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){90.0, 20.0, 1000.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 20000.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[3];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 90.0 * x[0] + 20.0 * x[1] + 1000.0 * x[2];
        ok = fabs(z + 14000.0) < 1e-6 && fabs(x[0] - 0.0) < 1e-6 &&
             fabs(x[1] - 3500.0) < 1e-6 && fabs(x[2] + 50.0) < 1e-6 &&
             budget <= 20000.0 + 1e-6;
        printf("omf_ex410 (i)   B=%.4g S=%.4g C=%.4g  profit=%.6g (book 14000) %s\n",
               x[0], x[1], x[2], -z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex410 (i)   rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* (iii) maximize (P1+P2+P3)/3 s.t. budget, P_i = scenario profits, P_i >= 2000 */
static int part_iii(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0=B 1=S 2=C 3=P1 4=P2 5=P3 ; rows: 0=budget, 1..3 = Pi definitions */
    PRIMAL_maketask(env, 4, 6, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);          /* B */
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);          /* S */
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_RA, -50.0, 50.0);            /* C */
    for (int k = 3; k < 6; k++) PRIMAL_putvarbound(t, k, PRIMAL_BK_LO, 2000.0, INFINITY);
    PRIMAL_putcj(t, 3, -1.0 / 3.0); PRIMAL_putcj(t, 4, -1.0 / 3.0); PRIMAL_putcj(t, 5, -1.0 / 3.0);
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){90.0, 20.0, 1000.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 20000.0);
    PRIMAL_putarow(t, 1, 4, (int[]){0, 1, 2, 3}, (double[]){10.0, 20.0, 1500.0, -1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 2, 3, (int[]){0, 2, 4}, (double[]){10.0, -500.0, -1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 3, 4, (int[]){0, 1, 2, 5}, (double[]){10.0, -8.0, -1000.0, -1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[6];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 90.0 * x[0] + 20.0 * x[1] + 1000.0 * x[2];
        /* book: 2800 shares, sell 36 calls, expected profit 11200, all P_i >= 2000 */
        ok = fabs(z + 11200.0) < 1e-6 && fabs(x[1] - 2800.0) < 1e-6 &&
             fabs(x[2] + 36.0) < 1e-6 && budget <= 20000.0 + 1e-6 &&
             x[3] >= 2000.0 - 1e-6 && x[4] >= 2000.0 - 1e-6 && x[5] >= 2000.0 - 1e-6;
        printf("omf_ex410 (iii) B=%.4g S=%.4g C=%.4g  P=(%.6g,%.6g,%.6g)  profit=%.6g (book 11200) %s\n",
               x[0], x[1], x[2], x[3], x[4], x[5], -z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex410 (iii) rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* (iv) maximize the riskless profit Z: P_i >= Z, budget, |C| <= 50 */
static int part_iv(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0=B 1=S 2=C 3=P1 4=P2 5=P3 6=Z ; rows: 0=budget, 1..3 Pi, 4..6 Pi>=Z */
    PRIMAL_maketask(env, 7, 7, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);          /* B */
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);          /* S */
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_RA, -50.0, 50.0);            /* C */
    for (int k = 3; k < 7; k++) PRIMAL_putvarbound(t, k, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, 6, -1.0);                                       /* max Z */
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){90.0, 20.0, 1000.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 20000.0);
    PRIMAL_putarow(t, 1, 4, (int[]){0, 1, 2, 3}, (double[]){10.0, 20.0, 1500.0, -1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 2, 3, (int[]){0, 2, 4}, (double[]){10.0, -500.0, -1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putarow(t, 3, 4, (int[]){0, 1, 2, 5}, (double[]){10.0, -8.0, -1000.0, -1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    for (int k = 0; k < 3; k++) {                                   /* P_i - Z >= 0 */
        PRIMAL_putarow(t, 4 + k, 2, (int[]){3 + k, 6}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 4 + k, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[7];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double budget = 90.0 * x[0] + 20.0 * x[1] + 1000.0 * x[2];
        double exp_profit = (x[3] + x[4] + x[5]) / 3.0;
        /* book rounds to 2273 shares / sell 25.45 calls / riskless 7272 /
         * expected 9091; the exact vertex is S=2272.73, C=-25.4545, Z=7272.73,
         * expected 9090.91, with the budget exactly tight. */
        ok = fabs(z + 7272.727) < 1e-2 && fabs(exp_profit - 9090.909) < 1e-2 &&
             fabs(budget - 20000.0) < 1e-6 &&
             x[3] >= x[6] - 1e-6 && x[4] >= x[6] - 1e-6 && x[5] >= x[6] - 1e-6;
        printf("omf_ex410 (iv)  B=%.4g S=%.4g C=%.4g  riskless=%.6g (book 7272)  "
               "expected=%.6g (book 9091) %s\n",
               x[0], x[1], x[2], -z, exp_profit, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex410 (iv)  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    int a = part_i(), b = part_iii(), c = part_iv();
    return (a && b && c) ? 0 : 1;
}