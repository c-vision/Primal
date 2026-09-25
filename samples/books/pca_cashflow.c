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

/* pca_cashflow.c - cash-flow matching, an independent LP exercise
 * (model taken from "Portfolio Construction and Analytics" (2016),
 * Section 15.2.1.4, lines 7107-7130):
 *
 *   minimize   sum_i p_i x_i
 *   subject to sum_i c_it x_i >= m_t,   t = 1..8
 *              x_i >= 0
 *
 * five semi-annual-coupon bonds with ask prices
 *   p = (102.36, 110.83, 96.94, 114.65, 96.63),
 * cash-flow matrix c_it (rows t=1..8, cols bonds 1..5) and obligations
 *   m = (100000, 200000, 100000, 200000, 800000, 1200000, 400000, 1000000).
 *
 * This sample solves the LP on its own terms and reports the optimum it finds.
 * (The book that states this example prints an allocation matching ours to the
 * cent, x = (6000, 28103.61, 3555.18, 0, 9661.84), but an objective that
 * exceeds the cost of those same numbers by ~$148 — an internal inconsistency
 * of the source, ignored here.)
 *
 * Usage: pca_cashflow   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double p[5] = {102.36, 110.83, 96.94, 114.65, 96.63};
    static const double c[8][5] = {
        {  2.50,   5.00,   3.00,   4.00,   3.50},
        {  2.50,   5.00,   3.00,   4.00,   3.50},
        {  2.50,   5.00,   3.00,   4.00,   3.50},
        {  2.50,   5.00,   3.00,   4.00,   3.50},
        {102.50,   5.00,   3.00,   4.00,   3.50},
        {  0.00, 105.00,   3.00,   4.00,   3.50},
        {  0.00,   0.00, 103.00,   4.00,   3.50},
        {  0.00,   0.00,   0.00, 104.00, 103.50}};
    static const double m[8] = {100000, 200000, 100000, 200000,
                                800000, 1200000, 400000, 1000000};

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 8, 5, &t);
    for (int j = 0; j < 5; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, j, p[j]);
    }
    for (int i = 0; i < 8; i++) {
        PRIMAL_putarow(t, i, 5, (int[]){0, 1, 2, 3, 4}, c[i]);
        PRIMAL_putconbound(t, i, PRIMAL_BK_LO, m[i], INFINITY);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[5];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 6000.0) < 1.0 && fabs(x[1] - 28103.61) < 1.0 &&
             fabs(x[2] - 3555.18) < 1.0 && fabs(x[3]) < 1.0 &&
             fabs(x[4] - 9661.84) < 1.0 && fabs(z - 5007145.84) < 1.0;
        printf("pca_cashflow  x=(%.0f,%.2f,%.2f,%.0f,%.2f) cost=%.2f (independent LP; book allocation matches) %s\n",
               x[0], x[1], x[2], x[3], x[4], z, ok ? "OK" : "FAIL");
    } else {
        printf("pca_cashflow  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
