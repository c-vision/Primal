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

/* omf_workforce.c - the workforce planning LP of "Optimization Methods in
 * Finance" (Cornuejols & Tutuncu, 2007), Exercise 3.15 (Section 3.3.1).
 *
 * A restaurant is open seven days a week and needs, on each day, the following
 * number of workers:
 *
 *   Mon  Tue  Wed  Thu  Fri  Sat  Sun
 *    14   13   15   16   19   18   11
 *
 * Every worker works five consecutive days and then takes two days off,
 * repeating indefinitely. Shift i starts on day i (i = 1..7, day 1 = Monday)
 * and works days i, i+1, ..., i+4 (mod 7). Minimize the number of workers.
 *
 * Model:
 *   min  sum_{i=1..7} x_i
 *   s.t. x1      + x4 + x5 + x6 + x7 >= 14   (Mon)
 *        x1 + x2      + x5 + x6 + x7 >= 13   (Tue)
 *        x1 + x2 + x3      + x6 + x7 >= 15   (Wed)
 *        x1 + x2 + x3 + x4      + x7 >= 16   (Thu)
 *        x1 + x2 + x3 + x4 + x5      >= 19   (Fri)
 *             x2 + x3 + x4 + x5 + x6 >= 18   (Sat)
 *                  x3 + x4 + x5 + x6 + x7 >= 11   (Sun)
 *        x_i >= 0.
 *
 * The book's sensitivity report (Section 3.3.1) gives the optimal solution
 *   x = (4, 7, 1, 4, 3, 3, 0),  total = 22 workers,
 * shadow prices (0.333333, 0, 0.333333, 0, 0.333333, 0.333333, 0) and a
 * reduced cost of 0.333333 for Shift 7 (the unused shift).
 *
 * The sample solves the LP and checks the optimum value and the published
 * sensitivity data.
 *
 * Usage: omf_workforce   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define D 7

static const double DEMAND[D] = {14, 13, 15, 16, 19, 18, 11};
static const double XBOOK[D]  = {4, 7, 1, 4, 3, 3, 0};
static const double SPBOOK[D] = {0.333333, 0, 0.333333, 0, 0.333333, 0.333333, 0};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars 0..6 = x1..x7 (shift starting Mon..Sun); rows 0..6 = Mon..Sun */
    PRIMAL_maketask(env, D, D, &t);

    for (int i = 0; i < D; i++) {
        PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, i, 1.0);                      /* min sum x_i */
    }

    /* day d is worked by the shifts starting d, d-1, ..., d-4 (mod 7) */
    for (int d = 0; d < D; d++) {
        int idx[5]; double val[5];
        for (int k = 0; k < 5; k++) {
            idx[k] = ((d - k) % D + D) % D;
            val[k] = 1.0;
        }
        PRIMAL_putarow(t, d, 5, idx, val);
        PRIMAL_putconbound(t, d, PRIMAL_BK_LO, DEMAND[d], INFINITY);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double obj, x[D], y[D];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        PRIMAL_gety(t, PRIMAL_SOL_ITR, y);

        /* our dual sign convention on a >= min row is negative */
        int x_ok = 1;
        for (int i = 0; i < D; i++)
            if (fabs(x[i] - XBOOK[i]) > 1e-6) x_ok = 0;
        int sp_ok = 1;
        for (int d = 0; d < D; d++)
            if (fabs(fabs(y[d]) - SPBOOK[d]) > 1e-5) sp_ok = 0;

        /* Shift 7 is unused; its reduced cost is 0.333333 */
        double rcs[D] = {0};
        PRIMAL_getreducedcosts(t, PRIMAL_SOL_ITR, 0, D, rcs);
        int rc7_ok = fabs(rcs[6] - 0.333333) < 1e-5;

        ok = fabs(obj - 22.0) < 1e-6 && x_ok && sp_ok && rc7_ok;
        printf("omf_workforce  workers=%.4f (book 22)  %s\n",
               obj, ok ? "OK" : "FAIL");
        printf("  x: ");
        for (int i = 0; i < D; i++) printf("S%d=%.4g ", i + 1, x[i]);
        printf("  (book 4,7,1,4,3,3,0)  %s\n", x_ok ? "OK" : "FAIL");
        printf("  shadow: ");
        for (int d = 0; d < D; d++) printf("%.6f ", fabs(y[d]));
        printf("  %s\n", sp_ok ? "OK" : "FAIL");
        printf("  reduced cost Shift7 = %.6f (book 0.333333)  %s\n",
               rcs[6], rc7_ok ? "OK" : "FAIL");
    } else {
        printf("omf_workforce  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
