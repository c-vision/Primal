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

/* boyd_detector.c - minimax detector, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Example 7.4 (lines 11778-11792):
 *
 * Binary hypothesis testing with n = 4 and
 *   P = [ 0.70 0.10 ; 0.20 0.10 ; 0.05 0.70 ; 0.05 0.10 ]
 * (column j = distribution of the observation under hypothesis j+1).
 * A randomized detector T (2x4, each column a probability distribution)
 * minimizes the worst-case error; the minimax detector is
 *   T^(4) = [ 1  2/3  0  0 ; 0  1/3  1  1 ],
 * yielding P_fp = P_fn = 1/6 (every deterministic detector exceeds 1/6).
 *
 * LP form: variables T0_i, T1_i (>=0, T0_i + T1_i = 1) and t, with
 *   t >= sum_i P[i][1] T1_i   (false negative),
 *   t >= sum_i P[i][0] T0_i   (false positive),   minimize t.
 * Sample solves it and checks t = 1/6 and both errors = 1/6.
 *
 * Usage: boyd_detector   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    static const double P0[4] = {0.70, 0.20, 0.05, 0.05};  /* hyp 1 column */
    static const double P1[4] = {0.10, 0.10, 0.70, 0.10};  /* hyp 2 column */

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 6, 9, &t);   /* 4 col-balance + 2 error rows; 8 T + t */
    for (int j = 0; j < 8; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 8, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, 8, 1.0);   /* minimize t */
    for (int i = 0; i < 4; i++) {  /* T0_i + T1_i = 1 */
        PRIMAL_putarow(t, i, 2, (int[]){i, 4 + i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* false negative D[1][0] = sum_i P[i][0] T1_i : t - sum_i P0[i] T1_i >= 0 */
        PRIMAL_putarow(t, 4, 5, (int[]){4, 5, 6, 7, 8},
                       (double[]){-P0[0], -P0[1], -P0[2], -P0[3], 1.0});
        PRIMAL_putconbound(t, 4, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    {   /* false positive D[0][1] = sum_i P[i][1] T0_i : t - sum_i P1[i] T0_i >= 0 */
        PRIMAL_putarow(t, 5, 5, (int[]){0, 1, 2, 3, 8},
                       (double[]){-P1[0], -P1[1], -P1[2], -P1[3], 1.0});
        PRIMAL_putconbound(t, 5, PRIMAL_BK_LO, 0.0, INFINITY);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, v[9];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        double fn = 0.0, fp = 0.0;   /* recomputed error probabilities */
        for (int i = 0; i < 4; i++) { fn += P0[i] * v[4 + i]; fp += P1[i] * v[i]; }
        ok = fabs(z - 1.0 / 6.0) < 1e-6 && fabs(fn - 1.0 / 6.0) < 1e-6 &&
             fabs(fp - 1.0 / 6.0) < 1e-6;
        printf("boyd_detector  t=%.6g Pfp=%.6g Pfn=%.6g (book 1/6 all) %s\n",
               z, fp, fn, ok ? "OK" : "FAIL");
    } else {
        printf("boyd_detector  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
