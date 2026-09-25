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

/* omf_appendix_d.c - revised-simplex worked LP, "Optimization Methods in
 * Finance" (Cornuejols & Tutuncu, 2007), Appendix D (lines 7656-7802):
 *
 *   maximize   Z = x1 + 2 x2 + x3 - 2 x4
 *   subject to -2 x1 +   x2 + x3 + 2 x4 + x6      = 2
 *               -x1 + 2 x2 + x3      + x5 + x7   = 7
 *                x1       + x3 + x4 + x5      + x8 = 3
 *               x1,...,x8 >= 0
 *
 * The book runs the revised simplex and reports
 *   x1 = 3, x2 = 5, x6 = 3, all other x = 0, Z = 13.
 *
 * Usage: omf_appendix_d   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 8, &t);
    for (int j = 0; j < 8; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putcj(t, 0, 1.0); PRIMAL_putcj(t, 1, 2.0);
    PRIMAL_putcj(t, 2, 1.0); PRIMAL_putcj(t, 3, -2.0);
    PRIMAL_putarow(t, 0, 5, (int[]){0, 1, 2, 3, 5}, (double[]){-2, 1, 1, 2, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 2.0, 2.0);
    PRIMAL_putarow(t, 1, 5, (int[]){0, 1, 2, 4, 6}, (double[]){-1, 2, 1, 1, 1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 7.0, 7.0);
    PRIMAL_putarow(t, 2, 5, (int[]){0, 2, 3, 4, 7}, (double[]){1, 1, 1, 1, 1});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 3.0, 3.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[8];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - 3.0) < 1e-6 && fabs(x[1] - 5.0) < 1e-6 &&
             fabs(x[5] - 3.0) < 1e-6;
        for (int j = 2; j < 8; j++) if (j != 5) ok = ok && fabs(x[j]) < 1e-6;
        ok = ok && fabs(z - 13.0) < 1e-6;
        printf("omf_appendix_d  x1=%.6g x2=%.6g x6=%.6g Z=%.6g (book 3,5,3, 13) %s\n",
               x[0], x[1], x[5], z, ok ? "OK" : "FAIL");
    } else {
        printf("omf_appendix_d  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
