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

/* example_lp.c - the smallest end-to-end use of the public API and the shape
 * to copy for a first model.  It builds a 3-row, 4-variable LP
 *
 *     max  3 x1 + x2 + 5 x3 + x4
 *     s.t. 3x1 + x2 + 2x3         = 30
 *          2x1 + x2 + 3x3 + x4  >= 15
 *               2x2        + 3x4 <= 25
 *          0 <= x1, x3, x4;   0 <= x2 <= 10
 *
 * by hand (env -> maketask -> objective sense and cost -> row bounds -> column
 * bounds -> one putacol per column), installs a log callback on the task
 * stream, solves with PRIMAL_optimize and prints the status, the primal point
 * and the objective.  The model is the reference "lo1" example.
 */

#include <stdio.h>
#include "primal.h"

static void PRIMALAPI logfn(void *h, const char s[]) { (void)h; printf("%s", s); }

int main(void) {
    PRIMALenv_t env = NULL;
    PRIMALtask_t task = NULL;

    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 3, 4, &task);
    PRIMAL_linkfunctotaskstream(task, PRIMAL_STREAM_LOG, NULL, logfn);

    /* max 3x1 + x2 + 5x3 + x4
       s.t. 3x1 + x2 + 2x3      = 30
            2x1 + x2 + 3x3 + x4 >= 15
                 2x2      + 3x4 <= 25
       0 <= x1, x3, x4;  0 <= x2 <= 10  */
    PRIMAL_putobjsense(task, PRIMAL_OBJECTIVE_SENSE_MAXIMIZE);
    double c[] = {3.0, 1.0, 5.0, 1.0};
    for (int j = 0; j < 4; j++) PRIMAL_putcj(task, j, c[j]);

    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 30.0, 30.0);
    PRIMAL_putconbound(task, 1, PRIMAL_BK_LO, 15.0, 0.0);
    PRIMAL_putconbound(task, 2, PRIMAL_BK_UP, 0.0, 25.0);

    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, 0.0);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_RA, 0.0, 10.0);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, 0.0);
    PRIMAL_putvarbound(task, 3, PRIMAL_BK_LO, 0.0, 0.0);

    /* Colonna 0: 3x1 + 2x1 -> righe 0,1 */
    { PRIMALint32t sub[] = {0,1}; double v[] = {3.0,2.0};
      PRIMAL_putacol(task, 0, 2, sub, v); }
    /* Colonna 1: x2 + x2 + 2x2 -> righe 0,1,2 */
    { PRIMALint32t sub[] = {0,1,2}; double v[] = {1.0,1.0,2.0};
      PRIMAL_putacol(task, 1, 3, sub, v); }
    /* Colonna 2: 2x3 + 3x3 -> righe 0,1 */
    { PRIMALint32t sub[] = {0,1}; double v[] = {2.0,3.0};
      PRIMAL_putacol(task, 2, 2, sub, v); }
    /* Colonna 3: x4 + 3x4 -> righe 1,2 */
    { PRIMALint32t sub[] = {1,2}; double v[] = {1.0,3.0};
      PRIMAL_putacol(task, 3, 2, sub, v); }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc == PRIMAL_RES_OK) {
        double x[4], pobj; PRIMALsolstae sta;
        PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &pobj);
        PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
        printf("\nStato: %d\n", sta);
        for (int j = 0; j < 4; j++) printf("x[%d] = %g\n", j, x[j]);
        printf("Obiettivo = %g\n", pobj);
    } else {
        printf("Errore: %d\n", rc);
    }

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return 0;
}