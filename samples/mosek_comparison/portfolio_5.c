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
 *
 * Port of a MOSEK example (see the comment below), rewritten against
 * the PrimalSolver (PRIMAL_*) API.  The MOSEK examples are Copyright (c)
 * MOSEK ApS; this port re-implements the same optimization problem and is
 * distributed under the Apache License, Version 2.0.  PrimalSolver is not
 * affiliated with, or endorsed by, MOSEK.
 */

/* portfolio_5.c — porting dell'esempio "portfolio_5_card.jl" della Julia
 * MOSEK API: ottimizzazione di portafoglio con vincolo di CARDINALITA'
 * (al piu' k asset con peso non nullo) — MIP con variabili binarie.
 *
 *   max  r'x - gamma*x'Sigma x
 *   s.t. sum(x) = 1,  0 <= x_j <= y_j,  y_j binaria,  sum(y) <= k
 *
 * Dati (tutorial MOSEK, n=3): r, GT, gamma = 0.05, k = 2.
 * Verifica: cardinalita' rispettata (al piu' 2 asset non nulli), bilancio,
 * e confronto con la versione senza cardinalita' (portfolio_1: obj ~
 * 0.119066): il vincolo k=2 riduce l'obiettivo.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const int n = 3;
    const int k = 2;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    const double GT[9] = {0.1667, 0.0, 0.0,
                          0.0247, 0.1592, 0.0,
                          0.0197, 0.0158, 0.1327};
    const double gamma = 0.05;

    double S[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            S[i * 3 + j] = 0.0;
            for (int k2 = 0; k2 < 3; k2++) S[i * 3 + j] += GT[i * 3 + k2] * GT[j * 3 + k2];
        }

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: x (3), y (3 binarie) */
    PRIMAL_appendvars(task, 2 * n);
    /* righe: bilancio, cardinalita' sum(y) <= k, x_j <= y_j (3) */
    PRIMAL_appendcons(task, 2 + n);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(task, j, r[j]);
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(task, n + j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(task, n + j, PRIMAL_VAR_TYPE_INT_BIN);
    }

    {   /* bilancio: sum x = 1 */
        int sub[3] = {0, 1, 2};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 0, 3, sub, v);
        PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* cardinalita': sum y <= k */
        int sub[3] = {3, 4, 5};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 1, 3, sub, v);
        PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, (double)k);
    }
    for (int j = 0; j < n; j++) {   /* x_j - y_j <= 0 */
        int sub[2] = {j, n + j};
        double v[2] = {1.0, -1.0};
        PRIMAL_putarow(task, 2 + j, 2, sub, v);
        PRIMAL_putconbound(task, 2 + j, PRIMAL_BK_UP, -INFINITY, 0.0);
    }

    {   /* obj = r'x - gamma x'Sx (Q = -2*gamma*S) */
        int sub[6] = {0, 0, 0, 1, 1, 2};
        int subj[6] = {0, 1, 2, 1, 2, 2};
        double v[6] = {
            -2.0 * gamma * S[0 * 3 + 0], -2.0 * gamma * S[0 * 3 + 1],
            -2.0 * gamma * S[0 * 3 + 2], -2.0 * gamma * S[1 * 3 + 1],
            -2.0 * gamma * S[1 * 3 + 2], -2.0 * gamma * S[2 * 3 + 2]};
        PRIMAL_putqobj(task, 6, sub, subj, v);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[6], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f)\n", xx[0], xx[1], xx[2]);
    printf("y = (%.0f, %.0f, %.0f)\n", xx[3], xx[4], xx[5]);
    printf("obj = %.6f\n", obj);

    /* verifica: bilancio, cardinalita' (n asset attivi <= 2),
     * x_j <= y_j, y binarie, obj <= versione senza cardinalita' */
    int nact = 0;
    for (int j = 0; j < n; j++) {
        if (xx[j] > 1e-6) nact++;
        if (xx[n + j] > 1e-6 && xx[j] > xx[n + j] + 1e-6) { nact = 99; }
        if (fabs(xx[n + j] - floor(xx[n + j] + 0.5)) > 1e-6) nact = 99;
    }
    int ok = fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-5 && nact <= k &&
             obj <= 0.119066 + 1e-6;
    printf("%s (bilancio, cardinalita' <= %d, obj <= 0.119066)\n",
           ok ? "OK" : "FAIL", k);

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
