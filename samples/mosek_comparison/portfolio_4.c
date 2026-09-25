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

/* portfolio_4.c — porting dell'esempio "portfolio_4_transcost.jl" della
 * MOSEK Julia API: ottimizzazione di portafoglio con costi di transazione
 * — versione MIQP (MIP + obiettivo quadratico): budget di cardinalita'
 * limitata con variabili binarie oltre ai costi lineari di transazione.
 *
 *   max  r'x - gamma*x'Sigma x - f' |x - x0|
 *   s.t. sum(x) = 1,  0 <= x_j <= y_j,  y binaria,  sum(y) <= k
 *
 * Dati (tutorial MOSEK, n=3): r, GT, gamma = 0.05, f = 0.01, x0 uniforme,
 * k = 3 (nessun limite effettivo con n=3, ma esercita le binarie).
 * |x - x0| lineare con variabili ausiliarie u_j (2 righe per j).
 * Verifica: bilancio, semi-logica delle binarie, obj <= portfolio_1.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const int n = 3;
    const int k = 3;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    const double GT[9] = {0.1667, 0.0, 0.0,
                          0.0247, 0.1592, 0.0,
                          0.0197, 0.0158, 0.1327};
    const double gamma = 0.05;
    const double f = 0.01;
    const double x0[3] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

    double S[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            S[i * 3 + j] = 0.0;
            for (int q = 0; q < 3; q++) S[i * 3 + j] += GT[i * 3 + q] * GT[j * 3 + q];
        }

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: x (0..2), u (3..5), y (6..8) */
    PRIMAL_appendvars(task, 3 * n);
    /* righe: bilancio, cardinalita', x_j <= y_j (n), |x_j-x0_j| <= u_j (2n) */
    PRIMAL_appendcons(task, 2 + n + 2 * n);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(task, j, r[j]);
        PRIMAL_putcj(task, n + j, -f);
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(task, n + j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(task, 2 * n + j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(task, 2 * n + j, PRIMAL_VAR_TYPE_INT_BIN);
    }

    {   /* bilancio: sum x = 1 */
        int sub[3] = {0, 1, 2};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 0, 3, sub, v);
        PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* cardinalita': sum y <= k */
        int sub[3] = {6, 7, 8};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 1, 3, sub, v);
        PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, (double)k);
    }
    for (int j = 0; j < n; j++) {   /* x_j - y_j <= 0 */
        int sub[2] = {j, 2 * n + j};
        double v[2] = {1.0, -1.0};
        PRIMAL_putarow(task, 2 + j, 2, sub, v);
        PRIMAL_putconbound(task, 2 + j, PRIMAL_BK_UP, -INFINITY, 0.0);
    }
    for (int j = 0; j < n; j++) {   /* |x_j - x0_j| <= u_j */
        int sub[2] = {j, n + j};
        double v1[2] = {1.0, -1.0};
        PRIMAL_putarow(task, 2 + n + j, 2, sub, v1);
        PRIMAL_putconbound(task, 2 + n + j, PRIMAL_BK_UP, -INFINITY, x0[j]);
        double v2[2] = {-1.0, -1.0};
        PRIMAL_putarow(task, 2 + n + n + j, 2, sub, v2);
        PRIMAL_putconbound(task, 2 + n + n + j, PRIMAL_BK_UP, -INFINITY, -x0[j]);
    }

    {   /* obj = r'x - f'u - gamma x'Sx (Q = -2*gamma*S) */
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

    double xx[9], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f)\n", xx[0], xx[1], xx[2]);
    printf("u = (%.4f, %.4f, %.4f)\n", xx[3], xx[4], xx[5]);
    printf("y = (%.0f, %.0f, %.0f)\n", xx[6], xx[7], xx[8]);
    printf("obj = %.6f\n", obj);

    int ok = fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-5;
    int nact = 0;
    for (int j = 0; j < n; j++) {
        if (xx[j] > 1e-6) nact++;
        if (fabs(xx[2 * n + j] - floor(xx[2 * n + j] + 0.5)) > 1e-6) ok = 0;
        if (xx[j] > xx[2 * n + j] + 1e-6) ok = 0;
    }
    if (nact > k) ok = 0;
    if (obj > 0.119066 + 1e-6) ok = 0;
    printf("%s (MIQP: bilancio, binarie, |x-x0|<=u, obj <= 0.119066)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
