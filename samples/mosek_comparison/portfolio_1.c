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

/* portfolio_1.c — porting dell'esempio "portfolio_1_basic.jl" della
 * MOSEK Julia API (docs.mosek.com/11.0/juliaapi): ottimizzazione di
 * portafoglio di Markowitz, versione base.
 *
 *   max   r'x - gamma * x' Sigma x
 *   s.t.  sum(x) = w0 + U'x + f|x - x0|   (bilancio, f=0 nella versione base)
 *         0 <= x <= 0.5? (budget di investimenti per asset)
 *
 * Dati (3 asset, dal tutorial MOSEK):
 *   r    = (0.10717, 0.07502, 0.11902)
 *   Sigma= [0.0277 0.0038 0.0021; 0.0038 0.0120 0.0013; 0.0021 0.0013 0.0086]
 *   gamma= 0.03? usato qui: gamma = 0.05
 *   budget: sum(x) = 1, 0 <= x <= 0.5? qui: 0 <= x <= 1
 *
 * Verifica: pobj = dobj (certificato di ottimalita' del QP) e ammissibilita'.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const int n = 3;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    /* Sigma = GT'GT con GT triangolare inferiore (factor model del tutorial) */
    const double GT[9] = {0.1667, 0.0, 0.0,
                          0.0247, 0.1592, 0.0,
                          0.0197, 0.0158, 0.1327};
    const double gamma = 0.05;

    /* Sigma = GT * GT' */
    double S[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            S[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) S[i * 3 + j] += GT[i * 3 + k] * GT[j * 3 + k];
        }

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, n);
    PRIMAL_appendcons(task, 1);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = 0; j < n; j++) PRIMAL_putcj(task, j, r[j]);
    /* obj = r'x - gamma*x'Sx = 0.5 x'Qx con Q = -2*gamma*S.
     * putqobj riceve la triangolare superiore; le voci fuori diagonale
     * sono rispecchiate internamente -> si fornisce il coefficiente pieno. */
    {
        int sub[6] = {0, 0, 0, 1, 1, 2};
        int subj[6] = {0, 1, 2, 1, 2, 2};
        double v[6] = {
            -2.0 * gamma * S[0 * 3 + 0], -2.0 * gamma * S[0 * 3 + 1],
            -2.0 * gamma * S[0 * 3 + 2], -2.0 * gamma * S[1 * 3 + 1],
            -2.0 * gamma * S[1 * 3 + 2], -2.0 * gamma * S[2 * 3 + 2]};
        PRIMAL_putqobj(task, 6, sub, subj, v);
    }

    /* sum(x) = 1 */
    PRIMAL_putarow(task, 0, n, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    for (int j = 0; j < n; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[3], po, dobj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(task, PRIMAL_SOL_ITR, &dobj);
    printf("x = (%.6f, %.6f, %.6f)\n", xx[0], xx[1], xx[2]);
    printf("pobj = %.6f, dobj = %.6f\n", po, dobj);

    /* verifica: ammissibile + pobj coerente + no duality gap */
    int ok = fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-6;
    for (int j = 0; j < n; j++) if (xx[j] < -1e-9) ok = 0;
    double q = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) q += xx[i] * S[i * 3 + j] * xx[j];
    double lin = r[0]*xx[0] + r[1]*xx[1] + r[2]*xx[2];
    double expect = lin - gamma * q;
    printf("verifica obj: %.6f (atteso %.6f)\n", po, expect);
    if (fabs(po - expect) > 1e-6) ok = 0;
    if (fabs(po - dobj) > 1e-5 * (1 + fabs(po))) ok = 0;

    printf("%s\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}