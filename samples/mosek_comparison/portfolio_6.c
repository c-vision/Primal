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

/* portfolio_6.c — porting dell'esempio "portfolio_6_factor.jl" della
 * MOSEK Julia API: ottimizzazione di portafoglio con modello a FATTORI:
 * la covarianza e' x'(F P F' + D)x con fattori f = F'x, P = diag(p) e
 * rischio specifico D (diagonale). Qui: rischio via fattori con cono QUAD
 * + MIP (binarie di cardinalita' come portfolio_5 ma con fattori).
 *
 *   max  r'x - gamma*(||p^{1/2} F'x||^2 + x'Dx)
 *   s.t. sum(x) = 1, 0 <= x <= y, y binaria, sum(y) <= k
 *
 * F (3x2): 2 fattori; p = (0.05, 0.06); D = diag(0.01, 0.02, 0.005).
 * Il rischio fattoriale ||p^{1/2} F'x||^2 e' modellato con variabile v e
 * cono QUAD (v, z1, z2) con z = p^{1/2} F'x (righe di uguaglianza):
 * v >= ||z|| => v^2 >= z1^2+z2^2 -> nella forma clone: (v, z1, z2) in QUAD.
 * La parte x'Dx resta quadratica (putqobj). Verifica: bilancio, binarie,
 * obj coerente con la valutazione diretta del rischio in x*.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const int n = 3;      /* asset */
    const int m = 2;      /* fattori */
    const int k = 2;
    const double r[3] = {0.10717, 0.07502, 0.11902};
    const double F[3][2] = {{0.16, -0.23}, {0.28, 0.19}, {-0.11, 0.32}};
    const double p[2] = {0.05, 0.06};
    const double D[3] = {0.01, 0.02, 0.005};
    const double gamma = 0.05;

    /* variabili: x(0..2), v=3, z(4..5), y(6..8) */
    const int VV = 3, VZ = 4, VY = 6;
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, 9);
    /* righe: bilancio, cardinalita', x<=y (3), z_i = sqrt(p) F'_i x (2) */
    PRIMAL_appendcons(task, 2 + n + m);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    for (int j = 0; j < n; j++) {
        PRIMAL_putcj(task, j, r[j]);
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMAL_putcj(task, VV, -gamma);                 /* -gamma * v */
    PRIMAL_putvarbound(task, VV, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < m; i++)
        PRIMAL_putvarbound(task, VZ + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 0; j < n; j++) {
        PRIMAL_putvarbound(task, VY + j, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(task, VY + j, PRIMAL_VAR_TYPE_INT_BIN);
    }

    {   /* bilancio */
        int sub[3] = {0, 1, 2};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 0, 3, sub, v);
        PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* cardinalita' */
        int sub[3] = {VY, VY + 1, VY + 2};
        double v[3] = {1.0, 1.0, 1.0};
        PRIMAL_putarow(task, 1, 3, sub, v);
        PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, (double)k);
    }
    for (int j = 0; j < n; j++) {   /* x_j <= y_j */
        int sub[2] = {j, VY + j};
        double v[2] = {1.0, -1.0};
        PRIMAL_putarow(task, 2 + j, 2, sub, v);
        PRIMAL_putconbound(task, 2 + j, PRIMAL_BK_UP, -INFINITY, 0.0);
    }
    for (int i = 0; i < m; i++) {   /* z_i = sqrt(p_i) * (F'x)_i */
        int sub[4] = {0, 1, 2, VZ + i};
        double v[4] = {-sqrt(p[i]) * F[0][i], -sqrt(p[i]) * F[1][i],
                      -sqrt(p[i]) * F[2][i], 1.0};
        PRIMAL_putarow(task, 2 + n + i, 4, sub, v);
        PRIMAL_putconbound(task, 2 + n + i, PRIMAL_BK_FX, 0.0, 0.0);
    }

    /* cono: (v, z1, z2) in QUAD -> v >= ||z|| */
    PRIMAL_appendcone(task, PRIMAL_CT_QUAD, 0.0, 3, (int[]){VV, VZ, VZ + 1});

    {   /* parte specifica: -gamma * x'Dx (Q = -2*gamma*D) */
        int sub[3] = {0, 1, 2};
        int subj[3] = {0, 1, 2};
        double v[3] = {-2.0 * gamma * D[0], -2.0 * gamma * D[1],
                       -2.0 * gamma * D[2]};
        PRIMAL_putqobj(task, 3, sub, subj, v);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[9], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    printf("x = (%.4f, %.4f, %.4f)\n", xx[0], xx[1], xx[2]);
    printf("v = %.4f, y = (%.0f, %.0f, %.0f)\n", xx[VV], xx[VY], xx[VY + 1], xx[VY + 2]);
    printf("obj = %.6f\n", obj);

    /* verifica indipendente: rischio in x* da F, P, D e obj = r'x - gamma*rischio.
     * L'IPM conico lascia un piccolo residuo sul bordo del cono (v >= ||z||):
     * v puo' restare leggermente sopra ||z||, quindi obj_solver <= exp_obj
     * con scarto ~ gamma*(v - ||z||). Tolleranza 5e-3 su entrambi i lati. */
    double fx[2] = {0.0, 0.0};
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) fx[i] += F[j][i] * xx[j];
    double risk = 0.0;
    for (int i = 0; i < m; i++) risk += p[i] * fx[i] * fx[i];
    for (int j = 0; j < n; j++) risk += D[j] * xx[j] * xx[j];
    double exp_obj = 0.0;
    for (int j = 0; j < n; j++) exp_obj += r[j] * xx[j];
    exp_obj -= gamma * risk;
    printf("rischio diretto = %.6f, obj atteso = %.6f\n", risk, exp_obj);

    int ok = fabs(xx[0] + xx[1] + xx[2] - 1.0) < 1e-5 &&
             obj >= exp_obj - 5e-3 && obj <= exp_obj + 5e-3;
    int nact = 0;
    for (int j = 0; j < n; j++) {
        if (xx[j] > 1e-6) nact++;
        if (fabs(xx[VY + j] - floor(xx[VY + j] + 0.5)) > 1e-6) ok = 0;
        if (xx[j] > xx[VY + j] + 1e-6) ok = 0;
    }
    if (nact > k) ok = 0;
    printf("%s (fattori+MIP: obj = r'x - gamma*risk, cardinalita' <= %d)\n",
           ok ? "OK" : "FAIL", k);

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
