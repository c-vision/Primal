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

/* solvebasis.c — porting dell'esempio "solvebasis.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): leggere una base da file e risolvere
 * partendo da quella base.
 *
 * Flusso (come l'esempio ufficiale):
 *   1. costruisce un LP noto e lo risolve (PRIMAL_optimize)
 *   2. scrive la base della soluzione ottimale (writebasis)
 *   3. costruisce un NUOVO task e legge la base (readbasis)
 *   4. risolve con PRIMAL_solvebasis e verifica che la soluzione coincida
 *
 * Problema (ottimo verificato a mano):  max 3x0 + x1 + 5x2 + x3
 *   s.t. 3x0 + x1 + 2x2 + x3  = 30   (r0, sempre attiva)
 *        2x0 + x1 + 3x2 + x3  >= 15  (r1)
 *               2x1 +     x3  <= 9   (r2)
 * con x >= 0, x3 in [0,5].
 * x2 domina (5 per 2 unita' di r0); ottimo x* = (0, 0, 15, 0), pobj = 75
 * (r1: 45 >= 15 ok, r2: 0 <= 9 ok). Base: x2 BAS, r0 attiva (LOW/UPR = EQ).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

static void build(PRIMALtask_t t) {
    PRIMAL_appendvars(t, 4);
    PRIMAL_appendcons(t, 3);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    double c[4] = {3, 1, 5, 1};
    for (int j = 0; j < 4; j++) PRIMAL_putcj(t, j, c[j]);
    for (int j = 0; j < 4; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_RA, 0.0, 5.0);
    PRIMAL_putarow(t, 0, 4, (int[]){0, 1, 2, 3}, (double[]){3, 1, 2, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 30.0, 30.0);
    PRIMAL_putarow(t, 1, 4, (int[]){0, 1, 2, 3}, (double[]){2, 1, 3, 1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, 15.0, INFINITY);
    PRIMAL_putarow(t, 2, 2, (int[]){1, 3}, (double[]){2, 1});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_UP, -INFINITY, 9.0);
}

/* ricava la base dalla soluzione: var BAS se strettamente tra i bound,
 * LOW/UPR se al bound; riga BAS (slack di base) se inattiva, LOW/UPR se
 * attiva (lato inferiore/superiore) */
static void basis_from_solution(PRIMALtask_t t, PRIMALstakeye *skc, PRIMALstakeye *skx) {
    int nvar, ncon;
    PRIMAL_getnumvar(t, &nvar);
    PRIMAL_getnumcon(t, &ncon);
    double *x = calloc((size_t)nvar, sizeof(double));
    PRIMAL_getxx(t, PRIMAL_SOL_BAS, x);
    for (int j = 0; j < nvar; j++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
        if (bk == PRIMAL_BK_FX) { skx[j] = PRIMAL_SK_BAS; continue; }
        if (fabs(x[j] - bl) < 1e-6) skx[j] = PRIMAL_SK_LOW;
        else if (fabs(x[j] - bu) < 1e-6) skx[j] = PRIMAL_SK_UPR;
        else skx[j] = PRIMAL_SK_BAS;
    }
    for (int i = 0; i < ncon; i++) {
        double aij, sum = 0.0;
        for (int j = 0; j < nvar; j++) {
            PRIMAL_getaij(t, i, j, &aij);
            sum += aij * x[j];
        }
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        if (fabs(sum - bl) < 1e-6) skc[i] = PRIMAL_SK_LOW;
        else if (fabs(sum - bu) < 1e-6) skc[i] = PRIMAL_SK_UPR;
        else skc[i] = PRIMAL_SK_BAS;
    }
    free(x);
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    /* 1. risolvi e ricava la base */
    PRIMALtask_t t1;
    PRIMAL_maketask(env, 0, 0, &t1);
    build(t1);
    PRIMALrescodee rc = PRIMAL_optimize(t1);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double po1;
    PRIMAL_getprimalobj(t1, PRIMAL_SOL_BAS, &po1);
    printf("fase 1: pobj = %.4f (atteso 75)\n", po1);

    int nvar, ncon;
    PRIMAL_getnumvar(t1, &nvar);
    PRIMAL_getnumcon(t1, &ncon);
    PRIMALstakeye *skc = calloc((size_t)ncon, sizeof(PRIMALstakeye));
    PRIMALstakeye *skx = calloc((size_t)nvar, sizeof(PRIMALstakeye));
    basis_from_solution(t1, skc, skx);

    /* 2. scrivi la base */
    rc = PRIMAL_putskc(t1, PRIMAL_SOL_BAS, skc);
    rc |= PRIMAL_putskx(t1, PRIMAL_SOL_BAS, skx);
    if (rc) { printf("putsk rc=%d\n", rc); return 1; }
    rc = PRIMAL_writebasis(t1, "/tmp/mc_solvebasis.bas");
    if (rc != PRIMAL_RES_OK) { printf("writebasis rc=%d\n", rc); return 1; }
    printf("fase 2: base scritta in /tmp/mc_solvebasis.bas\n");

    /* 3. nuovo task: legge la base e risolve con solvebasis */
    PRIMALtask_t t2;
    PRIMAL_maketask(env, 0, 0, &t2);
    build(t2);
    rc = PRIMAL_readbasis(t2, "/tmp/mc_solvebasis.bas");
    if (rc != PRIMAL_RES_OK) { printf("readbasis rc=%d\n", rc); return 1; }
    rc = PRIMAL_solvebasis(t2);
    if (rc != PRIMAL_RES_OK) { printf("solvebasis rc=%d\n", rc); return 1; }
    double po2, x2[4];
    PRIMAL_getprimalobj(t2, PRIMAL_SOL_BAS, &po2);
    PRIMAL_getxx(t2, PRIMAL_SOL_BAS, x2);
    printf("fase 3: solvebasis pobj = %.4f, x = (%.4f, %.4f, %.4f, %.4f)\n",
           po2, x2[0], x2[1], x2[2], x2[3]);

    int ok = fabs(po1 - 75.0) < 1e-6 && fabs(po2 - po1) < 1e-6 &&
             fabs(x2[0]) < 1e-6 && fabs(x2[1]) < 1e-6 &&
             fabs(x2[2] - 15.0) < 1e-5 && fabs(x2[3]) < 1e-6;
    printf("%s (atteso pobj=75, x=(0,0,15,0), base ripristinata)\n",
           ok ? "OK" : "FAIL");

    free(skc); free(skx);
    PRIMAL_deletetask(&t1);
    PRIMAL_deletetask(&t2);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
