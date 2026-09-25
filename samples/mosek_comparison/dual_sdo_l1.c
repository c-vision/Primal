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

/* dual_sdo_l1.c — porting dell'esempio "dual_sdo_l1.jl" (C API
 * docs.mosek.com): dualita' SDP — il duale di
 *   min <C, X>  s.t.  <A_i, X> = b_i (i=1..m),  X PSD
 * e'  max b'y  s.t.  S = C - sum_i y_i A_i PSD.
 * L'esempio costruisce il duale come problema SEPARATO e verifica la
 * strong duality <C, X*> = b'y* tramite il solver del clone.
 *
 * Primal (piccolo, verificato a mano):  min X00  s.t.  X11 = 1, X PSD (2x2)
 *   -> ottimo X* = diag(0, 1), pobj = 0.
 * Duale: max y (b_1 = 1, A_1 = E_11, C = E_00):
 *   max y  s.t.  S = C - y A_1 = diag(0, -y) PSD  ->  -y >= 0 -> y <= 0
 *   -> ottimo y* = 0, dobj = 0. Strong duality: 0 = 0 ✓.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static double primal_sdp(double *Xout) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FX, 0.0, 0.0);   /* dummy scalar */
    int mC, mA;
    PRIMAL_appendsparsesymmat(task, 2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &mC);
    PRIMAL_appendsparsesymmat(task, 2, 1, (int[]){1}, (int[]){1}, (double[]){1.0}, &mA);
    int dim = 2;
    PRIMAL_appendbarvars(task, 1, &dim);
    PRIMAL_putbarcj(task, 0, 1, (int[]){mC}, (double[]){1.0});   /* min X00 */
    PRIMAL_appendcons(task, 1);
    PRIMAL_putbaraij(task, 0, 0, 1, (int[]){mA}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);           /* X11 = 1 */
    double po = -1e30;
    if (PRIMAL_optimize(task) == PRIMAL_RES_OK) {
        double X[4];
        PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, X);
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
        for (int k = 0; k < 4; k++) Xout[k] = X[k];
    }
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return po;
}

static double dual_sdp(double *y_out, double *S_out) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    /* max y s.t. S = C - y A_1 PSD: S bar 2x2 con
     * S_00 = 0, S_01 = 0, S_11 = -y, y <= 0 */
    PRIMAL_appendvars(task, 1);   /* y */
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);   /* max y (b_1 = 1) */
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    int dim = 2;
    PRIMAL_appendbarvars(task, 1, &dim);
    int mC, mA, mO;
    PRIMAL_appendsparsesymmat(task, 2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &mC);
    PRIMAL_appendsparsesymmat(task, 2, 1, (int[]){1}, (int[]){1}, (double[]){1.0}, &mA);
    PRIMAL_appendsparsesymmat(task, 2, 1, (int[]){0}, (int[]){1}, (double[]){1.0}, &mO);
    PRIMAL_appendcons(task, 3);
    /* S_00 = 0 (da C - 0) */
    PRIMAL_putbaraij(task, 0, 0, 1, (int[]){mC}, (double[]){1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 0.0, 0.0);
    /* S_01 = 0 */
    PRIMAL_putbaraij(task, 1, 0, 1, (int[]){mO}, (double[]){1.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* S_11 = -y: <mA, S> + y = 0 */
    PRIMAL_putbaraij(task, 2, 0, 1, (int[]){mA}, (double[]){1.0});
    PRIMAL_putarow(task, 2, 1, (int[]){0}, (double[]){1.0});
    PRIMAL_putconbound(task, 2, PRIMAL_BK_FX, 0.0, 0.0);
    double dobj = -1e30, y = 0;
    if (PRIMAL_optimize(task) == PRIMAL_RES_OK) {
        double S[4];
        PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, S);
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &dobj);
        double yy[1];
        PRIMAL_getxx(task, PRIMAL_SOL_ITR, yy);
        y = yy[0];
        for (int k = 0; k < 4; k++) S_out[k] = S[k];
    }
    *y_out = y;
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return dobj;
}

int main(void) {
    double X[4], S[4], y;
    double pobj = primal_sdp(X);
    double dobj = dual_sdp(&y, S);
    printf("primale: pobj = %.6f (atteso 0), X = diag(%.4f, %.4f)\n",
           pobj, X[0], X[3]);
    printf("duale:   dobj = %.6f (atteso 0), y = %.4f, S = diag(%.4f, %.4f)\n",
           dobj, y, S[0], S[3]);

    int ok = fabs(pobj) < 1e-6 && fabs(dobj) < 1e-6 &&
             fabs(pobj - dobj) < 1e-6 &&               /* strong duality */
             fabs(X[0]) < 1e-4 && fabs(X[3] - 1.0) < 1e-4 &&
             fabs(S[0]) < 1e-4 && fabs(S[3]) < 1e-4;   /* S = diag(0,0) PSD */
    printf("%s (strong duality: <C,X*> = b'y* = 0)\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
