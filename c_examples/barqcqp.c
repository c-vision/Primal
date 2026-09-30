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

/* barqcqp.c — symmetric bars and quadratic part (constraint and objective) in
 * the same model, solved by the QCQP-to-conic route through the public API
 * alone; plus the nonconvex domain, which must be refused.
 *
 * The optimal values are derived by hand below (no number copied from
 * another solver). A and B are the two forms the bar route used to silently
 * drop (T86); C is the form the QCQP encoder used to silently cut (T87).
 *
 * A) min -x0 - x1 - 4 <E00,B>
 *    s.t. x0^2 + x1^2 + <E00,B> <= 3,  <I,B> = 1,  B in S^2_+,  x >= 0
 *    With tr(B)=1 and B psd we have b = B00 in [0,1]; for fixed b the sphere
 *    is active at x0=x1=sqrt((3-b)/2) and it remains to maximize
 *    g(b) = 2 sqrt((3-b)/2) + 4b, with g'(b) = 4 - 1/(2 sqrt((3-b)/2)) >= 3.5
 *    on all of [0,1]. Hence b*=1 and s*=1, and
 *      x* = (1,1),  B = diag(1,0) (B00=1 and tr(B)=1 force B11=0 and, from
 *      det(B)>=0, B01=0),  obj = -(2 + 4) = -6.
 *
 * B) min x0^2 + x1^2 - 2 x0 - 2 x1 - 3 <E00,B>
 *    s.t. <I,B> = 1,  B in S^2_+,  x >= 0
 *    The two parts separate: the quadratic is (x0-1)^2 + (x1-1)^2 - 2,
 *    minimal at x*=(1,1) with value -2; the bar objective term is -3 b with
 *    b = B00 in [0,1], minimal at b=1, i.e. again B = diag(1,0).
 *      obj = -2 - 3 = -5.
 *    Here it is the quadratic objective term that must move the answer:
 *    if the bar route drops it, -x0-x1 on x >= 0 remains, i.e. an
 *    unbounded model. The number checked below is therefore a certificate
 *    of the whole encoding, not a detail.
 *
 * C) as A but with constraint x0^2 - 2 x1^2 + <E00,B> <= 3 and 0 <= x <= 2.
 *    The quadratic matrix diag(2,-4) has an eigenvalue on the wrong side: the
 *    domain is not convex and no cone can represent it. An encoder that
 *    cuts leaves only x0^2 + <E00,B> <= 3 and answers a number on
 *    somebody else's model. Here the refusal is measured: rc = ERR_ARG, and
 *    the solution getters say "no" with the same code.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* E00 and the 2x2 identity, shared by the three cases */
static void put_syms(PRIMALtask_t t, int *m00, int *mI) {
    PRIMAL_appendsparsesymmat(t, 2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, m00);
    PRIMAL_appendsparsesymmat(t, 2, 2, (int[]){0, 1}, (int[]){0, 1},
                              (double[]){1.0, 1.0}, mI);
}

/* Solve cases A (bar+quadratic row), B (bar+quadratic objective) and C (refusal). */
int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    int pass = 1;

    /* ---- A) quadratic row and bar in the same row, bar also in objective ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 2);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, 0, -1.0);
        PRIMAL_putcj(t, 1, -1.0);
        double bobj = -4.0;
        PRIMAL_putbarcj(t, 0, 1, &m00, &bobj);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 3.0);
        PRIMAL_putqconk(t, 0, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
        PRIMAL_putbaraij(t, 0, 0, 1, &m00, (double[]){1.0});
        PRIMAL_putbaraij(t, 1, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, B[4] = {0, 0, 0, 0}, po = 0.0;
        int ok = (PRIMAL_optimize(t) == PRIMAL_RES_OK) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == PRIMAL_RES_OK) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == PRIMAL_RES_OK) &&
                 (PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, B) == PRIMAL_RES_OK);
        ok = ok && fabs(x[0] - 1.0) < 1e-4 && fabs(x[1] - 1.0) < 1e-4 &&
             fabs(po + 6.0) < 1e-5 &&
             fabs(B[0] - 1.0) < 1e-5 && fabs(B[1]) < 1e-5 && fabs(B[3]) < 1e-5;
        printf("barqcqp A  obj=%.6f (expected -6)  x=(%.6f, %.6f)  "
               "B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
               po, x[0], x[1], B[0], B[1], B[2], B[3], ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    /* ---- B) convex quadratic objective + bar ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 1);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, 0, -2.0);
        PRIMAL_putcj(t, 1, -2.0);
        PRIMAL_putqobj(t, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
        double bobj = -3.0;
        PRIMAL_putbarcj(t, 0, 1, &m00, &bobj);
        PRIMAL_putbaraij(t, 0, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, B[4] = {0, 0, 0, 0}, po = 0.0;
        int ok = (PRIMAL_optimize(t) == PRIMAL_RES_OK) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == PRIMAL_RES_OK) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == PRIMAL_RES_OK) &&
                 (PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, B) == PRIMAL_RES_OK);
        ok = ok && fabs(x[0] - 1.0) < 1e-4 && fabs(x[1] - 1.0) < 1e-4 &&
             fabs(po + 5.0) < 1e-5 &&
             fabs(B[0] - 1.0) < 1e-5 && fabs(B[1]) < 1e-5 && fabs(B[3]) < 1e-5;
        printf("barqcqp B  obj=%.6f (expected -5)  x=(%.6f, %.6f)  "
               "B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
               po, x[0], x[1], B[0], B[1], B[2], B[3], ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    /* ---- C) nonconvex domain: refusal, not an answer ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 2);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 2.0);
        PRIMAL_putcj(t, 0, -1.0);
        PRIMAL_putcj(t, 1, -1.0);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 3.0);
        PRIMAL_putqconk(t, 0, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, -4.0});
        PRIMAL_putbaraij(t, 0, 0, 1, &m00, (double[]){1.0});
        PRIMAL_putbaraij(t, 1, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, po = 0.0;
        int rc = (int)PRIMAL_optimize(t);
        int ok = (rc == (int)PRIMAL_RES_ERR_ARG) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == (int)PRIMAL_RES_ERR_ARG) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == (int)PRIMAL_RES_ERR_ARG);
        printf("barqcqp C  rc=%d (expected %d = ERR_ARG)  obj not published  %s\n",
               rc, (int)PRIMAL_RES_ERR_ARG, ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    PRIMAL_deleteenv(&env);
    return pass ? 0 : 1;
}
