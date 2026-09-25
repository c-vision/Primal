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

/* omf_ex2001.c - Example 20.1 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 20.4 ("Relative robustness in
 * portfolio selection").
 *
 * Mean-variance portfolio with a tracking-error constraint:
 *
 *   maximize  mu1 x1 + mu2 x2 + mu3 x3
 *   s.t.      TE(x1,x2,x3) <= 0.10        (tracking error vs the 50/50 benchmark)
 *             x1 + x2 + x3 = 1            (budget)
 *             x >= 0
 *
 *   TE = sqrt( (x-0.5)' Sigma (x-0.5) ),  Sigma = [[0.1764, 0.09702, 0],
 *                                                  [0.09702, 0.1089, 0],
 *                                                  [0, 0, 0]]
 * (x3 is the uninvested fraction; the benchmark is x1=x2=0.5).
 *
 * The book states: for mu = (6,4,0) the optimum is (0.831, 0.169, 0) with
 * objective 5.662; for mu = (4,6,0) it is (0.169, 0.831, 0), objective 5.662;
 * for mu = (5,5,0) the objective is 5.0.
 *
 * TE is a second-order cone: with the Cholesky factor L of the 2x2 covariance
 * block (Sigma = L L'), TE = || L' (x1-0.5, x2-0.5) ||_2, so the constraint is
 * (0.10, r1, r2) in QUAD with r = L' (x1-0.5, x2-0.5).
 *
 * The sample solves the mu = (6,4,0) and mu = (5,5,0) cases and checks the
 * book's optima.
 *
 * Usage: omf_ex2001   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* 2x2 covariance block and its Cholesky factor (Sigma2 = L L') */
static const double S11 = 0.1764, S12 = 0.09702, S22 = 0.1089;
static double l11, l21, l22;

static int solve(const char *name, double mu1, double mu2,
                 double want_obj, double wx1, double wx2) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* vars: 0=x1 1=x2 2=x3 3=r1 4=r2 5=t(=0.10) ; rows: 0=budget, 1=r1, 2=r2 */
    PRIMAL_maketask(env, 3, 6, &t);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, 3, PRIMAL_BK_FR, -INFINITY, INFINITY);   /* r1 */
    PRIMAL_putvarbound(t, 4, PRIMAL_BK_FR, -INFINITY, INFINITY);   /* r2 */
    PRIMAL_putvarbound(t, 5, PRIMAL_BK_FX, 0.10, 0.10);            /* TE budget */
    PRIMAL_putcj(t, 0, -mu1); PRIMAL_putcj(t, 1, -mu2);            /* maximize mu'x */
    /* budget */
    PRIMAL_putarow(t, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* r1 = l11 (x1-0.5) + l21 (x2-0.5)  ->  r1 - l11 x1 - l21 x2 = -0.5(l11+l21) */
    PRIMAL_putarow(t, 1, 3, (int[]){0, 1, 3}, (double[]){-l11, -l21, 1.0});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, -0.5 * (l11 + l21), -0.5 * (l11 + l21));
    /* r2 = l22 (x2-0.5)                 ->  r2 - l22 x2 = -0.5 l22 */
    PRIMAL_putarow(t, 2, 2, (int[]){1, 4}, (double[]){-l22, 1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, -0.5 * l22, -0.5 * l22);
    /* (t, r1, r2) in QUAD  <=>  t >= ||(r1,r2)|| */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){5, 3, 4});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[6];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double d1 = x[0] - 0.5, d2 = x[1] - 0.5;
        double te = sqrt(S11 * d1 * d1 + 2.0 * S12 * d1 * d2 + S22 * d2 * d2);
        /* wx1 < 0 means the case has a continuum of optima (the book says so):
         * check only the objective and the constraints, not a specific point. */
        int xcheck = (wx1 < 0.0) || (fabs(x[0] - wx1) < 2e-3 && fabs(x[1] - wx2) < 2e-3);
        ok = fabs(z + want_obj) < 1e-3 && xcheck && te <= 0.10 + 1e-6 &&
             fabs(x[0] + x[1] + x[2] - 1.0) < 1e-6;
        printf("omf_ex2001 %-10s x=(%.4f,%.4f,%.4f) TE=%.4f obj=%.4f (book %.4f) %s\n",
               name, x[0], x[1], x[2], te, -z, want_obj, ok ? "OK" : "FAIL");
    } else {
        printf("omf_ex2001 %-10s rc=%d FAIL\n", name, (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    l11 = sqrt(S11);
    l21 = S12 / l11;
    l22 = sqrt(S22 - l21 * l21);
    int a = solve("mu=(6,4,0)", 6.0, 4.0, 5.662, 0.831, 0.169);
    int b = solve("mu=(5,5,0)", 5.0, 5.0, 5.000, -1.0, -1.0);   /* continuum of optima */
    return (a && b) ? 0 : 1;
}
