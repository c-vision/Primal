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

/* sharpe_ratio.c - the max-Sharpe case (A3) of arXiv:2508.03704.
 *
 * Source: "Novel Risk Measures for Portfolio Optimization Using
 * Equal-Correlation Portfolio Strategy", arXiv:2508.03704, Section 4.4. The
 * convex cases are solved with MOSEK; case A3 maximizes the Sharpe ratio
 *
 *   max (mu'w - rf) / sqrt(w'Sigma w)   s.t.  e'w = 1,
 *
 * which is a non-convex RATIO of a linear and a quadratic function, not a
 * quadratic program. The standard convex reformulation is Charnes-Cooper:
 * with kappa = 1/sqrt(w'Sigma w) > 0 and y = kappa*w,
 *
 *   max mu'y - rf*kappa   s.t.  y'Sigma y <= 1,   e'y = kappa,   kappa >= 0,
 *
 * a quadratic constraint and a linear objective, and w = y/kappa recovers the
 * portfolio. The case here is hand-derived: mu = (3,1), rf = 0, Sigma = I.
 * The unconstrained max Sharpe is ||mu|| = sqrt(10), attained at w parallel to
 * mu; with e'w = 1 that is w = (3/4, 1/4).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double mu[2] = {3.0, 1.0};
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 3);   /* y0, y1, kappa */
    PRIMAL_appendcons(task, 2);   /* 0: y'Sigma y <= 1, 1: e'y - kappa = 0 */
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, 2, PRIMAL_BK_LO, 0.0, INFINITY);
    /* Sigma = I: 1/2 x'Qx with Q = 2I is y0^2 + y1^2 */
    PRIMAL_putqconk(task, 0, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, -INFINITY, 1.0);
    PRIMAL_putarow(task, 1, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, -1.0});
    PRIMAL_putconbound(task, 1, PRIMAL_BK_FX, 0.0, 0.0);
    PRIMAL_putcj(task, 0, mu[0]);
    PRIMAL_putcj(task, 1, mu[1]);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double sharpe = 0.0, x[4] = {0};
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &sharpe);
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
    double w0 = x[0] / x[2], w1 = x[1] / x[2];
    double norm = sqrt(w0 * w0 + w1 * w1);
    double ratio = (mu[0] * w0 + mu[1] * w1) / norm;

    int ok = fabs(sharpe - sqrt(10.0)) < 1e-6 &&
             fabs(w0 - 0.75) < 1e-5 && fabs(w1 - 0.25) < 1e-5 &&
             fabs(ratio - sharpe) < 1e-5 && fabs(w0 + w1 - 1.0) < 1e-6;
    printf("Sharpe = %.6f (atteso sqrt(10) = %.6f), w = (%.4f, %.4f) (atteso .75,.25)\n",
           sharpe, sqrt(10.0), w0, w1);
    printf("ricostruito (mu'w)/||w|| = %.6f, e'w = %.2e\n", ratio, w0 + w1);
    printf("%s (A3 non e' quadratico: risolto con Charnes-Cooper)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
