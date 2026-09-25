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

/* factor_model.c - minimum-variance portfolio under a factor risk model.
 *
 * The covariance is never formed: Sigma = F F' + diag(D), so
 *   w'Sigma w = ||F'w||^2 + sum_j D_j w_j^2,
 * and the minimum-variance portfolio is a QP on that operator. The sample
 * builds the QP with the explicit Sigma = F F' + diag(D) (so the operator is
 * visible) and cross-checks the optimum against a hand value.
 *
 * Hand case: N = 3, one factor F = (1,1,0), D = (0,0,1), so
 *   Sigma = [[1,1,0],[1,1,0],[0,0,1]].
 * With e'w = 1, w'Sigma w = (w0+w1)^2 + w2^2 = (1-w2)^2 + w2^2, minimised at
 * w2 = 1/2 with value 1/2 (any split of w0+w1 = 1/2).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    static const double F[3] = {1.0, 1.0, 0.0};
    static const double D[3] = {0.0, 0.0, 1.0};
    double Sigma[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Sigma[i][j] = F[i] * F[j] + (i == j ? D[i] : 0.0);

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 3);
    PRIMAL_appendcons(task, 1);
    for (int j = 0; j < 3; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    /* e'w = 1 */
    PRIMAL_putarow(task, 0, 3, (int[]){0, 1, 2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(task, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* objective w'Sigma w: 1/2 w'Qw with Q = 2 Sigma */
    for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++)
            PRIMAL_putqobj(task, 1, (int[]){i}, (int[]){j},
                           (double[]){2.0 * Sigma[i][j]});

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    double obj = 0.0, w[4] = {0};
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, w);
    /* the operator, rebuilt from the published w */
    double quad = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) quad += w[i] * Sigma[i][j] * w[j];
    int ok = fabs(obj - 0.5) < 1e-7 && fabs(w[2] - 0.5) < 1e-6 &&
             fabs(w[0] + w[1] - 0.5) < 1e-6 && fabs(w[0] + w[1] + w[2] - 1.0) < 1e-6 &&
             fabs(quad - obj) < 1e-7;
    printf("w = (%.4f, %.4f, %.4f), obj = %.6f (atteso 0.5)\n", w[0], w[1], w[2], obj);
    printf("w'Sigma w ricostruito = %.6f, e'w = %.2e\n", quad, w[0] + w[1] + w[2]);
    printf("%s (Sigma = F F' + diag(D), minimo a w2 = 1/2)\n", ok ? "OK" : "FAIL");
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
