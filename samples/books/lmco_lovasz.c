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

/* lmco_lovasz.c - Lovasz capacity number of C5, "Lectures on Modern Convex
 * Optimization 2020-2023" (Ben-Tal & Nemirovski), lines 6699-6725:
 *
 * With the 5-node cycle C5, associate the affine matrix
 *   L(x) = [ 1     xAB   1     1     xEA
 *            xAB   1     xBC   1     1
 *            1     xBC   1     xCD   1
 *            1     1     xCD   1     xDE
 *            xEA   1     1     xDE   1 ]
 * (entry 1 between non-adjacent nodes, the arc variable between adjacent
 * nodes). The Lovasz capacity number is
 *   theta(C5) = min_{lambda,x} { lambda : lambda I - L(x) >= 0 },
 * a semidefinite program. The book states theta(C5) = sqrt(5) exactly.
 *
 * Model: bar variable X (5x5, PSD) with one equality row per entry (i,j),
 * i <= j: <E_ij,X> = (lambda I - L(x))_ij.  On the diagonal that is
 * lambda - 1; on an arc x_arc it is -x_arc (times the doubled off-diagonal
 * <E_ij,X> = 2 X_ij); on a non-arc it is -1.
 *
 * Usage: lmco_lovasz   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* index of the arc variable for pair (i,j), i<j; -1 if non-adjacent */
static int arcidx(int i, int j) {
    static const int map[5][5] = {
        {-1, 1, -1, -1, 5},
        {-1, -1, 2, -1, -1},
        {-1, -1, -1, 3, -1},
        {-1, -1, -1, -1, 4},
        {-1, -1, -1, -1, -1}};
    return map[i][j];
}

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    int msym[15], mcnt = 0, row = 0;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    /* vars: 0 = lambda, 1..5 = xAB, xBC, xCD, xDE, xEA */
    PRIMAL_appendvars(t, 6);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 1.0, INFINITY);
    for (int k = 1; k < 6; k++) PRIMAL_putvarbound(t, k, PRIMAL_BK_RA, -10.0, 10.0);
    PRIMAL_putcj(t, 0, 1.0);   /* min lambda */
    for (int i = 0; i < 5; i++)
        for (int j = i; j < 5; j++) {
            int si[1] = {i}, sj[1] = {j};
            double sv[1] = {1.0};
            PRIMAL_appendsparsesymmat(t, 5, 1, si, sj, sv, &msym[mcnt++]);
        }
    int dim = 5;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, 15);
    for (int i = 0; i < 5; i++)
        for (int j = i; j < 5; j++) {
            PRIMAL_putbaraij(t, row, 0, 1, (int[]){msym[row]}, (double[]){1.0});
            if (i == j) {
                /* <E_ii,X> = lambda - 1 */
                PRIMAL_putarow(t, row, 1, (int[]){0}, (double[]){-1.0});
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, -1.0, -1.0);
            } else if (arcidx(i, j) >= 0) {
                /* <E_ij,X> = 2X_ij = -2 x_arc */
                PRIMAL_putarow(t, row, 1, (int[]){arcidx(i, j)}, (double[]){2.0});
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0);
            } else {
                /* non-arc: <E_ij,X> = -2 */
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, -2.0, -2.0);
            }
            row++;
        }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[6];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(x[0] - sqrt(5.0)) < 1e-6;
        printf("lmco_lovasz  theta(C5)=%.9g (book sqrt(5)=%.9g) %s\n",
               x[0], sqrt(5.0), ok ? "OK" : "FAIL");
    } else {
        printf("lmco_lovasz  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
