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

/* boyd_corr_sdp.c - bounding a correlation coefficient, "Convex Optimization"
 * (Boyd & Vandenberghe, 2004), Example 8.3 (lines 12910-12922):
 *
 * Find the minimum and maximum possible value of rho_14 given
 *   0.6 <= rho_12 <= 0.9,  0.8 <= rho_13 <= 0.9,
 *   0.5 <= rho_24 <= 0.7, -0.8 <= rho_34 <= -0.4,
 * and the symmetric 4x4 matrix with unit diagonal and off-diagonal rho_ij
 * positive semidefinite (an SDP). The book reports (2 significant digits)
 *   min rho_14 = -0.39,  max rho_14 = 0.23.
 *
 * Model: bar variable X (4x4, PSD) with rows <E_ij,X> = 2 rho_ij (off-diag)
 * and <E_ii,X> = 1; rho_ij are the scalar variables.
 *
 * Usage: boyd_corr_sdp   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* index of the correlation variable for pair (i,j), i<j */
static int ridx(int i, int j) {
    static const int map[4][4] = {
        {-1, 0, 1, 2},
        {-1, -1, 3, 4},
        {-1, -1, -1, 5},
        {-1, -1, -1, -1}};
    return map[i][j];
}

static int solve(double sgn, double *rho14, double *rho_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    int msym[10], cnt = 0;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    /* correlation variables rho_12, rho_13, rho_14, rho_23, rho_24, rho_34 */
    PRIMAL_appendvars(t, 6);
    static const double lo[6] = {0.6, 0.8, -INFINITY, -INFINITY, 0.5, -0.8};
    static const double up[6] = {0.9, 0.9,  INFINITY,  INFINITY, 0.7, -0.4};
    for (int k = 0; k < 6; k++) PRIMAL_putvarbound(t, k, PRIMAL_BK_RA, lo[k], up[k]);
    PRIMAL_putcj(t, 2, sgn);   /* minimize (sgn=+1) or maximize (-1) rho_14 */
    for (int i = 0; i < 4; i++)
        for (int j = i; j < 4; j++) {
            int si[1] = {i}, sj[1] = {j};
            double sv[1] = {1.0};
            PRIMAL_appendsparsesymmat(t, 4, 1, si, sj, sv, &msym[cnt++]);
        }
    int dim = 4;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, 10);
    int row = 0; cnt = 0;
    for (int i = 0; i < 4; i++)
        for (int j = i; j < 4; j++) {
            PRIMAL_putbaraij(t, row, 0, 1, (int[]){msym[cnt]}, (double[]){1.0});
            if (i == j) {
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0);
            } else {
                PRIMAL_putarow(t, row, 1, (int[]){ridx(i, j)}, (double[]){-2.0});
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0);
            }
            row++; cnt++;
        }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[6];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        *rho14 = x[2];
        if (rho_out) for (int k = 0; k < 6; k++) rho_out[k] = x[k];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double lo14 = 0.0, hi14 = 0.0;
    int ok = solve(+1.0, &lo14, NULL) && solve(-1.0, &hi14, NULL);
    ok = ok && fabs(lo14 + 0.39) < 0.01 && fabs(hi14 - 0.23) < 0.01;
    printf("boyd_corr_sdp  min rho14=%.4g max rho14=%.4g (book -0.39, 0.23) %s\n",
           lo14, hi14, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
