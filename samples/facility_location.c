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

/* facility_location.c - smallest k balls covering a point set (MISOCP).
 *
 * Source: MOSEK/Tutorials, facility-location/small_disks.py.  Given points
 * p_i in R^2 and a number k of balls, choose centers x_j and radii r_j, plus
 * binary assignment s_ij, so that every point is covered:
 *
 *   minimize  t
 *   subject to  || p_i - x_j ||_2 <= r_j + M (1 - s_ij)   for all i,j  (SOC)
 *               sum_j s_ij = 1,   s_ij in {0,1}
 *               t >= r_j,   r_j >= 0,
 *
 * the Big-M relaxation of "point i is assigned to ball j" (M large enough to
 * deactivate the cone when s_ij = 0).  This is the notebook's basicModel with
 * objective "minimize the largest radius".
 *
 * Hand-derived instance: four collinear points 0,1,2,3 and k=2 balls; the best
 * cover pairs neighbours and gives t = 0.5 (centers at 0.5 and 2.5).
 *
 * Usage: facility_location   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NP 4              /* points    */
#define NK 2              /* balls     */
#define DIM 2
#define BIGM 10.0

int main(void) {
    static const double P[NP][DIM] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    /* vars: T | R[NK] | X[NK*DIM] | S[NP*NK] | W[NP*NK] | D[NP*NK*DIM] */
    int VT = 0, VR = VT + 1, VX = VR + NK, VS = VX + NK * DIM,
        VW = VS + NP * NK, VD = VW + NP * NK, NV = VD + NP * NK * DIM;
    int nS = NP * NK, nW = NP * NK;
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, nS + NP * NK * DIM + NP + NK);   /* W + D + assign + t>=R */
    PRIMAL_putvarbound(t, VT, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int j = 0; j < NK; j++) PRIMAL_putvarbound(t, VR + j, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int q = 0; q < NK * DIM; q++) PRIMAL_putvarbound(t, VX + q, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int q = 0; q < nS; q++) {
        PRIMAL_putvarbound(t, VS + q, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, VS + q, PRIMAL_VAR_TYPE_INT_BIN);
    }
    for (int q = 0; q < nW; q++) PRIMAL_putvarbound(t, VW + q, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int q = 0; q < NP * NK * DIM; q++) PRIMAL_putvarbound(t, VD + q, PRIMAL_BK_FR, -INFINITY, INFINITY);

    int row = 0;
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < NK; j++) {        /* W_ij - R_j + M*S_ij = M */
            int vs = VS + i * NK + j, vw = VW + i * NK + j;
            PRIMAL_putarow(t, row, 3, (int[]){vw, VR + j, vs}, (double[]){1.0, -1.0, BIGM});
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, BIGM, BIGM); row++;
        }
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < NK; j++)          /* D_ijq + X_jq = P_iq */
            for (int q = 0; q < DIM; q++) {
                int vd = VD + (i * NK + j) * DIM + q, vx = VX + j * DIM + q;
                PRIMAL_putarow(t, row, 2, (int[]){vd, vx}, (double[]){1.0, 1.0});
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, P[i][q], P[i][q]); row++;
            }
    for (int i = 0; i < NP; i++) {            /* sum_j s_ij = 1 */
        int sub[NK]; double val[NK];
        for (int j = 0; j < NK; j++) { sub[j] = VS + i * NK + j; val[j] = 1.0; }
        PRIMAL_putarow(t, row, NK, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    for (int j = 0; j < NK; j++) {            /* t >= R_j */
        PRIMAL_putarow(t, row, 2, (int[]){VT, VR + j}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY); row++;
    }
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < NK; j++) {        /* W_ij >= || D_ij || */
            int vw = VW + i * NK + j, vd = VD + (i * NK + j) * DIM;
            PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){vw, vd, vd + 1});
        }
    PRIMAL_putcj(t, VT, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, R[NK] = {0}, X[NK * DIM] = {0};
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        obj = x[VT];
        for (int j = 0; j < NK; j++) { R[j] = x[VR + j]; X[2 * j] = x[VX + 2 * j]; X[2 * j + 1] = x[VX + 2 * j + 1]; }
    }
    /* every point must be within R[j] of its ball center */
    int covered = 1;
    for (int i = 0; i < NP && ok; i++) {
        double best = 1e300;
        for (int j = 0; j < NK; j++) {
            double dx = P[i][0] - X[2 * j], dy = P[i][1] - X[2 * j + 1];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist - R[j] < best) best = dist - R[j];
        }
        if (best > 1e-6) covered = 0;
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    int okall = ok && covered && fabs(obj - 0.5) < 1e-6;
    printf("facility_location  n=%d k=%d\n", NP, NK);
    printf("  t* = %.8f (atteso 0.5)   raggi=(%.4f,%.4f) copertura ok=%d\n",
           obj, R[0], R[1], covered);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
