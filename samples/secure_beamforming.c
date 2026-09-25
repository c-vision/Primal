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

/* secure_beamforming.c - the convex core of the SCA subproblem of
 * vkumar-ucd/secure_ISAC_GC23 (IRS-secure-ISAC beamforming, GLOBECOM 2023).
 *
 * The repository is an *iterative nonconvex* algorithm (successive convex
 * approximation): each step maximizes a beampattern gain subject to
 * communication-SINR and power constraints, which is a second-order cone
 * program.  This sample solves that convex subproblem at one step:
 *
 *     maximize   a'w                        (linearized beampattern gain)
 *     subject to h'w >= sqrt(gamma) ||w||   (communication SINR, phase-aligned)
 *                ||w|| <= sqrt(P)            (transmit power)
 *
 * The SINR constraint is the standard second-order-cone reformulation of
 * |h'w|^2/||sigma||^2 >= gamma (the extra leakage constraint of the paper is
 * the same cone with a "<= ", omitted here).
 *
 * Hand instance: Nb = 2, h = (1,0), a = (1,1)/sqrt(2), gamma = 1/2, P = 2.
 *   - SINR: w_0 >= sqrt(1/2)*||w||  <=>  w_0^2 >= w_1^2;
 *   - power: ||w||^2 <= 2;
 *   - maximize (w_0+w_1)/sqrt(2)  ->  w = (1,1), a'w = sqrt(2), gain = 2,
 *     with both constraints active.
 *
 * Usage: secure_beamforming   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const double h[2] = {1.0, 0.0};
    const double a[2] = {1.0 / sqrt(2.0), 1.0 / sqrt(2.0)};
    const double gamma = 0.5, P = 2.0;
    const double sg = sqrt(gamma), sP = sqrt(P);
    enum { W = 0, PB = 2, U = 3, Q = 5, NV = 6, NROW = 3 };

    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, NROW, NV, &t);
    for (int v = 0; v < NV; v++) PRIMAL_putvarbound(t, v, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, Q, PRIMAL_BK_FX, sP, sP);          /* sqrt(P) */
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = 0; j < 2; j++) PRIMAL_putcj(t, W + j, a[j]); /* maximize a'w */
    /* p = h'w */
    PRIMAL_putarow(t, 0, 3, (int[]){PB, W, W + 1}, (double[]){1.0, -h[0], -h[1]});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    /* u_j = sqrt(gamma) w_j */
    for (int j = 0; j < 2; j++) {
        PRIMAL_putarow(t, 1 + j, 2, (int[]){U + j, W + j}, (double[]){1.0, -sg});
        PRIMAL_putconbound(t, 1 + j, PRIMAL_BK_FX, 0.0, 0.0);
    }
    /* SINR: (h'w, sqrt(gamma) w) in QUAD ; power: (sqrt(P), w) in QUAD */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){PB, U, U + 1});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){Q, W, W + 1});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, w[2] = {0};
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, w);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    /* the optimum is degenerate (the objective gradient is radial on the power
     * circle), so w is accurate to ~1e-4 while the objective is to 1e-7. */
    int good = ok && fabs(obj - sqrt(2.0)) < 1e-6 && fabs(w[0] - 1.0) < 1e-3 &&
               fabs(w[1] - 1.0) < 1e-3 &&
               fabs(w[0] * w[0] + w[1] * w[1] - P) < 1e-6;   /* power active */
    printf("secure_beamforming: a'w=%.8f  w=(%.6f,%.6f)  gain=%.8f  (atteso sqrt2, w=(1,1), gain=2)  %s\n",
           obj, w[0], w[1], obj * obj, good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
