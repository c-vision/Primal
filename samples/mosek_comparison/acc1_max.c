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

/* acc1_max.c - the MOSEK "acc1.c" tutorial (affine conic constraints), the only
 * example in the OptimalCNC/MOSEK_EXAMPLE CMake template.  NOT the same model as
 * samples/mosek_comparison/acc1.c, which ports the Julia "acc1" (min x1):
 *
 *   maximize   c'x,             c = (2, 3, -1)
 *   subject to sum(x) = 1
 *              gamma >= || G x + h ||_2   (gamma = 0.03, one quadratic cone)
 *
 * with G = [[1.5, 0.1], [0.3, 2.1]], h = (0, 0.1).  In the MOSEK example the
 * cone is written with the ACC/AFE API (an affine expression F x + g in a
 * conic-quadratic domain); here the same rows are spelled as direct variables
 * (z1, z2, the two AFE rows) plus the cone, as samples/mosek_comparison/cqo1.c
 * does -- the ACC API (appendafes/putafefentry/appendacc) is equivalent.
 *
 * Hand value: the feasible set is the disk ||M w + v|| <= gamma in w = (x0,x1)
 * (x2 = 1 - w1 - w2), so the linear objective attains
 *     max = c2'w_c + gamma * || M^-T c2 ||,   w_c = -M^-1 v,
 * with c2 = (3,4), M = [[1.5,0.1],[-1.8,-2.1]], v = (0,2.2):
 *     w_c = (-0.07407407, 1.11111111),  ||M^-T c2|| = sqrt(33.3)/2.97,
 *     obj = 3.2805112648 at x = (-0.07838011, 1.12891290, -0.05053279).
 *
 * Usage: acc1_max   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { X0 = 0, X1 = 1, X2 = 2, Z1 = 3, Z2 = 4, GAMMA = 5 };
    PRIMAL_appendvars(t, 6);
    PRIMAL_appendcons(t, 3);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = X0; j <= Z2; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, GAMMA, PRIMAL_BK_FX, 0.03, 0.03);
    /* sum(x) = 1 */
    PRIMAL_putarow(t, 0, 3, (int[]){X0, X1, X2}, (double[]){1.0, 1.0, 1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    /* z1 = 1.5 x0 + 0.1 x1   <=>   z1 - 1.5 x0 - 0.1 x1 = 0 */
    PRIMAL_putarow(t, 1, 3, (int[]){Z1, X0, X1}, (double[]){1.0, -1.5, -0.1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* z2 = 0.3 x0 + 2.1 x2 + 0.1   <=>   z2 - 0.3 x0 - 2.1 x2 = 0.1 */
    PRIMAL_putarow(t, 2, 3, (int[]){Z2, X0, X2}, (double[]){1.0, -0.3, -2.1});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.1, 0.1);
    /* (gamma, z1, z2) in Q_3 :  gamma >= || (z1,z2) || */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){GAMMA, Z1, Z2});
    PRIMAL_putcj(t, X0, 2.0); PRIMAL_putcj(t, X1, 3.0); PRIMAL_putcj(t, X2, -1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double x[8] = {0}, obj = 0.0;
    if (ok) { PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj); }
    double z1 = 1.5*x[X0] + 0.1*x[X1], z2 = 0.3*x[X0] + 2.1*x[X2] + 0.1;
    double nrm = sqrt(z1*z1 + z2*z2), sum = x[X0] + x[X1] + x[X2];
    int good = ok && fabs(obj - 3.2805112648) < 1e-6 &&
               fabs(x[X0] + 0.0783801115) < 1e-6 &&
               fabs(x[X1] - 1.1289128998) < 1e-6 &&
               fabs(x[X2] + 0.0505327883) < 1e-6 &&
               nrm <= 0.03 + 1e-9 && fabs(sum - 1.0) < 1e-9;
    printf("acc1_max  obj=%.10f (atteso 3.2805112648)  x=(%.8f,%.8f,%.8f)\n", obj, x[X0], x[X1], x[X2]);
    printf("  ||Gx+h||=%.10f (<=0.03)  sum=%.10f\n", nrm, sum);
    printf("%s\n", good ? "OK" : "FAIL");
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return good ? 0 : 1;
}
