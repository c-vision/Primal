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

/* wasserstein.c - Wasserstein barycenter of discrete distributions (LP).
 *
 * Source: MOSEK/Tutorials, wasserstein/wasserstein-bary.py.  For distributions
 * nu_1..nu_K on a common finite support {x_1..x_n} and ground cost
 * d_ij = |x_i - x_j|, the Wasserstein-1 barycenter minimizes the total
 * transport cost
 *
 *   minimize  sum_k sum_ij d_ij * pi^k_ij
 *   subject to  sum_j pi^k_ij = mu_i        (row marginals, common mu)
 *               sum_i pi^k_ij = nu^k_j      (column marginals)
 *               pi^k_ij >= 0,  mu >= 0,  sum_i mu_i = 1
 *
 * a linear program (the multi-marginal coupling through the shared mu).
 *
 * Hand-derived instance: support x = (0,1,2), K=2 with
 * nu_1 = (1/2,1/2,0), nu_2 = (0,1/2,1/2).  With a = mu_0, c = mu_2, the cost is
 * |a-1/2| + a + c + |1/2-c| = 1 for every a,c <= 1/2 and c <= 1/2, so the
 * optimal value is 1 (the barycenter is not unique here, e.g. mu = x^1).
 *
 * Usage: wasserstein   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NB 3              /* support points */
#define NK 2              /* distributions  */

static const double XS[NB] = {0.0, 1.0, 2.0};
static const double NU[NK][NB] = {{0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    int npi = NB * NB;
    int VMU = 0, VPI = NB, NV = NB + NK * npi;
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, NK * (NB + NB));    /* row + col marginals per k */
    for (int i = 0; i < NB; i++) PRIMAL_putvarbound(t, VMU + i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int q = 0; q < NK * npi; q++) PRIMAL_putvarbound(t, VPI + q, PRIMAL_BK_LO, 0.0, INFINITY);

    int row = 0;
    for (int k = 0; k < NK; k++) {
        int base = VPI + k * npi;
        for (int i = 0; i < NB; i++) {        /* sum_j pi^k_ij - mu_i = 0 */
            int sub[NB + 1]; double val[NB + 1];
            for (int j = 0; j < NB; j++) { sub[j] = base + i * NB + j; val[j] = 1.0; }
            sub[NB] = VMU + i; val[NB] = -1.0;
            PRIMAL_putarow(t, row, NB + 1, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
        }
        for (int j = 0; j < NB; j++) {        /* sum_i pi^k_ij = nu^k_j */
            int sub[NB]; double val[NB];
            for (int i = 0; i < NB; i++) { sub[i] = base + i * NB + j; val[i] = 1.0; }
            PRIMAL_putarow(t, row, NB, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, NU[k][j], NU[k][j]); row++;
        }
    }
    for (int k = 0; k < NK; k++)              /* objective: sum d_ij pi^k_ij */
        for (int i = 0; i < NB; i++)
            for (int j = 0; j < NB; j++)
                PRIMAL_putcj(t, VPI + k * npi + i * NB + j, fabs(XS[i] - XS[j]));
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, mu[NB] = {0};
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < NB; i++) mu[i] = x[VMU + i];
        for (int k = 0; k < NK; k++)
            for (int i = 0; i < NB; i++)
                for (int j = 0; j < NB; j++)
                    obj += fabs(XS[i] - XS[j]) * x[VPI + k * npi + i * NB + j];
    }
    double msum = mu[0] + mu[1] + mu[2];
    int okall = ok && fabs(obj - 1.0) < 1e-8 && fabs(msum - 1.0) < 1e-7 &&
                mu[0] >= -1e-9 && mu[1] >= -1e-9 && mu[2] >= -1e-9;
    printf("wasserstein  n=%d K=%d\n", NB, NK);
    printf("  costo totale = %.10f (atteso 1.0)\n", obj);
    printf("  barycenter mu = (%.6f, %.6f, %.6f)  somma=%.8f\n", mu[0], mu[1], mu[2], msum);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
