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

/* risk_parity.c - equal risk contribution portfolio via exponential cones.
 *
 *   min  sqrt(x'Sx) - c * sum_i log(x_i)      (x_i > 0)
 * conic form:
 *   (t, G'x) in Q      (t >= sqrt(x'Sx),  S = F F' + diag(D), G = [F, sqrt(D)])
 *   (x_i, 1, s_i) in EXP   (s_i <= log x_i)
 *   min t - c * sum_i s_i
 *
 * The minimizer is scale-invariant; after rescaling to sum(x) = 1 the risk
 * contributions x_i (Sx)_i are equal (verified here).
 * Usage: risk_parity [n] [k]   (default 30 4)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 909090u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 20;
    int k = argc > 2 ? atoi(argv[2]) : 4;
    if (n < 2 || k < 1) return 2;
    const double c = argc > 3 ? atof(argv[3]) : 0.01;
    double *F = malloc((size_t)n * k * sizeof(double));
    double *D = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        D[i] = 0.001 + rnd() * 0.004;
        for (int j = 0; j < k; j++) F[i * k + j] = (rnd() - 0.5) * 0.2;
    }
    int m2 = k + n;

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    /* vars: x (n), z (m2), t (1), s (n), one (1) */
    int X0 = 0, Z0 = n, TV = n + m2, S0 = n + m2 + 1, ONE = n + m2 + 1 + n;
    PRIMAL_appendvars(t, n + m2 + 1 + n + 1);
    PRIMAL_appendcons(t, m2);
    for (int i = 0; i < n; i++) PRIMAL_putvarbound(t, X0 + i, PRIMAL_BK_RA, 1e-4, 1e4);
    for (int j = 0; j < m2; j++) PRIMAL_putvarbound(t, Z0 + j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, TV, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, TV, 1.0);
    for (int i = 0; i < n; i++) {
        /* big-M on the log variable: keeps the round-0 tangent LP bounded
         * (same device as the logistic benchmark) */
        PRIMAL_putvarbound(t, S0 + i, PRIMAL_BK_RA, -30.0, 10.0);
        PRIMAL_putcj(t, S0 + i, -c);
    }
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    for (int j = 0; j < m2; j++) {
        int *sub = malloc((size_t)(n + 1) * sizeof(int));
        double *val = malloc((size_t)(n + 1) * sizeof(double));
        int nz = 0;
        for (int i = 0; i < n; i++) {
            double g = (j < k) ? F[i * k + j]
                               : ((j - k == i) ? sqrt(D[i]) : 0.0);
            if (g != 0.0) { sub[nz] = X0 + i; val[nz] = -g; nz++; }
        }
        sub[nz] = Z0 + j; val[nz] = 1.0; nz++;
        PRIMAL_putarow(t, j, nz, sub, val);
        PRIMAL_putconbound(t, j, PRIMAL_BK_FX, 0.0, 0.0);
        free(sub); free(val);
    }
    {   /* (t, z) in QUAD */
        int *mem = malloc((size_t)(m2 + 1) * sizeof(int));
        mem[0] = TV;
        for (int j = 0; j < m2; j++) mem[1 + j] = Z0 + j;
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, m2 + 1, mem);
        free(mem);
    }
    for (int i = 0; i < n; i++)   /* (x_i, one, s_i) in EXP: x_i >= exp(s_i) */
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){X0 + i, ONE, S0 + i});

    struct timeval a, b;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b, NULL);
    double sec = (b.tv_sec - a.tv_sec) + 1e-6 * (b.tv_usec - a.tv_usec);

    int ok = (rc == PRIMAL_RES_OK);
    double spread = 0.0;
    if (ok) {
        double *x = malloc((size_t)(n + m2 + 1 + n + 1) * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        /* rescale to sum 1 */
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += x[i];
        for (int i = 0; i < n; i++) x[i] /= sum;
        /* Sx */
        double *Sx = calloc((size_t)n, sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int a2 = 0; a2 < k; a2++) {
                double s = 0.0;
                for (int l = 0; l < n; l++) s += F[l * k + a2] * x[l];
                Sx[i] += F[i * k + a2] * s;
            }
            Sx[i] += D[i] * x[i];
        }
        double rcmin = 1e300, rcmax = -1e300;
        for (int i = 0; i < n; i++) {
            double rc = x[i] * Sx[i];
            if (rc < rcmin) rcmin = rc;
            if (rc > rcmax) rcmax = rc;
        }
        spread = (rcmin > 0) ? (rcmax / rcmin - 1.0) : 1.0;
        ok = spread < 1e-3;   /* risk contributions equal within 0.1% */
        free(x); free(Sx);
    }
    printf("riskparity n=%d k=%d RC spread=%.3e rc=%d t=%.4fs %s\n",
           n, k, spread, (int)rc, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(F); free(D);
    return ok ? 0 : 1;
}
