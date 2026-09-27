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
 * License excerpt (Apache License 2.0, §2 "Grant of Copyright License"):
 *   "Subject to the terms and conditions of this License, each Contributor
 *    hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
 *    royalty-free, irrevocable copyright license to reproduce, prepare
 *    Derivative Works of, publicly display, publicly perform, sublicense, and
 *    distribute the Work and such Derivative Works in Source or Object form."
 * 
 * Disclaimer of liability and absence of warranty (Apache License 2.0, §7-§8):
 *   [§7] Unless required by applicable law or agreed to in writing, Licensor
 *   provides the Work (and each Contributor provides its Contributions) on an
 *   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 *   or implied, including, without limitation, any warranties or conditions of
 *   TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
 *   PURPOSE.  You are solely responsible for determining the appropriateness of
 *   using or redistributing the Work.
 *   [§8] In no event and under no legal theory, whether in tort (including
 *   negligence), contract, or otherwise, unless required by applicable law or
 *   agreed to in writing, shall any Contributor be liable to You for damages,
 *   including any direct, indirect, special, incidental, or consequential
 *   damages arising as a result of this License or out of the use or inability
 *   to use the Work.  This software is provided without any guarantee that it
 *   will operate correctly or be free of defects.
 */

/* conic_bench.c - PrimalSolver SOCP/SDP benchmark (MPS cannot express cones).
 * Deterministic instances from a fixed seed; prints CSV:
 *   class,n,size,obj,seconds,rc[,expected]
 *
 * SOCP: min sum_k t_k  s.t. (t_k, x_2k, x_2k+1) in QUAD, sum_i x_i = 1.
 *       Optimum = 1/sqrt(2) (equal split in one cone).
 * SDP:  min <C,X>  s.t. X_ii = 1, X >= 0, with C = -v v'.
 *       Optimum = -(sum_i |v_i|)^2 (rank-one X = sign(v) sign(v)'), an
 *       independent closed-form reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

/* Current wall-clock time in seconds. */
static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static unsigned rng_state;
/* Deterministic pseudo-random draw in [0,1] from rng_state. */
static double urand(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (double)((rng_state >> 16) & 0x7fff) / 32767.0;
}

/* Solve one SOCP instance of size n and print its CSV line. */
static int run_socp(int n, int seed) {
    (void)seed;
    int K = n / 2;
    PRIMALenv_t env = NULL; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t = NULL; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, n + K);           /* x_0..x_{n-1}, t_0..t_{K-1} */
    for (int j = 0; j < n; j++)
        PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int k = 0; k < K; k++) {
        PRIMAL_putvarbound(t, n + k, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, n + k, 1.0);
        int mem[3] = {n + k, 2 * k, 2 * k + 1};
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, mem);
    }
    PRIMAL_appendcons(t, 1);
    {
        int *sub = (int *)malloc((size_t)n * sizeof(int));
        double *val = (double *)malloc((size_t)n * sizeof(double));
        for (int j = 0; j < n; j++) { sub[j] = j; val[j] = 1.0; }
        PRIMAL_putarow(t, 0, n, sub, val);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
        free(sub); free(val);
    }
    double a = now();
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double sec = now() - a;
    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    printf("socp,%d,%d,%.8g,%.6f,rc=%d\n", n, K, obj, sec, (int)rc);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? 0 : 1;
}

/* Solve one SDP instance of dimension d and print its CSV line. */
static int run_sdp(int d, int seed) {
    rng_state = (unsigned)seed;
    double v[16];
    double sumabs = 0.0;
    for (int i = 0; i < d; i++) { v[i] = urand() * 2.0 - 1.0; sumabs += fabs(v[i]); }
    double expected = -sumabs * sumabs;

    PRIMALenv_t env = NULL; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t = NULL; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 1);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);   /* dummy scalar */
    int dim = d;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, d);                /* X_ii = 1 */

    /* objective <C,X> with C = -v v' (entries i<=j) */
    {
        int nz = 0, si[256], sj[256];
        double sv[256];
        for (int i = 0; i < d; i++)
            for (int j = i; j < d; j++) {
                si[nz] = i; sj[nz] = j; sv[nz] = -v[i] * v[j]; nz++;
            }
        int mC;
        PRIMAL_appendsparsesymmat(t, d, nz, si, sj, sv, &mC);
        PRIMAL_putbarcj(t, 0, 1, (int[]){mC}, (double[]){1.0});
    }
    /* X_ii = 1: <E_ii, X> = 1 */
    for (int i = 0; i < d; i++) {
        int mE;
        PRIMAL_appendsparsesymmat(t, d, 1, (int[]){i}, (int[]){i}, (double[]){1.0}, &mE);
        PRIMAL_putbaraij(t, i, 0, 1, (int[]){mE}, (double[]){1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    double a = now();
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double sec = now() - a;
    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    printf("sdp,%d,%d,%.8g,%.6f,rc=%d,expected=%.8g\n",
           d, d, obj, sec, (int)rc, expected);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? 0 : 1;
}

/* Run the SOCP/SDP benchmark grid and print the CSV header first. */
int main(void) {
    printf("class,n,size,obj,seconds,rc[,expected]\n");
    int rc = 0;
    for (int n = 40; n <= 200; n += 80) rc |= run_socp(n, 100 + n);
    for (int d = 4; d <= 8; d += 1) rc |= run_sdp(d, 200 + d);
    return rc;
}
