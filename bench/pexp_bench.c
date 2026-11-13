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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* pexp_bench.c -- the mixed SOC + exponential-cone regression instance of the
 * README table ("Mixed SOC and exponential cones"), plus one timing line per
 * solve.  This is the generator that used to live only in a gitignored
 * scratch directory: the instance is defined here, in the repository, so the
 * numbers are reproducible from a clean checkout.
 *
 * The model (one primal exponential cone per observation, one rotated
 * quadratic cone for the penalty):
 *
 *   min   0.1 * w + sum_i v_i
 *   s.t.  u_i = a_i' x + b_i                (n rows)
 *         (v_i, 1, u_i) in PEXP             =>  v_i >= exp(u_i)
 *         (w, 1, x_0 .. x_{d-1}) in RQUAD   =>  2*w >= ||x||^2
 *         x free, v_i >= 0, w free
 *
 * The rotated cone is written with the fixed-one variable, exactly as
 * examples/c/logistic.c does: 2*w*1 >= ||x||^2, so w >= 0.5*||x||^2 and the
 * objective's 0.1*w is the 0.05*||x||^2 penalty of the underlying problem.
 *
 * Instance data: the same 32-bit LCG as bench/pexp_bench.py (and as the
 * reference formulation there), state = s*1103515245 + 12345 mod 2^32,
 * draw = ((s >> 16) & 0x7fff) / 32767.  Features are drawn first, n*d of
 * them, then the n biases; feature i is (2*draw - 1) * 0.5, bias i is
 * 2*draw - 1.
 *
 * Output (one line, parsed by bench/pexp_bench.py):
 *   metrics <seconds> <pobj> <dobj> <primal infeas> <dual infeas>
 *
 * Build: make bench   (target out/bench/pexp_bench)
 * Run:   ./out/bench/pexp_bench 80 12
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "primal.h"

static uint32_t rng_state;
static double draw(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (double)((rng_state >> 16) & 0x7fffu) / 32767.0;
}

/* Monotonic seconds.  clock() would be CPU time and is coarse on this scale. */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 80;   /* observations */
    int d = (argc > 2) ? atoi(argv[2]) : 12;   /* features */
    if (n < 1 || d < 1) { fprintf(stderr, "usage: %s [obs] [features]\n", argv[0]); return 2; }

    /* Instance data. */
    rng_state = 11235813u;
    double *A = (double *)malloc((size_t)n * (size_t)d * sizeof(double));
    double *bias = (double *)malloc((size_t)n * sizeof(double));
    if (!A || !bias) { free(A); free(bias); return 2; }
    for (size_t q = 0; q < (size_t)n * (size_t)d; q++) A[q] = (draw() * 2.0 - 1.0) * 0.5;
    for (int i = 0; i < n; i++) bias[i] = draw() * 2.0 - 1.0;

    /* Variables: x (d, free), v (n, >= 0), u (n, free), one (fixed), w (free). */
    const int X0 = 0, V0 = d, U0 = d + n, ONE = d + 2 * n, W = ONE + 1;
    const int NV = d + 2 * n + 2;
    const int NC = n;

    PRIMALenv_t env;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) return 2;
    PRIMALtask_t task;
    if (PRIMAL_maketask(env, 0, 0, &task) != PRIMAL_RES_OK) return 2;
    if (PRIMAL_appendvars(task, NV) != PRIMAL_RES_OK) return 2;
    if (PRIMAL_appendcons(task, NC) != PRIMAL_RES_OK) return 2;

    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* c = (0, ..., 0, 1, ..., 1, 0, ..., 0, 0, 0.1) on (x, v, u, one, w). */
    for (int i = 0; i < n; i++) PRIMAL_putcj(task, V0 + i, 1.0);
    PRIMAL_putcj(task, W, 0.1);

    for (int j = 0; j < NV; j++) PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < n; i++) PRIMAL_putvarbound(task, V0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(task, ONE, PRIMAL_BK_FX, 1.0, 1.0);

    /* Row i: a_i' x - u_i = -b_i. */
    int *sub = (int *)malloc((size_t)(d + 1) * sizeof(int));
    double *val = (double *)malloc((size_t)(d + 1) * sizeof(double));
    if (!sub || !val) { free(sub); free(val); free(A); free(bias); return 2; }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) { sub[j] = X0 + j; val[j] = A[(size_t)i * d + j]; }
        sub[d] = U0 + i; val[d] = -1.0;
        PRIMAL_putarow(task, i, d + 1, sub, val);
        PRIMAL_putconbound(task, i, PRIMAL_BK_FX, -bias[i], -bias[i]);
    }

    /* Cones: one PEXP per observation, then the rotated quadratic penalty. */
    int *mem = (int *)malloc((size_t)(d + 2) * sizeof(int));
    if (!mem) { free(sub); free(val); free(A); free(bias); return 2; }
    for (int i = 0; i < n; i++) {
        mem[0] = V0 + i; mem[1] = ONE; mem[2] = U0 + i;
        PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, mem);
    }
    mem[0] = W; mem[1] = ONE;
    for (int j = 0; j < d; j++) mem[2 + j] = X0 + j;
    PRIMAL_appendcone(task, PRIMAL_CT_RQUAD, 0.0, d + 2, mem);

    double t0 = now_sec();
    PRIMALrescodee rc = PRIMAL_optimize(task);
    double t1 = now_sec();

    double pobj = 0.0, dobj = 0.0, pinf = 0.0, dinf = 0.0;
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &pobj);
    PRIMAL_getdualobj(task, PRIMAL_SOL_ITR, &dobj);
    PRIMAL_getprimalinfeas(task, PRIMAL_SOL_ITR, &pinf);
    PRIMAL_getdualinfeas(task, PRIMAL_SOL_ITR, &dinf);
    if (rc != PRIMAL_RES_OK) {
        fprintf(stderr, "pexp_bench: rc=%d n=%d d=%d\n", (int)rc, n, d);
        return 1;
    }
    printf("metrics %.9f %.17g %.17g %.3e %.3e\n", t1 - t0, pobj, dobj, pinf, dinf);

    /* --dump: the published point, so two builds can be compared field by
     * field (a reformulation that publishes the right objective with the wrong
     * vector is the failure mode this exists to catch). */
    if (argc > 3 && strcmp(argv[3], "--dump") == 0) {
        int nv = 0, nc = 0;
        if (PRIMAL_getnumvar(task, &nv) != PRIMAL_RES_OK) return 1;
        if (PRIMAL_getnumcon(task, &nc) != PRIMAL_RES_OK) return 1;
        double *v = (double *)malloc((size_t)(nv > nc ? nv : nc) * sizeof(double));
        if (!v) return 1;
        printf("point");
        PRIMAL_getxx(task, PRIMAL_SOL_ITR, v);
        printf(" x"); for (int j = 0; j < nv; j++) printf(" %.17g", v[j]);
        PRIMAL_gety(task, PRIMAL_SOL_ITR, v);
        printf(" y"); for (int i = 0; i < nc; i++) printf(" %.17g", v[i]);
        PRIMAL_getslx(task, PRIMAL_SOL_ITR, v);
        printf(" slx"); for (int j = 0; j < nv; j++) printf(" %.17g", v[j]);
        PRIMAL_getsux(task, PRIMAL_SOL_ITR, v);
        printf(" sux"); for (int j = 0; j < nv; j++) printf(" %.17g", v[j]);
        printf("\n");
        free(v);
    }

    free(mem); free(sub); free(val); free(A); free(bias);
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return 0;
}
