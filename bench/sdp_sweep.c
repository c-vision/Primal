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

/* bench/sdp_sweep.c - the pure-SDP family of bench/conic_bench.c (min <C,X>,
 * X_ii = 1, X >= 0, C = -v v', closed form -(sum |v_i|)^2) swept over the block
 * size d. conic_bench stops at d=8; this one exposes where the conic route
 * stops scaling -- the target of the Clarabel-style rewrite
 * (Piani/Piano-Rewrite-Solver-Conico-Stile-Clarabel.md, F0).
 *
 * Usage: sdp_sweep [dmin] [dmax]   (default 4 16)
 * Prints CSV: d,obj,seconds,rc,expected
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "primal.h"

static unsigned rng_state;
static double urand(void) { rng_state = rng_state * 1103515245u + 12345u; return (double)((rng_state >> 16) & 0x7fff) / 32767.0; }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

static int run_sdp(int d, int seed) {
    rng_state = (unsigned)seed;
    double v[128], sumabs = 0.0;
    for (int i = 0; i < d; i++) { v[i] = urand() * 2.0 - 1.0; sumabs += fabs(v[i]); }
    double expected = -sumabs * sumabs;

    PRIMALenv_t env = NULL; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t = NULL; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 1);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0);
    int dim = d;
    PRIMAL_appendbarvars(t, 1, &dim);
    PRIMAL_appendcons(t, d);
    {
        int nz = 0, si[4096], sj[4096]; double sv[4096];
        for (int i = 0; i < d; i++) for (int j = i; j < d; j++) { si[nz] = i; sj[nz] = j; sv[nz] = -v[i] * v[j]; nz++; }
        int mC; PRIMAL_appendsparsesymmat(t, d, nz, si, sj, sv, &mC);
        PRIMAL_putbarcj(t, 0, 1, (int[]){mC}, (double[]){1.0});
    }
    for (int i = 0; i < d; i++) {
        int mE; PRIMAL_appendsparsesymmat(t, d, 1, (int[]){i}, (int[]){i}, (double[]){1.0}, &mE);
        PRIMAL_putbaraij(t, i, 0, 1, (int[]){mE}, (double[]){1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    double a = now();
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double sec = now() - a;
    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    printf("%d,%.8g,%.6f,%d,%.8g\n", d, obj, sec, (int)rc, expected);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? 0 : 1;
}

int main(int argc, char **argv) {
    int dmin = argc > 1 ? atoi(argv[1]) : 4;
    int dmax = argc > 2 ? atoi(argv[2]) : 16;
    /* run_sdp writes v[128] and the d(d+1)/2 triangular triplets into si/sj/sv
     * [4096]; the coefficient arrays bind first at 90*91/2 = 4095, while
     * 91*92/2 = 4186 already overflows them. Reject the range before any model
     * is built (issue #12: `sdp_sweep 91 91` used to write past the arrays). */
    if (dmin < 1 || dmax < dmin || dmax > 90) {
        fprintf(stderr, "sdp_sweep: need 1 <= dmin <= dmax <= 90\n");
        return 2;
    }
    printf("d,obj,seconds,rc,expected\n");
    for (int d = dmin; d <= dmax; d++) run_sdp(d, 200 + d);
    return 0;
}
