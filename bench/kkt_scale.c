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

/* bench/kkt_scale.c - size scaling of the unified conic route (sdp.c), the
 * measurement behind its sparse KKT (SDP_SPARSE_WORK, attempts.md F81).  Two
 * families with a closed-form optimum:
 *
 *   kkt_scale exp N : max entropy, N exp cones (1, x_i, t_i) in PEXP,
 *                     sum x = 1, min -sum t           -> obj = -log N
 *   kkt_scale sdp N : X (2x2 PSD) min <diag(1,2),X>, tr X = 1, next to the
 *                     path LP min sum x, x_i + x_{i+1} >= 1, x >= 0
 *                                                     -> obj = 1 + floor(N/2)
 *
 * Prints rc, solsta, objective and its relative error, wall time and peak RSS.
 * GMB_SDP_SPARSE / GMB_SDP_DENSE force either form of the KKT; GMB_DBG prints
 * its size ("[kkt] ... n=") and the route that answered.
 *
 * Usage: kkt_scale exp|sdp N
 */
#ifndef __APPLE__   /* glibc hides clock_gettime under -std=c99; on macOS it would hide ru_maxrss */
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "primal.h"

/* Monotonic wall clock in seconds. */
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }

/* Build the requested family at size N, solve it and print one result line. */
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    int N = atoi(argv[2]), sdp = strcmp(argv[1], "sdp") == 0;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    double ref;
    if (!sdp) {
        PRIMAL_appendvars(t, 3 * N); PRIMAL_appendcons(t, 1);
        int *sub = malloc(N * sizeof(int)); double *one = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            int u = 3 * i, x = 3 * i + 1, tt = 3 * i + 2;
            PRIMAL_putvarbound(t, u, PRIMAL_BK_FX, 1.0, 1.0);
            PRIMAL_putvarbound(t, x, PRIMAL_BK_LO, 0.0, INFINITY);
            PRIMAL_putvarbound(t, tt, PRIMAL_BK_FR, -INFINITY, INFINITY);
            PRIMAL_putcj(t, tt, -1.0);
            sub[i] = x; one[i] = 1.0;
            /* (u, x, t) in PEXP: u >= x exp(t/x), so with u = 1, t <= -x log x;
             * min -sum t is the min of sum x log x over the simplex: -log N. */
            PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){u, x, tt});
        }
        PRIMAL_putarow(t, 0, N, sub, one); PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        ref = -log((double)N);
        free(sub); free(one);
    } else {
        PRIMAL_appendvars(t, N); PRIMAL_appendcons(t, N);   /* N-1 path rows + trace row */
        for (int i = 0; i < N; i++) { PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY); PRIMAL_putcj(t, i, 1.0); }
        for (int i = 0; i + 1 < N; i++) {
            PRIMAL_putarow(t, i, 2, (int[]){i, i + 1}, (double[]){1.0, 1.0});
            PRIMAL_putconbound(t, i, PRIMAL_BK_LO, 1.0, INFINITY);
        }
        int d = 2, ic, it;
        PRIMAL_appendbarvars(t, 1, &d);
        PRIMAL_appendsparsesymmat(t, 2, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){1.0, 2.0}, &ic);
        PRIMAL_appendsparsesymmat(t, 2, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){1.0, 1.0}, &it);
        PRIMAL_putbarcj(t, 0, 1, &ic, (double[]){1.0});
        PRIMAL_putbaraij(t, N - 1, 0, 1, &it, (double[]){1.0});
        PRIMAL_putconbound(t, N - 1, PRIMAL_BK_FX, 1.0, 1.0);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        ref = 1.0 + floor(N / 2.0);
    }
    double t0 = now();
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double sec = now() - t0;
    PRIMALsolstae ss = PRIMAL_SOL_STA_UNKNOWN; double obj = NAN;
    PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss); PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    double rss_mb = ru.ru_maxrss / 1048576.0;   /* bytes on macOS */
#else
    double rss_mb = ru.ru_maxrss / 1024.0;      /* kilobytes on Linux */
#endif
    printf("%s N=%d rc=%d solsta=%d obj=%.9g ref=%.9g relerr=%.1e t=%.3fs rss=%.0fMB\n",
           argv[1], N, rc, ss, obj, ref, fabs(obj - ref) / (1 + fabs(ref)), sec, rss_mb);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return 0;
}
