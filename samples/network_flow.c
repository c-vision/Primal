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

/* network_flow.c - min-cost flow on a random graph (LP, scaling test).
 *
 * min sum_e c_e f_e
 *   s.t. sum_{e out of v} f_e - sum_{e into v} f_e = b_v   (balance)
 *        0 <= f_e <= u_e
 *
 * b has zero sum (feasible by construction: f = 0 is feasible when all b=0;
 * here a random circulation plus a source/sink pair guarantees feasibility).
 * Verified: node balance, capacity, strong duality.
 * Usage: network_flow [nodes] [edges]   (default 200 800)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "primal.h"

static unsigned st = 27182818u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 200;
    int E = argc > 2 ? atoi(argv[2]) : 800;
    if (N < 2 || E < 1) return 2;
    int *eu = malloc((size_t)E * sizeof(int));
    int *ev = malloc((size_t)E * sizeof(int));
    double *cost = malloc((size_t)E * sizeof(double));
    double *cap = malloc((size_t)E * sizeof(double));
    double *flow0 = calloc((size_t)E, sizeof(double));   /* reference flow */
    double *b = calloc((size_t)N, sizeof(double));
    for (int e = 0; e < E; e++) {
        int u = (int)(rnd() * N), v = (int)(rnd() * N);
        if (v == u) v = (v + 1) % N;
        eu[e] = u; ev[e] = v;
        cost[e] = 0.5 + rnd() * 10.0;
        cap[e] = 5.0 + rnd() * 20.0;
        flow0[e] = rnd() * cap[e] * 0.5;
        b[u] += flow0[e];       /* node u sends flow0 out */
        b[v] -= flow0[e];       /* node v receives flow0 in */
    }
    /* flow0 is a feasible circulation for these balances */

    PRIMALenv_t env; PRIMAL_makeenv(&env, NULL);
    PRIMALtask_t t; PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, E);
    PRIMAL_appendcons(t, N);
    for (int e = 0; e < E; e++) {
        PRIMAL_putcj(t, e, cost[e]);
        PRIMAL_putvarbound(t, e, PRIMAL_BK_RA, 0.0, cap[e]);
    }
    for (int v = 0; v < N; v++) {
        int *sub = malloc((size_t)E * sizeof(int));
        double *val = malloc((size_t)E * sizeof(double));
        int nz = 0;
        for (int e = 0; e < E; e++) {
            if (eu[e] == v) { sub[nz] = e; val[nz] = 1.0; nz++; }
            if (ev[e] == v) { sub[nz] = e; val[nz] = -1.0; nz++; }
        }
        if (nz == 0) { sub[0] = 0; val[0] = 0.0; nz = 1; }
        PRIMAL_putarow(t, v, nz, sub, val);
        PRIMAL_putconbound(t, v, PRIMAL_BK_FX, b[v], b[v]);
        free(sub); free(val);
    }

    struct timeval a, b2;
    gettimeofday(&a, NULL);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    gettimeofday(&b2, NULL);
    double sec = (b2.tv_sec - a.tv_sec) + 1e-6 * (b2.tv_usec - a.tv_usec);

    double obj = 0.0, dobj = 0.0;
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj);
        double *f = malloc((size_t)E * sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, f);
        double maxbal = 0.0, maxcap = 0.0, minf = 0.0;
        double *bal = calloc((size_t)N, sizeof(double));
        for (int e = 0; e < E; e++) {
            bal[eu[e]] += f[e];
            bal[ev[e]] -= f[e];
            if (f[e] - cap[e] > maxcap) maxcap = f[e] - cap[e];
            if (f[e] < minf) minf = f[e];
        }
        for (int v = 0; v < N; v++)
            if (fabs(bal[v] - b[v]) > maxbal) maxbal = fabs(bal[v] - b[v]);
        ok = maxbal < 1e-5 && maxcap < 1e-5 && minf > -1e-6 &&
             fabs(obj - dobj) < 1e-5 * (1.0 + fabs(obj));
        free(f); free(bal);
    }
    printf("netflow    N=%d E=%d vars=%d cons=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
           N, E, E, N, obj, dobj, sec, ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    free(eu); free(ev); free(cost); free(cap); free(flow0); free(b);
    return ok ? 0 : 1;
}
