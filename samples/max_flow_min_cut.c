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

/* max_flow_min_cut.c - maximum flow / minimum cut as a linear program.
 *
 * Source: LightConvex (Doc 3 of the external-document list, read 2026-09-22),
 * whose example list names "Max Flow / Min Cut". LightConvex itself does not
 * publish the optimum of a specific instance, so the instance below is chosen
 * small enough that both the max flow and the min cut are derived BY HAND and
 * the two are checked against each other (the max-flow/min-cut theorem says
 * they are equal, so agreement is a genuine independent check).
 *
 * ---------------------------------------------------------------------------
 * MODEL
 * ---------------------------------------------------------------------------
 * Given a directed graph with node set V, arc set E, capacities u_e >= 0, a
 * source s and a sink t, the maximum flow is the LP
 *
 *   maximize    sum_{e out of s} f_e
 *   subject to  0 <= f_e <= u_e                    (capacity of each arc)
 *               sum_{e into v} f_e - sum_{e out of v} f_e = 0   (flow conservation
 *                                                                 at v != s, t)
 *
 * The dual of this LP is the minimum cut: a partition (S, T) with s in S, t in
 * T, minimizing the total capacity of the arcs from S to T. Strong duality
 * gives max flow = min cut.
 *
 * ---------------------------------------------------------------------------
 * INSTANCE (hand-derived max flow = min cut = 5)
 * ---------------------------------------------------------------------------
 *   nodes 0 = source, 3 = sink, 1 and 2 intermediate
 *   arcs:  f0: 0->1 cap 3
 *          f1: 0->2 cap 2
 *          f2: 1->3 cap 2
 *          f3: 2->3 cap 3
 *          f4: 1->2 cap 1
 *
 * A feasible flow of value 5:
 *   0->1 = 3, 0->2 = 2, 1->3 = 2, 1->2 = 1, 2->3 = 3.
 * Conservation: node 1: in 3 = out 2+1; node 2: in 2+1 = out 3. Value = 3+2 = 5.
 * The cut S = {0, 1} has capacity f1 + f4 + f2 = 2 + 1 + 2 = 5, so 5 is
 * optimal (no cut is smaller, hence no flow is larger).
 *
 * The sample solves the LP, checks the value is 5, recomputes conservation and
 * capacities from the published flow, and independently enumerates ALL cuts to
 * confirm the minimum cut is also 5.
 *
 * Usage: max_flow_min_cut   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NARC 5
/* arc table: from, to, capacity.  Node 0 = source, node 3 = sink. */
static const int    ARCF[NARC] = {0, 0, 1, 2, 1};
static const int    ARCT[NARC] = {1, 2, 3, 3, 2};
static const double CAP [NARC] = {3.0, 2.0, 2.0, 3.0, 1.0};
#define NNODE 4
#define SRC 0
#define SNK 3

/* minimum cut by brute force over the 2^(NNODE-2) partitions with source in S
 * and sink in T (intermediate nodes free).  This is the independent check. */
static double min_cut_bruteforce(void) {
    double best = 1e30;
    for (int mask = 0; mask < (1 << (NNODE - 2)); mask++) {
        int inS[NNODE];
        inS[SRC] = 1; inS[SNK] = 0;
        for (int v = 0; v < NNODE; v++)
            if (v != SRC && v != SNK) {
                int bit = (v < SRC) ? v : v - 1;
                inS[v] = (mask >> bit) & 1;
            }
        double cut = 0.0;
        for (int e = 0; e < NARC; e++)
            if (inS[ARCF[e]] && !inS[ARCT[e]]) cut += CAP[e];
        if (cut < best) best = cut;
    }
    return best;
}

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    /* 5 flow variables, 5 capacity rows + 2 conservation rows = 7 rows */
    PRIMAL_maketask(env, NARC + (NNODE - 2), NARC, &t);

    for (int e = 0; e < NARC; e++) PRIMAL_putvarbound(t, e, PRIMAL_BK_LO, 0.0, INFINITY);
    /* maximize f0 + f1 (the arcs out of the source): min -(f0 + f1) */
    PRIMAL_putcj(t, 0, -1.0);
    PRIMAL_putcj(t, 1, -1.0);

    int row = 0;
    /* capacity: f_e <= u_e */
    for (int e = 0; e < NARC; e++) {
        PRIMAL_putarow(t, row, 1, (int[]){e}, (double[]){1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, CAP[e]);
        row++;
    }
    /* conservation at every intermediate node v: in - out = 0 */
    for (int v = 1; v < NNODE - 1; v++) {
        int sub[NARC]; double val[NARC]; int nz = 0;
        for (int e = 0; e < NARC; e++) {
            double c = 0.0;
            if (ARCT[e] == v) c += 1.0;   /* inflow */
            if (ARCF[e] == v) c -= 1.0;   /* outflow */
            if (c != 0.0) { sub[nz] = e; val[nz] = c; nz++; }
        }
        PRIMAL_putarow(t, row, nz, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0);
        row++;
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double flow = 0.0;
    if (ok) {
        double x[NARC];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        flow = x[0] + x[1];

        /* recompute capacities and conservation from the published flow */
        double maxcapviol = 0.0, maxconsviol = 0.0;
        for (int e = 0; e < NARC; e++)
            if (x[e] > CAP[e] + 1e-7) maxcapviol = fmax(maxcapviol, x[e] - CAP[e]);
        for (int v = 1; v < NNODE - 1; v++) {
            double bal = 0.0;
            for (int e = 0; e < NARC; e++) {
                if (ARCT[e] == v) bal += x[e];
                if (ARCF[e] == v) bal -= x[e];
            }
            maxconsviol = fmax(maxconsviol, fabs(bal));
        }
        double cut = min_cut_bruteforce();
        ok = fabs(flow - 5.0) < 1e-6 && maxcapviol < 1e-7 && maxconsviol < 1e-7 &&
             fabs(cut - flow) < 1e-9;
        printf("max_flow_min_cut  flow=%.6f (atteso 5)  min_cut=%.6f  "
               "max_cap_viol=%.2e max_cons_viol=%.2e  f=(%.2f,%.2f,%.2f,%.2f,%.2f) %s\n",
               flow, cut, maxcapviol, maxconsviol, x[0], x[1], x[2], x[3], x[4],
               ok ? "OK" : "FAIL");
    } else {
        printf("max_flow_min_cut  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
