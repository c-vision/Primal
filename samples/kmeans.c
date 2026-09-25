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

/* kmeans.c - k-means clustering as a mixed-integer conic program with a
 * disjunction (MOSEK Tutorials, kmeans-clustering).
 *
 * The notebook assigns each point to a cluster by a disjunction, pays the
 * squared distance to the chosen centroid, and minimizes the total inertia:
 *
 *   minimize  sum_i d_i
 *   subject to  dAux_ij >= || p_i - c_j ||^2        (rotated cone per i,j)
 *               d_i >= dAux_i,j  AND  y_i == j       for SOME j   (DJC per i)
 *               y_i in {0,..,K-1},  c_0 <= c_1 (in the first coordinate)
 *
 * The "some j" is written with the reference DJC API (appendafes + putdjc),
 * exactly as the notebook's M.disjunction over the K (distance, label) terms.
 *
 * Hand-derived instance: four collinear points 0,1,2,3 and K=2 clusters.  The
 * best partition pairs neighbours (centroids 0.5 and 2.5), total inertia
 *   4 * (0.5)^2 = 1.
 *
 * Usage: kmeans   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NPTS 4
#define NK 2

static const double PT[NPTS] = {0.0, 1.0, 2.0, 3.0};

int main(void) {
    enum { C = 0, DIST = NK, AUX = NK + NPTS, D3 = NK + NPTS + NPTS * NK,
           Y = NK + NPTS + 2 * NPTS * NK, HALF = Y + NPTS, NV = HALF + 1 };
    int nAux = NPTS * NK;
    int nAfe = nAux + NPTS;   /* one per (i,j) distance + one label per i */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, nAux + 1);          /* D3 links + centroid ordering */
    /* Centroids are bound to the data range [0,3] (a k-means centroid is the
     * mean of the points it owns, hence always inside their convex hull).
     * DECLARED DEVIATION: the notebook leaves the centroids unbounded. */
    for (int j = 0; j < NK; j++) PRIMAL_putvarbound(t, C + j, PRIMAL_BK_RA, 0.0, 3.0);
    for (int i = 0; i < NPTS; i++) PRIMAL_putvarbound(t, DIST + i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int q = 0; q < nAux; q++) PRIMAL_putvarbound(t, AUX + q, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int q = 0; q < nAux; q++) PRIMAL_putvarbound(t, D3 + q, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < NPTS; i++) {
        PRIMAL_putvarbound(t, Y + i, PRIMAL_BK_RA, 0.0, (double)(NK - 1));
        PRIMAL_putvartype(t, Y + i, PRIMAL_VAR_TYPE_INT);
    }
    PRIMAL_putvarbound(t, HALF, PRIMAL_BK_FX, 0.5, 0.5);
    int row = 0;
    for (int i = 0; i < NPTS; i++)
        for (int j = 0; j < NK; j++) {       /* D3_ij - C_j = -point_i */
            int q = i * NK + j;
            PRIMAL_putarow(t, row, 2, (int[]){D3 + q, C + j}, (double[]){1.0, -1.0});
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, -PT[i], -PT[i]); row++;
        }
    PRIMAL_putarow(t, row, 2, (int[]){C, C + 1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;   /* c0 <= c1 */
    for (int i = 0; i < NPTS; i++)
        for (int j = 0; j < NK; j++) {       /* dAux_ij >= D3_ij^2 */
            int q = i * NK + j;
            PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 3, (int[]){AUX + q, HALF, D3 + q});
        }
    for (int i = 0; i < NPTS; i++) PRIMAL_putcj(t, DIST + i, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    /* disjunction: for each point, d_i >= dAux_ij AND y_i == j, for SOME j */
    PRIMAL_appendafes(t, nAfe);
    for (int i = 0; i < NPTS; i++)
        for (int j = 0; j < NK; j++) {       /* afe iK+j = d_i - dAux_ij */
            int a = i * NK + j;
            PRIMAL_putafefentry(t, a, DIST + i, 1.0);
            PRIMAL_putafefentry(t, a, AUX + a, -1.0);
        }
    for (int i = 0; i < NPTS; i++)           /* afe nAux+i = y_i */
        PRIMAL_putafefentry(t, nAux + i, Y + i, 1.0);
    PRIMALint64t rplus, rzero;
    PRIMAL_appendrplusdomain(t, 1, &rplus);
    PRIMAL_appendrzerodomain(t, 1, &rzero);
    PRIMAL_appenddjcs(t, NPTS);
    for (int i = 0; i < NPTS; i++) {
        PRIMALint64t dom[2 * NK], afe[2 * NK], ts[NK];
        double bb[2 * NK];
        for (int j = 0; j < NK; j++) {
            dom[2 * j] = rplus; afe[2 * j] = i * NK + j; bb[2 * j] = 0.0;
            dom[2 * j + 1] = rzero; afe[2 * j + 1] = nAux + i; bb[2 * j + 1] = (double)j;
            ts[j] = 2;
        }
        PRIMALrescodee drc = PRIMAL_putdjc(t, i, 2 * NK, dom, 2 * NK, afe, bb, NK, ts);
        if (drc != PRIMAL_RES_OK) { printf("kmeans  putdjc rc=%d FAIL\n", (int)drc); return 1; }
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, cen[NK] = {0};
    if (ok) {
        double x[128] = {0};
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        cen[0] = x[C]; cen[1] = x[C + 1];
        obj = 0.0; for (int i = 0; i < NPTS; i++) obj += x[DIST + i];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    int okall = ok && fabs(obj - 1.0) < 1e-6;
    printf("kmeans  n=%d K=%d\n", NPTS, NK);
    printf("  obiettivo (inerzia) = %.8f (atteso 1.0)   centri = (%.4f, %.4f)\n", obj, cen[0], cen[1]);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
