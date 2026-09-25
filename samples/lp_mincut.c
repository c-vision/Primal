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

/* lp_mincut.c - minimum s-t cut as a linear program, and the CORRELATED min-cut
 * (maxdan94/lp_mincut, which cites the LP formulation of the max-flow min-cut
 * theorem and MOSEK).
 *
 *   minimize    sum_e w_e d_e
 *   subject to  d_e - p_{s(e)} + p_{t(e)} >= 0     for every arc e   (d_e >= p_s - p_t)
 *               p_s - p_t = 1                                        (separate s from t)
 *               d_e, p_v >= 0
 *
 * The optimum is the min-cut capacity, equal to the max flow (max-flow/min-cut
 * theorem).  The repo's research part adds, for a pair of CORRELATED arcs,
 * the constraint d_ij = d_kl ("ij is cut iff kl is cut"), and asks whether the
 * resulting matrix stays totally unimodular (it does not, in general: fractional
 * solutions appear).
 *
 * Instance: the repo's graph.txt, s=0, t=3,
 *     arcs 0->1 (0.5), 0->2 (0.3), 1->3 (0.4), 2->3 (0.5).
 * Hand value: the max flow is the two disjoint paths 0-1-3 (0.4) and 0-2-3
 * (0.3) = 0.7; the cut {0,1}->{2,3} has capacity 0.3 + 0.4 = 0.7, so the min
 * cut is 0.7 (and the LP gives p0=1, p1=1, p2=0, p3=0).
 *
 * Usage: lp_mincut   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NE 4      /* arcs */
#define NN 4      /* nodes */
static const int SRC[NE] = {0, 0, 1, 2};
static const int DST[NE] = {1, 2, 3, 3};
static const double W[NE] = {0.5, 0.3, 0.4, 0.5};

/* solve min-cut with an optional list of correlated arc pairs forced equal;
 * returns the cut value (or -1). */
static double mincut(const int (*cor)[2], int ncor, double *dout) {
    int D0 = 0, P0 = NE, nv = NE + NN;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < nv; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int e = 0; e < NE; e++) PRIMAL_putcj(t, D0+e, W[e]);
    int nrow = NE + 1 + ncor;
    PRIMAL_appendcons(t, nrow);
    int r = 0;
    for (int e = 0; e < NE; e++) {                 /* d_e - p_s + p_t >= 0 */
        PRIMAL_putarow(t, r, 3, (int[]){D0+e, P0+SRC[e], P0+DST[e]}, (double[]){1.0, -1.0, 1.0});
        PRIMAL_putconbound(t, r, PRIMAL_BK_LO, 0.0, INFINITY); r++;
    }
    PRIMAL_putarow(t, r, 2, (int[]){P0+0, P0+3}, (double[]){1.0, -1.0});   /* p_s - p_t = 1 */
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 1.0, 1.0); r++;
    for (int c = 0; c < ncor; c++) {               /* d_ij = d_kl */
        PRIMAL_putarow(t, r, 2, (int[]){D0 + cor[c][0], D0 + cor[c][1]}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0.0, 0.0); r++;
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double val = -1.0;
    if (rc == PRIMAL_RES_OK) {
        double x[32] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        if (dout) for (int e = 0; e < NE; e++) dout[e] = x[D0+e];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &val);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return val;
}

int main(void) {
    printf("lp_mincut (min-cut LP; graph.txt di maxdan94/lp_mincut)\n");
    int all = 1;
    double d[NE], cut = mincut(NULL, 0, d);
    /* independent check: the cut d must be a feasible cut and the value must
     * equal the hand max flow 0.4 + 0.3 = 0.7 */
    int good = cut > 0 && fabs(cut - 0.7) < 1e-7;
    printf("  min cut = %.8f (atteso 0.7)  d=(%.4f,%.4f,%.4f,%.4f)  %s\n",
           cut, d[0], d[1], d[2], d[3], good ? "OK" : "FAIL");
    all &= good;

    /* correlated variant: force d(0->1) = d(2->3) (arcs 0 and 3) */
    int cor[1][2] = {{0, 3}};
    double d2[NE], cc = mincut(cor, 1, d2);
    /* the cut may rise (a correlation can only restrict the feasible set);
     * it must be feasible and >= the uncorrelated cut */
    int good2 = cc > 0 && cc >= cut - 1e-9 && fabs(d2[0] - d2[3]) < 1e-7;
    printf("  correlato d01=d23: cut = %.8f (d01=%.4f=d23=%.4f)  >= 0.7  %s\n",
           cc, d2[0], d2[3], good2 ? "OK" : "FAIL");
    all &= good2;

    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
