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

/* nesting_dotted_board.c - nesting of convex polygons on a dotted board, a
 * MILP (eellookkaa/mddm, the "Dotted-Board Model").
 *
 * A set of polygons is placed on dots (x,y) of a grid; each polygon i chooses
 * exactly one dot d, and the objective MINIMISES the board size:
 *
 *     minimise  z_length + z_width
 *     s.t.      sum_d delta[i,d] = 1                 (each polygon placed once)
 *               z_length >= (right edge of i at d) * delta[i,d]
 *               z_width  >= (top   edge of i at d) * delta[i,d]
 *               delta[i,d] + delta[j,e] <= 1          for e in NFP(i,j,d)
 *               delta[i,d] in {0,1}
 *
 * The last family is the no-overlap constraint: NFP(i,j,d) is the set of dots
 * the No-Fit Polygon (Minkowski sum) forbids for polygon j when i sits on d.
 * The upstream computes the IFP/NFP with shapely; here, for a tiny instance,
 * they are written by hand.
 *
 * Case: two unit squares on a board of 2 x 2, sampled by a FINE dot grid at
 * spacing 0.5 (indices 0..2 -> coordinates 0, 0.5, 1).  A square at index i
 * occupies [0.5 i, 0.5 i + 1], so it fits the board iff i <= 2 (IFP = 3x3), and
 * two squares on (i1,j1), (i2,j2) overlap iff |i1-i2| <= 1 AND |j1-j2| <= 1
 * (their interiors share a cell) -- the hand-written NFP of a square against
 * itself.  Placing them on (0,0) and (2,0) is legal and gives
 * z_length = 0.5*2+1 = 2, z_width = 1, so z = 3; z = 2.5 is impossible (both
 * would need i <= 1 and j = 0, which overlap), hence the optimum is EXACTLY 3.
 *
 * Usage: nesting_dotted_board   (no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

#define NG 3          /* dots per axis (indices 0..2) */
#define NDOT (NG*NG)  /* 9 dots */
#define NP 2          /* two squares */

int main(void) {
    int X0 = 0, ZL = NP*NDOT, ZW = NP*NDOT + 1, NV = NP*NDOT + 2;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < NP; i++)
        for (int d = 0; d < NDOT; d++) {
            PRIMAL_putvartype(t, X0 + i*NDOT + d, PRIMAL_VAR_TYPE_INT);
            PRIMAL_putvarbound(t, X0 + i*NDOT + d, PRIMAL_BK_RA, 0.0, 1.0);
        }
    PRIMAL_putvarbound(t, ZL, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ZW, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, ZL, 1.0);
    PRIMAL_putcj(t, ZW, 1.0);

    /* count rows: NP*NDOT placement-edge bounds x2 + NP placement + conflict pairs */
    int nrow = 2*NP*NDOT + NP;
    for (int i = 0; i < NP; i++) for (int j = 0; j < NP; j++) if (i != j)
        for (int d1 = 0; d1 < NDOT; d1++) for (int d2 = 0; d2 < NDOT; d2++) {
            int i1 = d1/NG, j1 = d1%NG, i2 = d2/NG, j2 = d2%NG;
            if (abs(i1-i2) <= 1 && abs(j1-j2) <= 1) nrow++;
        }
    PRIMAL_appendcons(t, nrow);
    int r = 0;
    /* zL >= (0.5*i+1)*delta ; zW >= (0.5*j+1)*delta  ->  (coef)*delta - z <= 0 */
    for (int i = 0; i < NP; i++)
        for (int d = 0; d < NDOT; d++) {
            int id = d/NG, jd = d%NG;
            PRIMAL_putarow(t, r, 2, (int[]){X0+i*NDOT+d, ZL}, (double[]){0.5*id + 1.0, -1.0});
            PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 0.0); r++;
            PRIMAL_putarow(t, r, 2, (int[]){X0+i*NDOT+d, ZW}, (double[]){0.5*jd + 1.0, -1.0});
            PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 0.0); r++;
        }
    /* sum_d delta[i,d] = 1 */
    for (int i = 0; i < NP; i++) {
        int sub[NDOT]; double val[NDOT];
        for (int d = 0; d < NDOT; d++) { sub[d] = X0 + i*NDOT + d; val[d] = 1.0; }
        PRIMAL_putarow(t, r, NDOT, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 1.0, 1.0); r++;
    }
    /* no-overlap: delta[i,d1] + delta[j,d2] <= 1 for overlapping dots */
    for (int i = 0; i < NP; i++) for (int j = 0; j < NP; j++) if (i != j)
        for (int d1 = 0; d1 < NDOT; d1++) for (int d2 = 0; d2 < NDOT; d2++) {
            int i1 = d1/NG, j1 = d1%NG, i2 = d2/NG, j2 = d2%NG;
            if (abs(i1-i2) <= 1 && abs(j1-j2) <= 1) {
                PRIMAL_putarow(t, r, 2, (int[]){X0+i*NDOT+d1, X0+j*NDOT+d2}, (double[]){1.0, 1.0});
                PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 1.0); r++;
            }
        }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, v[64] = {0};
    if (ok) { PRIMAL_getxx(t, PRIMAL_SOL_ITR, v); PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj); }
    /* recover the chosen dots and check the placement is legal (no overlap) */
    int chosen[NP], legal = 1;
    for (int i = 0; i < NP; i++) {
        chosen[i] = -1;
        for (int d = 0; d < NDOT; d++) if (v[X0+i*NDOT+d] > 0.5) chosen[i] = d;
        if (chosen[i] < 0) legal = 0;
    }
    if (legal) for (int i = 0; i < NP; i++) for (int j = 0; j < NP; j++) if (i != j && chosen[i] >= 0 && chosen[j] >= 0) {
        int i1 = chosen[i]/NG, j1 = chosen[i]%NG, i2 = chosen[j]/NG, j2 = chosen[j]%NG;
        if (abs(i1-i2) <= 1 && abs(j1-j2) <= 1) legal = 0;
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    int good = ok && fabs(obj - 3.0) < 1e-6 && legal;
    printf("nesting_dotted_board (2 quadrati, board 2x2, griglia fine)\n");
    printf("  z = %.6f (atteso 3)  dot scelti = (%d,%d) / (%d,%d)  legale=%d  %s\n",
           obj, chosen[0]>=0?chosen[0]/NG:-1, chosen[0]>=0?chosen[0]%NG:-1,
           chosen[1]>=0?chosen[1]/NG:-1, chosen[1]>=0?chosen[1]%NG:-1, legal, good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
