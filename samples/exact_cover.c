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

/* exact_cover.c - exact planar cover with rectangular bricks, by binary MIP.
 *
 * Source: MOSEK/Tutorials, exact-planar-cover/exactcover.py.  Given a board of
 * R x C unit cells and a set of brick shapes, choose placements so that every
 * cell is covered exactly once:
 *
 *   x_{s,r,c} in {0,1}   (shape s anchored at row r, column c)
 *   sum_{placements covering (i,j)} x = 1     for every cell (i,j)
 *   minimize  sum x                            (fewest bricks)
 *
 * a 0/1 exact-cover model solved as a MIP.  The notebook also reads a relaxed
 * infeasibility certificate when no cover exists (all variables made
 * continuous); the infeasible case is reported here by the solver status.
 *
 * Hand-derived instances:
 *   A) 2 x 3 board, dominoes 1x2 / 2x1  -> exactly 3 bricks (area/2)
 *   B) 3 x 3 board, 2 x 2 squares       -> infeasible (area 9 not divisible by 4)
 *
 * Usage: exact_cover   (no arguments)
 */
#include <stdio.h>
#include "primal.h"

#define MAXR 4
#define MAXC 4
#define MAXS 4
#define MAXV 128

/* returns 1 if solved to optimality, writes the number of bricks used;
 * returns 0 if the model is infeasible; returns -1 on another failure. */
static int cover(int R, int C, int ns, const int h[MAXS], const int w[MAXS],
                 int *nbricks_out) {
    /* enumerate placements and the coverage rows */
    int var[MAXS][MAXR][MAXC];   /* var index or -1 */
    int nv = 0;
    for (int s = 0; s < ns; s++)
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                var[s][r][c] = (r + h[s] <= R && c + w[s] <= C) ? nv++ : -1;
            }
    if (nv > MAXV) return -1;

    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_appendcons(t, R * C);
    for (int k = 0; k < nv; k++) {
        PRIMAL_putvarbound(t, k, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, k, PRIMAL_VAR_TYPE_INT_BIN);
        PRIMAL_putcj(t, k, 1.0);
    }
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            int sub[MAXV]; double val[MAXV]; int nn = 0;
            for (int s = 0; s < ns; s++)
                for (int r = 0; r <= i; r++)
                    for (int c = 0; c <= j; c++) {
                        if (var[s][r][c] < 0) continue;
                        if (i < r + h[s] && j < c + w[s]) { sub[nn] = var[s][r][c]; val[nn] = 1.0; nn++; }
                    }
            PRIMAL_putarow(t, i * C + j, nn, sub, val);
            PRIMAL_putconbound(t, i * C + j, PRIMAL_BK_FX, 1.0, 1.0);
        }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ret = -1;
    if (rc == PRIMAL_RES_OK) {
        double obj = 0.0;
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        *nbricks_out = (int)(obj + 0.5);
        ret = 1;
    } else if (rc == PRIMAL_RES_ERR_INFEASIBLE || rc == PRIMAL_RES_ERR_UNBOUNDED) {
        ret = 0;                  /* infeasible (no exact cover) */
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ret;
}

int main(void) {
    int h1[2] = {1, 2}, w1[2] = {2, 1};      /* dominoes        */
    int h2[1] = {2},   w2[1] = {2};          /* 2x2 squares     */
    int nb = 0;
    int a = cover(2, 3, 2, h1, w1, &nb);
    int na = nb;
    int b = cover(3, 3, 1, h2, w2, &nb);

    int ok = (a == 1 && na == 3) && (b == 0);
    printf("exact_cover\n");
    printf("  A) 2x3, domino: %s  bricks=%d (atteso 3)\n",
           a == 1 ? "ottimo" : (a == 0 ? "infeasible" : "errore"), na);
    printf("  B) 3x3, 2x2:    %s (atteso infeasible)\n",
           b == 1 ? "ottimo" : (b == 0 ? "infeasible" : "errore"));
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
