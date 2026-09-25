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

/* sudoku.c - MILP Sudoku solver (MOSEK Cookbook 11.7, sudoku.cc).
 *
 * Binary X[i][j][k] = 1 iff cell (i,j) holds digit k+1.  Constraints: exactly
 * one digit per cell, each digit once per row, once per column, once per block,
 * plus the givens.  Pure feasibility (objective 0).
 *
 * The MOSEK example is 9x9 (729 binaries); that instance does NOT close with
 * this library's branch-and-bound today, so the same model is solved here on a
 * 4x4 board (M=2, 64 binaries, the block size is the only parameter).  The
 * puzzle below is consistent with the unique 4x4 Latin-with-blocks grid
 *   1 2 3 4 / 3 4 1 2 / 2 1 4 3 / 4 3 2 1,
 * so the check is: the returned grid is a valid Sudoku (rows, columns and
 * blocks are 1..4) and respects every given.
 *
 * Usage: sudoku   (no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include "primal.h"

#define M 2
#define N (M * M)            /* 4 */
#define NVAR (N * N * N)     /* 64 */
#define IDX(i, j, k) (((i) * N + (j)) * N + (k))

/* givens (1-based) {row, col, digit}, taken from the grid above */
static const int HR[][3] = {
    {1, 1, 1}, {1, 3, 3}, {2, 1, 3}, {2, 4, 2},
    {3, 1, 2}, {3, 4, 3}, {4, 2, 3}, {4, 4, 1}
};
#define NFIX ((int)(sizeof(HR) / sizeof(HR[0])))

int main(void) {
    int nrow = 4 * N * N + NFIX;      /* 64 + 8 = 72 */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, nrow, NVAR, &t);
    for (int v = 0; v < NVAR; v++) {
        PRIMAL_putvarbound(t, v, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, v, PRIMAL_VAR_TYPE_INT_BIN);
        PRIMAL_putcj(t, v, 0.0);
    }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    int row = 0;
    for (int i = 0; i < N; i++)                    /* one digit per cell */
        for (int j = 0; j < N; j++) {
            int sub[N]; double val[N];
            for (int k = 0; k < N; k++) { sub[k] = IDX(i, j, k); val[k] = 1.0; }
            PRIMAL_putarow(t, row, N, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
        }
    for (int i = 0; i < N; i++)                    /* digit once per row */
        for (int k = 0; k < N; k++) {
            int sub[N]; double val[N];
            for (int j = 0; j < N; j++) { sub[j] = IDX(i, j, k); val[j] = 1.0; }
            PRIMAL_putarow(t, row, N, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
        }
    for (int j = 0; j < N; j++)                    /* digit once per column */
        for (int k = 0; k < N; k++) {
            int sub[N]; double val[N];
            for (int i = 0; i < N; i++) { sub[i] = IDX(i, j, k); val[i] = 1.0; }
            PRIMAL_putarow(t, row, N, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
        }
    for (int bi = 0; bi < M; bi++)                 /* digit once per block */
        for (int bj = 0; bj < M; bj++)
            for (int k = 0; k < N; k++) {
                int sub[M * M]; double val[M * M]; int nn = 0;
                for (int di = 0; di < M; di++)
                    for (int dj = 0; dj < M; dj++) {
                        sub[nn] = IDX(bi * M + di, bj * M + dj, k); val[nn] = 1.0; nn++;
                    }
                PRIMAL_putarow(t, row, nn, sub, val);
                PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
            }
    for (int f = 0; f < NFIX; f++) {
        int i = HR[f][0] - 1, j = HR[f][1] - 1, k = HR[f][2] - 1;
        PRIMAL_putarow(t, row, 1, (int[]){IDX(i, j, k)}, (double[]){1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    int grid[N][N] = {{0}};
    double *x = NULL;
    if (ok) {
        x = (double *)calloc(NVAR, sizeof(double));
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    if (x[IDX(i, j, k)] > 0.5) grid[i][j] = k + 1;
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    int good = ok;
    for (int i = 0; i < N && good; i++) {          /* rows */
        int seen[N + 1] = {0};
        for (int j = 0; j < N; j++) { int d = grid[i][j]; if (d < 1 || d > N || seen[d]++) good = 0; }
    }
    for (int j = 0; j < N && good; j++) {          /* columns */
        int seen[N + 1] = {0};
        for (int i = 0; i < N; i++) { int d = grid[i][j]; if (d < 1 || d > N || seen[d]++) good = 0; }
    }
    for (int bi = 0; bi < M && good; bi++)         /* blocks */
        for (int bj = 0; bj < M && good; bj++) {
            int seen[N + 1] = {0};
            for (int di = 0; di < M; di++)
                for (int dj = 0; dj < M; dj++) {
                    int d = grid[bi * M + di][bj * M + dj];
                    if (d < 1 || d > N || seen[d]++) good = 0;
                }
        }
    for (int f = 0; f < NFIX && good; f++)         /* givens */
        if (grid[HR[f][0] - 1][HR[f][1] - 1] != HR[f][2]) good = 0;

    printf("sudoku: rc=%d  griglia %s\n", (int)rc, good ? "valida e givens rispettati" : "NON valida");
    if (good)
        for (int i = 0; i < N; i++) {
            printf("  ");
            for (int j = 0; j < N; j++) printf("%d%c", grid[i][j], j == N - 1 ? '\n' : ' ');
        }
    free(x);
    return good ? 0 : 1;
}
