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

/* omf_nearestcorr.c - nearest correlation matrix, "Optimization Methods in
 * Finance" (Cornuejols & Tutuncu, 2007), Example 10.1 (lines 4443-4451):
 *
 * Given the (invalid) correlation estimate
 *   Sigma_hat = [ 1.0 0.8 0.5 0.2
 *                 0.8 1.0 0.9 0.1
 *                 0.5 0.9 1.0 0.7
 *                 0.2 0.1 0.7 1.0 ]   (smallest eigenvalue -0.1337),
 * find the nearest valid correlation matrix, i.e.
 *   min ||Sigma - Sigma_hat||_F   s.t.  Sigma >= 0,  diag(Sigma) = 1.
 *
 * The book (via SDPT3) reports (approximately)
 *   Sigma = [ 1.00 0.76 0.53 0.18
 *             0.76 1.00 0.82 0.15
 *             0.53 0.82 1.00 0.65
 *             0.18 0.15 0.65 1.00 ].
 *
 * Model: bar variable X (4x4, PSD); for each i<=j an auxiliary d_ij with
 *   X_ii + d_ii = Sigma_hat_ii,   2 X_ij + sqrt(2) d_ij = 2 Sigma_hat_ij,
 * so that sum_{i<=j} d_ij^2 = ||Sigma_hat - X||_F^2; the Frobenius norm is the
 * epigraph t of a quadratic cone (t, d) in QUAD.  Diagonal rows X_kk = 1.
 *
 * Usage: omf_nearestcorr   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define N 4

static const double A[N][N] = {
    {1.0, 0.8, 0.5, 0.2},
    {0.8, 1.0, 0.9, 0.1},
    {0.5, 0.9, 1.0, 0.7},
    {0.2, 0.1, 0.7, 1.0}};

/* reference from the book (2 significant digits) */
static const double REF[N][N] = {
    {1.00, 0.76, 0.53, 0.18},
    {0.76, 1.00, 0.82, 0.15},
    {0.53, 0.82, 1.00, 0.65},
    {0.18, 0.15, 0.65, 1.00}};

/* independent reference: Higham alternating projections (computed offline) */
static const double HIGHAM[N][N] = {
    {1.0000, 0.7698, 0.5301, 0.1823},
    {0.7698, 1.0000, 0.8169, 0.1488},
    {0.5301, 0.8169, 1.0000, 0.6513},
    {0.1823, 0.1488, 0.6513, 1.0000}};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t task;
    const int nd = N * (N + 1) / 2;
    int msym[nd], cnt;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &task);

    /* variables: 0 = t (Frobenius norm), 1..nd = d_ij */
    PRIMAL_appendvars(task, 1 + nd);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);
    for (int k = 0; k < nd; k++)
        PRIMAL_putvarbound(task, 1 + k, PRIMAL_BK_FR, -INFINITY, INFINITY);

    cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
            int si[1] = {i}, sj[1] = {j};
            double sv[1] = {1.0};
            PRIMAL_appendsparsesymmat(task, N, 1, si, sj, sv, &msym[cnt++]);
        }
    int dim = N;
    PRIMAL_appendbarvars(task, 1, &dim);
    PRIMAL_appendcons(task, nd + N);

    int row = 0;
    cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
            double w = (i == j) ? 1.0 : 2.0;
            double rhs = (i == j) ? A[i][j] : 2.0 * A[i][j];
            PRIMAL_putbaraij(task, row, 0, 1, (int[]){msym[cnt]}, (double[]){1.0});
            PRIMAL_putarow(task, row, 1, (int[]){1 + cnt}, (double[]){sqrt(w)});
            PRIMAL_putconbound(task, row, PRIMAL_BK_FX, rhs, rhs);
            row++; cnt++;
        }
    for (int k = 0; k < N; k++) {           /* X_kk = 1 */
        PRIMAL_putbaraij(task, row, 0, 1, (int[]){msym[k * (2 * N - k + 1) / 2]}, (double[]){1.0});
        PRIMAL_putconbound(task, row, PRIMAL_BK_FX, 1.0, 1.0);
        row++;
    }
    {   /* quadratic cone (t, d_00..): t >= ||d|| = ||Sigma_hat - X||_F */
        int mem[1 + nd];
        mem[0] = 0;
        for (int k = 0; k < nd; k++) mem[1 + k] = 1 + k;
        PRIMAL_appendcone(task, PRIMAL_CT_QUAD, 0.0, 1 + nd, mem);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double Xr[N * N], po, fro = 0.0, dev_book = 0.0, dev_high = 0.0;
        PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, Xr);
        PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double xij = Xr[i * N + j];
                double d = xij - A[i][j]; fro += d * d;
                if (fabs(xij - REF[i][j]) > dev_book) dev_book = fabs(xij - REF[i][j]);
                if (fabs(xij - HIGHAM[i][j]) > dev_high) dev_high = fabs(xij - HIGHAM[i][j]);
            }
        fro = sqrt(fro);
        ok = dev_high < 5e-3 && fabs(Xr[0] - 1.0) < 1e-5 && fabs(po - fro) < 1e-4;
        printf("omf_nearestcorr  ||Sig-X||_F=%.6g  max|X-Higham|=%.4g  max|X-book|=%.4g (book 2 sig figs) %s\n",
               fro, dev_high, dev_book, ok ? "OK" : "FAIL");
    } else {
        printf("omf_nearestcorr  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&task); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
