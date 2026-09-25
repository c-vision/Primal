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
 *
 * Port of a MOSEK example (see the comment below), rewritten against
 * the PrimalSolver (PRIMAL_*) API.  The MOSEK examples are Copyright (c)
 * MOSEK ApS; this port re-implements the same optimization problem and is
 * distributed under the Apache License, Version 2.0.  PrimalSolver is not
 * affiliated with, or endorsed by, MOSEK.
 */

/* nearestcorrelation.c — porting dell'esempio "nearestcorrelation.jl"
 * (C API docs.mosek.com): nearest correlation matrix (Higham).
 *
 *   min  ||A - X||_F  s.t.  X PSD, diag(X) = 1
 * modellato con:  min t  s.t. (t, vec(A-X)) in QUAD (norma Frobenius),
 * X bar PSD, righe FX X_ii = 1.
 *
 * A (3x3, dal tutorial MOSEK):
 *   A = [ 2  -1    0
 *        -1   2   -1
 *         0  -1   2 ] / scaling? qui: A come sopra ma con diag fuori scala
 * per rendere il problema non banale. La proiezione Higham (alternating
 * projections su PSD ∩ diag=1) fornisce la verifica indipendente.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define N 3

static double A[N][N] = {{2.0, -1.0, 0.0},
                         {-1.0, 2.0, -1.0},
                         {0.0, -1.0, 2.0}};

/* Frobenius norm */
static double fro(const double X[N][N]) {
    double s = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double d = X[i][j] - A[i][j];
            s += d * d;
        }
    return sqrt(s);
}

/* eigen decomposition wrapper (Jacobi via the library is static; use a
 * simple power-iteration deflation for the verification) */
static void eig3(const double M[N][N], double eval[N], double evec[N][N]) {
    /* Jacobi rotations, self-contained 3x3 */
    double W[N][N], V[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) { W[i][j] = M[i][j]; V[i][j] = (i == j); }
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < N; p++)
            for (int q = p + 1; q < N; q++) off += W[p][q] * W[p][q];
        if (off < 1e-30) break;
        for (int p = 0; p < N; p++)
            for (int q = p + 1; q < N; q++) {
                double apq = W[p][q];
                if (fabs(apq) < 1e-32) continue;
                double theta = (W[q][q] - W[p][p]) / (2.0 * apq);
                double t = (theta >= 0.0 ? 1.0 : -1.0) /
                           (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), sn = t * c;
                for (int k = 0; k < N; k++) {
                    double wp = W[p][k], wq = W[q][k];
                    W[p][k] = c * wp - sn * wq;
                    W[q][k] = sn * wp + c * wq;
                }
                for (int k = 0; k < N; k++) {
                    double wp = W[k][p], wq = W[k][q];
                    W[k][p] = c * wp - sn * wq;
                    W[k][q] = sn * wp + c * wq;
                }
                for (int k = 0; k < N; k++) {
                    double vp = V[k][p], vq = V[k][q];
                    V[k][p] = c * vp - sn * vq;
                    V[k][q] = sn * vp + c * vq;
                }
            }
    }
    for (int i = 0; i < N; i++) eval[i] = W[i][i];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) evec[i][j] = V[i][j];
}

/* Higham alternating projection: nearest correlation matrix (reference) */
static void higham(double X[N][N]) {
    /* start from A, then alternate: (1) project on PSD, (2) set diag = 1 */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) X[i][j] = A[i][j];
    for (int it = 0; it < 500; it++) {
        /* PSD projection: eigendecompose, clip negative eigenvalues */
        double ev[N], V[N][N], S[N][N];
        eig3((const double (*)[N])X, ev, V);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) S[i][j] = 0.0;
        for (int k = 0; k < N; k++) {
            double lk = ev[k] > 0.0 ? ev[k] : 0.0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    S[i][j] += lk * V[i][k] * V[j][k];
        }
        /* diag = 1 */
        for (int i = 0; i < N; i++) S[i][i] = 1.0;
        /* convergence check */
        double diff = 0.0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double d = fabs(S[i][j] - X[i][j]);
                if (d > diff) diff = d;
            }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) X[i][j] = S[i][j];
        if (diff < 1e-12) break;
    }
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: t (norma) + bar X 3x3.
     * ||A - X||_F = sqrt(sum_{i<=j} w_ij (A_ij - X_ij)^2) con w = 1 (diag)
     * o 2 (offdiag, per il doppio conteggio nel Frobenius).
     * Cono QUAD su (t, d_00..d_22) con d_ij = A_ij - X_ij (righe FX con
     * termini barA): 9 membri + t. Il primo membro e' t: t >= ||d||. */
    PRIMAL_appendvars(task, 1);         /* t */
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 1.0);         /* min t */

    /* matrici del matrix store: E_ij per ogni entrata (i<=j), UN SOLO
     * triangolo (la simmetria implicita conta l'offdiag due volte) */
    int msym[N * (N + 1) / 2];
    int cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
            int si[1], sj[1];
            double sv[1];
            si[0] = i; sj[0] = j; sv[0] = 1.0;
            PRIMAL_appendsparsesymmat(task, N, 1, si, sj, sv, &msym[cnt]);
            cnt++;
        }
    int dim = N;
    PRIMAL_appendbarvars(task, 1, &dim);

    /* righe: per ogni entrata (i,j): w_ij * <E_ij, X> + w_ij * d_ij = w_ij * A_ij
     * e per la diag: riga extra X_kk = 1.
     * d_ij sono variabili ausiliarie: aggiungiamole al task (9).
     * Layout variabili: [0]=t, [1..9]=d_ij (row-major i<=j), poi bar.
     * Ricalcolo: le d sono 6 (i<=j). variabili: t + 6 d. */
    PRIMAL_appendvars(task, N * (N + 1) / 2);
    for (int k = 0; k < N * (N + 1) / 2; k++)
        PRIMAL_putvarbound(task, 1 + k, PRIMAL_BK_FR, -INFINITY, INFINITY);

    /* righe: per (i,i):   X_ii + d_ii = A_ii            (d = A_ii - X_ii)
     *       per (i,j) i<j: 2 X_ij + sqrt(2) d_ij = 2 A_ij
     *         (<E_ij,X> = 2 X_ij per la simmetria; d_ij = sqrt(2)(A_ij - X_ij))
     * cosi' ||A-X||_F^2 = sum_{i<=j} d_ij^2 (offdiag contato due volte) e
     * il cono t >= ||d|| da' la norma Frobenius. */
    PRIMAL_appendcons(task, N * (N + 1) / 2 + N);
    int row = 0;
    cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
            double w = (i == j) ? 1.0 : 2.0;
            double rhs = (i == j) ? A[i][j] : 2.0 * A[i][j];
            PRIMAL_putbaraij(task, row, 0, 1, (int[]){msym[cnt]}, (double[]){1.0});
            int sub[1] = {1 + cnt};
            double val[1] = {sqrt(w)};
            PRIMAL_putarow(task, row, 1, sub, val);
            PRIMAL_putconbound(task, row, PRIMAL_BK_FX, rhs, rhs);
            row++; cnt++;
        }
    /* righe diag: X_kk = 1: <E_kk, X> = 1 */
    {
        int me[N];
        int k2 = 0;
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++) { if (i == j) me[i] = msym[k2]; k2++; }
        for (int k = 0; k < N; k++) {
            PRIMAL_putbaraij(task, row, 0, 1, (int[]){me[k]}, (double[]){1.0});
            PRIMAL_putconbound(task, row, PRIMAL_BK_FX, 1.0, 1.0);
            row++;
        }
    }

    /* cono: (t, d_00, d_11, d_22, d_01, d_02, d_12) in QUAD con pesi sqrt(w):
     * t >= ||(sqrt(w_ij) d_ij)||, w=1 per diag, w=2 per offdiag.
     * L'utente deve pre-scalare: d'_ij = sqrt(w)*d via... il cono QUAD non
     * pesa i membri: aggiungiamo i coefficienti nelle righe: d_ij definita
     * gia' come sqrt(w)*(A - X): riscrivo le righe sopra con -sqrt(w) su d.
     * Implementazione: coef su d = sqrt(w_ij). */
    {
        int mem[1 + N * (N + 1) / 2];
        mem[0] = 0;
        for (int k = 0; k < N * (N + 1) / 2; k++) mem[1 + k] = 1 + k;
        PRIMAL_appendcone(task, PRIMAL_CT_QUAD, 0.0, 1 + N * (N + 1) / 2, mem);
    }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double Xr[9], po;
    PRIMAL_getbarxj(task, PRIMAL_SOL_ITR, 0, Xr);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    double X[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) X[i][j] = Xr[i * N + j];
    printf("X =\n");
    for (int i = 0; i < N; i++) {
        printf("   ");
        for (int j = 0; j < N; j++) printf(" %8.4f", X[i][j]);
        printf("\n");
    }
    printf("obj (norma Frobenius) = %.6f\n", po);

    /* verifica indipendente: Higham alternating projections */
    double Xh[N][N];
    higham(Xh);
    printf("Higham:\n");
    for (int i = 0; i < N; i++) {
        printf("   ");
        for (int j = 0; j < N; j++) printf(" %8.4f", Xh[i][j]);
        printf("\n");
    }
    double fh = fro(Xh);
    double fx = fro(X);
    printf("||A-X_higham||_F = %.6f, ||A-X_clone||_F = %.6f\n", fh, fx);

    /* il clone deve essere almeno buono quanto Higham (entro tolleranza)
     * e ammissibile: diag=1, PSD (autovalori >= -1e-6) */
    double ev[N], V[N][N];
    eig3((const double (*)[N])X, ev, V);
    int psd_ok = 1;
    for (int k = 0; k < N; k++) if (ev[k] < -1e-6) psd_ok = 0;
    int diag_ok = 1;
    for (int i = 0; i < N; i++)
        if (fabs(X[i][i] - 1.0) > 1e-5) diag_ok = 0;
    int ok = psd_ok && diag_ok && fx <= fh + 1e-3;
    printf("%s (PSD=%s, diag=1=%s, norma ~ Higham)\n",
           ok ? "OK" : "FAIL", psd_ok ? "si" : "NO", diag_ok ? "si" : "NO");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
