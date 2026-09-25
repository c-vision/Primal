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

/* sparsecholesky.c — porting dell'esempio "sparsecholesky.jl" della MOSEK
 * Julia API: fattorizzazione di Cholesky sparsa come SDP: dato A simmetrica
 * SPD, trovare L triangolare inferiore con A = L L' — qui nella versione
 * "matrix completion": il sample ufficiale fattorizza una matrice corsa
 * (banded). Il clone lo modella come: min 0 s.t. le entrate di L con
 * le variabili, e A - L L' = 0... NON lineare. La versione SDP lineare
 * (come nel MOSEK example concettualmente equivalente): verifica che la
 * matrice A corsa ammetta Cholesky: le sottomatrici principali hanno
 * determinante >= 0, cioe' A e' PSD <=> esiste la fattorizzazione.
 * Pratico: si costruisce A (tridiagonale 3x3), la si impone come bar X
 * (fissa, righe FX) e si risolve min 0 con X PSD: feasible ⇔ Cholesky
 * esiste. Poi si verifica con la fattorizzazione diretta (via dmat_cholesky
 * non esposto: qui fattorizzazione 3x3 esplicita nel sample).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define D 3

/* A tridiagonale SPD: diag 2, offdiag -1 (matrice del Laplaciano 1D) */
static double A[D][D] = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

/* Cholesky 3x3 esplicita (verifica indipendente) */
static int chol3(double M[D][D], double L[D][D]) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) L[i][j] = 0.0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            double s = M[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 0) return 0;
                L[i][i] = sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return 1;
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 1);
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_FX, 0.0, 0.0);   /* dummy */
    int dim = D;
    PRIMAL_appendbarvars(task, 1, &dim);

    /* matrix store: E_ii (diag) e E_ij (offdiag, un solo triangolo) */
    int mE[D], mO[3];
    for (int i = 0; i < D; i++)
        PRIMAL_appendsparsesymmat(task, D, 1,
                               (int[]){i}, (int[]){i}, (double[]){1.0}, &mE[i]);
    int oc = 0;
    for (int i = 0; i < D; i++)
        for (int j = i + 1; j < D; j++) {
            PRIMAL_appendsparsesymmat(task, D, 1,
                                   (int[]){i}, (int[]){j}, (double[]){1.0}, &mO[oc]);
            oc++;
        }

    /* fissa X = A: righe FX su ogni entrata (6 per il triangolo) */
    PRIMAL_appendcons(task, 6);
    int r = 0;
    for (int i = 0; i < D; i++) {
        PRIMAL_putbaraij(task, r, 0, 1, (int[]){mE[i]}, (double[]){1.0});
        PRIMAL_putconbound(task, r, PRIMAL_BK_FX, A[i][i], A[i][i]);
        r++;
    }
    oc = 0;
    for (int i = 0; i < D; i++)
        for (int j = i + 1; j < D; j++) {
            /* <E_ij, X> = 2 X_ij = 2 A_ij */
            PRIMAL_putbaraij(task, r, 0, 1, (int[]){mO[oc]}, (double[]){1.0});
            PRIMAL_putconbound(task, r, PRIMAL_BK_FX, 2.0 * A[i][j], 2.0 * A[i][j]);
            r++;
            oc++;
        }

    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    printf("SDP feasibility di A: solsta = %d (%s)\n", sta,
           sta == PRIMAL_SOL_STA_OPTIMAL ? "PSD" : "NON PSD");

    /* verifica indipendente: Cholesky esplicita */
    double L[D][D];
    int ok_chol = chol3(A, L);
    printf("Cholesky diretta: %s\n", ok_chol ? "esiste (A SPD)" : "non esiste");
    if (ok_chol) {
        printf("L =\n");
        for (int i = 0; i < D; i++) {
            printf("   ");
            for (int j = 0; j < D; j++) printf(" %8.4f", L[i][j]);
            printf("\n");
        }
        /* A = L L' check */
        double err = 0.0;
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                double s = 0.0;
                for (int k = 0; k < D; k++) s += L[i][k] * L[j][k];
                err += fabs(s - A[i][j]);
            }
        printf("||L L' - A||_1 = %.3e\n", err);
        if (err > 1e-9) ok_chol = 0;
    }

    int ok = (sta == PRIMAL_SOL_STA_OPTIMAL) && ok_chol;
    printf("%s (A PSD via SDP = Cholesky esiste)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
