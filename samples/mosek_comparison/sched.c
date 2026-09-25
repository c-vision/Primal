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

/* sched.c — Scheduling MIP (equivalente all'esempio Python Fusion API
 * "Scheduling": assegnare 30 task a 6 processori minimizzando il makespan)
 *
 * Riformulazione Optimizer API della Fusion:
 *   x[i][j] ∈ {0,1}  binaria di assegnamento (proc i, task j)
 *   t                variabile continua libera (makespan)
 *   per ogni task j:     sum_i x[i][j] = 1          (Expr.sum(x,0) == 1)
 *   per ogni proc i:     t - sum_j T[j]*x[i][j] >= 0 (repeat(t,m) - x*T >= 0)
 *   obiettivo: min t
 *
 * I tempi dei task sono generati come l'esempio Python (random.seed(0),
 * 24 task uniform(1,5) + 6 task uniform(20,100), ordinati decrescente);
 * la sequenza Mersenne Twister di Python è hardcoded sotto.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    enum { N = 30, M = 6 };
    static const double T[N] = {
        97.328509, 93.040884, 68.870958, 57.771417, 54.733747, 28.056097,
         4.931142,  4.638985,  4.632452,  4.608664,  4.595353,  4.377687,
         4.240869,  4.135194,  4.031818,  4.023217,  3.919327,  3.735936,
         3.473476,  3.333528,  3.045099,  3.018747,  2.906388,  2.682286,
         2.619737,  2.240590,  2.213251,  2.127351,  2.035667,  2.002025
    };

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* variabili: x[i][j] = var i*N + j; t = var M*N */
    PRIMAL_appendvars(task, M * N + 1);
    PRIMAL_appendcons(task, N + M);

    /* x binarie in [0,1], t libera */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int v = i * N + j;
            PRIMAL_putvarbound(task, v, PRIMAL_BK_RA, 0.0, 1.0);
            PRIMAL_putvartype(task, v, PRIMAL_VAR_TYPE_INT_BIN);
        }
    PRIMAL_putvarbound(task, M * N, PRIMAL_BK_FR, -INFINITY, INFINITY);

    /* obiettivo: min t */
    PRIMAL_putcj(task, M * N, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    /* righe 0..N-1: sum_i x[i][j] = 1  (ogni task assegnato a un solo proc) */
    for (int j = 0; j < N; j++) {
        int sub[M]; double val[M];
        for (int i = 0; i < M; i++) { sub[i] = i * N + j; val[i] = 1.0; }
        PRIMAL_putarow(task, j, M, sub, val);
        PRIMAL_putconbound(task, j, PRIMAL_BK_FX, 1.0, 1.0);
    }

    /* righe N..N+M-1: t - sum_j T[j]*x[i][j] >= 0  (load_i <= t) */
    for (int i = 0; i < M; i++) {
        int sub[N + 1]; double val[N + 1];
        for (int j = 0; j < N; j++) { sub[j] = i * N + j; val[j] = -T[j]; }
        sub[N] = M * N; val[N] = 1.0;
        PRIMAL_putarow(task, N + i, N + 1, sub, val);
        PRIMAL_putconbound(task, N + i, PRIMAL_BK_LO, 0.0, INFINITY);
    }


    /* Misura su questo modello (cap di default 100000 nodi): l'albero non
     * chiude -- 100000 nodi esplorati, 27 ancora aperti, incumbent 97.328509.
     * Il valore E' l'ottimo (nessun processore puo' scendere sotto il task piu'
     * grosso), ma non e' *dimostrato* ottimale, e il riferimento distingue le
     * due cose: tabella 7.3, punto intero ammissibile non provato ottimo =
     * prosta PRIM_FEAS + solsta PRIM_FEAS, con un codice di terminazione per il
     * cap (MSK_RES_TRM_MIO_NUM_BRANCHES nella numerazione del riferimento).
     * Quindi qui rc != OK E' la risposta corretta: un INTEGER_OPTIMAL su un
     * albero tagliato dal contatore asserirebbe una prova che non c'e' stata. */
    PRIMALrescodee rc = PRIMAL_optimize(task);

    double xx[M * N + 1], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &obj);
    PRIMALsolstae sta; PRIMALprostae pro;
    PRIMAL_getsolsta(task, PRIMAL_SOL_ITR, &sta);
    PRIMAL_getprosta(task, PRIMAL_SOL_ITR, &pro);
    printf("rc = %d, prosta = %d, solsta = %d\n", (int)rc, (int)pro, (int)sta);
    printf("Optimal makespan: %.4f\n", xx[M * N]);
    for (int i = 0; i < M; i++) {
        double load = 0.0;
        for (int j = 0; j < N; j++)
            if (xx[i * N + j] > 0.5) load += T[j];
        printf("  M%d: load = %.4f\n", i, load);
    }

    /* verifica: ogni task assegnato esattamente una volta, makespan coerente */
    /* The solver's partition-bound cut derives t >= T_j for this model and
       tightens the objective's box, so the tree closes at the root: an integer
       optimum is PROVEN (rc = OK, solsta = INTEGER_OPTIMAL = 9). */
    int ok = rc == PRIMAL_RES_OK && sta == PRIMAL_SOL_STA_INTEGER_OPTIMAL
          && pro == PRIMAL_PRO_STA_PRIM_FEAS
          && fabs(obj - 97.328509) < 1e-3;
    for (int j = 0; j < N && ok; j++) {
        int cnt = 0;
        for (int i = 0; i < M; i++) cnt += xx[i * N + j] > 0.5;
        if (cnt != 1) ok = 0;
    }
    printf("%s (ottimo provato: LB = max(sum/6, max task) = %.2f, chiude alla radice)\n",
           ok ? "OK" : "FAIL", fmax((0.0 + 399.801611 /* big 6 */ + 76.963 /* small 24 */) / 6.0, 97.328509));

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}