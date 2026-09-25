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

/* logistic.c — porting dell'esempio "logistic.jl" della MOSEK Julia API
 * (docs.mosek.com/11.0/juliaapi): regressione logistica in forma conica
 * tramite la funzione softmax (log-sum-exp).
 *
 *   min  t + 0.5*gamma*||x||^2
 *   s.t. y_i = a_i' x                    (i = 1..m)
 *        t >= sum_i exp(y_i)             (vincolo softmax)
 *
 * Il softmax si modella con coni esponenziali: per ogni i,
 * (v_i, 1, y_i) in PEXP  =>  v_i >= exp(y_i),  t = sum_i v_i.
 * Il termine quadratico (QP + coni non supportato, deviazione documentata
 * del clone) si assorbe nel cono RQUAD: (w, 0.5, x0, x1) in RQUAD  =>
 * 2*w*0.5 >= x0^2+x1^2, quindi 0.5*gamma*||x||^2 = gamma*w.
 *
 * Verifica indipendente: la funzione obiettivo in x e' liscia (softmax),
 * quindi l'ottimo si conferma con discesa numerica a gradiente finito.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define M 4
#define NX 2   /* variabili x */

static const double A[M][NX] = {
    { 2.0,  1.0},
    {-1.0,  1.0},
    { 1.0, -2.0},
    {-2.0, -1.0},
};
static const double GAMMA = 0.1;

/* obiettivo liscio f(x) = sum_i exp(a_i'x) + 0.5*gamma*||x||^2
 * (la forma conica minimizza direttamente sum exp, non il logsumexp) */
static double fval(const double *x) {
    double s = 0.0;
    for (int i = 0; i < M; i++)
        s += exp(A[i][0] * x[0] + A[i][1] * x[1]);
    return s + 0.5 * GAMMA * (x[0] * x[0] + x[1] * x[1]);
}

int main(void) {
    /* variabili: [x0, x1, v_0..v_{M-1}, u_0..u_{M-1}, one, w]
     *   v_i >= exp(u_i) tramite (v_i, one, u_i) in PEXP
     *   u_i = a_i'x      tramite righe lineari
     *   w >= 0.5*||x||^2 tramite (w, one, x0, x1) in RQUAD
     *   obj = sum_i v_i + gamma*w */
    const int NV = 2 + 2 * M + 2;
    const int V0 = 2, U0 = 2 + M, ONE = 2 + 2 * M, W = ONE + 1;

    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    PRIMAL_appendvars(task, NV);
    PRIMAL_appendcons(task, M);

    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < M; i++) PRIMAL_putcj(task, V0 + i, 1.0);
    PRIMAL_putcj(task, W, GAMMA);

    for (int i = 0; i < M; i++) {
        int sub[3] = {0, 1, U0 + i};
        double val[3] = {A[i][0], A[i][1], -1.0};
        PRIMAL_putarow(task, i, 3, sub, val);          /* a_i'x - u_i = 0 */
        PRIMAL_putconbound(task, i, PRIMAL_BK_FX, 0.0, 0.0);
    }
    for (int j = 0; j < NV; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    /* big-M sulle u_i: rende il LP di round 0 limitato (l'ottimo vero e'
     * vicino a 0, il taglio tangente di exp a round 0 non basta a limitare
     * x libera). Gli u_i restano liberi nella semantica del cono PEXP. */
    for (int i = 0; i < M; i++)
        PRIMAL_putvarbound(task, U0 + i, PRIMAL_BK_RA, -30.0, 30.0);

    for (int i = 0; i < M; i++) {
        int mem[3] = {V0 + i, ONE, U0 + i};
        PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, mem);
    }
    int qmem[4] = {W, ONE, 0, 1};
    PRIMAL_appendcone(task, PRIMAL_CT_RQUAD, 0.0, 4, qmem);

    PRIMAL_putintparam(task, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 5000);
    PRIMALrescodee rc = PRIMAL_optimize(task);
    if (rc != PRIMAL_RES_OK) { printf("optimize rc=%d\n", rc); return 1; }

    double xx[16], po;
    PRIMAL_getxx(task, PRIMAL_SOL_ITR, xx);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &po);
    printf("x = (%.6f, %.6f), obj = %.6f\n", xx[0], xx[1], po);

    /* check indipendente: discesa a gradiente su f */
    double xg[2] = {0.0, 0.0}, fbest = fval(xg), step = 0.5;
    for (int it = 0; it < 20000; it++) {
        double g[2];
        for (int d = 0; d < 2; d++) {
            double h = 1e-6, xp[2] = {xg[0], xg[1]}, xm[2] = {xg[0], xg[1]};
            xp[d] += h; xm[d] -= h;
            g[d] = (fval(xp) - fval(xm)) / (2 * h);
        }
        double xn[2] = {xg[0] - step * g[0], xg[1] - step * g[1]};
        double fn = fval(xn);
        if (fn < fbest) { fbest = fn; xg[0] = xn[0]; xg[1] = xn[1]; step *= 1.05; }
        else if (fn < fbest + 1e-12) { /* vicino: scala lo step */ step *= 0.7; }
        else step *= 0.5;
        if (step < 1e-12) break;
    }
    printf("gradiente numerico: f* = %.6f in (%.6f, %.6f)\n", fbest, xg[0], xg[1]);

    int ok = fabs(po - fbest) < 1e-3 &&
             fabs(xx[0] - xg[0]) < 1e-2 && fabs(xx[1] - xg[1]) < 1e-2;
    printf("%s (conico = gradiente numerico)\n", ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}