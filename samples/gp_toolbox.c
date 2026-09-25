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

/* gp_toolbox.c - a small geometric program (MOSEK Tutorials, gp-toolbox).
 *
 *   minimize    x + y^2 z
 *   subject to  0.1 sqrt(x) + 2 y^-1 <= 1
 *               z^-1 + y x^-2 <= 1
 *
 * A GP is convex under y = log x: every posynomial becomes a log-sum-exp,
 *   sum_i c_i exp(a_i . y) <= 1  <=>  p_i >= exp(a_i . y + ln c_i), sum p_i <= 1
 * i.e. one exponential cone (PRIMAL_CT_PEXP) per monomial plus a linear row.
 * The objective adds a -t and minimizes t.  This is the same encoder used by
 * samples/transformer_design.c, on the tutorial's 3-variable example.
 *
 * Usage: gp_toolbox   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

typedef struct { double coef; int n; int var[4]; double alpha[4]; } Mono;

/* pairs = {var0,exp0, var1,exp1, ...} */
static Mono mon(double coef, int n, const double *p) {
    Mono r; r.coef = coef; r.n = n;
    for (int i = 0; i < n; i++) { r.var[i] = (int)p[2 * i]; r.alpha[i] = p[2 * i + 1]; }
    return r;
}
#define MO(coef, ...) mon((coef), (int)(sizeof((double[]){__VA_ARGS__}) / sizeof(double) / 2), (double[]){__VA_ARGS__})

#define NV 3
int main(void) {
    /* objective and constraints, each a posynomial */
    Mono OBJ[2] = { MO(1.0, 0,1), MO(1.0, 1,2, 2,1) };          /* x + y^2 z */
    Mono C0[2]  = { MO(0.1, 0,0.5), MO(2.0, 1,-1) };            /* 0.1 sqrt(x) + 2/y */
    Mono C1[2]  = { MO(1.0, 2,-1), MO(1.0, 0,-2, 1,1) };        /* 1/z + y/x^2 */
    Mono *all[3] = { OBJ, C0, C1 };
    int nm[3] = { 2, 2, 2 };
    int isobj[3] = { 1, 0, 0 };
    int nposy = 3, nmulti = 6;

    /* vars: y[NV] | t | e_i (per monomial) | p_i (per monomial) | ONE */
    int YBASE = 0, TV = NV, EBASE = NV + 1, PBASE = EBASE + nmulti, ONEV = PBASE + nmulti;
    int NVAR = ONEV + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NVAR);
    for (int i = 0; i < NV; i++) PRIMAL_putvarbound(t, YBASE + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, TV, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nmulti; i++) PRIMAL_putvarbound(t, EBASE + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nmulti; i++) PRIMAL_putvarbound(t, PBASE + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ONEV, PRIMAL_BK_FX, 1.0, 1.0);

    int nrow = nposy + 2 * nmulti;               /* one sum row per posynomial + one link row per monomial */
    PRIMAL_appendcons(t, nrow);
    int r = 0, ecur = 0, pcur = 0;
    for (int i = 0; i < nposy; i++) {
        int p0 = pcur;
        for (int q = 0; q < nm[i]; q++) {
            int sub[5]; double val[5]; int nn = 0;
            sub[nn] = EBASE + ecur; val[nn] = 1.0; nn++;
            for (int z = 0; z < all[i][q].n; z++) { sub[nn] = YBASE + all[i][q].var[z]; val[nn] = -all[i][q].alpha[z]; nn++; }
            if (isobj[i]) { sub[nn] = TV; val[nn] = 1.0; nn++; }
            PRIMAL_putarow(t, r, nn, sub, val);
            PRIMAL_putconbound(t, r, PRIMAL_BK_FX, log(all[i][q].coef), log(all[i][q].coef)); r++;
            PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){PBASE + pcur, ONEV, EBASE + ecur});
            ecur++; pcur++;
        }
        int sub[3]; double val[3];
        for (int q = 0; q < nm[i]; q++) { sub[q] = PBASE + p0 + q; val[q] = 1.0; }
        PRIMAL_putarow(t, r, nm[i], sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 1.0); r++;
    }
    PRIMAL_putcj(t, TV, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double x[NV] = {0}, topt = 0.0, v[64] = {0};
    if (ok) { PRIMAL_getxx(t, PRIMAL_SOL_ITR, v); for (int i = 0; i < NV; i++) x[i] = exp(v[YBASE + i]); topt = v[TV]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* objective and the two constraint posynomials, by direct substitution */
    double obj = x[0] + x[1] * x[1] * x[2];
    double c0 = 0.1 * sqrt(x[0]) + 2.0 / x[1];
    double c1 = 1.0 / x[2] + x[1] / (x[0] * x[0]);
    int okall = ok && c0 < 1.0 + 1e-8 && c1 < 1.0 + 1e-8 && fabs(obj - exp(topt)) < 1e-7 * (1.0 + obj);
    printf("gp_toolbox  x=%.6f y=%.6f z=%.6f  obj = %.6f\n", x[0], x[1], x[2], obj);
    printf("  c0 = %.9f  c1 = %.9f  (entrambi <= 1)   exp(t) = %.6f\n", c0, c1, exp(topt));
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
