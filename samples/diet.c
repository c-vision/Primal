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

/* diet.c - Stigler's nutrition model (the DIET LP), the classic MOSEK example
 * (AlexHahnPublic/PortfolioOptimizationMosek, Examples/diet.py, from the GAMS
 * library DIET / Dantzig 1963):
 *
 *   minimize    c' x
 *   subject to  N x >= b,   x >= 0
 * where N[i][j] is the amount of nutrient i per unit of food j, b is the daily
 * allowance of each nutrient, c the cost per unit of each food.
 *
 * Its LP dual is the "shadow price" problem
 *   maximize    b' y   s.t.  N' y <= c,  y >= 0,
 * and strong duality gives the same optimum -- an independent cross-check.
 *
 * Hand case: 3 foods A,B,C, 3 nutrients (protein, calories, vitamin):
 *   N = [[2,1,0],[1,2,0],[0,0,1]],  b = (4,5,1),  c = (1,1,2).
 * The vitamin only comes from C, so C = 1 (cost 2); the first two nutrients
 * give the 2x2 system 2A+B=4, A+2B=5 -> A=1, B=2 (cost 3).  Total cost 5 at
 * x = (1,2,1), all three constraints tight.  The dual optimum is y = (1/3,1/3,2)
 * with value 4/3 + 5/3 + 2 = 5.
 *
 * Usage: diet   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NF 3
#define NN 3

static const double N[NN][NF] = {{2.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 1.0}};
static const double BB[NN] = {4.0, 5.0, 1.0};
static const double CC[NF] = {1.0, 1.0, 2.0};

int main(void) {
    printf("diet (Stigler, 3 alimenti x 3 nutrienti)\n");
    int all = 1;

    /* primal: min c'x s.t. N x >= b, x >= 0 */
    double xp[NF] = {0}, cost = 0; int ok = 0;
    {
        PRIMALenv_t env; PRIMALtask_t t;
        PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
        PRIMAL_appendvars(t, NF);
        PRIMAL_appendcons(t, NN);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < NF; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        for (int i = 0; i < NN; i++) {
            int sub[NF]; double val[NF];
            for (int j = 0; j < NF; j++) { sub[j] = j; val[j] = N[i][j]; }
            PRIMAL_putarow(t, i, NF, sub, val);
            PRIMAL_putconbound(t, i, PRIMAL_BK_LO, BB[i], INFINITY);
        }
        for (int j = 0; j < NF; j++) PRIMAL_putcj(t, j, CC[j]);
        if (PRIMAL_optimize(t) == PRIMAL_RES_OK) {
            double x[8] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
            for (int j = 0; j < NF; j++) xp[j] = x[j];
            PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &cost);
            ok = 1;
        }
        PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    }
    /* nutrient totals at xp and the deficit */
    double nut[NN], mindef = 0;
    for (int i = 0; i < NN; i++) { double s = 0; for (int j = 0; j < NF; j++) s += N[i][j]*xp[j];
        nut[i] = s; if (BB[i]-s > mindef) mindef = BB[i]-s; }
    int good = ok && fabs(cost - 5.0) < 1e-7 && fabs(xp[0]-1.0) < 1e-7 &&
               fabs(xp[1]-2.0) < 1e-7 && fabs(xp[2]-1.0) < 1e-7 && mindef < 1e-9;
    printf("  primale: cost=%.6f (atteso 5)  x=(%.4f,%.4f,%.4f)  deficit=%.2e  %s\n",
           cost, xp[0], xp[1], xp[2], mindef, good ? "OK" : "FAIL");
    all &= good;

    /* dual: max b'y s.t. N'y <= c, y >= 0 */
    double yd[NN] = {0}, value = 0; ok = 0;
    {
        PRIMALenv_t env; PRIMALtask_t t;
        PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
        PRIMAL_appendvars(t, NN);
        PRIMAL_appendcons(t, NF);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
        for (int i = 0; i < NN; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
        for (int j = 0; j < NF; j++) {
            int sub[NN]; double val[NN];
            for (int i = 0; i < NN; i++) { sub[i] = i; val[i] = N[i][j]; }   /* N'y : row j = sum_i N[i][j] y_i */
            PRIMAL_putarow(t, j, NN, sub, val);
            PRIMAL_putconbound(t, j, PRIMAL_BK_UP, -INFINITY, CC[j]);
        }
        for (int i = 0; i < NN; i++) PRIMAL_putcj(t, i, BB[i]);
        if (PRIMAL_optimize(t) == PRIMAL_RES_OK) {
            double y[8] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, y);
            for (int i = 0; i < NN; i++) yd[i] = y[i];
            PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &value);
            ok = 1;
        }
        PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    }
    double vmax = 0;
    for (int j = 0; j < NF; j++) { double s = 0; for (int i = 0; i < NN; i++) s += N[i][j]*yd[i];
        if (s-CC[j] > vmax) vmax = s-CC[j]; }
    good = ok && fabs(value - 5.0) < 1e-7 && vmax < 1e-9 &&
           fabs(value - cost) < 1e-7;
    printf("  duale:   value=%.6f (atteso 5)  y=(%.4f,%.4f,%.4f)  viol=%.2e  dualita' forte %s\n",
           value, yd[0], yd[1], yd[2], vmax, good ? "OK" : "FAIL");
    all &= good;

    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
