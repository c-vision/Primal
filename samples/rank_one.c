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

/* rank_one.c - best-subset selection by rank-one convexification (MOSEK
 * Tutorials, rank-one-regression).  The notebook solves, with a disjunction
 * (DJC) selecting the support,
 *
 *   minimize  y'y - 2 y'X b + t + mu * sum_i u_i
 *   subject to  t >= b' (X'X + lambda I) b     (rotated cone)
 *               sum_i z_i <= k,   z_i in {0,1}
 *               |b_i| <= u_i
 *               (z_i == 1) OR (b_i == 0)        (one DJC per i)
 *
 * The disjunction is written with the reference DJC API (appendafes +
 * putdjc on RZERO domains), exactly as the notebook's
 * `M.disjunction((z == 1) | (B == 0))`.
 *
 * Hand-derived instance: X = [[1,0],[1,1],[0,1]], y = (1,2,1), k=1,
 * lambda = 0, mu = 0.  The full least squares fit is b = (1,1); with
 * card(b) <= 1 the best is b = (3/2, 0) (or (0, 3/2)) giving
 *   y'y - 2 y'Xb + b'X'Xb = 6 - 9 + 9/2 = 3/2.
 *
 * Usage: rank_one   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NPTS 3
#define NF 2
#define LAM 0.0
#define MU 0.0
#define KCAP 1

static const double XP[NPTS + NF][NF] = {{1, 0}, {1, 1}, {0, 1}, {0, 0}, {0, 0}};
static const double Y[NPTS] = {1, 2, 1};

int main(void) {
    enum { B = 0, U = NF, TT = 2 * NF, Z = 2 * NF + 1, W = 3 * NF + 1,
           HT = 3 * NF + 1 + (NPTS + NF), ONE = HT + 1, NV = HT + 2 };
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, (NPTS + NF) + 1 + 1 + 2 * NF);   /* W links + HT + card + |b|<=u */
    for (int i = 0; i < NF; i++) {
        PRIMAL_putvarbound(t, B + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, U + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(t, Z + i, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, Z + i, PRIMAL_VAR_TYPE_INT_BIN);
    }
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int k = 0; k < NPTS + NF; k++) PRIMAL_putvarbound(t, W + k, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, HT, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    int row = 0;
    for (int k = 0; k < NPTS + NF; k++) {        /* W_k - sum_j Xp[k][j] B_j = 0 */
        int sub[NF + 1]; double val[NF + 1]; int nn = 0;
        sub[nn] = W + k; val[nn] = 1.0; nn++;
        for (int j = 0; j < NF; j++) if (XP[k][j] != 0.0) { sub[nn] = B + j; val[nn] = -XP[k][j]; nn++; }
        PRIMAL_putarow(t, row, nn, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
    }
    PRIMAL_putarow(t, row, 2, (int[]){HT, TT}, (double[]){1.0, -0.5});
    PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
    {   int sub[NF]; double val[NF];
        for (int i = 0; i < NF; i++) { sub[i] = Z + i; val[i] = 1.0; }
        PRIMAL_putarow(t, row, NF, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, KCAP); row++;
    }
    for (int i = 0; i < NF; i++) {               /* |B_i| <= U_i */
        PRIMAL_putarow(t, row, 2, (int[]){B + i, U + i}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;
        PRIMAL_putarow(t, row, 2, (int[]){B + i, U + i}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0); row++;
    }
    {   int mem[2 + (NPTS + NF)]; mem[0] = ONE; mem[1] = HT;
        for (int k = 0; k < NPTS + NF; k++) mem[2 + k] = W + k;
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 2 + (NPTS + NF), mem);
    }
    double coeff[NF];
    for (int j = 0; j < NF; j++) { coeff[j] = 0.0; for (int k = 0; k < NPTS; k++) coeff[j] += XP[k][j] * Y[k]; }
    for (int j = 0; j < NF; j++) PRIMAL_putcj(t, B + j, -2.0 * coeff[j]);
    PRIMAL_putcj(t, TT, 1.0);
    for (int i = 0; i < NF; i++) PRIMAL_putcj(t, U + i, MU);
    {   double yy = 0.0; for (int k = 0; k < NPTS; k++) yy += Y[k] * Y[k]; PRIMAL_putcfix(t, yy); }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    /* disjunction (z_i == 1) OR (b_i == 0) */
    PRIMAL_appendafes(t, 2 * NF);
    for (int i = 0; i < NF; i++) {
        PRIMAL_putafefentry(t, i, Z + i, 1.0);
        PRIMAL_putafefentry(t, NF + i, B + i, 1.0);
    }
    PRIMALint64t rzero;
    PRIMAL_appendrzerodomain(t, 1, &rzero);
    PRIMAL_appenddjcs(t, NF);
    for (int i = 0; i < NF; i++) {
        PRIMALint64t dom[2] = {rzero, rzero};
        PRIMALint64t afe[2] = {i, NF + i};
        double bb[2] = {1.0, 0.0};
        PRIMALint64t ts[2] = {1, 1};
        PRIMALrescodee drc = PRIMAL_putdjc(t, i, 2, dom, 2, afe, bb, 2, ts);
        if (drc != PRIMAL_RES_OK) { printf("rank_one  putdjc rc=%d FAIL\n", (int)drc); return 1; }
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, b0 = 0, b1 = 0;
    if (ok) {
        double x[64] = {0};
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        b0 = x[B]; b1 = x[B + 1];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    int onezero = (fabs(b0) < 1e-6) || (fabs(b1) < 1e-6);
    int okall = ok && fabs(obj - 1.5) < 1e-6 && onezero &&
                (fabs(fabs(b0) - 1.5) < 1e-5 || fabs(fabs(b1) - 1.5) < 1e-5);
    printf("rank_one  k=%d\n", KCAP);
    printf("  b = (%.8f, %.8f)   obiettivo = %.10f (atteso 1.5)\n", b0, b1, obj);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
