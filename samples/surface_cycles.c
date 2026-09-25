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

/* surface_cycles.c - shortest cycle in a homology class (MOSEK Tutorials,
 * surfacecycles).
 *
 * Given a triangulated surface and a 1-chain C, the shortest homologous chain
 * is the LP
 *      min  sum_e w_e |x_e|   s.t.  x - C = D2 y
 * where D2 is the boundary operator (edges x triangles) and y a 2-chain.  The
 * absolute value is linearized with xabs_e >= +-x_e.
 *
 * This sample builds a triangulated square cylinder (an open surface with
 * betti1 = 1) directly in code, so no .ply parsing is needed, and shortens a
 * cycle C = (top ring) + D2*(one triangle) -- homologous to the ring but not
 * minimal.  The shortest representative of the "around" class is the ring of 4
 * unit edges: min-cost homologous cycle = min-cut between the two ends = 4.
 *
 * Usage: surface_cycles   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define N 4                    /* sides of the cylinder */
#define NV (2 * N)             /* 8 vertices */
#define NT (2 * N)             /* 8 triangles */
#define NE (4 * N)             /* 2 rings + verticals + diagonals = 16 */

static int F[NT][3];
static int E[NE][2];
static double W[NE];
static int D1[NV][NE];         /* vertices x edges */
static int D2[NE][NT];         /* edges x triangles */

static int edge_id(int a, int b) {         /* the undirected edge {a,b} */
    if (a > b) { int t = a; a = b; b = t; }
    for (int i = 0; i < NE; i++) if (E[i][0] == a && E[i][1] == b) return i;
    return -1;
}
/* signed edge: coefficient +1 if a->b matches the stored orientation (min,max) */
static int edge_sgn(int a, int b, double *s) { *s = (a <= b) ? 1.0 : -1.0; return edge_id(a, b); }

int main(void) {
    double xyz[NV][3];
    const double sq[4][2] = {{0.5, 0.5}, {-0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}};
    for (int i = 0; i < N; i++) {
        xyz[i][0] = sq[i][0]; xyz[i][1] = sq[i][1]; xyz[i][2] = 0.0;       /* bottom ring */
        xyz[N + i][0] = sq[i][0]; xyz[N + i][1] = sq[i][1]; xyz[N + i][2] = 1.0; /* top ring */
    }
    /* faces: each side quad (i, i+1, N+i+1, N+i) -> 2 triangles */
    int nf = 0;
    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N;
        F[nf][0] = i; F[nf][1] = j; F[nf][2] = N + j; nf++;
        F[nf][0] = i; F[nf][1] = N + j; F[nf][2] = N + i; nf++;
    }
    /* unique edges from faces */
    int ne = 0;
    for (int t = 0; t < nf; t++)
        for (int q = 0; q < 3; q++) {
            int a = F[t][q], b = F[t][(q + 1) % 3];
            if (edge_id(a, b) < 0) { if (a > b) { int s = a; a = b; b = s; } E[ne][0] = a; E[ne][1] = b; ne++; }
        }
    if (ne != NE) { printf("surface_cycles  edge count %d != %d FAIL\n", ne, NE); return 1; }
    for (int e = 0; e < NE; e++) {
        int a = E[e][0], b = E[e][1];
        W[e] = sqrt((xyz[a][0]-xyz[b][0])*(xyz[a][0]-xyz[b][0]) +
                    (xyz[a][1]-xyz[b][1])*(xyz[a][1]-xyz[b][1]) +
                    (xyz[a][2]-xyz[b][2])*(xyz[a][2]-xyz[b][2]));
    }
    /* D1 (vertices x edges): boundary of edge (a,b) = b - a */
    for (int i = 0; i < NV; i++) for (int e = 0; e < NE; e++) D1[i][e] = 0;
    for (int e = 0; e < NE; e++) { D1[E[e][0]][e] = -1; D1[E[e][1]][e] = +1; }
    /* D2 (edges x triangles): boundary of (i,j,k) = (j,k) - (i,k) + (i,j) */
    for (int e = 0; e < NE; e++) for (int t = 0; t < nf; t++) D2[e][t] = 0;
    for (int t = 0; t < nf; t++) {
        double s0, s1, s2;
        int e0 = edge_sgn(F[t][1], F[t][2], &s0);
        int e1 = edge_sgn(F[t][0], F[t][2], &s1);
        int e2 = edge_sgn(F[t][0], F[t][1], &s2);
        D2[e0][t] += (int)s0; D2[e1][t] -= (int)s1; D2[e2][t] += (int)s2;
    }
    /* C = top ring (edges N..2N-1) + D2 * (one triangle) : homologous, not minimal */
    double Cring[NE]; for (int e = 0; e < NE; e++) Cring[e] = 0.0;
    for (int i = 0; i < N; i++) { double sg; int e = edge_sgn(N + i, N + (i + 1) % N, &sg); Cring[e] += sg; }
    double y0[NT]; for (int t = 0; t < nf; t++) y0[t] = 0.0; y0[0] = 1.0;
    double C[NE]; for (int e = 0; e < NE; e++) { C[e] = Cring[e]; for (int t = 0; t < nf; t++) C[e] += D2[e][t] * y0[t]; }
    {   /* the ring itself, x=Cring with y=-y0, must be a feasible point: value 4 */
        double rv = 0.0, rval = 0.0;
        for (int e = 0; e < NE; e++) {
            double s = Cring[e] - C[e];
            for (int t = 0; t < nf; t++) s -= D2[e][t] * (-y0[t]);
            rv = fmax(rv, fabs(s)); rval += W[e] * fabs(Cring[e]);
        }
        printf("  ring candidate: value=%.6f  residual=%.2e\n", rval, rv);
    }

    /* ---- LP: vars x(E), xabs(E), y(T) ---- */
    int XB = 0, AB = NE, YB = 2 * NE, NVAR = 2 * NE + nf;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NVAR);
    for (int i = 0; i < NVAR; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    int nrow = 2 * NE + NE;
    PRIMAL_appendcons(t, nrow);
    int r = 0;
    /* xabs >= x  ->  x - xabs <= 0 */
    for (int e = 0; e < NE; e++) {
        PRIMAL_putarow(t, r, 2, (int[]){XB + e, AB + e}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 0.0); r++;
    }
    /* xabs >= -x  ->  -x - xabs <= 0 */
    for (int e = 0; e < NE; e++) {
        PRIMAL_putarow(t, r, 2, (int[]){XB + e, AB + e}, (double[]){-1.0, -1.0});
        PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 0.0); r++;
    }
    /* x - D2 y = C */
    for (int e = 0; e < NE; e++) {
        int sub[1 + NT]; double val[1 + NT]; int nn = 0;
        sub[nn] = XB + e; val[nn] = 1.0; nn++;
        for (int tt = 0; tt < nf; tt++) if (D2[e][tt]) { sub[nn] = YB + tt; val[nn] = -(double)D2[e][tt]; nn++; }
        PRIMAL_putarow(t, r, nn, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, C[e], C[e]); r++;
    }
    for (int e = 0; e < NE; e++) PRIMAL_putcj(t, AB + e, W[e]);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    { PRIMALprostae ps = PRIMAL_PRO_STA_UNKNOWN; PRIMALsolstae ss = PRIMAL_SOL_STA_UNKNOWN;
      PRIMAL_getprosta(t, PRIMAL_SOL_ITR, &ps); PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss);
      printf("  rc=%d prosta=%d solsta=%d\n", (int)rc, (int)ps, (int)ss); }
    double x[NE] = {0}, xabs[NE] = {0}, y[NT] = {0}, v[NVAR];
    if (ok) {
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        for (int e = 0; e < NE; e++) { x[e] = v[XB + e]; xabs[e] = v[AB + e]; }
        for (int tt = 0; tt < nf; tt++) y[tt] = v[YB + tt];
    }
    double obj = 0.0; for (int e = 0; e < NE; e++) obj += W[e] * xabs[e];
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* independent checks at the returned point */
    double cyc = 0.0;                      /* D1 x = 0 : x is a cycle */
    for (int i = 0; i < NV; i++) { double s = 0; for (int e = 0; e < NE; e++) s += D1[i][e] * x[e]; cyc = fmax(cyc, fabs(s)); }
    double hom = 0.0;                      /* x - C - D2 y = 0 : same class */
    for (int e = 0; e < NE; e++) { double s = x[e] - C[e]; for (int tt = 0; tt < nf; tt++) s -= D2[e][tt] * y[tt]; hom = fmax(hom, fabs(s)); }
    double absok = 0.0;                    /* xabs = |x| */
    for (int e = 0; e < NE; e++) absok = fmax(absok, fabs(xabs[e] - fabs(x[e])));
    /* the reference value: the shortest around-cycle is the 4-edge top/bottom ring */
    int okall = ok && cyc < 1e-8 && hom < 1e-8 && absok < 1e-8 && fabs(obj - 4.0) < 1e-6;
    printf("surface_cycles  e=%d t=%d  shortest cycle = %.6f\n", NE, nf, obj);
    printf("  |D1 x|max = %.2e  |x - C - D2 y|max = %.2e  |xabs-|x||max = %.2e\n", cyc, hom, absok);
    printf("  support = {");
    for (int e = 0; e < NE; e++) if (fabs(x[e]) > 1e-6) printf("%d%s", e, (e == NE - 1) ? "" : ",");
    printf("}\n");
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
