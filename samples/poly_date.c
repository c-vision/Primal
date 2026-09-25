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

/* poly_date.c - "poly-date": the shortest homogeneous three-variable polynomial
 * equation with all coefficients 1 satisfied by a date (d, m, y)
 * (aszek/poly-date, Python + MOSEK Fusion).
 *
 * For each degree k, enumerate the monomials d^a m^b y^c with a+b+c = k and
 * their values val_i at (d,m,y).  Two DISJOINT subsets s1, s2 with
 *     sum_{i in s1} val_i = sum_{j in s2} val_j,   s1 non-empty
 * give a valid homogeneous equation  P_s1 = P_s2 ; the model finds the shortest:
 *
 *     minimize   sum_i (s1_i + s2_i)
 *     subject to sum_i (s1_i - s2_i) val_i = 0
 *                s1_i + s2_i <= 1          (disjoint)
 *                sum_i s1_i >= 1           (an equation, not 0 = 0)
 *                s1, s2 in {0,1}
 *
 * The smallest degree with a solution is reported.  The coefficients are
 * verified in EXACT integer arithmetic on the returned support.
 *
 * Known case (the repo's example): 8/11/15 has no equation of degree 1 or 2,
 * and at degree 3 gives  y^3 + m^3 + d m y + d^2 m = m y^2 + m^2 y + d m^2 +
 * d^2 y + d^3, both sides equal to 6730 (the found support is exactly that,
 * 4 terms = 5 terms).
 *
 * Usage: poly_date [d m y maxDeg]   (default 8 11 15 5)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

#define NMAX 32
typedef struct { int a, b, c; long val; } Mono;

static int gen_monomials(int d, int m, int y, int k, Mono *mo) {
    int n = 0;
    for (int a = 0; a <= k; a++) for (int b = 0; b <= k - a; b++) {
        int c = k - a - b;
        long v = 1;
        for (int i = 0; i < a; i++) v *= d;
        for (int i = 0; i < b; i++) v *= m;
        for (int i = 0; i < c; i++) v *= y;
        mo[n].a = a; mo[n].b = b; mo[n].c = c; mo[n].val = v; n++;
    }
    return n;
}

/* MILP for one degree; returns 1 if feasible, writing the two supports and
 * their sizes, and the exact sums L = sum_{s1} val, R = sum_{s2} val. */
static int solve_degree(int d, int m, int y, int k, int *nmono_out, int *n1_out, int *n2_out,
                        long *L_out, long *R_out) {
    Mono mo[NMAX];
    int n = gen_monomials(d, m, y, k, mo);
    int S1 = 0, S2 = n, nv = 2 * n;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < nv; i++) {
        PRIMAL_putvartype(t, i, PRIMAL_VAR_TYPE_INT);
        PRIMAL_putvarbound(t, i, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putcj(t, i, 1.0);
    }
    PRIMAL_appendcons(t, 1 + n + 1);
    int r = 0;
    {   int sub[2*NMAX]; double val[2*NMAX]; int nn = 0;
        for (int i = 0; i < n; i++) { sub[nn] = S1+i; val[nn] = (double)mo[i].val; nn++; }
        for (int i = 0; i < n; i++) { sub[nn] = S2+i; val[nn] = -(double)mo[i].val; nn++; }
        PRIMAL_putarow(t, r, nn, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0.0, 0.0); r++;
    }
    for (int i = 0; i < n; i++) {
        PRIMAL_putarow(t, r, 2, (int[]){S1+i, S2+i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 1.0); r++;
    }
    {   int sub[NMAX]; double val[NMAX];
        for (int i = 0; i < n; i++) { sub[i] = S1+i; val[i] = 1.0; }
        PRIMAL_putarow(t, r, n, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_LO, 1.0, INFINITY); r++;
    }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = 0, n1 = 0, n2 = 0;
    long L = 0, R = 0;
    if (rc == PRIMAL_RES_OK) {
        double x[2*NMAX] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < n; i++) {
            if (x[S1+i] > 0.5) { L += mo[i].val; n1++; }
            if (x[S2+i] > 0.5) { R += mo[i].val; n2++; }
        }
        ok = n1 > 0 && L == R;
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *nmono_out = n; *n1_out = n1; *n2_out = n2; *L_out = L; *R_out = R;
    return ok;
}

int main(int argc, char **argv) {
    int d = argc > 1 ? atoi(argv[1]) : 8;
    int m = argc > 2 ? atoi(argv[2]) : 11;
    int y = argc > 3 ? atoi(argv[3]) : 15;
    int maxDeg = argc > 4 ? atoi(argv[4]) : 5;
    printf("poly_date  d=%d m=%d y=%d maxDeg=%d\n", d, m, y, maxDeg);
    int all = 1, found = 0;
    for (int k = 1; k <= maxDeg && !found; k++) {
        int n, n1, n2; long L, R;
        if (solve_degree(d, m, y, k, &n, &n1, &n2, &L, &R)) {
            int good = (L == R) && L > 0 && n1 > 0;
            printf("  grado %d: %d monomi, %d + %d termini, LHS=RHS=%ld  %s\n",
                   k, n, n1, n2, L, good ? "OK" : "FAIL");
            all &= good; found = 1;
        } else {
            printf("  grado %d: nessuna equazione\n", k);
        }
    }
    if (!found) { printf("  nessuna equazione fino al grado %d\n", maxDeg); all = 0; }
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
