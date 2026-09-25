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

/* welzl.c - Welzl's exact minimum enclosing circle (tschm/min_circle).
 *
 * Welzl's algorithm is expected-linear-time and exact: it processes the points
 * in random order, keeping the (up to 3) points known to lie on the boundary,
 * and returns the circle determined by the boundary set as soon as three
 * points fix it.
 *
 *     welzl(P, R):
 *       if P empty or |R| = 3:  return circle(R)
 *       p = first(P)
 *       D = welzl(P without p, R)
 *       if p inside D: return D
 *       return welzl(P without p, R + {p})
 *
 * Here it is cross-checked against this library's SOCP
 * (min r s.t. ||x-p_i|| <= r, samples/min_circle.c) on several deterministic
 * pseudo-random point sets -- two independent methods must agree exactly.
 *
 * Usage: welzl   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

typedef struct { double c[2]; double r; } Circ;

/* ---- circle through 0, 1, 2 or 3 boundary points ---- */
static Circ circ_of(int m, const double R[3][2]) {
    Circ z;
    if (m == 0) { z.c[0] = z.c[1] = 0.0; z.r = -1.0; return z; }   /* empty: contains nothing */
    if (m == 1) { z.c[0] = R[0][0]; z.c[1] = R[0][1]; z.r = 0.0; return z; }
    if (m == 2) {
        for (int d = 0; d < 2; d++) z.c[d] = 0.5 * (R[0][d] + R[1][d]);
        double s = 0; for (int d = 0; d < 2; d++) s += (R[0][d] - R[1][d]) * (R[0][d] - R[1][d]);
        z.r = 0.5 * sqrt(s); return z;
    }
    double ax = R[0][0], ay = R[0][1], bx = R[1][0], by = R[1][1], cx = R[2][0], cy = R[2][1];
    double dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (fabs(dd) < 1e-14) { z.r = INFINITY; return z; }            /* collinear */
    double a2 = ax*ax + ay*ay, b2 = bx*bx + by*by, c2 = cx*cx + cy*cy;
    z.c[0] = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd;
    z.c[1] = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd;
    z.r = sqrt((ax - z.c[0]) * (ax - z.c[0]) + (ay - z.c[1]) * (ay - z.c[1]));
    return z;
}
static int inside(const Circ *z, const double p[2]) {
    if (z->r < 0) return 0;
    double dx = p[0] - z->c[0], dy = p[1] - z->c[1];
    return sqrt(dx * dx + dy * dy) <= z->r + 1e-9;
}
static Circ welzl(const double (*P)[2], int n, double R[3][2], int m) {
    if (n == 0 || m == 3) return circ_of(m, (const double (*)[2])R);
    Circ D = welzl(P + 1, n - 1, R, m);
    if (inside(&D, P[0])) return D;
    for (int d = 0; d < 2; d++) R[m][d] = P[0][d];
    return welzl(P + 1, n - 1, R, m + 1);
}

/* ---- SOCP oracle (same formulation as samples/min_circle.c) ---- */
static double socp_circle(int k, const double (*P)[2], int *ok_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { PX = 0, PY = 1, R = 2, D = 3 };
    PRIMAL_appendvars(t, 3 + 2 * k);
    PRIMAL_appendcons(t, 2 * k);
    PRIMAL_putvarbound(t, PX, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, PY, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, R, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < k; i++) {
        PRIMAL_putvarbound(t, D + 2 * i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, D + 2 * i + 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putarow(t, 2 * i, 2, (int[]){D + 2 * i, PX}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 2 * i, PRIMAL_BK_FX, -P[i][0], -P[i][0]);
        PRIMAL_putarow(t, 2 * i + 1, 2, (int[]){D + 2 * i + 1, PY}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, 2 * i + 1, PRIMAL_BK_FX, -P[i][1], -P[i][1]);
    }
    for (int i = 0; i < k; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){R, D + 2 * i, D + 2 * i + 1});
    PRIMAL_putcj(t, R, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double r = 0.0;
    if (ok) { double x[64]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); r = x[R]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = ok; return r;
}

#define KP 10
static unsigned long g_seed;
static double rnd(void) { g_seed = g_seed * 1103515245 + 12345; return (double)((g_seed >> 16) % 1000) / 1000.0; }

int main(void) {
    int all = 1, nsets = 20;
    printf("welzl\n");
    for (int s = 0; s < nsets; s++) {
        double P[KP][2];
        g_seed = 1000 + 7 * s;
        for (int i = 0; i < KP; i++) { P[i][0] = rnd(); P[i][1] = rnd(); }
        /* Fisher-Yates shuffle with the same deterministic PRNG */
        for (int i = KP - 1; i > 0; i--) { int j = (int)(rnd() * (i + 1)); if (j > i) j = i;
            double tx = P[i][0], ty = P[i][1]; P[i][0] = P[j][0]; P[i][1] = P[j][1]; P[j][0] = tx; P[j][1] = ty; }
        double R[3][2]; Circ w = welzl((const double (*)[2])P, KP, R, 0);
        int ok = 0; double rs = socp_circle(KP, (const double (*)[2])P, &ok);
        int good = ok && fabs(w.r - rs) < 1e-6;
        if (!good) { printf("  set %2d  welzl=%.8f  socp=%.8f  FAIL\n", s, w.r, rs); all = 0; }
        else if (s == 0) printf("  set  0  welzl=%.8f  socp=%.8f  OK\n", w.r, rs);
    }
    printf("  %d point sets, Welzl == SOCP: %s\n", nsets, all ? "OK" : "FAIL");
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
