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

/* min_circle.c - minimum enclosing circle, SOCP vs exact (tschm/min_circle).
 *
 * Model (src/min_circle/msk.py, identical to the MOSEK minimum-ellipsoid
 * tutorial and this library's samples/min_enclosing_ball.c):
 *      minimize r  s.t.  || x - p_i ||_2 <= r  for every point p_i,
 * i.e. one quadratic cone [r, x - p_i] in Q^3 per point.
 *
 * The repository's point is to compare solvers; here the SOCP is
 * cross-checked against an EXACT oracle: the minimum enclosing circle is the
 * circumcircle of two or three of the points, so brute-forcing all pairs and
 * triples gives the true optimum independently of the optimization.
 *
 * Cases: the repo's tests ([[0,0],[4,0]] -> r=2; [[0,0],[4,0],[2,4]] -> r=2.5)
 * plus a deterministic pseudo-random point set.
 *
 * Usage: min_circle   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define KMAX 10
#define ND 2

/* ---- SOCP (the reference formulation) ---- */
static double socp_circle(int k, const double P[KMAX][ND], double c_out[ND], int *ok_out) {
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
    if (ok) { double x[32]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); r = x[R]; c_out[0] = x[PX]; c_out[1] = x[PY]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = ok;
    return r;
}

/* ---- exact oracle: circumcircle of a pair / a triple ---- */
static double circum(int m, const double P[KMAX][ND], int idx[3], double c[ND]) {
    if (m == 2) {
        for (int d = 0; d < ND; d++) c[d] = 0.5 * (P[idx[0]][d] + P[idx[1]][d]);
        double s = 0; for (int d = 0; d < ND; d++) s += (P[idx[0]][d] - P[idx[1]][d]) * (P[idx[0]][d] - P[idx[1]][d]);
        return 0.5 * sqrt(s);
    }
    double ax = P[idx[0]][0], ay = P[idx[0]][1];
    double bx = P[idx[1]][0], by = P[idx[1]][1];
    double cx = P[idx[2]][0], cy = P[idx[2]][1];
    double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (fabs(d) < 1e-14) return INFINITY;
    double a2 = ax*ax + ay*ay, b2 = bx*bx + by*by, c2 = cx*cx + cy*cy;
    c[0] = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    c[1] = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    double dx = ax - c[0], dy = ay - c[1];
    return sqrt(dx * dx + dy * dy);
}
static int contains_all(int k, const double P[KMAX][ND], double c[ND], double r) {
    for (int i = 0; i < k; i++) {
        double dx = P[i][0] - c[0], dy = P[i][1] - c[1];
        if (sqrt(dx * dx + dy * dy) > r + 1e-9) return 0;
    }
    return 1;
}
static double exact_circle(int k, const double P[KMAX][ND]) {
    double best = INFINITY;
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++) {
            int idx[3] = {i, j, 0}; double c[ND];
            double r = circum(2, P, idx, c);
            if (r < best && contains_all(k, P, c, r)) best = r;
        }
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++)
            for (int l = j + 1; l < k; l++) {
                int idx[3] = {i, j, l}; double c[ND];
                double r = circum(3, P, idx, c);
                if (r < best && contains_all(k, P, c, r)) best = r;
            }
    return best;
}

static int run(const char *name, int k, double P[KMAX][ND]) {
    double c[ND]; int ok = 0;
    double rs = socp_circle(k, P, c, &ok);
    double re = exact_circle(k, P);
    int good = ok && fabs(rs - re) < 1e-6;
    printf("  %-14s k=%d  SOCP r=%.8f  exact r=%.8f  %s\n", name, k, rs, re, good ? "OK" : "FAIL");
    return good;
}

int main(void) {
    int all = 1;
    printf("min_circle\n");
    double P2[KMAX][ND] = {{0.0, 0.0}, {4.0, 0.0}};
    all &= run("two points", 2, P2);
    double P3[KMAX][ND] = {{0.0, 0.0}, {4.0, 0.0}, {2.0, 4.0}};
    all &= run("three points", 3, P3);
    /* deterministic pseudo-random 8 points (LCG) */
    double PR[KMAX][ND]; unsigned long s0 = 12345;
    for (int i = 0; i < 8; i++)
        for (int d = 0; d < 2; d++) { s0 = s0 * 1103515245 + 12345; PR[i][d] = (double)((s0 >> 16) % 1000) / 1000.0; }
    all &= run("random (seed)", 8, PR);
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
