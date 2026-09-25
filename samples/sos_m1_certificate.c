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

/* sos_m1_certificate.c - exact rational SOS certificate of the axisymmetric
 * sub-case of the "KKT-infused flag algebra" P2 project
 * (https://github.com/jcpaik/p2-kkt-flag-sos, MIT), verified with this library.
 *
 * The project studies copositivity of K(t) = 32 t^6 - 48 t^4 + 20 t^2 - 4/3
 * for probability measures on RP^2.  On the axisymmetric branch the energy is
 * the bicubic
 *   C(s,t) = sum_{i,j<=3} M0[i,j] s^i t^j   on [0,1]^2,
 * and the committed rational certificate proves
 *   int int C dtau dtau >= 0   for every probability measure tau on [0,1]
 * via the identity (all data rational)
 *   C(s,t) = Phi(s)^T G Phi(t) + sum_beta lambda_beta B_beta(s,t),
 *   G = P W P^T,  Phi = (1,s,s^2,s^3),
 * with W >= 0 (3x3), G v* = 0 for v* = (1,1/3,1/3,1/3), lambda >= 0, and
 * B_beta the symmetrized corner-vanishing Handelman products
 * s^i(1-s)^j t^k(1-t)^l.  The witness is certificates/m1_axisymmetric.json.
 *
 * This sample reproduces the repository's `cylinder_cert.py m1 --verify`
 * (6/6): it re-checks the polynomial identity coefficient by coefficient, W
 * PSD (eigenvalues), G v* = 0, lambda >= 0, and the corner vanishing, and
 * reports the certified margin min(lambda_min(W), min lambda) = 0.0670...
 *
 * Usage: sos_m1_certificate   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* mode-0 bicubic coefficients */
static const double M0[4][4] = {
    {2.0 / 3, -4, 12, -10}, {-4, 84, -270, 210},
    {12, -270, 840, -630}, {-10, 210, -630, 462}};
/* face_P: columns span v*^perp, v* = (1, 1/3, 1/3, 1/3) */
static const double FACE_P[4][3] = {{1, 0, 0}, {-3, 1, 0}, {0, -1, 1}, {0, 0, -1}};
/* committed rational witness (certificates/m1_axisymmetric.json), denom 12852161 */
static const double W[3][3] = {
    {0.666666666667, -4.634368181351, 12.634368181351},
    {-4.634368181351, 68.817301074893, -175.449591239948},
    {12.634368181351, -175.449591239948, 450.081881405003}};
/* Handelman terms: (i,j),(k,l),lambda */
static const int HL[8][4] = {
    {0, 5, 1, 2}, {1, 2, 1, 6}, {1, 2, 2, 6}, {1, 2, 4, 0},
    {1, 2, 5, 0}, {1, 2, 5, 1}, {1, 2, 5, 2}, {1, 2, 5, 3}};
static const double LAM[8] = {
    5.268736362702, 7.720171650511, 8.532848755941, 0.067019857594,
    13.395507417002, 20.316405933601, 17.878374617311, 8.532848755941};

static double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    double r = 1.0;
    for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1);
    return r;
}
static double s_coef(int a, int j, int x) {       /* coeff of s^x in s^a (1-s)^j */
    int e = x - a;
    if (e < 0 || e > j) return 0.0;
    double s = binom(j, e);
    return (e % 2) ? -s : s;
}
static double handelman(const int p[4], int m, int n) {
    double f = s_coef(p[0], p[1], m) * s_coef(p[2], p[3], n);
    double g = s_coef(p[2], p[3], m) * s_coef(p[0], p[1], n);
    return 0.5 * (f + g);
}
/* value of the symmetrized Handelman basis at (s,t) */
static double handelman_value(const int p[4], double s, double t) {
    double f = pow(s, p[0]) * pow(1 - s, p[1]) * pow(t, p[2]) * pow(1 - t, p[3]);
    double g = pow(s, p[2]) * pow(1 - s, p[3]) * pow(t, p[0]) * pow(1 - t, p[1]);
    return 0.5 * (f + g);
}
/* eigenvalues of a symmetric 3x3 matrix (Jacobi) */
static void eig3(const double A[3][3], double ev[3]) {
    double a[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) a[i][j] = A[i][j];
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                if (fabs(a[p][q]) < 1e-300) continue;
                double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {
                    double ap = a[p][k], aq = a[q][k];
                    a[p][k] = c * ap - s * aq; a[q][k] = s * ap + c * aq;
                }
                for (int k = 0; k < 3; k++) {
                    double ap = a[k][p], aq = a[k][q];
                    a[k][p] = c * ap - s * aq; a[k][q] = s * ap + c * aq;
                }
            }
    }
    for (int i = 0; i < 3; i++) ev[i] = a[i][i];
}

int main(void) {
    double G[4][4];
    double vstar[4] = {1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3};
    /* G = P W P^T */
    for (int p = 0; p < 4; p++)
        for (int q = 0; q < 4; q++) {
            double s = 0.0;
            for (int r = 0; r < 3; r++)
                for (int t = 0; t < 3; t++) s += FACE_P[p][r] * W[r][t] * FACE_P[q][t];
            G[p][q] = s;
        }
    int nres = 0;
    double maxres = 0.0;
    for (int m = 0; m <= 8; m++)
        for (int n = m; n <= 8; n++) {
            double gram = (m <= 3 && n <= 3) ? G[m][n] : 0.0;
            double hand = 0.0;
            for (int b = 0; b < 8; b++) hand += LAM[b] * handelman(HL[b], m, n);
            double c = (m <= 3 && n <= 3) ? M0[m][n] : 0.0;
            double res = fabs(gram + hand - c);
            if (res > maxres) maxres = res;
            nres++;
        }
    double ev[3];
    eig3(W, ev);
    double lam_min = LAM[0];
    for (int b = 1; b < 8; b++) if (LAM[b] < lam_min) lam_min = LAM[b];
    double evmin = ev[0];
    for (int i = 1; i < 3; i++) if (ev[i] < evmin) evmin = ev[i];
    double margin = (evmin < lam_min) ? evmin : lam_min;

    double gv[4];
    for (int i = 0; i < 4; i++) { double s = 0;
        for (int j = 0; j < 4; j++) s += G[i][j] * vstar[j]; gv[i] = s; }
    double gvmax = 0.0;
    for (int i = 0; i < 4; i++) if (fabs(gv[i]) > gvmax) gvmax = fabs(gv[i]);
    int corner_ok = 1;
    for (int b = 0; b < 8; b++) {
        double cs[2] = {0.0, 1.0};
        for (int a = 0; a < 2; a++)
            for (int d = 0; d < 2; d++)
                if (fabs(handelman_value(HL[b], cs[a], cs[d])) > 1e-12) corner_ok = 0;
    }
    int lam_ok = 1;
    for (int b = 0; b < 8; b++) if (LAM[b] < 0.0) lam_ok = 0;

    int id_ok = maxres < 1e-6;
    int w_ok = evmin > -1e-9;
    int gv_ok = gvmax < 1e-6;
    int ok = id_ok && w_ok && gv_ok && lam_ok && corner_ok;

    printf("sos_m1_certificate (deg 8, %d monomials)\n", nres);
    printf("  (i)   C = Phi^T G Phi + sum lam*B        max|res|=%.3e  %s\n",
           maxres, id_ok ? "OK" : "FAIL");
    printf("  (ii)  W PSD                              lambda_min=%.6f  %s\n",
           evmin, w_ok ? "OK" : "FAIL");
    printf("  (iii) G v* = 0                           max|=%.3e  %s\n",
           gvmax, gv_ok ? "OK" : "FAIL");
    printf("  (iv)  lambda >= 0                        min=%.6f  %s\n",
           lam_min, lam_ok ? "OK" : "FAIL");
    printf("  (v)   Handelman corner-vanishing         %s\n", corner_ok ? "OK" : "FAIL");
    printf("  certified margin min(lmin(W), min lam) = %.6f  ->  int int C dtau dtau >= 0  %s\n",
           margin, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
