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

/* bounded_real_lmi.c - the Bounded Real Lemma (H-infinity norm) as an SDP
 * (S-Salavati/Convex-Optimization-with-LMIs-in-{C-plus-plus,Python}, whose code
 * writes the induced-L2 LMI M(x) <= 0 as a PSD bar variable R = -M(x) equated
 * entrywise to M(x)).
 *
 * For a stable LTI system (A,B,C,D), ||G||_inf < gamma iff there is P > 0 with
 *     [[A'P + PA + C'C,  PB + C'D],
 *      [B'P + D'C,       D'D - gamma^2 I]] <= 0.
 * (The scaled form: the exit block is -gamma^2 I, as the Python writes
 * `L_77 = -gamma_sq*eye`.  The naive 3x3 with -gamma I is NOT this lemma and
 * sends gamma -> 0.)  The model minimises gamma^2; the LMI is written as a PSD
 * bar variable R = -M with entrywise equalities, entries addressed through
 * symmetric matrices (appendsparsesymmat + putbaraij).
 *
 * Case: SISO first order A = -1, B = 1, C = 1, D = 0, whose H-infinity norm is
 *     ||G||_inf = |C B| / |A| = 1.
 * The LMI at (P, g=gamma^2) is [[-2P+1, P],[P, -g]] <= 0, i.e. a Schur
 * condition g >= P^2/(2P-1) for P > 1/2, minimised at P = 1 giving g = 1.
 *
 * Usage: bounded_real_lmi   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    const double B = 1.0, C = 1.0, D = 0.0;   /* SISO, A=-1, |CB|/|A| = 1 */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    int r00, r01, r11, p00;
    PRIMAL_appendsparsesymmat(t, 2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &r00);
    PRIMAL_appendsparsesymmat(t, 2, 1, (int[]){0}, (int[]){1}, (double[]){1.0}, &r01);
    PRIMAL_appendsparsesymmat(t, 2, 1, (int[]){1}, (int[]){1}, (double[]){1.0}, &r11);
    int d1 = 1;
    PRIMAL_appendsparsesymmat(t, d1, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &p00);
    int d2 = 2;
    PRIMAL_appendbarvars(t, 1, &d1);   /* barvar 0 : P (1x1 PSD) */
    PRIMAL_appendbarvars(t, 1, &d2);   /* barvar 1 : R (2x2 PSD) = -M */
    int G = 0; PRIMAL_appendvars(t, 1);
    PRIMAL_putvarbound(t, G, PRIMAL_BK_LO, 0.0, INFINITY);   /* g = gamma^2 >= 0 */
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMAL_putcj(t, G, 1.0);
    PRIMAL_appendcons(t, 3);
    /* M = [[A'P+PA+C'C, PB+C'D],[B'P+D'C, D'D - g]]
     *   = [[-2P + C^2, P*B + C*D],[P*B + D*C, D^2 - g]]  (A=-1) */
    /* row 0 : R00 = -M00 = 2P - C^2  ->  R00 - 2P = -C^2 */
    PRIMAL_putbaraij(t, 0, 1, 1, &r00, (double[]){1.0});
    PRIMAL_putbaraij(t, 0, 0, 1, &p00, (double[]){-2.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, -C*C, -C*C);
    /* row 1 : R01 = -M01 = -(P*B + C*D)  ->  R01 + P*B = -C*D */
    PRIMAL_putbaraij(t, 1, 1, 1, &r01, (double[]){0.5});   /* <E01,R>/2 = R01 */
    PRIMAL_putbaraij(t, 1, 0, 1, &p00, (double[]){B});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, -C*D, -C*D);
    /* row 2 : R11 = -M11 = g - D^2  ->  R11 - g = -D^2 */
    PRIMAL_putbaraij(t, 2, 1, 1, &r11, (double[]){1.0});
    PRIMAL_putarow(t, 2, 1, (int[]){G}, (double[]){-1.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, -D*D, -D*D);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double g2 = 0.0, P[4] = {0};
    if (ok) {
        double x[4] = {0}; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); g2 = x[G];
        PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, P);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    /* independent check: the LMI at the returned point must be <= 0 (2x2), and
     * gamma = sqrt(g2) must equal |CB|/|A| = 1 */
    double m00 = -2*P[0] + C*C, m01 = P[0]*B + C*D, m11 = D*D - g2;
    double tr = m00 + m11, det = m00*m11 - m01*m01;
    double lmax = tr/2 + sqrt(fmax(0.0, tr*tr/4 - det));
    double gamma = sqrt(g2);
    int good = ok && fabs(gamma - 1.0) < 1e-6 && lmax <= 1e-6;
    printf("bounded_real_lmi  gamma^2*=%.8f  gamma*=%.8f (atteso 1)  P=%.6f  lmax=%.2e  %s\n",
           g2, gamma, P[0], lmax, good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
