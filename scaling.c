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

/* scaling.c — equilibratura geometrica righe/colonne con potenze di 2.
 *
 * Poche iterazioni di equilibratura: ogni pass computa per ogni riga
 * (poi colonna) la radice quadrata della media di |a_ij|^2 e arrotonda
 * il fattore 1/rms alla potenza di 2 più vicina, così lo scaling è
 * esatto in floating point.
 */
#include <math.h>
#include <string.h>

#include "scaling.h"

static double pow2_round(double x) {
    /* potenza di 2 più vicina a x (>0), clampata a [2^-24, 2^24] */
    if (!(x > 0.0) || x != x) return 1.0;
    double e = floor(log2(x) + 0.5);
    if (e > 24.0) e = 24.0;
    if (e < -24.0) e = -24.0;
    return ldexp(1.0, (int)e);
}

/**
 * Performs geometric row/column equilibration with power-of-2 scaling factors.
 *
 * @param nvar   [in]  Number of variables.
 * @param ncon   [in]  Number of constraints.
 * @param ptr    [in]  Column pointers for A (CSC, size nvar+1).
 * @param sub    [in]  Row indices for A (size ptr[nvar]).
 * @param val    [in/out] Values for A (size ptr[nvar]). Modified in-place.
 * @param lc     [in/out] Constraint lower bounds (size ncon). Modified in-place.
 * @param uc     [in/out] Constraint upper bounds (size ncon). Modified in-place.
 * @param lx     [in/out] Variable lower bounds (size nvar). Modified in-place.
 * @param ux     [in/out] Variable upper bounds (size nvar). Modified in-place.
 * @param c      [in/out] Linear objective (size nvar). Modified in-place.
 * @param Q      [in/out] Quadratic objective matrix (nvar x nvar, row-major). May be NULL.
 * @param r      [out] Row scaling factors (size ncon). Must be pre-allocated.
 * @param d      [out] Column scaling factors (size nvar). Must be pre-allocated.
 *
 * @note Performs 4 iterations of alternating row/column scaling:
 *       Row pass: scales each row by nearest power-of-2 to make RMS ~ 1.
 *       Column pass: scales each column similarly, updates bounds and Q.
 *
 *       Scaling factors are powers of 2 (exact in floating point).
 *       Clamped to [2^-24, 2^24] to prevent overflow/underflow.
 *       Modifies A, bounds, c, and Q in-place. Outputs scaling factors r, d.
 *
 * @example
 * double r[m], d[n];
 * scale_equilibrate(nvar, ncon, Aptr, Arow, Aval, lc, uc, lx, ux, c, Q, r, d);
 * // A, lc, uc, lx, ux, c, Q are now scaled
 */
void scale_equilibrate(int nvar, int ncon,
                       const int *ptr, const int *sub, double *val,
                       double *lc, double *uc, double *lx, double *ux,
                       double *c, double *Q,
                       double *r, double *d) {
    for (int i = 0; i < ncon; i++) r[i] = 1.0;
    for (int j = 0; j < nvar; j++) d[j] = 1.0;

    for (int it = 0; it < 4; it++) {
        /* ---- row pass ---- */
        for (int i = 0; i < ncon; i++) {
            double sum = 0.0;
            int cnt = 0;
            for (int j = 0; j < nvar; j++)
                for (int k = ptr[j]; k < ptr[j + 1]; k++)
                    if (sub[k] == i) { sum += val[k] * val[k]; cnt++; }
            if (cnt == 0) continue;
            double f = pow2_round(1.0 / sqrt(sum / (double)cnt));
            if (f == 1.0) continue;
            r[i] *= f;
            lc[i] *= f; uc[i] *= f;
            for (int j = 0; j < nvar; j++)
                for (int k = ptr[j]; k < ptr[j + 1]; k++)
                    if (sub[k] == i) val[k] *= f;
        }
        /* ---- column pass ---- */
        for (int j = 0; j < nvar; j++) {
            double sum = 0.0;
            int cnt = 0;
            for (int k = ptr[j]; k < ptr[j + 1]; k++) { sum += val[k] * val[k]; cnt++; }
            if (cnt == 0) continue;
            double f = pow2_round(1.0 / sqrt(sum / (double)cnt));
            if (f == 1.0) continue;
            d[j] *= f;
            c[j] *= f;
            lx[j] /= f; ux[j] /= f;
            for (int k = ptr[j]; k < ptr[j + 1]; k++) val[k] *= f;
            if (Q) {
                /* Q' = D Q D: riga j e colonna j scalate da d[j] (parziale) */
                for (int k = 0; k < nvar; k++) {
                    Q[j * nvar + k] *= f;
                    Q[k * nvar + j] *= f;
                }
            }
        }
    }
}