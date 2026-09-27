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

/* scaling.h -- row/column equilibration (powers of 2) for LP/QP.
 *
 * Transforms the problem into min-form before the solve:
 *   A' = R A D,  b' = R b  (row bounds: [lc,uc] *= r_i)
 *   c' = D c,    Q' = D Q D (var bounds: [lx,ux] /= d_j)
 * with r_i, d_j exact powers of 2 (no round-off error).
 * After the solve: x_orig = D x', y_orig = R^{-1} y'.
 */
#ifndef SCALING_H
#define SCALING_H

/* Edits CSC (val), bounds, and c/Q in place; writes r[ncon], d[nvar] (caller-allocated). */
void scale_equilibrate(int nvar, int ncon,
                       const int *ptr, const int *sub, double *val,
                       double *lc, double *uc, double *lx, double *ux,
                       double *c, double *Q /* nullable, dense nvar*nvar */,
                       double *r, double *d);

#endif