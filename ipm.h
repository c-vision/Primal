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

/* ipm.h - Mehrotra predictor-corrector interior point on the standard form
 *   min 1/2 x'Qx + c'x  s.t.  A x = b, x >= 0   (Q = NULL for LP)
 * statuses: 0 optimal, 1 no convergence, 2 memory, 3 singular system */
#ifndef IPM_H
#define IPM_H

/* Wall-clock deadline (absolute clock() ticks; <0 = none) shared by the IPM
 * loops; PRIMAL_optimize sets it from PRIMAL_DPAR_OPTIMIZER_MAX_TIME. */
void ipm_set_deadline(double abs_deadline);
/* Objective cuts in min-space (-1e308/1e308 = none); the flag says whether one
 * fired (so PRIMAL_optimize can report TRM_OBJECTIVE_RANGE). */
void ipm_set_obj_cuts(double lower, double upper);
int  ipm_obj_cut_hit(void);
/* Numero di correttori dell'IPM (>=1; MSK_IPAR_INTPNT_MAX_NUM_COR). */
void ipm_set_max_cor(int ncor);

int ipm_solve_std(const double *A, const double *Q, int m, int n,
                  const double *b, const double *c,
                  double tol_gap, double tol_pfeas, double tol_dfeas,
                  int max_iter,
                  double *x /*n*/, double *y /*m*/, double *z /*n*/,
                  const double *x0 /*nullable warm start, n*/,
                  const double *y0 /*nullable warm start, m*/);

/* ---- sparse LP variant: A in CSC (col_ptr n+1, row_idx, val) ----
 * Mehrotra predictor-corrector with normal equations
 *   (A Theta A' + delta I) dy = rhs,  Theta = X/S,
 * solved with the sparse Cholesky of linalg.h. LP only (Q = NULL).
 * Same statuses as ipm_solve_std. */
int ipm_solve_std_csc(const int *Aptr, const int *Arow, const double *Aval,
                      int m, int n,
                      const double *b, const double *c,
                      double tol_gap, double tol_pfeas, double tol_dfeas,
                      int max_iter,
                      double *x /*n*/, double *y /*m*/, double *z /*n*/,
                      const double *x0 /*nullable*/, const double *y0 /*nullable*/);

/* ---- sparse QP variant: A in CSC (n+1) and Q in LOWER-triangle CSC (n+1) ----
 * min 1/2 x'Qx + c'x, A x = b, x >= 0, Q symmetric PSD with its diagonal
 * present in the CSC (even if zero). Mehrotra predictor-corrector on the
 * normal equations: factor M = Q + D (D = Z/X, sparse Cholesky), W = M^-1 A',
 * K = A W + delta I (Cholesky), K dy = -rp - A M^-1 rhs1, dx = M^-1 rhs1 + W dy,
 * dz from complementarity. Suited to large QP with sparse Q and m << n.
 * Same statuses as ipm_solve_std. */
int ipm_solve_qp_csc(const int *Aptr, const int *Arow, const double *Aval,
                     const int *Qptr, const int *Qrow, const double *Qval,
                     int m, int n,
                     const double *b, const double *c,
                     double tol_gap, double tol_pfeas, double tol_dfeas,
                     int max_iter,
                     double *x /*n*/, double *y /*m*/, double *z /*n*/,
                     const double *x0 /*nullable*/, const double *y0 /*nullable*/);

#endif /* IPM_H */
