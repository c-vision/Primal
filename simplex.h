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

/* simplex.h - two-phase primal simplex on the standard form
 *   min c'x  s.t.  A x = b, x >= 0, b >= 0
 * statuses: 0 optimal, 1 infeasible (Farkas witness in dray),
 *           2 unbounded (recession direction in pray),
 *           3 max iterations, 4 out of memory.
 * dray (m) and pray (n) may be NULL and are written only by the status that
 * owns them, so on any other status they hold nothing the caller may read.
 * dray is unnormalized (y'A <= 0, y'b > 0); pray is normalized to 1 in the
 * entering column (rho >= 0, A rho = 0, c'rho < 0). */
#ifndef SIMPLEX_H
#define SIMPLEX_H

/* Primal simplex wrapper on the standard form (no basis/tableau output).
 * Solves for primal x and dual y, with optional Farkas rays. */
int simplex_solve_std(const double *A, int m, int n,
                      const double *b, const double *c,
                      int max_iter, double *x /*n*/, double *y /*m*/,
                      double *dray /*m, status 1*/, double *pray /*n, status 2*/);
/* As above, but also returns the final basis (`basis_out`, m column
 * indices) and the tableau (`tab_out`, m rows of stride n+m+1: reduced
 * coefficients in the first n+m columns, RHS in [n+m]). Needed for Gomory cuts.
 * Both optional (NULL). */
int simplex_solve_std_tab(const double *A, int m, int n,
                          const double *b, const double *c,
                          int max_iter, double *x, double *y,
                          double *dray, double *pray,
                          int *basis_out, double *tab_out);

/* Dual simplex: min c'x s.t. A x = b, x >= 0, from a DUAL-feasible basis
 * (`basis[m]`, column indices); `b` may have negative entries. Builds
 * B^-1 with Gauss-Jordan on [B|I], picks the row with the most negative RHS and
 * runs the ratio test on the reduced costs. Statuses as above (0 optimal,
 * 1 infeasible, 2 unbounded, 3 max iter, 4 memory). x[n]. Ported from gmbortools
 * `gor_lp_dual_simplex` (itself from apurvasijaria/Operation_Research_Lab). */
int simplex_dual_solve_std(const double *A, int m, int n,
                           const double *b, const double *c,
                           const int *basis, int max_iter, double *x /*n*/,
                           int *basis_out /*m, final basis, optional*/,
                           double *y /*m, duals cB^T B^-1 via LU, optional*/);

/* Revised simplex: min c'x s.t. A x = b, x >= 0, from a PRIMAL-feasible basis
 * (`basis[m]`). Keeps B^-1 with eta updates (one factorization per
 * iteration, not a dense tableau). Statuses as above; 1 = primal-infeasible
 * basis. x[n]. Ported from gmbortools `gor_lp_revised_simplex`
 * (from athityakumar/or_lab). */
int simplex_revised_solve_std(const double *A, int m, int n,
                              const double *b, const double *c,
                              const int *basis, int max_iter, double *x /*n*/,
                              double *y /*m, duals, optional (NULL ok)*/);

/* Reduced costs c - A'y (n). Ported from gmbortools `gor_lp_reduced_costs`. */
void simplex_reduced_costs(const double *A, int m, int n, const double *c,
                           const double *y, double *red /*n*/);

/* Crash basis: m linearly independent columns of A (Gaussian elimination
 * with column pivoting). Returns 1 and fills `basis[m]` (column indices)
 * when rank(A) == m, 0 otherwise. Gives the revised simplex a valid
 * starting basis without phase 1 (the MSK_IPAR_SIM_PRIMAL_CRASH concept). */
int simplex_crash_basis(const double *A, int m, int n, int *basis /*m*/);

#endif /* SIMPLEX_H */
