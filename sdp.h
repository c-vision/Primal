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

/* sdp.h - primal-dual interior point for conic problems (standard form)
 *
 *   min  c'x + sum_j <C_j, X_j> + sum_i <c_i, z_i>
 *   s.t. A x + sum_j <A_kj, X_j> + sum_i <a_ki, z_i> = b_k
 *        x >= 0,  X_j >= 0 (PSD),  z_i in SOC
 *
 * A    : m x n row-major.
 * Cbar : nb symmetric matrices, Cbar[j] -> d_j*d_j row-major.
 * Abar : m*nb symmetric matrices, Abar[k*nb+j] -> d_j*d_j row-major.
 * Csoc : nsoc vectors, Csoc[i] -> k_i.
 * Asoc : m*nsoc vectors, Asoc[k*nsoc+i] -> k_i.
 *
 * Unified conic path: R_+, SOC and PSD blocks in the same interior point.
 * R_+ and PSD use Nesterov-Todd scaling + normal equations on the m equality
 * rows.  SOC blocks use the socp.c arrow-matrix augmented KKT (the NT scaling
 * point diverges at the cone boundary), so the SOC direction stays robust
 * near the optimum.
 *
 * Outputs: x (n), Xbar[j] (d_j*d_j primal PSD), y (m), Sbar[j] (d_j*d_j dual
 * PSD), Zsoc[i] (k_i primal SOC), Ssoc[i] (k_i dual SOC).
 *
 * tol is the absolute Newton tolerance the loop accepts an iterate at.  The
 * three rtol_* are the RELATIVE criteria a point is judged by: primal
 * infeasibility /(1+|b|_inf), dual infeasibility /(1+|c|_inf) and duality gap
 * |pobj-dobj|/(1+|pobj|+|dobj|).  With exp/power blocks a point that misses
 * them is reported as 1 (not solved), which sends the caller to the tangent-cut
 * outer approximation.  Without them the same three criteria are still
 * measured, and near_rel is what decides the verdict: the reference's rule is
 * that a point satisfying the criteria with every tolerance multiplied by
 * near_rel is declared optimal, so a model with no alternative route is
 * reported 1 only when it misses rtol_* * near_rel.  near_rel = 1 disables the
 * rule and makes the declared tolerances the verdict.
 * Returns 0 optimal, 1 max-iter, 2 memory, 3 singular. */
#ifndef SDP_H
#define SDP_H

int sdp_ipm(int m, int n, const double *A, const double *b, const double *c,
            int nb, const int *dims,
            const double *const *Cbar, const double *const *Abar,
            int nsoc, const int *socdims,
            const double *const *Csoc, const double *const *Asoc,
            int nep, const int *ekind, const double *ealpha,
            const double *const *Cexp, const double *const *Aexp,
            int max_iter, double tol,
            double rtol_pri, double rtol_dual, double rtol_gap,
            double near_rel,
            double *x, double *const *Xbar, double *y, double *const *Sbar,
            double *const *Zsoc, double *const *Ssoc,
            double *const *Zexp, double *const *Sexp, const double *xwarm,
            const double *ywarm,
            int *fb_ok /* out: un punto near-optimal e' stato salvato come fallback */);

#endif /* SDP_H */
