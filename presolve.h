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

/* presolve.h - LP presolve on the standard form  min c'x, A x = b, x >= 0,
 * with A SPARSE in CSC. Safe, structure-triggered reductions with exact
 * primal+dual postsolve; the reduced problem is returned in CSC too, so the
 * whole pipeline stays sparse (no dense m x n is ever materialized).
 * See presolve.c for the reduction set and the postsolve proof. */
#ifndef PRESOLVE_H
#define PRESOLVE_H

typedef struct Presolve Presolve;

/* Reduce  min c'x, A x = b, x >= 0  (A in CSC: Aptr[n+1], Arow, Aval).
 * Returns 0 on success (*out set), -1 on allocation failure (then *out is NULL
 * and the caller should solve the original). tol is the value-check tolerance. */
int lp_presolve(const int *Aptr, const int *Arow, const double *Aval,
                const double *b, const double *c, int m, int n, double tol,
                int level, Presolve **out);

/* Non-zero if at least one reduction was applied (otherwise the reduced problem
 * equals the input and the caller may solve the original directly). */
int presolve_changed(const Presolve *p);

/* Access the reduced problem (CSC: Arptr[nr+1], Arrow, Arval). */
void presolve_reduced(const Presolve *p, const int **Arptr, const int **Arrow,
                      const double **Arval, const double **br, const double **cr,
                      int *mr, int *nr);

/* Expand a reduced solution (xred[nr], yred[mr]; either may be NULL when the
 * corresponding reduced dimension is 0) into the full space xfull[n], yfull[m]. */
void presolve_postsolve(const Presolve *p, const double *xred, const double *yred,
                        double *xfull, double *yfull);

/* Lift a reduced DIRECTION (a Farkas ray) from the reduced space to the full
 * one.  Columns: kept ones take the reduced value, removed ones stay 0 (a
 * substituted column is fixed to a CONSTANT, so its direction is 0).  Rows: the
 * reduction log is replayed in reverse with the HOMOGENEOUS form of the primal
 * recovery, y_i = -(sum_{k != i} A_kj y_k)/A_ij — the c_j term that recovery
 * solves for is exactly what a homogeneous witness must not inherit.  Every row
 * recorded in an operation was still alive when the operation was created, so
 * the reverse pass has all of them.  That lift leaves b'y unchanged (the reduced
 * RHS is b_k - A_kj·x_j^fix, so the recovered y_i pays back y_i·b_i), which is
 * what lets a witness found after a singleton substitution still name the
 * original rows.  Either input may be NULL; the caller re-measures the result in
 * the full space, so a ray that needs a removed column costs publication, never
 * correctness. */
void presolve_postsolve_dir(const Presolve *p, const double *rred, const double *yred,
                            double *rfull, double *yfull);

void presolve_free(Presolve *p);

#endif /* PRESOLVE_H */
