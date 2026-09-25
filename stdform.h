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

/* stdform.h - transformation of a general LP/QP into standard form
 *   min  1/2 xt' Qs xt + cs' xt  (+ const)
 *   s.t. As xt = bs,  xt >= 0
 * Handles: variable shifts (finite lower bound), negated columns (upper-bound-
 * only variables), split free variables, fixed-variable substitution,
 * ranged/equality/free constraints, per-variable upper-bound rows,
 * row normalization (bs >= 0), and dual mapping back to the original rows.
 *
 * STORAGE IS SPARSE: As is held in CSC (Aptr/Arow/Aval) and Qs in lower-triangle
 * CSC with an explicit diagonal (Qptr/Qrow/Qval), so a large sparse LP/QP never
 * materializes a dense m x n or n x n matrix. The dense solvers (tableau simplex,
 * dense augmented IPM) materialize on demand via stdform_dense_A/Q. */
#ifndef STDFORM_H
#define STDFORM_H

typedef enum { SFRK_EQ = 0, SFRK_LO, SFRK_UP, SFRK_VARUB } SfrRowKind;

typedef struct {
    int        orig;   /* constraint index (EQ/LO/UP) or variable index (VARUB) */
    SfrRowKind kind;
    double     sigma;  /* row normalization sign (+1/-1) */
} SfrRow;

typedef enum { SFCK_VAR = 0, SFCK_SLACK } SfrColKind;

typedef struct {
    SfrColKind kind;
    int    idx;    /* original variable j (VAR) or owning std row r (SLACK) */
    double tau;    /* VAR: sign of x_j = shift_j + tau*xt (split: two cols +1/-1) */
} SfrCol;

typedef struct {
    int    ncols;        /* 0 (fixed), 1, or 2 (split free var) */
    int    col[2];
    double tau[2];
    double shift;
    int    fixed;
    double fixval;
} SfrVar;

typedef struct {
    int     m, n, nvar, ncon;
    int    *Aptr, *Arow;  /* A in CSC: Aptr[n+1], Arow/Aval[nnz(A)] (rows sorted) */
    double *Aval;
    double *b;            /* m, >= 0 */
    double *c;            /* n */
    int    *Qptr, *Qrow;  /* Q in lower CSC (i>=j, explicit diagonal), or NULL (LP) */
    double *Qval;
    double  cfix;         /* objective constant of the min-form problem (diagnostic) */
    SfrRow *rows;         /* m */
    SfrCol *cols;         /* n */
    SfrVar *vars;         /* nvar */
} StdForm;

/* Bounds use +/- INFINITY convention. A is given by columns (CSC): col_ptr has
 * nvar+1 entries. The quadratic objective is given as nq sparse triplets
 * (qi,qj,qv) interpreted symmetrically (Q[i][j]=Q[j][i]=v); pass nq=0 / NULL
 * for an LP. Returns NULL on invalid bounds or alloc fail. */
StdForm *stdform_build(int nvar, int ncon,
                       const double *c_int,
                       const int *qi, const int *qj, const double *qv, int nq,
                       const double *lx, const double *ux,
                       const double *lc, const double *uc,
                       const int *col_ptr, const int *sub, const double *val);
void stdform_free(StdForm *sf);

/* map standard-form primal solution to original variables */
void stdform_map_x(const StdForm *sf, const double *xt, double *x);
/* map a standard-form recession DIRECTION (xt, n entries) to original variables.
 * Same column signs as stdform_map_x, without the variable shifts: a shift moves
 * a point, and it is not part of a direction. */
void stdform_map_dir(const StdForm *sf, const double *dxt, double *dx);
/* map standard-form row duals (min form, c = A'y + w) to per-constraint duals
 * in min form: ymin[i] = -sum_r sigma_r * ystd_r over rows of constraint i */
void stdform_map_y(const StdForm *sf, const double *ystd, double *ymin);

/* Dense materializations for the dense solvers (caller frees the result).
 * stdform_dense_A returns m x n row-major (NULL on alloc fail).
 * stdform_dense_Q returns n x n symmetric (NULL if there is no Q or on fail). */
double *stdform_dense_A(const StdForm *sf);
double *stdform_dense_Q(const StdForm *sf);

#endif /* STDFORM_H */
