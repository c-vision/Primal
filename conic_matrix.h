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

#ifndef CONIC_MATRIX_H
#define CONIC_MATRIX_H
#include <stddef.h>
#include "socp.h"

/* Internal master rows: unused future cut slots have no coefficient buffers. */
typedef struct { int col; double value; } ConicEntry;
typedef struct {
    int nz, cap, maxcol; /* maxcol remains an upper bound after cancellation */
    ConicEntry *entry;
} ConicRow;
typedef struct {
    int rows, cols, used, failed;
    size_t bytes;
    ConicRow *row;
} ConicMatrix;
ConicMatrix *conic_matrix_new(int rows, int cols);
void conic_matrix_free(ConicMatrix *m);
void conic_matrix_set(ConicMatrix *m, int row, int col, double value);
void conic_matrix_add(ConicMatrix *m, int row, int col, double value);
void conic_matrix_clear(ConicMatrix *m, int first);
void conic_matrix_sort(ConicMatrix *m);
/* out must be empty; release any previous export before calling again. */
int conic_matrix_csr(ConicMatrix *m, int rows, SocpCsr *out);
void conic_csr_free(SocpCsr *m);
double *conic_matrix_dense(const ConicMatrix *m, int rows);
#endif
