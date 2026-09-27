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
/* primal_put.c - model input: columns/rows, objective, bounds.
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* ---------------- data input ---------------- */

/**
 * Sets a single linear objective coefficient.
 *
 * @param t  [in] Task handle.
 * @param j  [in] Variable index (0 <= j < numvar).
 * @param cj [in] Coefficient value for variable j in the linear objective.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if j is out of bounds.
 *
 * @note This directly sets c[j] in the linear objective vector. For setting
 *       multiple coefficients at once, use PRIMAL_putclist or PRIMAL_putcslice.
 *
 * @example
 * // Set coefficient of x_3 to 2.5
 * PRIMAL_putcj(task, 3, 2.5);
 */
PRIMALrescodee PRIMAL_putcj(PRIMALtask_t t, int j, double cj) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    t->c[j] = cj;
    return PRIMAL_RES_OK;
}

/**
 * Sets multiple linear objective coefficients by index list.
 *
 * @param t   [in] Task handle.
 * @param num [in] Number of coefficients to set. Must be non-negative.
 * @param subj [in] Array of variable indices (length num). Each must satisfy
 *                   0 <= subj[k] < numvar.
 * @param val  [in] Array of coefficient values (length num).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or (num > 0 and subj/val is NULL),
 *         PRIMAL_RES_ERR_ARG if num < 0 or any index is out of bounds.
 *
 * @note The entire input is validated before any modification is made. If any
 *       index is invalid, the function returns an error and the objective
 *       vector remains unchanged. This is an atomic operation.
 *
 * @example
 * int idx[] = {0, 2, 4};
 * double vals[] = {1.0, -2.0, 3.5};
 * PRIMAL_putclist(task, 3, idx, vals);
 * // Sets c[0]=1.0, c[2]=-2.0, c[4]=3.5
 */
PRIMALrescodee PRIMAL_putclist(PRIMALtask_t t, int num, const int *subj, const PRIMALrealt *val) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subj || !val))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++)
        if (subj[k] < 0 || subj[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < num; k++) t->c[subj[k]] = val[k];
    return PRIMAL_RES_OK;
}

/**
 * Sets a contiguous slice of the linear objective coefficients.
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in] Ending index (exclusive), first <= last <= numvar.
 * @param c     [in] Array of coefficient values (length last - first).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or c is NULL,
 *         PRIMAL_RES_ERR_ARG if indices are out of bounds or first > last.
 *
 * @note Copies c[k] to objective variable (first + k) for k = 0..(last-first-1).
 *       The slice is [first, last), i.e., last is exclusive. An empty slice
 *       (first == last) is valid and does nothing.
 *
 * @example
 * double vals[] = {1.0, 2.0, 3.0, 4.0};
 * PRIMAL_putcslice(task, 0, 4, vals);
 * // Sets c[0]=1.0, c[1]=2.0, c[2]=3.0, c[3]=4.0
 */
PRIMALrescodee PRIMAL_putcslice(PRIMALtask_t t, int first, int last, const PRIMALrealt *c) {
    model_changed(t);
    if (!t || !c) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    for (int j = first; j < last; j++) t->c[j] = c[j - first];
    return PRIMAL_RES_OK;
}

/**
 * Sets the fixed constant term in the objective function.
 *
 * @param t    [in] Task handle.
 * @param cfix [in] Constant term added to the objective value.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note The constant term is added to both primal and dual objective values
 *       reported by the solver. It does not affect the optimization problem
 *       itself (the optimal x is unchanged), but shifts the objective value.
 *       In the dual, the constant appears in the dual objective.
 *
 *       When converting a ranged variable l <= x <= u to the conic form,
 *       the substitution x = l + u' introduces a constant term cfix += c_j * l_j
 *       that must be accounted for in the dual objective.
 *
 * @example
 * // Add constant 5.0 to objective: min c'x + 5.0
 * PRIMAL_putcfix(task, 5.0);
 */
PRIMALrescodee PRIMAL_putcfix(PRIMALtask_t t, double cfix) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    t->cfix = cfix;
    return PRIMAL_RES_OK;
}

/**
 * Replaces an entire column of the constraint matrix A.
 *
 * @param t   [in] Task handle.
 * @param j   [in] Column index (variable index), 0 <= j < numvar.
 * @param nz  [in] Number of non-zero entries in this column. Must be non-negative.
 * @param sub [in] Array of row indices (length nz). Each must satisfy
 *                 0 <= sub[k] < numcon.
 * @param val [in] Array of coefficient values (length nz).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds, nz < 0, or any row index invalid,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note This REPLACES the entire column j. Any previous non-zero entries in
 *       column j are removed. The column storage uses a simple dense vector
 *       of (row, value) pairs. If nz exceeds the current column capacity,
 *       the column is reallocated.
 *
 *       Duplicate row indices within the same call are allowed and stored as
 *       separate entries. The solver's internal matrix-vector products will
 *       sum them (the operator is the sum of entries at the same position).
 *
 * @example
 * // Column 2: A[0,2] = 1.0, A[3,2] = -2.0
 * int rows[] = {0, 3};
 * double vals[] = {1.0, -2.0};
 * PRIMAL_putacol(task, 2, 2, rows, vals);
 */
PRIMALrescodee PRIMAL_putacol(PRIMALtask_t t, int j, int nz, const int *sub, const double *val) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar || nz < 0) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < nz; k++)
        if (sub[k] < 0 || sub[k] >= t->numcon) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    Col *c = &t->cols[j];
    if (nz > c->cap) {
        int *s2 = (int *)realloc(c->sub, (size_t)nz * sizeof(int));
        double *v2 = (double *)realloc(c->val, (size_t)nz * sizeof(double));
        if (!s2 || !v2) return PRIMAL_RES_ERR_ALLOC;
        c->sub = s2; c->val = v2; c->cap = nz;
    }
    c->nz = nz;
    for (int k = 0; k < nz; k++) { c->sub[k] = sub[k]; c->val[k] = val[k]; }
    return PRIMAL_RES_OK;
}

/**
 * Replaces an entire row of the constraint matrix A.
 *
 * @param t   [in] Task handle.
 * @param i   [in] Row index (constraint index), 0 <= i < numcon.
 * @param nz  [in] Number of non-zero entries in this row. Must be non-negative.
 * @param sub [in] Array of column indices (variable indices, length nz).
 *                 Each must satisfy 0 <= sub[k] < numvar.
 * @param val [in] Array of coefficient values (length nz).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds, nz < 0, or any column index invalid,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note This REPLACES the entire row i. The implementation first removes all
 *       entries in row i from every column (scanning all columns), then adds
 *       the new entries. This means putarow has O(numvar) complexity due to
 *       the column-clearing step. For setting multiple rows efficiently, use
 *       PRIMAL_putarowlist or PRIMAL_putarowslice.
 *
 *       Duplicate column indices within the same call are allowed and stored
 *       as separate entries. The solver's internal operations sum them.
 *
 * @example
 * // Row 1: A[1,0] = 2.0, A[1,2] = -1.0
 * int cols[] = {0, 2};
 * double vals[] = {2.0, -1.0};
 * PRIMAL_putarow(task, 1, 2, cols, vals);
 */
PRIMALrescodee PRIMAL_putarow(PRIMALtask_t t, int i, int nz, const int *sub, const double *val) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || nz < 0) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < nz; k++)
        if (sub[k] < 0 || sub[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    /* replace semantics: clear row i from every column, then set */
    for (int j = 0; j < t->numvar; j++) {
        Col *c = &t->cols[j];
        int w = 0;
        for (int k = 0; k < c->nz; k++)
            if (c->sub[k] != i) { c->sub[w] = c->sub[k]; c->val[w] = c->val[k]; w++; }
        c->nz = w;
    }
    for (int k = 0; k < nz; k++) {
        Col *c = &t->cols[sub[k]];
        if (c->nz == c->cap) {
            int ncap = c->cap ? c->cap * 2 : 4;
            int *s2 = (int *)realloc(c->sub, (size_t)ncap * sizeof(int));
            if (!s2) return PRIMAL_RES_ERR_ALLOC;
            c->sub = s2;
            double *v2 = (double *)realloc(c->val, (size_t)ncap * sizeof(double));
            if (!v2) return PRIMAL_RES_ERR_ALLOC;
            c->val = v2; c->cap = ncap;
        }
        c->sub[c->nz] = i; c->val[c->nz] = val[k]; c->nz++;
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets multiple rows of the constraint matrix A in one call (CSR format).
 *
 * @param t    [in] Task handle.
 * @param num  [in] Number of rows to set. Must be non-negative.
 * @param sub  [in] Array of row indices (length num). Each must satisfy
 *                 0 <= sub[k] < numcon.
 * @param ptrb [in] Array of start pointers (length num). ptrb[k] is the start
 *                  index in asub/aval for row k.
 * @param ptre [in] Array of end pointers (length num). ptre[k] is the end
 *                  index (exclusive) in asub/aval for row k.
 * @param asub [in] Concatenated column indices for all rows (length ptre[num-1]).
 * @param aval [in] Concatenated coefficient values for all rows (length ptre[num-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if any index or pointer range is invalid.
 *
 * @note The entire input is validated before any modification. Each row is set
 *       using putarow semantics (replace). The CSR format uses half-open
 *       intervals [ptrb[k], ptre[k]) for each row. Rows not in sub are unchanged.
 *
 *       This is more efficient than repeated putarow calls because validation
 *       is batched, but each row still incurs the column-clearing cost.
 *
 * @example
 * // Set row 0: A[0,0]=1, A[0,1]=2; row 2: A[2,1]=3
 * int rows[] = {0, 2};
 * int ptrb[] = {0, 2};
 * int ptre[] = {2, 3};
 * int cols[] = {0, 1, 1};
 * double vals[] = {1.0, 2.0, 3.0};
 * PRIMAL_putarowlist(task, 2, rows, ptrb, ptre, cols, vals);
 */
PRIMALrescodee PRIMAL_putarowlist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !ptrb || !ptre || !asub || !aval))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->numcon) return PRIMAL_RES_ERR_ARG;
        if (ptrb[k] < 0 || ptre[k] < ptrb[k]) return PRIMAL_RES_ERR_ARG;
        for (int e = ptrb[k]; e < ptre[k]; e++)
            if (asub[e] < 0 || asub[e] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++)
        PRIMAL_putarow(t, sub[k], ptre[k] - ptrb[k], asub + ptrb[k], aval + ptrb[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets multiple columns of the constraint matrix A in one call (CSC format).
 *
 * @param t    [in] Task handle.
 * @param num  [in] Number of columns to set. Must be non-negative.
 * @param sub  [in] Array of column indices (length num). Each must satisfy
 *                 0 <= sub[k] < numvar.
 * @param ptrb [in] Array of start pointers (length num). ptrb[k] is the start
 *                  index in asub/aval for column k.
 * @param ptre [in] Array of end pointers (length num). ptre[k] is the end
 *                  index (exclusive) in asub/aval for column k.
 * @param asub [in] Concatenated row indices for all columns (length ptre[num-1]).
 * @param aval [in] Concatenated coefficient values for all columns (length ptre[num-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if any index or pointer range is invalid.
 *
 * @note The entire input is validated before any modification. Each column is
 *       set using putacol semantics (replace). The CSC format uses half-open
 *       intervals [ptrb[k], ptre[k]) for each column. Columns not in sub are unchanged.
 *
 *       This is more efficient than repeated putacol calls because validation
 *       is batched and no row-clearing is needed (columns are independent).
 *
 * @example
 * // Set col 0: A[0,0]=1, A[1,0]=2; col 2: A[1,2]=3
 * int cols[] = {0, 2};
 * int ptrb[] = {0, 2};
 * int ptre[] = {2, 3};
 * int rows[] = {0, 1, 1};
 * double vals[] = {1.0, 2.0, 3.0};
 * PRIMAL_putacollist(task, 2, cols, ptrb, ptre, rows, vals);
 */
PRIMALrescodee PRIMAL_putacollist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !ptrb || !ptre || !asub || !aval))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (ptrb[k] < 0 || ptre[k] < ptrb[k]) return PRIMAL_RES_ERR_ARG;
        for (int e = ptrb[k]; e < ptre[k]; e++)
            if (asub[e] < 0 || asub[e] >= t->numcon) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++)
        PRIMAL_putacol(t, sub[k], ptre[k] - ptrb[k], asub + ptrb[k], aval + ptrb[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets a contiguous slice of rows of the constraint matrix A (CSR format).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting row index (inclusive), 0 <= first <= numcon.
 * @param last  [in] Ending row index (exclusive), first <= last <= numcon.
 * @param ptrb  [in] Array of start pointers (length last-first). ptrb[k] is the
 *                   start index in asub/aval for row (first+k).
 * @param ptre  [in] Array of end pointers (length last-first). ptre[k] is the
 *                   end index (exclusive) in asub/aval for row (first+k).
 * @param asub  [in] Concatenated column indices for all rows in the slice.
 * @param aval  [in] Concatenated coefficient values for all rows in the slice.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or pointers invalid.
 *
 * @note This is a convenience wrapper for putarowlist where the rows are
 *       a contiguous range [first, last). Each row is set with replace semantics.
 *       An empty slice (first == last) is valid and does nothing.
 *
 * @example
 * // Set rows 1..3 (1, 2, 3)
 * int ptrb[] = {0, 1, 2};
 * int ptre[] = {1, 2, 3};
 * int cols[] = {0, 1, 2};
 * double vals[] = {1.0, 2.0, 3.0};
 * PRIMAL_putarowslice(task, 1, 4, ptrb, ptre, cols, vals);
 */
PRIMALrescodee PRIMAL_putarowslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numcon) return PRIMAL_RES_ERR_ARG;
    if (last > first && (!ptrb || !ptre || !asub || !aval)) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++) {
        if (ptrb[k] < 0 || ptre[k] < ptrb[k]) return PRIMAL_RES_ERR_ARG;
        for (int e = ptrb[k]; e < ptre[k]; e++)
            if (asub[e] < 0 || asub[e] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    }
    for (int i = first; i < last; i++) {
        int k = i - first;
        PRIMAL_putarow(t, i, ptre[k] - ptrb[k], asub + ptrb[k], aval + ptrb[k]);
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets a contiguous slice of columns of the constraint matrix A (CSC format).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting column index (inclusive), 0 <= first <= numvar.
 * @param last  [in] Ending column index (exclusive), first <= last <= numvar.
 * @param ptrb  [in] Array of start pointers (length last-first). ptrb[k] is the
 *                   start index in asub/aval for column (first+k).
 * @param ptre  [in] Array of end pointers (length last-first). ptre[k] is the
 *                   end index (exclusive) in asub/aval for column (first+k).
 * @param asub  [in] Concatenated row indices for all columns in the slice.
 * @param aval  [in] Concatenated coefficient values for all columns in the slice.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or pointers invalid.
 *
 * @note This is a convenience wrapper for putacollist where the columns are
 *       a contiguous range [first, last). Each column is set with replace semantics.
 *       An empty slice (first == last) is valid and does nothing.
 *
 * @example
 * // Set columns 1..3 (1, 2, 3)
 * int ptrb[] = {0, 1, 2};
 * int ptre[] = {1, 2, 3};
 * int rows[] = {0, 1, 2};
 * double vals[] = {1.0, 2.0, 3.0};
 * PRIMAL_putacolslice(task, 1, 4, ptrb, ptre, rows, vals);
 */
PRIMALrescodee PRIMAL_putacolslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numvar) return PRIMAL_RES_ERR_ARG;
    if (last > first && (!ptrb || !ptre || !asub || !aval)) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++) {
        if (ptrb[k] < 0 || ptre[k] < ptrb[k]) return PRIMAL_RES_ERR_ARG;
        for (int e = ptrb[k]; e < ptre[k]; e++)
            if (asub[e] < 0 || asub[e] >= t->numcon) return PRIMAL_RES_ERR_ARG;
    }
    for (int j = first; j < last; j++) {
        int k = j - first;
        PRIMAL_putacol(t, j, ptre[k] - ptrb[k], asub + ptrb[k], aval + ptrb[k]);
    }
    return PRIMAL_RES_OK;
}

/**
 * Helper: converts 64-bit pointer arrays to 32-bit with range checking.
 *
 * @param num  [in] Number of elements to convert.
 * @param p64  [in] Source array of PRIMALint64t (length num).
 * @param p32  [out] Destination array of int (length num).
 *
 * @return 1 on success (all values fit in int), 0 if any value is negative
 *         or exceeds INT_MAX.
 *
 * @note Internal helper used by the *64 variants of matrix input functions.
 *       The reference API uses MSKint64t for pointer arrays to support very
 *       large models; this solver uses int internally, so conversion with
 *       range checking is required.
 */
static int ptr64_to_int(int num, const PRIMALint64t *p64, int *p32) {
    for (int k = 0; k < num; k++) {
        if (p64[k] < 0 || p64[k] > INT_MAX) return 0;
        p32[k] = (int)p64[k];
    }
    return 1;
}
/**
 * Sets a contiguous slice of rows using 64-bit pointer arrays (CSR format).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting row index (inclusive), 0 <= first <= numcon.
 * @param last  [in] Ending row index (exclusive), first <= last <= numcon.
 * @param ptrb  [in] Array of 64-bit start pointers (length last-first).
 * @param ptre  [in] Array of 64-bit end pointers (length last-first).
 * @param asub  [in] Concatenated column indices (int, length ptre[last-first-1]).
 * @param aval  [in] Concatenated coefficient values (length ptre[last-first-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds, any 64-bit pointer
 *         exceeds INT_MAX, or pointer ranges invalid.
 *
 * @note This is the 64-bit variant of PRIMAL_putarowslice for compatibility
 *       with the reference API. The pointer arrays (ptrb, ptre) use
 *       PRIMALint64t to support models with more than 2^31 non-zeros.
 *       The column indices (asub) and values (aval) remain 32-bit.
 *       Any 64-bit pointer value that doesn't fit in a signed 32-bit int
 *       causes ERR_ARG.
 *
 * @example
 * // Same as putarowslice but with 64-bit pointers
 * PRIMALint64t ptrb64[] = {0, 1, 2};
 * PRIMALint64t ptre64[] = {1, 2, 3};
 * PRIMAL_putarowslice64(task, 1, 4, ptrb64, ptre64, cols, vals);
 */
PRIMALrescodee PRIMAL_putarowslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numcon) return PRIMAL_RES_ERR_ARG;
    int n = last - first;
    if (n == 0) return PRIMAL_RES_OK;
    if (!ptrb || !ptre || !asub || !aval) return PRIMAL_RES_ERR_NULL;
    int *b = (int *)malloc((size_t)n * sizeof(int));
    int *e = (int *)malloc((size_t)n * sizeof(int));
    if (!b || !e) { free(b); free(e); return PRIMAL_RES_ERR_ALLOC; }
    if (!ptr64_to_int(n, ptrb, b) || !ptr64_to_int(n, ptre, e)) {
        free(b); free(e); return PRIMAL_RES_ERR_ARG;
    }
    PRIMALrescodee rc = PRIMAL_putarowslice(t, first, last, b, e, asub, aval);
    free(b); free(e);
    return rc;
}
/**
 * Sets a contiguous slice of columns using 64-bit pointer arrays (CSC format).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting column index (inclusive), 0 <= first <= numvar.
 * @param last  [in] Ending column index (exclusive), first <= last <= numvar.
 * @param ptrb  [in] Array of 64-bit start pointers (length last-first).
 * @param ptre  [in] Array of 64-bit end pointers (length last-first).
 * @param asub  [in] Concatenated row indices (int, length ptre[last-first-1]).
 * @param aval  [in] Concatenated coefficient values (length ptre[last-first-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds, any 64-bit pointer
 *         exceeds INT_MAX, or pointer ranges invalid.
 *
 * @note This is the 64-bit variant of PRIMAL_putacolslice. See putarowslice64
 *       for details on the 64-bit pointer array convention.
 */
PRIMALrescodee PRIMAL_putacolslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numvar) return PRIMAL_RES_ERR_ARG;
    int n = last - first;
    if (n == 0) return PRIMAL_RES_OK;
    if (!ptrb || !ptre || !asub || !aval) return PRIMAL_RES_ERR_NULL;
    int *b = (int *)malloc((size_t)n * sizeof(int));
    int *e = (int *)malloc((size_t)n * sizeof(int));
    if (!b || !e) { free(b); free(e); return PRIMAL_RES_ERR_ALLOC; }
    if (!ptr64_to_int(n, ptrb, b) || !ptr64_to_int(n, ptre, e)) {
        free(b); free(e); return PRIMAL_RES_ERR_ARG;
    }
    PRIMALrescodee rc = PRIMAL_putacolslice(t, first, last, b, e, asub, aval);
    free(b); free(e);
    return rc;
}
/**
 * Sets multiple rows using 64-bit pointer arrays (CSR format).
 *
 * @param t    [in] Task handle.
 * @param num  [in] Number of rows to set. Must be non-negative.
 * @param sub  [in] Array of row indices (int, length num).
 * @param ptrb [in] Array of 64-bit start pointers (length num).
 * @param ptre [in] Array of 64-bit end pointers (length num).
 * @param asub [in] Concatenated column indices (int, length ptre[num-1]).
 * @param aval [in] Concatenated coefficient values (length ptre[num-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if any index out of bounds or 64-bit pointer
 *         exceeds INT_MAX.
 *
 * @note 64-bit variant of PRIMAL_putarowlist. The row indices (sub) remain
 *       32-bit; only the pointer arrays use PRIMALint64t.
 */
PRIMALrescodee PRIMAL_putarowlist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !ptrb || !ptre || !asub || !aval))) return PRIMAL_RES_ERR_NULL;
    int *b = (int *)malloc((size_t)(num > 0 ? num : 1) * sizeof(int));
    int *e = (int *)malloc((size_t)(num > 0 ? num : 1) * sizeof(int));
    if (!b || !e) { free(b); free(e); return PRIMAL_RES_ERR_ALLOC; }
    if (!ptr64_to_int(num, ptrb, b) || !ptr64_to_int(num, ptre, e)) {
        free(b); free(e); return PRIMAL_RES_ERR_ARG;
    }
    PRIMALrescodee rc = PRIMAL_putarowlist(t, num, sub, b, e, asub, aval);
    free(b); free(e);
    return rc;
}
/**
 * Sets multiple columns using 64-bit pointer arrays (CSC format).
 *
 * @param t    [in] Task handle.
 * @param num  [in] Number of columns to set. Must be non-negative.
 * @param sub  [in] Array of column indices (int, length num).
 * @param ptrb [in] Array of 64-bit start pointers (length num).
 * @param ptre [in] Array of 64-bit end pointers (length num).
 * @param asub [in] Concatenated row indices (int, length ptre[num-1]).
 * @param aval [in] Concatenated coefficient values (length ptre[num-1]).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if any index out of bounds or 64-bit pointer
 *         exceeds INT_MAX.
 *
 * @note 64-bit variant of PRIMAL_putacollist.
 */
PRIMALrescodee PRIMAL_putacollist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !ptrb || !ptre || !asub || !aval))) return PRIMAL_RES_ERR_NULL;
    int *b = (int *)malloc((size_t)(num > 0 ? num : 1) * sizeof(int));
    int *e = (int *)malloc((size_t)(num > 0 ? num : 1) * sizeof(int));
    if (!b || !e) { free(b); free(e); return PRIMAL_RES_ERR_ALLOC; }
    if (!ptr64_to_int(num, ptrb, b) || !ptr64_to_int(num, ptre, e)) {
        free(b); free(e); return PRIMAL_RES_ERR_ARG;
    }
    PRIMALrescodee rc = PRIMAL_putacollist(t, num, sub, b, e, asub, aval);
    free(b); free(e);
    return rc;
}
/* 64-bit count variant of putaijlist: checks the count fits in an int,
 * then delegates to PRIMAL_putaijlist with the narrowed count. */
PRIMALrescodee PRIMAL_putaijlist64(PRIMALtask_t t, PRIMALint64t num,
    const int *subi, const int *subj, const PRIMALrealt *valij) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_putaijlist(t, (int)num, subi, subj, valij);
}

/**
 * Sets a single entry of the constraint matrix A (replaces any existing entries at (i,j)).
 *
 * @param t   [in] Task handle.
 * @param i   [in] Row index (constraint), 0 <= i < numcon.
 * @param j   [in] Column index (variable), 0 <= j < numvar.
 * @param aij [in] Coefficient value. If zero, any existing entry at (i,j) is removed.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if i or j out of bounds.
 *
 * @note This REPLACES any existing entries at position (i,j). Since the column
 *       storage can hold multiple entries for the same (i,j) (from putarow/
 *       putacol calls), putaij first removes ALL entries in column j with row i,
 *       then stores a single new entry if aij != 0.
 *
 *       This means putaij and getaij are consistent: getaij returns the sum
 *       of all stored entries at (i,j), and putaij ensures that sum equals aij.
 *
 * @example
 * PRIMAL_putaij(task, 2, 5, 3.14); // A[2,5] = 3.14
 * PRIMAL_putaij(task, 2, 5, 0.0);  // Removes A[2,5]
 */
PRIMALrescodee PRIMAL_putaij(PRIMALtask_t t, int i, int j, double aij) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    Col *c = &t->cols[j];
    int w = 0;
    for (int k = 0; k < c->nz; k++)
        if (c->sub[k] != i) { c->sub[w] = c->sub[k]; c->val[w] = c->val[k]; w++; }
    c->nz = w;
    if (aij != 0.0) {
        if (c->nz == c->cap) {
            int ncap = c->cap ? c->cap * 2 : 4;
            int *s2 = (int *)realloc(c->sub, (size_t)ncap * sizeof(int));
            if (!s2) return PRIMAL_RES_ERR_ALLOC;
            c->sub = s2;
            double *v2 = (double *)realloc(c->val, (size_t)ncap * sizeof(double));
            if (!v2) return PRIMAL_RES_ERR_ALLOC;
            c->val = v2; c->cap = ncap;
        }
        c->sub[c->nz] = i; c->val[c->nz] = aij; c->nz++;
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets multiple (i,j) entries of the constraint matrix A in one call.
 *
 * @param t     [in] Task handle.
 * @param num   [in] Number of entries to set. Must be non-negative.
 * @param subi  [in] Array of row indices (length num). Each must satisfy
 *                  0 <= subi[k] < numcon.
 * @param subj  [in] Array of column indices (length num). Each must satisfy
 *                  0 <= subj[k] < numvar.
 * @param valij [in] Array of coefficient values (length num). NaN is not allowed.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0, any index out of bounds, or NaN in valij.
 *
 * @note The entire list is validated before any modification. Each entry calls
 *       putaij semantics (replace the single (i,j) entry). If valij[k] == 0,
 *       the entry is removed. Duplicate (i,j) pairs in the list are processed
 *       sequentially (last write wins). For setting many entries, this is
 *       more convenient than repeated putaij calls.
 *
 * @example
 * int rows[] = {0, 1, 2};
 * int cols[] = {0, 1, 2};
 * double vals[] = {1.0, 2.0, 3.0};
 * PRIMAL_putaijlist(task, 3, rows, cols, vals);
 */
PRIMALrescodee PRIMAL_putaijlist(PRIMALtask_t t, int num, const int *subi,
                                 const int *subj, const PRIMALrealt *valij) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subi || !subj || !valij))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (subi[k] < 0 || subi[k] >= t->numcon ||
            subj[k] < 0 || subj[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (valij[k] != valij[k]) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++) PRIMAL_putaij(t, subi[k], subj[k], valij[k]);
    return PRIMAL_RES_OK;
}

/* Materialize the dense n x n quadratic objective from the sparse triplet store
 * (cached in t->qobj). Returns the dense matrix, or NULL if none / alloc fail.
 * Used only by the consumers that genuinely need a dense Q (conic/QCQP
 * eigendecomposition, getqobjij, shadow tasks). */
double *ensure_dense_qobj(PRIMALtask_t t) {
    if (!t || !t->has_qobj) return NULL;
    if (t->qobj) return t->qobj;
    int n = t->numvar;
    t->qobj = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    if (!t->qobj) return NULL;
    for (int e = 0; e < t->qt_n; e++) {
        int i = t->qt_i[e], j = t->qt_j[e];
        t->qobj[(size_t)i * n + j] += t->qt_v[e];
        if (i != j) t->qobj[(size_t)j * n + i] += t->qt_v[e];
    }
    return t->qobj;
}

/* x'(Q)x from the sparse triplet store (Q symmetric). */
double task_xQx(PRIMALtask_t t, const double *x) {
    double s = 0.0;
    for (int e = 0; e < t->qt_n; e++) {
        int i = t->qt_i[e], j = t->qt_j[e];
        s += t->qt_v[e] * x[i] * x[j] * (i == j ? 1.0 : 2.0);
    }
    return s;
}

/* out = Q*x (out pre-zeroed) from the sparse triplet store. */
void task_Qx(PRIMALtask_t t, const double *x, double *out) {
    for (int e = 0; e < t->qt_n; e++) {
        int i = t->qt_i[e], j = t->qt_j[e];
        out[i] += t->qt_v[e] * x[j];
        if (i != j) out[j] += t->qt_v[e] * x[i];
    }
}

/* Build the min-form scaled quadratic-objective VALUES over the task's sparse
 * triplets (t->qt_i/t->qt_j are reused as the indices): qv[e] = s * qt_v[e] *
 * ds[i]*ds[j] (ds = column scaling, may be NULL). Returns a malloc'd array the
 * caller frees, or NULL if there is no Q / on alloc failure. */
double *scaled_qvals(PRIMALtask_t t, double s, const double *ds) {
    if (!t->has_qobj) return NULL;
    double *qv = (double *)malloc((size_t)(t->qt_n > 0 ? t->qt_n : 1) * sizeof(double));
    if (!qv) return NULL;
    for (int e = 0; e < t->qt_n; e++) {
        double sc = s;
        if (ds) sc *= ds[t->qt_i[e]] * ds[t->qt_j[e]];
        qv[e] = t->qt_v[e] * sc;
    }
    return qv;
}

/**
 * Adds entries to the quadratic objective function (x'Qx).
 *
 * @param t       [in] Task handle.
 * @param numqcnz [in] Number of quadratic terms to add. Must be non-negative.
 * @param qi      [in] Array of row indices (length numqcnz). Each must satisfy
 *                    0 <= qi[k] < numvar.
 * @param qj      [in] Array of column indices (length numqcnz). Each must satisfy
 *                    0 <= qj[k] < numvar.
 * @param qoval   [in] Array of coefficient values (length numqcnz).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if numqcnz < 0, any index out of bounds, or NaN in qoval.
 *
 * @note The quadratic objective is stored as a SPARSE triplet list (qt_i, qt_j, qt_v)
 *       that ACCUMULATES values for duplicate (i,j) pairs. The full symmetric Q matrix
 *       is constructed by adding qoval[k] to both Q[qi, qj] and Q[qj, qi] when i != j.
 *       The diagonal entries are stored once (coefficient applies to x_i^2).
 *
 *       The reference API putqobj uses the FULL coefficient: to get x^2 + y^2, pass
 *       qoval = {2.0, 2.0} for the diagonal (NOT 1.0, 1.0). The solver does NOT
 *       divide by 2 for off-diagonals; the user provides the operator coefficients.
 *
 *       The dense cache (qobj) is invalidated on each call and rebuilt lazily
 *       only when needed (e.g., by the conic IPM or getqobjij).
 *
 * @example
 * // Objective: x0^2 + x1^2 + 2*x0*x1  =>  Q = [[2,2],[2,2]]
 * // Full coefficients: q00=2, q11=2, q01=2 (=> Q[0,1]+Q[1,0]=4 => 2*x0*x1)
 * int qi[] = {0, 1, 0};
 * int qj[] = {0, 1, 1};
 * double qoval[] = {2.0, 2.0, 2.0};
 * PRIMAL_putqobj(task, 3, qi, qj, qoval);
 */
PRIMALrescodee PRIMAL_putqobj(PRIMALtask_t t, int numqcnz, const int *qi, const int *qj, const double *qoval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numqcnz < 0) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < numqcnz; k++) {
        if (qi[k] < 0 || qi[k] >= t->numvar || qj[k] < 0 || qj[k] >= t->numvar)
            return PRIMAL_RES_ERR_ARG;
        if (qoval[k] != qoval[k]) return PRIMAL_RES_ERR_ARG; /* NaN */
    }
    if (numqcnz == 0) { t->has_qobj = 0; t->qt_n = 0; free(t->qobj); t->qobj = NULL; return PRIMAL_RES_OK; }
    if (t->qt_n + numqcnz > t->qt_cap) {
        int nc = t->qt_cap ? t->qt_cap : 16;
        while (nc < t->qt_n + numqcnz) nc *= 2;
        int *ni = (int *)realloc(t->qt_i, (size_t)nc * sizeof(int));
        int *nj = (int *)realloc(t->qt_j, (size_t)nc * sizeof(int));
        double *nv = (double *)realloc(t->qt_v, (size_t)nc * sizeof(double));
        if (!ni || !nj || !nv) { free(ni); free(nj); free(nv); return PRIMAL_RES_ERR_ALLOC; }
        t->qt_i = ni; t->qt_j = nj; t->qt_v = nv; t->qt_cap = nc;
    }
    for (int k = 0; k < numqcnz; k++) {
        t->qt_i[t->qt_n] = qi[k]; t->qt_j[t->qt_n] = qj[k]; t->qt_v[t->qt_n] = qoval[k]; t->qt_n++;
    }
    free(t->qobj); t->qobj = NULL;   /* invalidate the dense cache */
    t->has_qobj = 1;
    return PRIMAL_RES_OK;
}

/**
 * Sets a single entry of the quadratic objective in the lower triangle (replaces the symmetric pair).
 *
 * @param t   [in] Task handle.
 * @param i   [in] Row index, 0 <= i < numvar. Must satisfy i >= j (lower triangle).
 * @param j   [in] Column index, 0 <= j < numvar.
 * @param qoij [in] Coefficient for the (i,j) and (j,i) pair. If zero, removes the pair.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds, i < j, or NaN.
 *
 * @note This operates on the LOWER triangle only (i >= j). It REPLACES the
 *       symmetric pair (i,j) and (j,i) -- all existing entries for both
 *       positions are removed, and if qoij != 0, a single (i,j) entry is stored.
 *       The operator x'Qx will count this entry once for the diagonal (i==j)
 *       or twice for off-diagonals (i!=j, because Q[i,j]+Q[j,i] = 2*qoij).
 *
 *       This is different from putqobj which takes the FULL operator coefficient.
 *       For example, to get x0*x1 with coefficient 2, use qoij=1 with putqobjij
 *       (since Q[0,1]+Q[1,0] = 2*1 = 2), but use qoval=2 with putqobj.
 *
 * @example
 * // Set Q[1,0] = 1.5 => x'Qx includes 2*1.5*x0*x1 = 3*x0*x1
 * PRIMAL_putqobjij(task, 1, 0, 1.5);
 * // Remove Q[2,2]
 * PRIMAL_putqobjij(task, 2, 2, 0.0);
 */
PRIMALrescodee PRIMAL_putqobjij(PRIMALtask_t t, int i, int j, double qoij) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || j < 0 || i >= t->numvar || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (qoij != qoij) return PRIMAL_RES_ERR_ARG;   /* NaN */
    if (i < j) return PRIMAL_RES_ERR_ARG;          /* only lower triangle */
    int w = 0;
    for (int k = 0; k < t->qt_n; k++) {
        int a = t->qt_i[k], b = t->qt_j[k];
        if ((a == i && b == j) || (a == j && b == i)) continue;
        t->qt_i[w] = a; t->qt_j[w] = b; t->qt_v[w] = t->qt_v[k]; w++;
    }
    t->qt_n = w;
    if (qoij != 0.0) {
        if (t->qt_n + 1 > t->qt_cap) {
            int nc = t->qt_cap ? t->qt_cap * 2 : 16;
            int *ni = (int *)realloc(t->qt_i, (size_t)nc * sizeof(int));
            int *nj = (int *)realloc(t->qt_j, (size_t)nc * sizeof(int));
            double *nv = (double *)realloc(t->qt_v, (size_t)nc * sizeof(double));
            if (!ni || !nj || !nv) { free(ni); free(nj); free(nv); return PRIMAL_RES_ERR_ALLOC; }
            t->qt_i = ni; t->qt_j = nj; t->qt_v = nv; t->qt_cap = nc;
        }
        t->qt_i[t->qt_n] = i; t->qt_j[t->qt_n] = j; t->qt_v[t->qt_n] = qoij; t->qt_n++;
    }
    free(t->qobj); t->qobj = NULL;
    t->has_qobj = t->qt_n > 0;
    return PRIMAL_RES_OK;
}

/**
 * Validates a bound key and bound values.
 *
 * @param bk [in] Bound key (PRIMAL_BK_LO, UP, FX, FR, RA).
 * @param bl [in] Lower bound value.
 * @param bu [in] Upper bound value.
 *
 * @return 1 if valid, 0 if invalid.
 *
 * @note Valid bound keys are LO, UP, FX, FR, RA. For ranged bounds (RA),
 *       the lower bound must not exceed the upper bound (with a small
 *       numerical tolerance of 1e-12 * (1 + |bl|)). This validator is
 *       shared by putvarbound, putconbound, and their slice forms to ensure
 *       consistent behavior across all bound-setting APIs.
 *
 * @internal Used by putvarbound, putconbound, putvarboundslice, putconboundslice.
 */
static int bound_ok(PRIMALboundkeye bk, double bl, double bu) {
    if ((int)bk < (int)PRIMAL_BK_LO || (int)bk > (int)PRIMAL_BK_RA) return 0;
    if (bk == PRIMAL_BK_RA && bl > bu + 1e-12 * (1.0 + fabs(bl))) return 0;
    return 1;
}

/**
 * Sets the bound for a single variable.
 *
 * @param t  [in] Task handle.
 * @param j  [in] Variable index, 0 <= j < numvar.
 * @param bk [in] Bound key:
 *               - PRIMAL_BK_LO:  x_j >= bl
 *               - PRIMAL_BK_UP:  x_j <= bu
 *               - PRIMAL_BK_FX:  x_j = bl = bu (fixed)
 *               - PRIMAL_BK_FR:  free (-INF <= x_j <= +INF)
 *               - PRIMAL_BK_RA:  bl <= x_j <= bu (ranged)
 * @param bl [in] Lower bound (used for LO, FX, RA). Ignored for UP, FR.
 * @param bu [in] Upper bound (used for UP, FX, RA). Ignored for LO, FR.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds, bk invalid, or RA with bl > bu.
 *
 * @note The bound arrays are grown lazily via ensure_size if this is the first
 *       bound operation. Invalid bounds (e.g., RA with bl > bu) are rejected
 *       without modifying the model.
 *
 * @example
 * // x_3 >= 0
 * PRIMAL_putvarbound(task, 3, PRIMAL_BK_LO, 0.0, 0.0);
 * // x_5 <= 10
 * PRIMAL_putvarbound(task, 5, PRIMAL_BK_UP, 0.0, 10.0);
 * // x_2 = 5 (fixed)
 * PRIMAL_putvarbound(task, 2, PRIMAL_BK_FX, 5.0, 5.0);
 * // 1 <= x_4 <= 7 (ranged)
 * PRIMAL_putvarbound(task, 4, PRIMAL_BK_RA, 1.0, 7.0);
 */
PRIMALrescodee PRIMAL_putvarbound(PRIMALtask_t t, int j, PRIMALboundkeye bk, double bl, double bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (!bound_ok(bk, bl, bu)) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    t->bkx[j] = bk;
    switch (bk) {
        case PRIMAL_BK_LO: t->blx[j] = bl; t->bux[j] = INF; break;
        case PRIMAL_BK_UP: t->blx[j] = -INF; t->bux[j] = bu; break;
        case PRIMAL_BK_FR: t->blx[j] = -INF; t->bux[j] = INF; break;
        case PRIMAL_BK_RA: t->blx[j] = bl; t->bux[j] = bu; break;
        case PRIMAL_BK_FX: t->blx[j] = bl; t->bux[j] = bl; break;
        default: return PRIMAL_RES_ERR_ARG;
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets the bound for a single constraint.
 *
 * @param t  [in] Task handle.
 * @param i  [in] Constraint index, 0 <= i < numcon.
 * @param bk [in] Bound key (same semantics as variables):
 *               - PRIMAL_BK_LO:  constraint i >= bl
 *               - PRIMAL_BK_UP:  constraint i <= bu
 *               - PRIMAL_BK_FX:  constraint i = bl = bu (fixed)
 *               - PRIMAL_BK_FR:  free (-INF <= constraint i <= +INF)
 *               - PRIMAL_BK_RA:  bl <= constraint i <= bu (ranged)
 * @param bl [in] Lower bound (used for LO, FX, RA). Ignored for UP, FR.
 * @param bu [in] Upper bound (used for UP, FX, RA). Ignored for LO, FR.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds, bk invalid, or RA with bl > bu.
 *
 * @note Same semantics as putvarbound but for constraint bounds (blc, buc).
 *
 * @example
 * // Constraint 0: x0 + x1 >= 5
 * PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 5.0, 0.0);
 * // Constraint 1: x2 <= 10
 * PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, 0.0, 10.0);
 * // Constraint 2: 3 <= x3 + x4 <= 7 (ranged)
 * PRIMAL_putconbound(task, 2, PRIMAL_BK_RA, 3.0, 7.0);
 */
PRIMALrescodee PRIMAL_putconbound(PRIMALtask_t t, int i, PRIMALboundkeye bk, double bl, double bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    if (!bound_ok(bk, bl, bu)) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    t->bkc[i] = bk;
    switch (bk) {
        case PRIMAL_BK_LO: t->blc[i] = bl; t->buc[i] = INF; break;
        case PRIMAL_BK_UP: t->blc[i] = -INF; t->buc[i] = bu; break;
        case PRIMAL_BK_FR: t->blc[i] = -INF; t->buc[i] = INF; break;
        case PRIMAL_BK_RA: t->blc[i] = bl; t->buc[i] = bu; break;
        case PRIMAL_BK_FX: t->blc[i] = bl; t->buc[i] = bl; break;
        default: return PRIMAL_RES_ERR_ARG;
    }
    return PRIMAL_RES_OK;
}

/**
 * Determines the bound key from lower/upper bound values.
 *
 * @param lo [in] Lower bound (can be -INF).
 * @param up [in] Upper bound (can be INF).
 *
 * @return The appropriate bound key:
 *         - FR if lo=-INF and up=INF
 *         - FX if lo=up (both finite)
 *         - UP if lo=-INF and up finite
 *         - LO if lo finite and up=INF
 *         - RA otherwise (both finite, lo < up)
 *
 * @note Internal helper used by chgvarbound and chgconbound to recompute
 *       the bound key after changing one side of a bound.
 */
static PRIMALboundkeye boundkey_of(double lo, double up) {
    if (lo == -INF && up == INF) return PRIMAL_BK_FR;
    if (lo > -INF && up < INF && lo == up) return PRIMAL_BK_FX;
    if (lo == -INF) return PRIMAL_BK_UP;
    if (up == INF) return PRIMAL_BK_LO;
    return PRIMAL_BK_RA;
}

/**
 * Changes one side of a variable bound.
 *
 * @param t      [in] Task handle.
 * @param j      [in] Variable index, 0 <= j < numvar.
 * @param lower  [in] If non-zero, modify the LOWER bound; otherwise modify the UPPER bound.
 * @param finite [in] If non-zero, the new value is finite (value); if zero, the
 *                    new bound is infinite (-INF for lower, +INF for upper).
 * @param value  [in] New finite bound value (used only if finite != 0).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds.
 *
 * @note The other side of the bound is preserved. The bound key is recomputed
 *       based on the new (lo, up) pair: equal finite sides -> FX, both
 *       infinite -> FR, one infinite -> LO/UP, both finite different -> RA.
 *
 * @example
 * // Change lower bound of x_3 to 2.0
 * PRIMAL_chgvarbound(task, 3, 1, 1, 2.0);
 * // Remove upper bound of x_5 (make it +INF)
 * PRIMAL_chgvarbound(task, 5, 0, 0, 0.0);
 */
PRIMALrescodee PRIMAL_chgvarbound(PRIMALtask_t t, int j, int lower, int finite, double value) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    double lo, up;
    bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
    if (lower) lo = finite ? value : -INF;
    else       up = finite ? value : INF;
    return PRIMAL_putvarbound(t, j, boundkey_of(lo, up), lo, up);
}

/**
 * Changes one side of a constraint bound.
 *
 * @param t      [in] Task handle.
 * @param i      [in] Constraint index, 0 <= i < numcon.
 * @param lower  [in] If non-zero, modify the LOWER bound; otherwise modify the UPPER bound.
 * @param finite [in] If non-zero, the new value is finite (value); if zero, the
 *                    new bound is infinite (-INF for lower, +INF for upper).
 * @param value  [in] New finite bound value (used only if finite != 0).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds.
 *
 * @note Same semantics as chgvarbound but for constraint bounds.
 */
PRIMALrescodee PRIMAL_chgconbound(PRIMALtask_t t, int i, int lower, int finite, double value) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    double lo, up;
    bound_range(t->bkc[i], t->blc[i], t->buc[i], &lo, &up);
    if (lower) lo = finite ? value : -INF;
    else       up = finite ? value : INF;
    return PRIMAL_putconbound(t, i, boundkey_of(lo, up), lo, up);
}

/**
 * Retrieves a contiguous slice of variable bounds.
 *
 * @param t     [in]  Task handle.
 * @param first [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in]  Ending index (exclusive), first <= last <= numvar.
 * @param bk    [out] Array of bound keys (length last-first). Must not be NULL.
 * @param bl    [out] Array of lower bounds (length last-first). Must not be NULL.
 * @param bu    [out] Array of upper bounds (length last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any output array is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds.
 *
 * @note The slice is [first, last), i.e., last is exclusive. The function
 *       copies (bkx, blx, bux) for indices first..last-1 into the provided
 *       buffers. An empty slice (first == last) is valid and does nothing.
 *
 * @example
 * int n = 10;
 * PRIMALboundkeye *bk = malloc(n * sizeof(PRIMALboundkeye));
 * double *bl = malloc(n * sizeof(double));
 * double *bu = malloc(n * sizeof(double));
 * PRIMAL_getvarboundslice(task, 0, n, bk, bl, bu);
 * // bk[0..9], bl[0..9], bu[0..9] now hold bounds for variables 0..9
 */
PRIMALrescodee PRIMAL_getvarboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu) {
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numvar) return PRIMAL_RES_ERR_ARG;
    for (int j = first; j < last; j++) {
        int k = j - first;
        bk[k] = t->bkx[j]; bl[k] = t->blx[j]; bu[k] = t->bux[j];
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves a contiguous slice of constraint bounds.
 *
 * @param t     [in]  Task handle.
 * @param first [in]  Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in]  Ending index (exclusive), first <= last <= numcon.
 * @param bk    [out] Array of bound keys (length last-first). Must not be NULL.
 * @param bl    [out] Array of lower bounds (length last-first). Must not be NULL.
 * @param bu    [out] Array of upper bounds (length last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any output array is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds.
 *
 * @note Same semantics as getvarboundslice but for constraints (bkc, blc, buc).
 */
PRIMALrescodee PRIMAL_getconboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu) {
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numcon) return PRIMAL_RES_ERR_ARG;
    for (int i = first; i < last; i++) {
        int k = i - first;
        bk[k] = t->bkc[i]; bl[k] = t->blc[i]; bu[k] = t->buc[i];
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets a contiguous slice of variable bounds (atomic validation).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in] Ending index (exclusive), first <= last <= numvar.
 * @param bk    [in] Array of bound keys (length last-first). Must not be NULL.
 * @param bl    [in] Array of lower bounds (length last-first). Must not be NULL.
 * @param bu    [in] Array of upper bounds (length last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any input array is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or any bound invalid.
 *
 * @note The ENTIRE slice is validated FIRST (via bound_ok) before ANY modification.
 *       If any entry is invalid, the function returns ERR_ARG and the model is
 *       UNCHANGED. This is the "a rejected write leaves the model untouched" rule: a rejected
 *       write leaves no partial modifications.
 *
 * @example
 * PRIMALboundkeye bk[] = {PRIMAL_BK_LO, PRIMAL_BK_UP, PRIMAL_BK_FX};
 * double bl[] = {0.0, 0.0, 5.0};
 * double bu[] = {0.0, 10.0, 5.0};
 * PRIMAL_putvarboundslice(task, 0, 3, bk, bl, bu);
 * // Sets: x_0 >= 0, x_1 <= 10, x_2 = 5
 */
PRIMALrescodee PRIMAL_putvarboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu) {
    model_changed(t);
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numvar) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++)
        if (!bound_ok(bk[k], bl[k], bu[k])) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++)
        PRIMAL_putvarbound(t, first + k, bk[k], bl[k], bu[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets a contiguous slice of constraint bounds (atomic validation).
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in] Ending index (exclusive), first <= last <= numcon.
 * @param bk    [in] Array of bound keys (length last-first). Must not be NULL.
 * @param bl    [in] Array of lower bounds (length last-first). Must not be NULL.
 * @param bu    [in] Array of upper bounds (length last-first). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any input array is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or any bound invalid.
 *
 * @note Same semantics as putvarboundslice but for constraints. The entire
 *       slice is validated before any modification. A rejection leaves the
 *       model untouched.
 */
PRIMALrescodee PRIMAL_putconboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu) {
    model_changed(t);
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numcon) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++)
        if (!bound_ok(bk[k], bl[k], bu[k])) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < last - first; k++)
        PRIMAL_putconbound(t, first + k, bk[k], bl[k], bu[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets bounds for a list of variables (atomic validation).
 *
 * @param t   [in] Task handle.
 * @param num [in] Number of variables to set. Must be non-negative.
 * @param sub [in] Array of variable indices (length num). Each must satisfy
 *                 0 <= sub[k] < numvar.
 * @param bk  [in] Array of bound keys (length num).
 * @param bl  [in] Array of lower bounds (length num).
 * @param bu  [in] Array of upper bounds (length num).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0, any index out of bounds, or any bound invalid.
 *
 * @note The ENTIRE list is validated first before any modification. This is
 *       the same atomic validation rule as the slice functions. The indices
 *       in sub need not be contiguous or sorted.
 *
 * @example
 * int idx[] = {0, 2, 5};
 * PRIMALboundkeye bk[] = {PRIMAL_BK_LO, PRIMAL_BK_FX, PRIMAL_BK_UP};
 * double bl[] = {0.0, 3.0, 0.0};
 * double bu[] = {0.0, 3.0, 10.0};
 * PRIMAL_putvarboundlist(task, 3, idx, bk, bl, bu);
 * // Sets: x_0 >= 0, x_2 = 3, x_5 <= 10
 */
PRIMALrescodee PRIMAL_putvarboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !bk || !bl || !bu))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (!bound_ok(bk[k], bl[k], bu[k])) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++) PRIMAL_putvarbound(t, sub[k], bk[k], bl[k], bu[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets bounds for a list of constraints (atomic validation).
 *
 * @param t   [in] Task handle.
 * @param num [in] Number of constraints to set. Must be non-negative.
 * @param sub [in] Array of constraint indices (length num). Each must satisfy
 *                 0 <= sub[k] < numcon.
 * @param bk  [in] Array of bound keys (length num).
 * @param bl  [in] Array of lower bounds (length num).
 * @param bu  [in] Array of upper bounds (length num).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0, any index out of bounds, or any bound invalid.
 *
 * @note Same semantics as putvarboundlist but for constraints.
 */
PRIMALrescodee PRIMAL_putconboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!sub || !bk || !bl || !bu))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->numcon) return PRIMAL_RES_ERR_ARG;
        if (!bound_ok(bk[k], bl[k], bu[k])) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++) PRIMAL_putconbound(t, sub[k], bk[k], bl[k], bu[k]);
    return PRIMAL_RES_OK;
}

/**
 * Sets the same bound for a contiguous slice of variables.
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in] Ending index (exclusive), first <= last <= numvar.
 * @param bk    [in] Bound key to apply to all variables in the slice.
 * @param bl    [in] Lower bound value for all variables.
 * @param bu    [in] Upper bound value for all variables.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or bound invalid.
 *
 * @note Convenience function equivalent to calling putvarbound with the same
 *       parameters for each index in [first, last). The bound is validated
 *       once before applying to all variables.
 *
 * @example
 * // Set x_0..x_9 >= 0
 * PRIMAL_putvarboundsliceconst(task, 0, 10, PRIMAL_BK_LO, 0.0, 0.0);
 */
PRIMALrescodee PRIMAL_putvarboundsliceconst(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numvar) return PRIMAL_RES_ERR_ARG;
    if (!bound_ok(bk, bl, bu)) return PRIMAL_RES_ERR_ARG;
    for (int j = first; j < last; j++) PRIMAL_putvarbound(t, j, bk, bl, bu);
    return PRIMAL_RES_OK;
}

/**
 * Sets the same bound for a contiguous slice of constraints.
 *
 * @param t     [in] Task handle.
 * @param first [in] Starting index (inclusive), 0 <= first <= numcon.
 * @param last  [in] Ending index (exclusive), first <= last <= numcon.
 * @param bk    [in] Bound key to apply to all constraints in the slice.
 * @param bl    [in] Lower bound value for all constraints.
 * @param bu    [in] Upper bound value for all constraints.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds or bound invalid.
 *
 * @note Same semantics as putvarboundsliceconst but for constraints.
 */
PRIMALrescodee PRIMAL_putconboundsliceconst(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numcon) return PRIMAL_RES_ERR_ARG;
    if (!bound_ok(bk, bl, bu)) return PRIMAL_RES_ERR_ARG;
    for (int i = first; i < last; i++) PRIMAL_putconbound(t, i, bk, bl, bu);
    return PRIMAL_RES_OK;
}

/**
 * Sets the optimization sense (minimize or maximize).
 *
 * @param t    [in] Task handle.
 * @param sense [in] Optimization sense:
 *                 - PRIMAL_OPTIMIZE_MINIMIZE (default)
 *                 - PRIMAL_OPTIMIZE_MAXIMIZE
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if sense is invalid.
 *
 * @note This affects how the solver interprets the objective coefficients
 *       and the dual problem. The default is MINIMIZE.
 *
 * @example
 * PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
 * // Now maximizes c'x instead of minimizing
 */
PRIMALrescodee PRIMAL_putobjsense(PRIMALtask_t t, PRIMALobjsensee sense) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (sense != PRIMAL_OPTIMIZE_MINIMIZE && sense != PRIMAL_OPTIMIZE_MAXIMIZE) return PRIMAL_RES_ERR_ARG;
    t->sense = sense;
    return PRIMAL_RES_OK;
}

