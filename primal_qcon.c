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
/* primal_qcon.c - quadratic constraint terms (putqconk family).
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* ---------------- quadratic constraint terms ---------------- */

/**
 * Internal: ensures the qcon row-pointer array exists with sufficient capacity.
 *
 * @param t [in] Task handle.
 * @return 1 on success, 0 on allocation failure.
 *
 * @note Grows the qcon array to at least numcon entries (or 1 if numcon=0).
 *       New entries are zeroed (NULL), meaning no quadratic terms yet.
 */
static int qcon_alloc(PRIMALtask_t t) {
    int need = t->numcon > 0 ? t->numcon : 1;
    double **q = (double **)lazy_grow(t->qcon, &t->qcon_cap, need, sizeof(double *));
    if (!q) return 0;
    t->qcon = q;
    return 1;
}

/**
 * Adds quadratic terms to a constraint (sets the Q matrix for constraint k).
 *
 * @param t       [in] Task handle.
 * @param k       [in] Constraint index, 0 <= k < numcon.
 * @param numqcnz [in] Number of quadratic terms to add. Must be non-negative.
 * @param qsubi   [in] Array of row indices (length numqcnz). Each must satisfy
 *                    0 <= qsubi[e] < numvar.
 * @param qsubj   [in] Array of column indices (length numqcnz). Each must satisfy
 *                    0 <= qsubj[e] < numvar.
 * @param qval    [in] Array of coefficient values (length numqcnz).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds, numqcnz < 0, indices invalid, or NaN.
 *
 * @note The quadratic terms are ACCUMULATED into the dense Q matrix for
 *       constraint k. The matrix is symmetric: qval[e] is added to both
 *       Q[qsubi, qsubj] and Q[qsubj, qsubi] (unless i==j). The dense matrix
 *       is allocated lazily on first use (numvar x numvar).
 *
 *       If numqcnz == 0, the quadratic part of constraint k is CLEARED
 *       (has_qcon is decremented, the dense block is freed).
 *
 *       The dense symmetric Q matrix for constraint k is rebuilt from the
 *       accumulated triplets when needed (e.g., by the conic IPM or QCQP encoder).
 *
 * @example
 * // Constraint 0: x0^2 + x1^2 + 2*x0*x1 <= 1  =>  Q = [[2,2],[2,2]]
 * int qsubi[] = {0, 1, 0};
 * int qsubj[] = {0, 1, 1};
 * double qval[] = {2.0, 2.0, 2.0};
 * PRIMAL_putqconk(task, 0, 3, qsubi, qsubj, qval);
 * // And set the bound: x'Qx <= 1
 * PRIMAL_putconbound(task, 0, PRIMAL_BK_UP, 0.0, 1.0);
 */
PRIMALrescodee PRIMAL_putqconk(PRIMALtask_t t, int k, int numqcnz,
                         const int *qsubi, const int *qsubj,
                         const double *qval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcon || numqcnz < 0) return PRIMAL_RES_ERR_ARG;
    if (numqcnz > 0 && (!qsubi || !qsubj || !qval)) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    /* reject NaN / out-of-range indices */
    for (int e = 0; e < numqcnz; e++) {
        if (qsubi[e] < 0 || qsubi[e] >= t->numvar ||
            qsubj[e] < 0 || qsubj[e] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (qval[e] != qval[e]) return PRIMAL_RES_ERR_ARG;
    }
    if (!qcon_alloc(t)) return PRIMAL_RES_ERR_ALLOC;
    if (!t->qcon[k]) {
        t->qcon[k] = (double *)calloc((size_t)t->numvar * (size_t)t->numvar, sizeof(double));
        if (!t->qcon[k]) return PRIMAL_RES_ERR_ALLOC;
    }
    int had = 0;
    for (int e = 0; e < t->numvar * t->numvar; e++) if (t->qcon[k][e] != 0.0) { had = 1; break; }
    /* replace semantics: clear row k first (PRIMAL putqconk overwrites) */
    if (had) memset(t->qcon[k], 0, (size_t)t->numvar * (size_t)t->numvar * sizeof(double));
    for (int e = 0; e < numqcnz; e++) {
        int i = qsubi[e], j = qsubj[e];
        t->qcon[k][i * t->numvar + j] += qval[e];
        if (i != j) t->qcon[k][j * t->numvar + i] += qval[e];
    }
    int now = 0;
    for (int e = 0; e < t->numvar * t->numvar; e++) if (t->qcon[k][e] != 0.0) { now = 1; break; }
    t->has_qcon += now - had;
    if (t->has_qcon < 0) t->has_qcon = 0;
    return PRIMAL_RES_OK;
}

/**
 * Replaces all quadratic constraint terms from a single triplet list.
 *
 * @param t       [in] Task handle.
 * @param numqcnz [in] Number of quadratic terms. Must be non-negative.
 * @param qcsubk  [in] Array of constraint indices (length numqcnz). Each must satisfy
 *                    0 <= qcsubk[e] < numcon.
 * @param qcsubi  [in] Array of row indices (length numqcnz). Must satisfy
 *                    0 <= qcsubi[e] < numvar AND qcsubi[e] >= qcsubj[e]
 *                    (LOWER triangle only, i >= j).
 * @param qcsubj  [in] Array of column indices (length numqcnz). Must satisfy
 *                    0 <= qcsubj[e] < numvar.
 * @param qcval   [in] Array of coefficient values (length numqcnz).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if numqcnz < 0, any index out of bounds,
 *                    i < j (not lower triangle), or NaN in qcval.
 *
 * @note This REPLACES all quadratic terms for ALL constraints. The input uses
 *       LOWER TRIANGLE ONLY (i >= j). Duplicates within the call accumulate.
 *       The symmetric Q matrices are built by adding qcval to both (i,j) and (j,i).
 *
 *       If numqcnz == 0, all quadratic constraint terms are cleared.
 *       The entire list is validated before any modification.
 *
 * @example
 * // Constraint 0: x0^2 + x1^2 <= 1 (Q = [[2,0],[0,2]])
 * // Constraint 1: 2*x0*x1 <= 1 (Q = [[0,2],[2,0]])
 * int qcsubk[] = {0, 0, 1};
 * int qcsubi[] = {0, 1, 1};
 * int qcsubj[] = {0, 0, 0};
 * double qcval[] = {2.0, 2.0, 2.0};
 * PRIMAL_putqcon(task, 3, qcsubk, qcsubi, qcsubj, qcval);
 */
PRIMALrescodee PRIMAL_putqcon(PRIMALtask_t t, int numqcnz,
                              const int *qcsubk, const int *qcsubi, const int *qcsubj,
                              const PRIMALrealt *qcval) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numqcnz < 0 || (numqcnz > 0 && (!qcsubk || !qcsubi || !qcsubj || !qcval)))
        return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < numqcnz; e++) {
        if (qcsubk[e] < 0 || qcsubk[e] >= t->numcon) return PRIMAL_RES_ERR_ARG;
        if (qcsubi[e] < 0 || qcsubi[e] >= t->numvar ||
            qcsubj[e] < 0 || qcsubj[e] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (qcsubi[e] < qcsubj[e]) return PRIMAL_RES_ERR_ARG;   /* lower triangle only */
        if (qcval[e] != qcval[e]) return PRIMAL_RES_ERR_ARG;    /* NaN */
    }
    if (numqcnz == 0) {
        if (t->qcon) for (int k = 0; k < t->qcon_cap; k++) { free(t->qcon[k]); t->qcon[k] = NULL; }
        t->has_qcon = 0;
        return PRIMAL_RES_OK;
    }
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    if (!qcon_alloc(t)) return PRIMAL_RES_ERR_ALLOC;
    for (int k = 0; k < t->numcon; k++)
        if (t->qcon[k]) memset(t->qcon[k], 0, (size_t)t->numvar * (size_t)t->numvar * sizeof(double));
    t->has_qcon = 0;
    for (int e = 0; e < numqcnz; e++) {
        int k = qcsubk[e], i = qcsubi[e], j = qcsubj[e];
        if (!t->qcon[k]) {
            t->qcon[k] = (double *)calloc((size_t)t->numvar * (size_t)t->numvar, sizeof(double));
            if (!t->qcon[k]) return PRIMAL_RES_ERR_ALLOC;
        }
        t->qcon[k][i * t->numvar + j] += qcval[e];
        if (i != j) t->qcon[k][j * t->numvar + i] += qcval[e];
    }
    for (int k = 0; k < t->numcon; k++) {
        if (!t->qcon[k]) continue;
        for (int e = 0; e < t->numvar * t->numvar; e++)
            if (t->qcon[k][e] != 0.0) { t->has_qcon++; break; }
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the number of non-zero quadratic terms in a constraint's Q matrix (upper triangle).
 *
 * @param t       [in]  Task handle.
 * @param k       [in]  Constraint index, 0 <= k < numcon.
 * @param numqcnz [out] Pointer to int receiving the count of non-zeros in upper triangle.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or numqcnz is NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds.
 *
 * @note Counts entries in the upper triangle (i <= j) of the dense Q matrix
 *       for constraint k. Each off-diagonal non-zero counts as 1 (the symmetric
 *       pair is one term). The count matches what getqconk would return.
 *
 * @example
 * int nnz;
 * PRIMAL_getnumqconknz(task, 0, &nnz);
 * printf("Constraint 0 has %d quadratic terms\n", nnz);
 */
PRIMALrescodee PRIMAL_getnumqconknz(PRIMALtask_t t, int k, int *numqcnz) {
    if (!t || !numqcnz) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcon) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    if (t->qcon && t->qcon[k]) {
        for (int i = 0; i < t->numvar; i++)
            for (int j = i; j < t->numvar; j++)
                if (t->qcon[k][i * t->numvar + j] != 0.0) n++;
    }
    *numqcnz = n;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves a single entry of a constraint's Q matrix.
 *
 * @param t  [in]  Task handle.
 * @param k  [in]  Constraint index, 0 <= k < numcon.
 * @param i  [in]  Row index, 0 <= i < numvar.
 * @param j  [in]  Column index, 0 <= j < numvar.
 * @param qij [out] Pointer to double receiving Q[i,j].
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or qij is NULL,
 *         PRIMAL_RES_ERR_ARG if any index out of bounds.
 *
 * @note Returns the value from the dense symmetric Q matrix (which is built
 *       by adding to both halves at put time). The matrix is symmetrized,
 *       so Q[i,j] == Q[j,i] always holds.
 */
PRIMALrescodee PRIMAL_getqconkij(PRIMALtask_t t, int k, int i, int j, double *qij) {
    if (!t || !qij) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcon || i < 0 || i >= t->numvar || j < 0 || j >= t->numvar)
        return PRIMAL_RES_ERR_ARG;
    *qij = (t->qcon && t->qcon[k]) ? t->qcon[k][i * t->numvar + j] : 0.0;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the upper triangle of a constraint's Q matrix as triplets.
 *
 * @param t      [in]  Task handle.
 * @param k      [in]  Constraint index, 0 <= k < numcon.
 * @param qi     [out] Array for row indices (length maxnum). Must not be NULL.
 * @param qj     [out] Array for column indices (length maxnum). Must not be NULL.
 * @param qval   [out] Array for coefficient values (length maxnum). Must not be NULL.
 * @param maxnum [in]  Maximum number of entries the buffers can hold.
 * @param numret [out] Pointer to int receiving actual number of entries written.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any output array is NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds or maxnum < 0 or maxnum < required capacity.
 *
 * @note The function FIRST counts the required capacity via getnumqconknz.
 *       If maxnum < required capacity, returns ERR_ARG without writing
 *       anything to the buffers or *numret (a rejection writes nothing).
 *       The output is the upper triangle (i <= j) in ascending order.
 *
 * @example
 * int nnz;
 * PRIMAL_getnumqconknz(task, 0, &nnz);
 * int *qi = malloc(nnz * sizeof(int));
 * int *qj = malloc(nnz * sizeof(int));
 * double *qval = malloc(nnz * sizeof(double));
 * int numret;
 * PRIMAL_getqconk(task, 0, qi, qj, qval, nnz, &numret);
 * // qi, qj, qval now hold the upper triangle entries
 */
PRIMALrescodee PRIMAL_getqconk(PRIMALtask_t t, int k, int *qi, int *qj, double *qval,
                               int maxnum, int *numret) {
    if (!t || !qi || !qj || !qval || !numret) return PRIMAL_RES_ERR_NULL;
    if (maxnum < 0) return PRIMAL_RES_ERR_ARG;
    int want = 0;
    PRIMALrescodee rc = PRIMAL_getnumqconknz(t, k, &want);   /* validates k as well */
    if (rc != PRIMAL_RES_OK) return rc;
    if (want > maxnum) return PRIMAL_RES_ERR_ARG;   /* buffers and *numret untouched */
    int w = 0;
    if (t->qcon && t->qcon[k])
        for (int i = 0; i < t->numvar; i++)
            for (int j = i; j < t->numvar; j++)
                if (t->qcon[k][i * t->numvar + j] != 0.0) {
                    qi[w] = i; qj[w] = j;
                    qval[w] = t->qcon[k][i * t->numvar + j];
                    w++;
                }
    *numret = w;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the upper triangle of a constraint's Q matrix (64-bit variant).
 *
 * @param t      [in]  Task handle.
 * @param k      [in]  Constraint index, 0 <= k < numcon.
 * @param qi     [out] Array for row indices (length maxnum). Must not be NULL.
 * @param qj     [out] Array for column indices (length maxnum). Must not be NULL.
 * @param qval   [out] Array for coefficient values (length maxnum). Must not be NULL.
 * @param maxnum [in]  Maximum number of entries (64-bit).
 * @param numret [out] Pointer to 64-bit int receiving actual number written.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or any output array is NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds, maxnum < 0, or maxnum < required capacity.
 *
 * @note 64-bit variant of getqconk. The counts and capacity use PRIMALint64t.
 *       The row/column indices (qi, qj) remain 32-bit int.
 */
PRIMALrescodee PRIMAL_getqconk64(PRIMALtask_t t, int k, int *qi, int *qj, double *qval,
                                  PRIMALint64t maxnum, PRIMALint64t *numret) {
    if (!t || !qi || !qj || !qval || !numret) return PRIMAL_RES_ERR_NULL;
    int want = 0;
    PRIMALrescodee rc = PRIMAL_getnumqconknz(t, k, &want);
    if (rc != PRIMAL_RES_OK) return rc;
    if (maxnum < want) return PRIMAL_RES_ERR_ARG;
    int w = 0;
    if (t->qcon && t->qcon[k])
        for (int i = 0; i < t->numvar; i++)
            for (int j = i; j < t->numvar; j++)
                if (t->qcon[k][i * t->numvar + j] != 0.0) {
                    qi[w] = i; qj[w] = j; qval[w] = t->qcon[k][i * t->numvar + j]; w++;
                }
    *numret = w;
    return PRIMAL_RES_OK;
}

