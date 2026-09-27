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
/* primal_get.c - model data getters: rows/columns/slices/counters, 64-bit variants.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* ---------------- data getters ---------------- */

/**
 * Retrieves a single linear objective coefficient.
 *
 * @param t  [in]  Task handle.
 * @param j  [in]  Variable index, 0 <= j < numvar.
 * @param cj [out] Pointer to double receiving the coefficient.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or cj is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds.
 */
PRIMALrescodee PRIMAL_getcj(PRIMALtask_t t, int j, double *cj) {
    if (!t || !cj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    *cj = t->c[j]; return PRIMAL_RES_OK;
}

/**
 * Retrieves the entire linear objective vector.
 *
 * @param t [in]  Task handle.
 * @param c [out] Array of size numvar receiving the coefficients. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or c is NULL.
 *
 * @note If t->c is NULL (no variables yet), fills with zeros.
 *
 * @example
 * double *c = malloc(task->numvar * sizeof(double));
 * PRIMAL_getc(task, c);
 * // c[0..numvar-1] now holds the objective coefficients
 */
PRIMALrescodee PRIMAL_getc(PRIMALtask_t t, PRIMALrealt *c) {
    if (!t || !c) return PRIMAL_RES_ERR_NULL;
    for (int j = 0; j < t->numvar; j++) c[j] = t->c ? t->c[j] : 0.0;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves a contiguous slice of the linear objective vector.
 *
 * @param t     [in]  Task handle.
 * @param first [in]  Starting index (inclusive), 0 <= first <= numvar.
 * @param last  [in]  Ending index (exclusive), first <= last <= numvar.
 * @param c     [out] Array of size (last-first) receiving coefficients. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or c is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds.
 *
 * @note The slice is [first, last). Empty slice (first == last) is valid.
 */
PRIMALrescodee PRIMAL_getcslice(PRIMALtask_t t, int first, int last, PRIMALrealt *c) {
    if (!t || !c) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    for (int j = first; j < last; j++) c[j - first] = t->c ? t->c[j] : 0.0;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the operator coefficient A[i,j] (sum of all stored entries at (i,j)).
 *
 * @param t   [in]  Task handle.
 * @param i   [in]  Row index, 0 <= i < numcon.
 * @param j   [in]  Column index, 0 <= j < numvar.
 * @param aij [out] Pointer to double receiving the sum of entries at (i,j).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or aij is NULL,
 *         PRIMAL_RES_ERR_ARG if indices out of bounds.
 *
 * @note This returns the OPERATOR coefficient (what the solver actually uses),
 *       which is the SUM of all stored entries at position (i,j). Since
 *       putarow/putacol can write multiple entries for the same (i,j), the
 *       store may have duplicates. This getter sums them, matching what the
 *       matrix-vector products compute. This is the paired getter for
 *       PRIMAL_getarow/PRIMAL_getacol (which return the store entries) and
 *       PRIMAL_getnumanz (which counts store entries).
 *
 *       The reference's WRITE-time policy (coalesce vs accumulate) was not read;
 *       this solver's store accumulates, and getaij sums to match the operator.
 *
 * @example
 * double a;
 * PRIMAL_getaij(task, 2, 5, &a);
 * // a = sum of all entries at row 2, column 5
 */
PRIMALrescodee PRIMAL_getaij(PRIMALtask_t t, int i, int j, double *aij) {
    if (!t || !aij) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    double v = 0.0;
    const Col *c = &t->cols[j];
    for (int k = 0; k < c->nz; k++)
        if (c->sub[k] == i) v += c->val[k];
    *aij = v;
    return PRIMAL_RES_OK;
}

/* Reads one entry of the quadratic objective, summing the triplet store. */
PRIMALrescodee PRIMAL_getqobjij(PRIMALtask_t t, int i, int j, double *qij) {
    if (!t || !qij) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || j < 0 || i >= t->numvar || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    double v = 0.0;
    if (t->has_qobj)
        for (int e = 0; e < t->qt_n; e++)
            if ((t->qt_i[e] == i && t->qt_j[e] == j) || (t->qt_i[e] == j && t->qt_j[e] == i))
                v += t->qt_v[e];
    *qij = v;
    return PRIMAL_RES_OK;
}

/* Whole Q of the objective, read from the SAME table PRIMAL_getnumqobjnz counts:
 * the triplet store as the user wrote it, in write order (no sorting is
 * promised, exactly as for getacol -- this solver does not require sorted
 * input). Two things the signature does not say:
 *  - a triplet whose value is 0.0 IS returned and IS counted: the number is the
 *    count of the user's writes, not of the nonzeros of the operator;
 *  - a cross term (i,j) with i != j is ONE entry here while getqobjij answers
 *    the same number on both halves. The store is not symmetrized and the
 *    operator is: returning the mirror would state the user's decision twice
 *    and double what x'Qx already multiplies by 2. */
PRIMALrescodee PRIMAL_getqobj(PRIMALtask_t t, int *qi, int *qj, double *qval,
                              int maxnum, int *numret) {
    if (!t || !qi || !qj || !qval || !numret) return PRIMAL_RES_ERR_NULL;
    if (maxnum < 0 || t->qt_n > maxnum) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < t->qt_n; e++) {
        qi[e] = t->qt_i[e]; qj[e] = t->qt_j[e]; qval[e] = t->qt_v[e];
    }
    *numret = t->qt_n;
    return PRIMAL_RES_OK;
}

/* 64-bit variant of getqobj. */
PRIMALrescodee PRIMAL_getqobj64(PRIMALtask_t t, int *qi, int *qj, double *qval,
                                PRIMALint64t maxnum, PRIMALint64t *numret) {
    if (!t || !qi || !qj || !qval || !numret) return PRIMAL_RES_ERR_NULL;
    if (maxnum < t->qt_n) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < t->qt_n; e++) {
        qi[e] = t->qt_i[e]; qj[e] = t->qt_j[e]; qval[e] = t->qt_v[e];
    }
    *numret = t->qt_n;
    return PRIMAL_RES_OK;
}

/* Reads the bound key and values of one variable into the outputs. */
PRIMALrescodee PRIMAL_getvarbound(PRIMALtask_t t, int j, PRIMALboundkeye *bk, double *bl, double *bu) {
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    *bk = t->bkx[j]; *bl = t->blx[j]; *bu = t->bux[j];
    return PRIMAL_RES_OK;
}

/* Reads the bound key and values of one constraint into the outputs. */
PRIMALrescodee PRIMAL_getconbound(PRIMALtask_t t, int i, PRIMALboundkeye *bk, double *bl, double *bu) {
    if (!t || !bk || !bl || !bu) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    *bk = t->bkc[i]; *bl = t->blc[i]; *bu = t->buc[i];
    return PRIMAL_RES_OK;
}

/* Reads the stored optimization sense (minimize or maximize). */
PRIMALrescodee PRIMAL_getobjsense(PRIMALtask_t t, PRIMALobjsensee *sense) {
    if (!t || !sense) return PRIMAL_RES_ERR_NULL;
    *sense = t->sense; return PRIMAL_RES_OK;
}

/* Reads the fixed constant term of the objective function. */
PRIMALrescodee PRIMAL_getcfix(PRIMALtask_t t, double *cfix) {
    if (!t || !cfix) return PRIMAL_RES_ERR_NULL;
    *cfix = t->cfix; return PRIMAL_RES_OK;
}

/* Reads one integer parameter from its table slot into *value. */
PRIMALrescodee PRIMAL_getintparam(PRIMALtask_t t, int param, int *value) {
    if (!t || !value) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_INT, param);
    if (!d) return PRIMAL_RES_ERR_ARG;
    *value = *(int *)param_slot(d, t);
    return PRIMAL_RES_OK;
}

/* Reads one double parameter from its table slot into *value. */
PRIMALrescodee PRIMAL_getdouparam(PRIMALtask_t t, int param, double *value) {
    if (!t || !value) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_DOU, param);
    if (!d) return PRIMAL_RES_ERR_ARG;
    *value = (d->kind == P_DOUI) ? (double)*(int *)param_slot(d, t)
                                 : *(double *)param_slot(d, t);
    return PRIMAL_RES_OK;
}

/* `kind` is an input, not an output: int ids and double ids are separate
 * namespaces that share numbers (PRIMAL_IPAR_OPTIMIZER and
 * PRIMAL_DPAR_INTPNT_TOL_PFEAS are both 0). */
PRIMALrescodee PRIMAL_getparaminfo(PRIMALtask_t t, int kind, int param,
                                   double *dflt, double *lo, double *hi) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    /* the kind is the reference's parameter-type enum, where 0 is
     * MSK_PAR_INVALID_TYPE and 3 is a string: only the two kinds this solver
     * has answers for are a query, anything else is refused rather than read
     * as "int" by default. */
    if (kind != PRIMAL_PARAM_KIND_DOU && kind != PRIMAL_PARAM_KIND_INT)
        return PRIMAL_RES_ERR_ARG;
    const PrimalParam *d = param_find(kind == PRIMAL_PARAM_KIND_DOU ? P_DOU : P_INT,
                                      param);
    if (!d) return PRIMAL_RES_ERR_ARG;
    if (dflt) *dflt = d->dflt;
    if (lo) *lo = d->lo;
    if (hi) *hi = d->hi;
    return PRIMAL_RES_OK;
}

/* ---------------- model data accessors ----------------
 * A lives column-major (Col *cols), so a column is read directly and a row is
 * a scan. Both slices answer about the STORAGE, not about a canonicalized A:
 * an entry written twice is stored twice, is counted twice and is returned
 * twice. That is deliberate (see the duplicate note in primal.h): the number
 * the user gets is the number of their own writes, and getaij's first-match is
 * the same rule read from the other end.
 */

/* Entries of row i whose column index falls in [first,last). Ascending by
 * column, because the scan itself runs over the columns. */
static int row_slice_count(const PRIMALtask_t t, int i, int first, int last) {
    int n = 0;
    if (!t->cols) return 0;
    for (int j = first; j < last; j++) {
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) if (c->sub[k] == i) n++;
    }
    return n;
}

/* Entries of column j whose row index falls in [first,last), in stored order:
 * this solver does not require (or enforce) sorted input in putacol/putarow, so
 * the storage order is the only order that can be promised. */
static int col_slice_count(const PRIMALtask_t t, int j, int first, int last) {
    if (!t->cols) return 0;
    const Col *c = &t->cols[j];
    int n = 0;
    for (int k = 0; k < c->nz; k++) if (c->sub[k] >= first && c->sub[k] < last) n++;
    return n;
}

/* Slice argument check, shared by the two readers. */
static PRIMALrescodee slice_args_ok(int idx, int nidx, int first, int last,
                                    int slice_len, int offset, int maxnum) {
    if (idx < 0 || idx >= nidx) return PRIMAL_RES_ERR_ARG;
    if (first < 0 || last > slice_len || first > last) return PRIMAL_RES_ERR_ARG;
    if (offset < 0 || maxnum < offset) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/* Counts every stored entry of A, duplicates included, into *numanzs. */
PRIMALrescodee PRIMAL_getnumanz(PRIMALtask_t t, int *numanzs) {
    if (!t || !numanzs) return PRIMAL_RES_ERR_NULL;
    long total = 0;
    if (t->cols)
        for (int j = 0; j < t->numvar; j++) total += t->cols[j].nz;
    if (total > 2147483647L) return PRIMAL_RES_ERR_ARG;   /* not an int answer */
    *numanzs = (int)total;
    return PRIMAL_RES_OK;
}

/* Row/column nonzero counts and A in triplet form (reference getarownumnz,
 * getacolnumnz, getarowslicenumnz, getacolslicenumnz, getatrip). They count the
 * STORE -- entries, duplicates included -- exactly as getnumanz does, so the
 * three cannot disagree. getatrip has no count output in the reference; the
 * caller sizes with getnumanz, and an insufficient maxnumnz is refused without
 * writing. */
PRIMALrescodee PRIMAL_getacolnumnz(PRIMALtask_t t, int j, int *nzj) {
    if (!t || !nzj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    *nzj = t->cols ? t->cols[j].nz : 0;
    return PRIMAL_RES_OK;
}

/* Counts the stored entries of one row, duplicates included, into *nzi. */
PRIMALrescodee PRIMAL_getarownumnz(PRIMALtask_t t, int i, int *nzi) {
    if (!t || !nzi) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    if (t->cols)
        for (int j = 0; j < t->numvar; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++) if (c->sub[k] == i) n++;
        }
    *nzi = n;
    return PRIMAL_RES_OK;
}

/* Counts the stored entries of the column range [first,last) into *numnz. */
PRIMALrescodee PRIMAL_getacolslicenumnz(PRIMALtask_t t, int first, int last, int *numnz) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    if (t->cols) for (int j = first; j < last; j++) n += t->cols[j].nz;
    *numnz = n;
    return PRIMAL_RES_OK;
}
/* 64-bit variant of getacolslicenumnz; widens the int count. */
PRIMALrescodee PRIMAL_getacolslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz) {
    if (!numnz) return PRIMAL_RES_ERR_NULL;
    int n = 0;
    PRIMALrescodee rc = PRIMAL_getacolslicenumnz(t, first, last, &n);
    if (rc == PRIMAL_RES_OK) *numnz = n;
    return rc;
}
/* 64-bit variant of getarowslicenumnz; widens the int count. */
PRIMALrescodee PRIMAL_getarowslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz) {
    if (!numnz) return PRIMAL_RES_ERR_NULL;
    int n = 0;
    PRIMALrescodee rc = PRIMAL_getarowslicenumnz(t, first, last, &n);
    if (rc == PRIMAL_RES_OK) *numnz = n;
    return rc;
}
/* 64-bit versions of getarowslice/getacolslice in CSR form (int64 ptrb/ptre):
 * same read, count first and refuse without writing. */
PRIMALrescodee PRIMAL_getarowslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    if (last > first && (!ptrb || !ptre || !sub || !val)) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t need = 0;
    for (int i = first; i < last; i++) { int n = 0; PRIMAL_getarowslicenumnz(t, i, i + 1, &n); need += n; }
    if (maxnumnz < need) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t w = 0;
    for (int i = first; i < last; i++) {
        int n = 0;
        ptrb[i - first] = w;
        PRIMAL_getarowslice(t, i, 0, t->numvar, 0, t->numvar, &n, sub + w, val + w);
        w += n;
        ptre[i - first] = w;
    }
    return PRIMAL_RES_OK;
}
/* 64-bit column-slice read in CSC form; same entries as getacolslice. */
PRIMALrescodee PRIMAL_getacolslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    if (last > first && (!ptrb || !ptre || !sub || !val)) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t need = 0;
    for (int j = first; j < last; j++) { int n = 0; PRIMAL_getacolslicenumnz(t, j, j + 1, &n); need += n; }
    if (maxnumnz < need) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t w = 0;
    for (int j = first; j < last; j++) {
        int n = 0;
        ptrb[j - first] = w;
        PRIMAL_getacolslice(t, j, 0, t->numcon, 0, t->numcon, &n, sub + w, val + w);
        w += n;
        ptre[j - first] = w;
    }
    return PRIMAL_RES_OK;
}

/* Counts the stored entries of the row range [first,last) into *numnz. */
PRIMALrescodee PRIMAL_getarowslicenumnz(PRIMALtask_t t, int first, int last, int *numnz) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    if (t->cols)
        for (int j = 0; j < t->numvar; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] >= first && c->sub[k] < last) n++;
        }
    *numnz = n;
    return PRIMAL_RES_OK;
}

/* Reads the whole store of A in triplet form, sized via getnumanz. */
PRIMALrescodee PRIMAL_getatrip(PRIMALtask_t t, PRIMALint64t maxnumnz,
                               int *subi, int *subj, PRIMALrealt *val) {
    if (!t || !subi || !subj || !val) return PRIMAL_RES_ERR_NULL;
    if (maxnumnz < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t total = 0;
    if (t->cols) for (int j = 0; j < t->numvar; j++) total += t->cols[j].nz;
    if (total > maxnumnz) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t n = 0;
    for (int j = 0; j < t->numvar; j++) {
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            subi[n] = c->sub[k]; subj[n] = j; val[n] = c->val[k]; n++;
        }
    }
    return PRIMAL_RES_OK;
}

/* Reads the reserved entry capacity of A, summed over the columns. */
PRIMALrescodee PRIMAL_getmaxnumanz(PRIMALtask_t t, int *maxnumanzs) {
    if (!t || !maxnumanzs) return PRIMAL_RES_ERR_NULL;
    long total = 0;
    if (t->cols)
        for (int j = 0; j < t->numvar; j++) total += t->cols[j].cap;
    if (total > 2147483647L) return PRIMAL_RES_ERR_ARG;
    *maxnumanzs = (int)total;
    return PRIMAL_RES_OK;
}

/* Reads the number of triplets in the quadratic-objective store. */
PRIMALrescodee PRIMAL_getnumqobjnz(PRIMALtask_t t, int *numqobjnz) {
    if (!t || !numqobjnz) return PRIMAL_RES_ERR_NULL;
    *numqobjnz = t->qt_n;
    return PRIMAL_RES_OK;
}

/* ---- 64-bit counter variants (reference getnumanz64/getmaxnumanz64/
 * getnumqobjnz64/getnumqconknz64). This solver's sizes fit an int, so these are
 * the same numbers widened to 64 bits; the point is that a caller written against
 * a 64-bit reference build links and reads them. */
PRIMALrescodee PRIMAL_getnumanz64(PRIMALtask_t t, PRIMALint64t *numanzs) {
    if (!t || !numanzs) return PRIMAL_RES_ERR_NULL;
    int v = 0; PRIMALrescodee rc = PRIMAL_getnumanz(t, &v);
    if (rc != PRIMAL_RES_OK) return rc;
    *numanzs = v; return PRIMAL_RES_OK;
}
/* 64-bit variant of getmaxnumanz; widens the int capacity. */
PRIMALrescodee PRIMAL_getmaxnumanz64(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    int v = 0; PRIMALrescodee rc = PRIMAL_getmaxnumanz(t, &v);
    if (rc != PRIMAL_RES_OK) return rc;
    *n = v; return PRIMAL_RES_OK;
}
/* 64-bit variant of getnumqobjnz; widens the triplet count. */
PRIMALrescodee PRIMAL_getnumqobjnz64(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    *n = t->qt_n; return PRIMAL_RES_OK;
}
/* 64-bit variant of getnumqconknz; widens the int count. */
PRIMALrescodee PRIMAL_getnumqconknz64(PRIMALtask_t t, int k, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    int v = 0; PRIMALrescodee rc = PRIMAL_getnumqconknz(t, k, &v);
    if (rc != PRIMAL_RES_OK) return rc;
    *n = v; return PRIMAL_RES_OK;
}

/* Reads the entries of row i in [first,last), ascending by column. */
PRIMALrescodee PRIMAL_getarowslice(PRIMALtask_t t, int i, int first, int last,
                                   int offset, int maxnum, int *numret,
                                   int *sub, double *val) {
    if (!t || !numret || !sub || !val) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = slice_args_ok(i, t->numcon, first, last, t->numvar, offset, maxnum);
    if (rc != PRIMAL_RES_OK) return rc;
    int want = row_slice_count(t, i, first, last);
    /* The refusal leaves the buffers and *numret untouched: writing what fits
     * and then failing would hand back a prefix nobody asked for. */
    if (want > maxnum - offset) return PRIMAL_RES_ERR_ARG;
    int w = offset;
    if (t->cols)
        for (int j = first; j < last; j++) {
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++)
                if (c->sub[k] == i) { sub[w] = j; val[w] = c->val[k]; w++; }
        }
    *numret = w - offset;
    return PRIMAL_RES_OK;
}

/* Reads the entries of column j in [first,last), in stored order. */
PRIMALrescodee PRIMAL_getacolslice(PRIMALtask_t t, int j, int first, int last,
                                   int offset, int maxnum, int *numret,
                                   int *sub, double *val) {
    if (!t || !numret || !sub || !val) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = slice_args_ok(j, t->numvar, first, last, t->numcon, offset, maxnum);
    if (rc != PRIMAL_RES_OK) return rc;
    int want = col_slice_count(t, j, first, last);
    if (want > maxnum - offset) return PRIMAL_RES_ERR_ARG;
    const Col *c = &t->cols[j];
    int w = offset;
    for (int k = 0; k < c->nz; k++)
        if (c->sub[k] >= first && c->sub[k] < last) {
            sub[w] = c->sub[k]; val[w] = c->val[k]; w++;
        }
    *numret = w - offset;
    return PRIMAL_RES_OK;
}

/* Fixed-buffer form of the same read: the whole row / column from 0. */
PRIMALrescodee PRIMAL_getarow(PRIMALtask_t t, int i, int *sub, double *val,
                              int maxnum, int *numret) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_getarowslice(t, i, 0, t->numvar, 0, maxnum, numret, sub, val);
}

/* Reads one whole column of A via the column-slice reader. */
PRIMALrescodee PRIMAL_getacol(PRIMALtask_t t, int j, int *sub, double *val,
                              int maxnum, int *numret) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_getacolslice(t, j, 0, t->numcon, 0, maxnum, numret, sub, val);
}

/* A over a slice of rows/columns in triplets (reference getarowslicetrip/
 * getacolslicetrip). Counts first, and refuses without writing when
 * `maxnumnz` is short: a prefix of A is another matrix. */
PRIMALrescodee PRIMAL_getarowslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numcon || first > last) return PRIMAL_RES_ERR_ARG;
    if (!subi || !subj || !val) return PRIMAL_RES_ERR_NULL;
    int nv = t->numvar > 0 ? t->numvar : 1;
    int *sub = (int *)malloc((size_t)nv * sizeof(int));
    double *v = (double *)malloc((size_t)nv * sizeof(double));
    if (!sub || !v) { free(sub); free(v); return PRIMAL_RES_ERR_ALLOC; }
    PRIMALint64t need = 0;
    for (int i = first; i < last; i++) {
        int nr = 0;
        PRIMAL_getarowslice(t, i, 0, t->numvar, 0, t->numvar, &nr, sub, v);
        need += nr;
    }
    if (maxnumnz < need) { free(sub); free(v); return PRIMAL_RES_ERR_ARG; }
    PRIMALint64t w = 0;
    for (int i = first; i < last; i++) {
        int nr = 0;
        PRIMAL_getarowslice(t, i, 0, t->numvar, 0, t->numvar, &nr, sub, v);
        for (int k = 0; k < nr; k++) { subi[w] = i; subj[w] = sub[k]; val[w] = v[k]; w++; }
    }
    free(sub); free(v);
    return PRIMAL_RES_OK;
}
/* Reads the column range [first,last) of A in triplet form. */
PRIMALrescodee PRIMAL_getacolslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last > t->numvar || first > last) return PRIMAL_RES_ERR_ARG;
    if (!subi || !subj || !val) return PRIMAL_RES_ERR_NULL;
    int nc = t->numcon > 0 ? t->numcon : 1;
    int *sub = (int *)malloc((size_t)nc * sizeof(int));
    double *v = (double *)malloc((size_t)nc * sizeof(double));
    if (!sub || !v) { free(sub); free(v); return PRIMAL_RES_ERR_ALLOC; }
    PRIMALint64t need = 0;
    for (int j = first; j < last; j++) {
        int nr = 0;
        PRIMAL_getacolslice(t, j, 0, t->numcon, 0, t->numcon, &nr, sub, v);
        need += nr;
    }
    if (maxnumnz < need) { free(sub); free(v); return PRIMAL_RES_ERR_ARG; }
    PRIMALint64t w = 0;
    for (int j = first; j < last; j++) {
        int nr = 0;
        PRIMAL_getacolslice(t, j, 0, t->numcon, 0, t->numcon, &nr, sub, v);
        for (int k = 0; k < nr; k++) { subi[w] = sub[k]; subj[w] = j; val[w] = v[k]; w++; }
    }
    free(sub); free(v);
    return PRIMAL_RES_OK;
}

/* ---------------- names ----------------
 * name_put and name_find are the only two places where the naming rules live:
 * one table per kind, univocal inside its own table, an unnamed slot reading
 * back as the empty string, and a refusal that never mutates.
 */
