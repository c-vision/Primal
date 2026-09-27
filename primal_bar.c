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
/* primal_bar.c - bar variables, symmetric store, A-bar/C-bar triplets.
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* =====================================================================
 * SDP: bar variables, matrix store, inner-product terms
 * ===================================================================== */

/* Appends one symmetric matrix to the store and returns its index.
 * Validates dimensions and triplet bounds; grows the store on demand. */
PRIMALrescodee PRIMAL_appendsparsesymmat(PRIMALtask_t t, int dim, int nnz,
                                   const int *subi, const int *subj,
                                   const PRIMALrealt *val, int *idx) {
    if (!t || !idx) return PRIMAL_RES_ERR_NULL;
    if (dim <= 0 || nnz < 0 || (nnz > 0 && (!subi || !subj || !val))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < nnz; k++)
        if (subi[k] < 0 || subi[k] >= dim || subj[k] < 0 || subj[k] >= dim)
            return PRIMAL_RES_ERR_ARG;
    if (t->nsym >= t->symcap) {
        int nc = t->symcap ? t->symcap * 2 : 4;
        int **a1 = (int **)realloc(t->sym_subi, (size_t)nc * sizeof(int *));
        int **a2 = (int **)realloc(t->sym_subj, (size_t)nc * sizeof(int *));
        double **a3 = (double **)realloc(t->sym_val, (size_t)nc * sizeof(double *));
        int *a4 = (int *)realloc(t->sym_dim, (size_t)nc * sizeof(int));
        int *a5 = (int *)realloc(t->sym_nnz, (size_t)nc * sizeof(int));
        int *a6 = (int *)realloc(t->sym_cap, (size_t)nc * sizeof(int));
        if (!a1 || !a2 || !a3 || !a4 || !a5 || !a6) {
            free(a1); free(a2); free(a3); free(a4); free(a5); free(a6); return PRIMAL_RES_ERR_ALLOC;
        }
        t->sym_subi = a1; t->sym_subj = a2; t->sym_val = a3;
        t->sym_dim = a4; t->sym_nnz = a5; t->sym_cap = a6; t->symcap = nc;
    }
    int k = t->nsym;
    t->sym_subi[k] = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    t->sym_subj[k] = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    t->sym_val[k] = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (!t->sym_subi[k] || !t->sym_subj[k] || !t->sym_val[k]) return PRIMAL_RES_ERR_ALLOC;
    for (int m = 0; m < nnz; m++) {
        t->sym_subi[k][m] = subi[m];
        t->sym_subj[k][m] = subj[m];
        t->sym_val[k][m] = val[m];
    }
    t->sym_dim[k] = dim; t->sym_nnz[k] = nnz; t->sym_cap[k] = nnz;
    t->nsym++;
    *idx = k;
    return PRIMAL_RES_OK;
}

/* Reference getsparsesymmat: read one symmetric matrix from the store as
 * (subi, subj, valij) in lower-triangle form, `maxlen` being the buffer size. */
PRIMALrescodee PRIMAL_getsparsesymmat(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxlen,
                                      int *subi, int *subj, PRIMALrealt *valij) {
    if (!t || !subi || !subj || !valij) return PRIMAL_RES_ERR_NULL;
    if (idx < 0 || idx >= t->nsym) return PRIMAL_RES_ERR_ARG;
    if (maxlen < 0 || t->sym_nnz[idx] > maxlen) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < t->sym_nnz[idx]; e++) {
        subi[e] = t->sym_subi[idx][e];
        subj[e] = t->sym_subj[idx][e];
        valij[e] = t->sym_val[idx][e];
    }
    return PRIMAL_RES_OK;
}

/* Reference appendsparsesymmatlist: append several symmetric matrices from a flat
 * triplet list (dims[k] + nz[k] give the shape, subi/subj/valij are concatenated,
 * idx[k] returns each stored id). The whole list is validated before appending. */
PRIMALrescodee PRIMAL_appendsparsesymmatlist(PRIMALtask_t t, int num, const int *dims,
    const PRIMALint64t *nz, const int *subi, const int *subj, const PRIMALrealt *valij,
    PRIMALint64t *idx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!dims || !nz || !subi || !subj || !valij || !idx)))
        return PRIMAL_RES_ERR_ARG;
    PRIMALint64t off = 0;
    for (int k = 0; k < num; k++) {
        if (dims[k] <= 0 || nz[k] < 0) return PRIMAL_RES_ERR_ARG;
        for (PRIMALint64t e = 0; e < nz[k]; e++) {
            int i = subi[off + e], j = subj[off + e];
            if (i < 0 || i >= dims[k] || j < 0 || j >= dims[k]) return PRIMAL_RES_ERR_ARG;
        }
        off += nz[k];
    }
    off = 0;
    for (int k = 0; k < num; k++) {
        int id = -1;
        PRIMALrescodee rc = PRIMAL_appendsparsesymmat(t, dims[k], (int)nz[k],
            subi + off, subj + off, valij + off, &id);
        if (rc != PRIMAL_RES_OK) return rc;
        idx[k] = id;
        off += nz[k];
    }
    return PRIMAL_RES_OK;
}

/* Appends bar variables of the given dimensions with zeroed solution blocks.
 * Grows the shared bar storage and invalidates any published solution. */
PRIMALrescodee PRIMAL_appendbarvars(PRIMALtask_t t, int num, const int *dim) {
    if (num != 0) model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !dim)) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) if (dim[k] <= 0) return PRIMAL_RES_ERR_ARG;
    int oldcap = t->barcap;
    if (t->numbarvar + num > t->barcap) {
        int nc = t->barcap ? t->barcap : 4;
        while (nc < t->numbarvar + num) nc *= 2;
        int *a1 = (int *)realloc(t->barDim, (size_t)nc * sizeof(int));
        double **a2 = (double **)realloc(t->barx, (size_t)nc * sizeof(double *));
        double **a3 = (double **)realloc(t->barsj, (size_t)nc * sizeof(double *));
        if (!a1 || !a2 || !a3) { free(a1); free(a2); free(a3); return PRIMAL_RES_ERR_ALLOC; }
        t->barDim = a1; t->barx = a2; t->barsj = a3; t->barcap = nc;
    }
    /* The name table shares barcap with the other three: a single capacity,
     * not two to keep in sync. The pointer table moves, not the strings: a name
     * borrowed before this point stays at the same address after. */
    if (t->barcap && (t->barcap != oldcap || !t->barname)) {
        int had = t->barname != NULL;
        char **a4 = (char **)realloc(t->barname, (size_t)t->barcap * sizeof(char *));
        if (!a4) { t->barcap = oldcap; return PRIMAL_RES_ERR_ALLOC; }   /* numbarvar not yet moved */
        t->barname = a4;
        for (int k = had ? oldcap : 0; k < t->barcap; k++) t->barname[k] = NULL;
    }
    for (int k = 0; k < num; k++) {
        t->barDim[t->numbarvar + k] = dim[k];
        t->barx[t->numbarvar + k] = (double *)calloc((size_t)dim[k] * (size_t)dim[k], sizeof(double));
        t->barsj[t->numbarvar + k] = (double *)calloc((size_t)dim[k] * (size_t)dim[k], sizeof(double));
        if (!t->barx[t->numbarvar + k] || !t->barsj[t->numbarvar + k]) return PRIMAL_RES_ERR_ALLOC;
    }
    t->numbarvar += num;
    /* Published bars are one per block: an added block grows as zero,
     * and a zero nobody has solved is not an answer of the model. */
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/* Reference removebarvars: remove the symmetric-matrix variables at the given
 * indices. The kept ones are compacted, the A-bar/C-bar terms on removed
 * variables are dropped and the remaining bar indices are remapped. */
PRIMALrescodee PRIMAL_removebarvars(PRIMALtask_t t, int num, const int *subset) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subset)) return PRIMAL_RES_ERR_ARG;
    for (int a = 0; a < num; a++) {
        if (subset[a] < 0 || subset[a] >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
        for (int b = a + 1; b < num; b++) if (subset[a] == subset[b]) return PRIMAL_RES_ERR_ARG;
    }
    int oldn = t->numbarvar;
    char *del = (char *)calloc((size_t)(oldn > 0 ? oldn : 1), 1);
    int *remap = (int *)malloc((size_t)(oldn > 0 ? oldn : 1) * sizeof(int));
    if (!del || !remap) { free(del); free(remap); return PRIMAL_RES_ERR_ALLOC; }
    for (int a = 0; a < num; a++) del[subset[a]] = 1;
    int w = 0;
    for (int j = 0; j < oldn; j++) {
        if (del[j]) {
            remap[j] = -1;
            free(t->barx[j]); free(t->barsj[j]); free(t->barname[j]);
            continue;
        }
        remap[j] = w;
        t->barDim[w] = t->barDim[j];
        t->barx[w] = t->barx[j];
        t->barsj[w] = t->barsj[j];
        t->barname[w] = t->barname[j];
        w++;
    }
    for (int j = w; j < oldn; j++) { t->barx[j] = NULL; t->barsj[j] = NULL; t->barname[j] = NULL; }
    t->numbarvar = w;
    int wa = 0;
    for (int k = 0; k < t->nbarA; k++) {
        int b = t->barA_bar[k];
        if (b < 0 || b >= oldn || remap[b] < 0) continue;
        t->barA_con[wa] = t->barA_con[k];
        t->barA_bar[wa] = remap[b];
        t->barA_sym[wa] = t->barA_sym[k];
        t->barA_coef[wa] = t->barA_coef[k];
        wa++;
    }
    t->nbarA = wa;
    int wc = 0;
    for (int k = 0; k < t->nbarC; k++) {
        int b = t->barC_bar[k];
        if (b < 0 || b >= oldn || remap[b] < 0) continue;
        t->barC_bar[wc] = remap[b];
        t->barC_sym[wc] = t->barC_sym[k];
        t->barC_coef[wc] = t->barC_coef[k];
        wc++;
    }
    t->nbarC = wc;
    for (int a = 0; a < t->numafe; a++) {
        int count = 0;
        for (int e = 0; e < t->afe_barnz[a]; e++) {
            int j = t->afe_baridx[a][e];
            if (j >= 0 && j < oldn && remap[j] >= 0) {
                t->afe_baridx[a][count] = remap[j];
                t->afe_barsym[a][count] = t->afe_barsym[a][e];
                t->afe_barcoef[a][count++] = t->afe_barcoef[a][e];
            }
        }
        t->afe_barnz[a] = count;
    }
    free(del); free(remap);
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/* Appends inner-product terms linking constraint i to bar variable j.
 * Validates stored-matrix ids and dimensions; appends to the A-bar list. */
PRIMALrescodee PRIMAL_putbaraij(PRIMALtask_t t, int i, int j, int num,
                          const int *sub, const PRIMALrealt *val) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numbarvar ||
        num < 0 || (num > 0 && (!sub || !val))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->nsym) return PRIMAL_RES_ERR_ARG;
        if (t->sym_dim[sub[k]] != t->barDim[j]) return PRIMAL_RES_ERR_ARG;
    }
    if (t->nbarA + num > t->capbarA) {
        int nc = t->capbarA ? t->capbarA : 4;
        while (nc < t->nbarA + num) nc *= 2;
        int *a1 = (int *)realloc(t->barA_con, (size_t)nc * sizeof(int));
        int *a2 = (int *)realloc(t->barA_bar, (size_t)nc * sizeof(int));
        int *a3 = (int *)realloc(t->barA_sym, (size_t)nc * sizeof(int));
        double *a4 = (double *)realloc(t->barA_coef, (size_t)nc * sizeof(double));
        if (!a1 || !a2 || !a3 || !a4) { free(a1); free(a2); free(a3); free(a4); return PRIMAL_RES_ERR_ALLOC; }
        t->barA_con = a1; t->barA_bar = a2; t->barA_sym = a3; t->barA_coef = a4; t->capbarA = nc;
    }
    for (int k = 0; k < num; k++) {
        t->barA_con[t->nbarA] = i;
        t->barA_bar[t->nbarA] = j;
        t->barA_sym[t->nbarA] = sub[k];
        t->barA_coef[t->nbarA] = val[k];
        t->nbarA++;
    }
    return PRIMAL_RES_OK;
}

/* The bar-term readers answer with a LIST, and the list is the store: the pair
 * (symidx,coef) is one term, so `num` says how many terms this (i,j) has.
 * These two used to stop at `maxnum` and answer PRIMAL_RES_OK with a truncated
 * list -- the reading that lies, because the caller cannot tell "these are all
 * the terms" from "you stopped at the space I gave you", and a <A,X> built from
 * a prefix is a different model. They now keep the contract every other
 * buffer-filling getter of this API was given (PRIMAL_getarow, the *slice
 * forms, PRIMAL_getqobj/PRIMAL_getqconk): count first, refuse with ERR_ARG
 * without touching the buffers or *num when the space does not suffice.
 * Passing NULL for BOTH symidx and val is the count door -- this pair has no
 * separate getnum... for one (i,j) -- and it never refuses, because it writes
 * nothing. An empty (i,j) is an empty answer, not a refusal. */
static int bar_term_count(PRIMALtask_t t, int con, int bar, int bycon) {
    int want = 0;
    if (bycon) {
        for (int k = 0; k < t->nbarA; k++)
            if (t->barA_con[k] == con && t->barA_bar[k] == bar) want++;
    } else {
        for (int k = 0; k < t->nbarC; k++)
            if (t->barC_bar[k] == bar) want++;
    }
    return want;
}

/* Reads the A-bar term list for pair (i,j) as (symidx, coef) entries.
 * Counts first and refuses without writing when maxnum is too small. */
PRIMALrescodee PRIMAL_getbaraidxij(PRIMALtask_t t, int i, int j, int maxnum,
                             int *num, int *symidx, double *val) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numbarvar)
        return PRIMAL_RES_ERR_ARG;
    if (maxnum < 0) return PRIMAL_RES_ERR_ARG;
    int want = bar_term_count(t, i, j, 1);
    if ((symidx || val) && want > maxnum) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    for (int k = 0; k < t->nbarA; k++)
        if (t->barA_con[k] == i && t->barA_bar[k] == j) {
            if (symidx) symidx[n] = t->barA_sym[k];
            if (val) val[n] = t->barA_coef[k];
            n++;
        }
    *num = n;
    return PRIMAL_RES_OK;
}

/* Reads the C-bar term list for bar variable j as (symidx, coef) entries.
 * Counts first and refuses without writing when maxnum is too small. */
PRIMALrescodee PRIMAL_getbarcidxj(PRIMALtask_t t, int j, int maxnum,
                             int *num, int *symidx, double *val) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    if (maxnum < 0) return PRIMAL_RES_ERR_ARG;
    int want = bar_term_count(t, -1, j, 0);
    if ((symidx || val) && want > maxnum) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    for (int k = 0; k < t->nbarC; k++)
        if (t->barC_bar[k] == j) {
            if (symidx) symidx[n] = t->barC_sym[k];
            if (val) val[n] = t->barC_coef[k];
            n++;
        }
    *num = n;
    return PRIMAL_RES_OK;
}

/* ---- bar sparsity and per-block index info (reference getbarasparsity/
 * getbaraidxinfo/getbaraidx and the C variants) ----
 * A-bar is a sparse matrix of symmetric matrices; a nonzero BLOCK (i,j) is a
 * weighted sum of stored symmetric matrices. `idx` names a block: this solver
 * uses the row-major index `idx = i*numbarvar + j` for A-bar and `idx = j` for
 * C-bar (the reference's exact vectorisation was not read, declared deviation);
 * getbaraidx decodes it, so a caller using this solver's functions is
 * self-consistent. */
static int barA_block_terms(const PRIMALtask_t t, int i, int j) {
    int n = 0;
    for (int k = 0; k < t->nbarA; k++)
        if (t->barA_con[k] == i && t->barA_bar[k] == j) n++;
    return n;
}

/* Lists the distinct nonzero A-bar blocks as row-major indices i*numbarvar+j.
 * Counts first and refuses without writing when maxnumnz is too small. */
PRIMALrescodee PRIMAL_getbarasparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxij) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (maxnumnz < 0) return PRIMAL_RES_ERR_ARG;
    /* distinct (i,j) blocks */
    PRIMALint64t need = 0;
    for (int k = 0; k < t->nbarA; k++) {
        int i = t->barA_con[k], j = t->barA_bar[k], seen = 0;
        for (int q = 0; q < k && !seen; q++)
            if (t->barA_con[q] == i && t->barA_bar[q] == j) seen = 1;
        if (!seen) need++;
    }
    if (idxij && need > maxnumnz) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t n = 0;
    for (int k = 0; k < t->nbarA; k++) {
        int i = t->barA_con[k], j = t->barA_bar[k], seen = 0;
        for (int q = 0; q < k && !seen; q++)
            if (t->barA_con[q] == i && t->barA_bar[q] == j) seen = 1;
        if (seen) continue;
        if (idxij) idxij[n] = (PRIMALint64t)i * t->numbarvar + j;
        n++;
    }
    *numnz = n;
    return PRIMAL_RES_OK;
}

/* Returns the number of stored terms in the A-bar block named by idx.
 * Decodes idx into the (constraint, bar variable) pair. */
PRIMALrescodee PRIMAL_getbaraidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (idx < 0 || t->numbarvar <= 0) return PRIMAL_RES_ERR_ARG;
    int i = (int)(idx / t->numbarvar), j = (int)(idx % t->numbarvar);
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    *num = barA_block_terms(t, i, j);
    return PRIMAL_RES_OK;
}

/* Reads one A-bar block by idx: decodes (i,j) and lists its matrix ids
 * with weights. Refuses without writing when maxnum is too small. */
PRIMALrescodee PRIMAL_getbaraidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *i, int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (idx < 0 || t->numbarvar <= 0) return PRIMAL_RES_ERR_ARG;
    int ii = (int)(idx / t->numbarvar), jj = (int)(idx % t->numbarvar);
    if (ii < 0 || ii >= t->numcon || jj < 0 || jj >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    int want = barA_block_terms(t, ii, jj);
    if ((sub || weights) && want > maxnum) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    for (int k = 0; k < t->nbarA; k++)
        if (t->barA_con[k] == ii && t->barA_bar[k] == jj) {
            if (sub) sub[n] = t->barA_sym[k];
            if (weights) weights[n] = t->barA_coef[k];
            n++;
        }
    if (i) *i = ii;
    if (j) *j = jj;
    *num = n;
    return PRIMAL_RES_OK;
}

/* Lists the distinct bar variables carrying C-bar terms as indices j.
 * Counts first and refuses without writing when maxnumnz is too small. */
PRIMALrescodee PRIMAL_getbarcsparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxj) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (maxnumnz < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t need = 0;
    for (int k = 0; k < t->nbarC; k++) {
        int j = t->barC_bar[k], seen = 0;
        for (int q = 0; q < k && !seen; q++) if (t->barC_bar[q] == j) seen = 1;
        if (!seen) need++;
    }
    if (idxj && need > maxnumnz) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t n = 0;
    for (int k = 0; k < t->nbarC; k++) {
        int j = t->barC_bar[k], seen = 0;
        for (int q = 0; q < k && !seen; q++) if (t->barC_bar[q] == j) seen = 1;
        if (seen) continue;
        if (idxj) idxj[n] = j;
        n++;
    }
    *numnz = n;
    return PRIMAL_RES_OK;
}

/* Returns the number of stored C-bar terms on bar variable idx. */
PRIMALrescodee PRIMAL_getbarcidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (idx < 0 || idx >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    for (int k = 0; k < t->nbarC; k++) if (t->barC_bar[k] == (int)idx) n++;
    *num = n;
    return PRIMAL_RES_OK;
}

/* Reads one C-bar block by idx: reports j and lists its matrix ids
 * with weights. Refuses without writing when maxnum is too small. */
PRIMALrescodee PRIMAL_getbarcidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (idx < 0 || idx >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    int want = 0;
    for (int k = 0; k < t->nbarC; k++) if (t->barC_bar[k] == (int)idx) want++;
    if ((sub || weights) && want > maxnum) return PRIMAL_RES_ERR_ARG;
    int n = 0;
    for (int k = 0; k < t->nbarC; k++)
        if (t->barC_bar[k] == (int)idx) {
            if (sub) sub[n] = t->barC_sym[k];
            if (weights) weights[n] = t->barC_coef[k];
            n++;
        }
    if (j) *j = (int)idx;
    *num = n;
    return PRIMAL_RES_OK;
}

/* ---- block-triplet form of A-bar and C-bar (reference:
 * getbarablocktriplet/getbarcblocktriplet and their counts) ----
 * A-bar is a sparse matrix of symmetric matrices; the triplet lists one entry
 * per stored (lower-triangle) element of each block: for A (i, j, k, l, val),
 * for C (j, k, l, val), where (k,l) is the element inside the block. The counts
 * are the exact number of such entries, an upper bound in the reference's own
 * sense. A refusal (capacity too small, with buffers supplied) writes nothing. */
static PRIMALint64t barA_triplet_count(const PRIMALtask_t t) {
    PRIMALint64t n = 0;
    for (int q = 0; q < t->nbarA; q++) {
        int m = t->barA_sym[q];
        if (m >= 0 && m < t->nsym) n += t->sym_nnz[m];
    }
    return n;
}
/* Counts stored lower-triangle entries over all C-bar terms.
 * Sums the triplet counts of the referenced symmetric matrices. */
static PRIMALint64t barC_triplet_count(const PRIMALtask_t t) {
    PRIMALint64t n = 0;
    for (int q = 0; q < t->nbarC; q++) {
        int m = t->barC_sym[q];
        if (m >= 0 && m < t->nsym) n += t->sym_nnz[m];
    }
    return n;
}

/* Returns the exact number of A-bar block-triplet entries in the store. */
PRIMALrescodee PRIMAL_getnumbarablocktriplets(PRIMALtask_t t, PRIMALint64t *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = barA_triplet_count(t);
    return PRIMAL_RES_OK;
}
/* Returns the exact number of C-bar block-triplet entries in the store. */
PRIMALrescodee PRIMAL_getnumbarcblocktriplets(PRIMALtask_t t, PRIMALint64t *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = barC_triplet_count(t);
    return PRIMAL_RES_OK;
}

/* Expands every A-bar term into (i,j,k,l,val) block triplets, one row per
 * stored lower-triangle element. Refuses without writing if maxnum is short. */
PRIMALrescodee PRIMAL_getbarablocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subi, int *subj, int *subk, int *subl, PRIMALrealt *valijkl) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (maxnum < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t need = barA_triplet_count(t);
    if ((subi || subj || subk || subl || valijkl) && need > maxnum) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t n = 0;
    for (int q = 0; q < t->nbarA; q++) {
        int i = t->barA_con[q], j = t->barA_bar[q], m = t->barA_sym[q];
        if (m < 0 || m >= t->nsym) continue;
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            if (subi) subi[n] = i;
            if (subj) subj[n] = j;
            if (subk) subk[n] = t->sym_subi[m][e];
            if (subl) subl[n] = t->sym_subj[m][e];
            if (valijkl) valijkl[n] = t->barA_coef[q] * t->sym_val[m][e];
            n++;
        }
    }
    *num = n;
    return PRIMAL_RES_OK;
}

/* Expands every C-bar term into (j,k,l,val) block triplets, one row per
 * stored lower-triangle element. Refuses without writing if maxnum is short. */
PRIMALrescodee PRIMAL_getbarcblocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subj, int *subk, int *subl, PRIMALrealt *valjkl) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    if (maxnum < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t need = barC_triplet_count(t);
    if ((subj || subk || subl || valjkl) && need > maxnum) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t n = 0;
    for (int q = 0; q < t->nbarC; q++) {
        int j = t->barC_bar[q], m = t->barC_sym[q];
        if (m < 0 || m >= t->nsym) continue;
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            if (subj) subj[n] = j;
            if (subk) subk[n] = t->sym_subi[m][e];
            if (subl) subl[n] = t->sym_subj[m][e];
            if (valjkl) valjkl[n] = t->barC_coef[q] * t->sym_val[m][e];
            n++;
        }
    }
    *num = n;
    return PRIMAL_RES_OK;
}

/* Appends a list of stored symmetric matrices to A-bar block (i,j).
 * Same per-block semantics as putbaraij; appends instead of replacing. */
PRIMALrescodee PRIMAL_putbarablockij(PRIMALtask_t t, int i, int j, int num,
                               const int *blk_sub, const double *blk_val) {
    model_changed(t);
    /* il clone rappresenta ogni termine come (con, bar, sym, coef): il
     * blocco (i,j) e' la lista di matrici per la coppia (i,j) — stessa
     * semantica di putbaraij (che e' gia' per-block). Deviazione PRIMAL:
     * appende ai termini esistenti invece di sostituirli. */
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || j < 0 || j >= t->numbarvar ||
        num < 0 || (num > 0 && (!blk_sub || !blk_val))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (blk_sub[k] < 0 || blk_sub[k] >= t->nsym) return PRIMAL_RES_ERR_ARG;
        if (t->sym_dim[blk_sub[k]] != t->barDim[j]) return PRIMAL_RES_ERR_ARG;
    }
    if (t->nbarA + num > t->capbarA) {
        int nc = t->capbarA ? t->capbarA : 4;
        while (nc < t->nbarA + num) nc *= 2;
        int *a1 = (int *)realloc(t->barA_con, (size_t)nc * sizeof(int));
        int *a2 = (int *)realloc(t->barA_bar, (size_t)nc * sizeof(int));
        int *a3 = (int *)realloc(t->barA_sym, (size_t)nc * sizeof(int));
        double *a4 = (double *)realloc(t->barA_coef, (size_t)nc * sizeof(double));
        if (!a1 || !a2 || !a3 || !a4) { free(a1); free(a2); free(a3); free(a4); return PRIMAL_RES_ERR_ALLOC; }
        t->barA_con = a1; t->barA_bar = a2; t->barA_sym = a3; t->barA_coef = a4; t->capbarA = nc;
    }
    for (int k = 0; k < num; k++) {
        t->barA_con[t->nbarA] = i;
        t->barA_bar[t->nbarA] = j;
        t->barA_sym[t->nbarA] = blk_sub[k];
        t->barA_coef[t->nbarA] = blk_val[k];
        t->nbarA++;
    }
    return PRIMAL_RES_OK;
}

/* Appends inner-product terms to the objective on bar variable j.
 * Validates stored-matrix ids and dimensions; appends to the C-bar list. */
PRIMALrescodee PRIMAL_putbarcj(PRIMALtask_t t, int j, int num,
                         const int *sub, const PRIMALrealt *val) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar || num < 0 || (num > 0 && (!sub || !val)))
        return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (sub[k] < 0 || sub[k] >= t->nsym) return PRIMAL_RES_ERR_ARG;
        if (t->sym_dim[sub[k]] != t->barDim[j]) return PRIMAL_RES_ERR_ARG;
    }
    if (t->nbarC + num > t->capbarC) {
        int nc = t->capbarC ? t->capbarC : 4;
        while (nc < t->nbarC + num) nc *= 2;
        int *a1 = (int *)realloc(t->barC_bar, (size_t)nc * sizeof(int));
        int *a2 = (int *)realloc(t->barC_sym, (size_t)nc * sizeof(int));
        double *a3 = (double *)realloc(t->barC_coef, (size_t)nc * sizeof(double));
        if (!a1 || !a2 || !a3) { free(a1); free(a2); free(a3); return PRIMAL_RES_ERR_ALLOC; }
        t->barC_bar = a1; t->barC_sym = a2; t->barC_coef = a3; t->capbarC = nc;
    }
    for (int k = 0; k < num; k++) {
        t->barC_bar[t->nbarC] = j;
        t->barC_sym[t->nbarC] = sub[k];
        t->barC_coef[t->nbarC] = val[k];
        t->nbarC++;
    }
    return PRIMAL_RES_OK;
}

/* Block writes of bars (reference putbarablocktriplet/putbarcblocktriplet/
 * putbaraijlist): they append terms to the store, like `putbaraij`/`putbarcj`.
 * `putbarablocktriplet` describes A-bar by entries: (con, bar, k, l, val);
 * `putbarcblocktriplet` C-bar: (bar, k, l, val). Each entry becomes a
 * single-entry symmetric matrix. */
PRIMALrescodee PRIMAL_putbarablocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const int *subk, const int *subl,
        const PRIMALrealt *valijkl) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subi || !subj || !subk || !subl || !valijkl)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < num; k++) {
        int i = subi[k], j = subj[k], p = subk[k], q = subl[k];
        if (i < 0 || i >= t->numcon || j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
        int d = t->barDim[j];
        if (p < 0 || p >= d || q < 0 || q >= d || valijkl[k] != valijkl[k])
            return PRIMAL_RES_ERR_ARG;
        int idx = -1;
        PRIMALrescodee rc = PRIMAL_appendsparsesymmat(t, d, 1, &p, &q, &valijkl[k], &idx);
        if (rc != PRIMAL_RES_OK) return rc;
        double one = 1.0;
        rc = PRIMAL_putbaraij(t, i, j, 1, &idx, &one);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Adds C-bar entries from (bar,k,l,val) triplets, storing each entry as
 * a single-entry symmetric matrix appended to the C-bar list. */
PRIMALrescodee PRIMAL_putbarcblocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subj, const int *subk, const int *subl, const PRIMALrealt *valjkl) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subj || !subk || !subl || !valjkl))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < num; k++) {
        int j = subj[k], p = subk[k], q = subl[k];
        if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
        int d = t->barDim[j];
        if (p < 0 || p >= d || q < 0 || q >= d || valjkl[k] != valjkl[k]) return PRIMAL_RES_ERR_ARG;
        int idx = -1;
        PRIMALrescodee rc = PRIMAL_appendsparsesymmat(t, d, 1, &p, &q, &valjkl[k], &idx);
        if (rc != PRIMAL_RES_OK) return rc;
        double one = 1.0;
        rc = PRIMAL_putbarcj(t, j, 1, &idx, &one);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* putbaraijlist: for each i, the terms matidx[alphaptrb[i]..alphaptre[i]) with
 * the weights weights, appended to block (subi[i], subj[i]). */
PRIMALrescodee PRIMAL_putbaraijlist(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const PRIMALint64t *alphaptrb,
        const PRIMALint64t *alphaptre, const PRIMALint64t *matidx,
        const PRIMALrealt *weights) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subi || !subj || !alphaptrb || !alphaptre || !matidx || !weights)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t i = 0; i < num; i++) {
        int con = subi[i], bar = subj[i];
        if (con < 0 || con >= t->numcon || bar < 0 || bar >= t->numbarvar)
            return PRIMAL_RES_ERR_ARG;
        for (PRIMALint64t m = alphaptrb[i]; m < alphaptre[i]; m++) {
            int sidx = (int)matidx[m];
            if (sidx < 0 || sidx >= t->nsym) return PRIMAL_RES_ERR_ARG;
            double w = weights[m];
            PRIMALrescodee rc = PRIMAL_putbaraij(t, con, bar, 1, &sidx, &w);
            if (rc != PRIMAL_RES_OK) return rc;
        }
    }
    return PRIMAL_RES_OK;
}

/* Copies the published dual block of bar variable j into the dense d*d
 * buffer sj. Refuses when no solution has been published. */
PRIMALrescodee PRIMAL_getbarsj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *sj) {
    (void)which;
    if (!t || !sj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int d = t->barDim[j];
    for (int k = 0; k < d * d; k++) sj[k] = t->barsj[j][k];
    return PRIMAL_RES_OK;
}

/* Returns the dimension d of bar variable j. */
PRIMALrescodee PRIMAL_getbarsize(PRIMALtask_t t, int j, int *dim) {
    if (!t || !dim) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    *dim = t->barDim[j];
    return PRIMAL_RES_OK;
}

/* Reference: getdimbarvarj (same value as getbarsize) and getlenbarvarj, the
 * number of elements in the LOWER TRIANGLE, d*(d+1)/2 -- the same count the
 * compressed block of a bar variable occupies. */
PRIMALrescodee PRIMAL_getdimbarvarj(PRIMALtask_t t, int j, int *dimbarvarj) {
    if (!t || !dimbarvarj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    *dimbarvarj = t->barDim[j];
    return PRIMAL_RES_OK;
}

/* Returns the lower-triangle length d*(d+1)/2 of bar variable j,
 * the size of its compressed conic block. */
PRIMALrescodee PRIMAL_getlenbarvarj(PRIMALtask_t t, int j, PRIMALint64t *lenbarvarj) {
    if (!t || !lenbarvarj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t d = (PRIMALint64t)t->barDim[j];
    *lenbarvarj = d * (d + 1) / 2;
    return PRIMAL_RES_OK;
}

/* Returns the number of stored C-bar terms. */
PRIMALrescodee PRIMAL_getnumbarcterm(PRIMALtask_t t, int *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = t->nbarC;
    return PRIMAL_RES_OK;
}

/* Reads the k-th C-bar term as (bar index, matrix id, coefficient). */
PRIMALrescodee PRIMAL_getbarcitem(PRIMALtask_t t, int k, int *jbar, int *msym, PRIMALrealt *coef) {
    if (!t || !jbar || !msym || !coef) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->nbarC) return PRIMAL_RES_ERR_ARG;
    *jbar = t->barC_bar[k]; *msym = t->barC_sym[k]; *coef = t->barC_coef[k];
    return PRIMAL_RES_OK;
}

/* Returns the number of stored A-bar terms. */
PRIMALrescodee PRIMAL_getnumbaraterm(PRIMALtask_t t, int *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = t->nbarA;
    return PRIMAL_RES_OK;
}

/* Reads the k-th A-bar term as (constraint, bar index, matrix id, coef). */
PRIMALrescodee PRIMAL_getbaraitem(PRIMALtask_t t, int k, int *con, int *jbar, int *msym, PRIMALrealt *coef) {
    if (!t || !con || !jbar || !msym || !coef) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->nbarA) return PRIMAL_RES_ERR_ARG;
    *con = t->barA_con[k]; *jbar = t->barA_bar[k]; *msym = t->barA_sym[k]; *coef = t->barA_coef[k];
    return PRIMAL_RES_OK;
}

/* Returns the number of symmetric matrices held in the store. */
PRIMALrescodee PRIMAL_getnumsymmat(PRIMALtask_t t, int *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = t->nsym;
    return PRIMAL_RES_OK;
}

/* Returns the dimension and entry count of stored symmetric matrix m. */
PRIMALrescodee PRIMAL_getsymmatinfo(PRIMALtask_t t, int m, int *dim, int *nnz) {
    if (!t || !dim || !nnz) return PRIMAL_RES_ERR_NULL;
    if (m < 0 || m >= t->nsym) return PRIMAL_RES_ERR_ARG;
    *dim = t->sym_dim[m]; *nnz = t->sym_nnz[m];
    return PRIMAL_RES_OK;
}

/* Reads entry e of stored symmetric matrix m as (row, column, value). */
PRIMALrescodee PRIMAL_getsymmatentry(PRIMALtask_t t, int m, int e, int *i, int *j, PRIMALrealt *val) {
    if (!t || !i || !j || !val) return PRIMAL_RES_ERR_NULL;
    if (m < 0 || m >= t->nsym || e < 0 || e >= t->sym_nnz[m]) return PRIMAL_RES_ERR_ARG;
    *i = t->sym_subi[m][e]; *j = t->sym_subj[m][e]; *val = t->sym_val[m][e];
    return PRIMAL_RES_OK;
}

/* Returns the number of bar variables in the task. */
PRIMALrescodee PRIMAL_getnumbarvar(PRIMALtask_t t, int *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = t->numbarvar;
    return PRIMAL_RES_OK;
}

/* Copies the published primal block of bar variable j into the dense d*d
 * buffer xj. Refuses when no solution has been published. */
PRIMALrescodee PRIMAL_getbarxj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *xj) {
    (void)which;
    if (!t || !xj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int d = t->barDim[j];
    for (int k = 0; k < d * d; k++) xj[k] = t->barx[j][k];
    return PRIMAL_RES_OK;
}

/* ---- reference bar surface (names, slices, warm start, counters) ----
 * No new rule: `getnumbaranz`/`getnumbarcnz` are the same counters as
 * `getnumbaraterm`/`getnumbarcterm`; `*barvarname` the same `name_put`/
 * `name_find` as `putbarname`; `getbar?slice` concatenates the dense `d*d`
 * blocks of `barx`/`barsj` (the form of `getbarxj`/`getbarsj`); `putbarxj`/
 * `putbarsj` write the point; the `putmaxnum*` are capacity hints, no-ops
 * because this solver grows on its own. */
/* Returns the A-bar term count in 64-bit form (same counter as
 * getnumbaraterm). */
PRIMALrescodee PRIMAL_getnumbaranz(PRIMALtask_t t, PRIMALint64t *nz) {
    if (!t || !nz) return PRIMAL_RES_ERR_NULL;
    *nz = t->nbarA;
    return PRIMAL_RES_OK;
}
/* Returns the C-bar term count in 64-bit form (same counter as
 * getnumbarcterm). */
PRIMALrescodee PRIMAL_getnumbarcnz(PRIMALtask_t t, PRIMALint64t *nz) {
    if (!t || !nz) return PRIMAL_RES_ERR_NULL;
    *nz = t->nbarC;
    return PRIMAL_RES_OK;
}
/* Sets the name of bar variable j (delegates to the bar name table). */
PRIMALrescodee PRIMAL_putbarvarname(PRIMALtask_t t, int j, const char *name) {
    return PRIMAL_putbarname(t, j, name);
}
/* Copies the name of bar variable i into the caller buffer of sizename
 * bytes, including the terminator. Empty string when unnamed. */
PRIMALrescodee PRIMAL_getbarvarname(PRIMALtask_t t, int i, int sizename, char *name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    const char *nm = (t->barname && t->barname[i]) ? t->barname[i] : "";
    int len = (int)strlen(nm);
    if (sizename < len + 1) return PRIMAL_RES_ERR_ARG;
    memcpy(name, nm, (size_t)len + 1);
    return PRIMAL_RES_OK;
}
/* Looks up a bar variable by name and returns its index.
 * Reports a single assignment through asgn. */
PRIMALrescodee PRIMAL_getbarvarnameindex(PRIMALtask_t t, const char *somename,
                                         int *asgn, int *index) {
    if (!t || !somename || !index) return PRIMAL_RES_ERR_NULL;
    if (!t->barname) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = name_find((const char **)t->barname, t->numbarvar, somename, index);
    if (rc == PRIMAL_RES_OK && asgn) *asgn = 0;   /* single assignment */
    return rc;
}
/* Returns the length of the name of bar variable i, excluding the terminator. */
PRIMALrescodee PRIMAL_getbarvarnamelen(PRIMALtask_t t, int i, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->barname ? t->barname[i] : NULL);
    return PRIMAL_RES_OK;
}
/* Concatenates the dense primal blocks barx[first..last) into barxslice.
 * Refuses without writing when slicesize is smaller than needed. */
PRIMALrescodee PRIMAL_getbarxslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barxslice) {
    (void)which;
    if (!t || !barxslice) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numbarvar) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t need = 0;
    for (int j = first; j < last; j++) need += (PRIMALint64t)t->barDim[j] * t->barDim[j];
    if (slicesize < need) return PRIMAL_RES_ERR_ARG;   /* no space: does not write */
    PRIMALint64t w = 0;
    for (int j = first; j < last; j++) {
        int d = t->barDim[j];
        for (int k = 0; k < d * d; k++) barxslice[w++] = t->barx[j][k];
    }
    return PRIMAL_RES_OK;
}
/* Concatenates the dense dual blocks barsj[first..last) into barsslice.
 * Refuses without writing when slicesize is smaller than needed. */
PRIMALrescodee PRIMAL_getbarsslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barsslice) {
    (void)which;
    if (!t || !barsslice) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numbarvar) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t need = 0;
    for (int j = first; j < last; j++) need += (PRIMALint64t)t->barDim[j] * t->barDim[j];
    if (slicesize < need) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t w = 0;
    for (int j = first; j < last; j++) {
        int d = t->barDim[j];
        for (int k = 0; k < d * d; k++) barsslice[w++] = t->barsj[j][k];
    }
    return PRIMAL_RES_OK;
}
/* Overwrites the stored primal block of bar variable j from a dense d*d
 * warm-start buffer. */
PRIMALrescodee PRIMAL_putbarxj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barxj) {
    (void)which;
    if (!t || !barxj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar || !t->barx[j]) return PRIMAL_RES_ERR_ARG;
    int d = t->barDim[j];
    for (int k = 0; k < d * d; k++) t->barx[j][k] = barxj[k];
    return PRIMAL_RES_OK;
}
/* Overwrites the stored dual block of bar variable j from a dense d*d
 * warm-start buffer. */
PRIMALrescodee PRIMAL_putbarsj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barsj) {
    (void)which;
    if (!t || !barsj) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar || !t->barsj[j]) return PRIMAL_RES_ERR_ARG;
    int d = t->barDim[j];
    for (int k = 0; k < d * d; k++) t->barsj[j][k] = barsj[k];
    return PRIMAL_RES_OK;
}
/* Accepts a suggested bar-variable capacity; a no-op hint because
 * this solver grows its storage on its own. */
PRIMALrescodee PRIMAL_putmaxnumbarvar(PRIMALtask_t t, int maxnumbarvar) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumbarvar < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;   /* suggested capacity: this solver grows on its own */
}

