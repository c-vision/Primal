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
/* primal_afe.c - affine expressions, conic domains, ACC.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* ---- affine expressions (AFE): f_i = sum_j F_ij x_j + g_i ----
 * Rows of F are stored sparse per AFE; g is one double per AFE. This is the
 * storage the affine conic constraints (ACC) and disjunctive constraints (DJC)
 * build on. */
PRIMALrescodee PRIMAL_appendafes(PRIMALtask_t t, PRIMALint64t num) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX) return PRIMAL_RES_ERR_ARG;
    int need = t->numafe + (int)num;
    if (need > t->afecap) {
        int nc = t->afecap ? t->afecap : 4;
        while (nc < need) nc *= 2;
        int *n1 = (int *)realloc(t->afe_nz, (size_t)nc * sizeof(int));
        int *n2 = (int *)realloc(t->afe_cap, (size_t)nc * sizeof(int));
        int **n3 = (int **)realloc(t->afe_sub, (size_t)nc * sizeof(int *));
        double **n4 = (double **)realloc(t->afe_val, (size_t)nc * sizeof(double *));
        double *n5 = (double *)realloc(t->afeg, (size_t)nc * sizeof(double));
        int *n6 = (int *)realloc(t->afe_barnz, (size_t)nc * sizeof(int));
        int *n7 = (int *)realloc(t->afe_barcap, (size_t)nc * sizeof(int));
        int **n8 = (int **)realloc(t->afe_baridx, (size_t)nc * sizeof(int *));
        int **n9 = (int **)realloc(t->afe_barsym, (size_t)nc * sizeof(int *));
        double **n10 = (double **)realloc(t->afe_barcoef, (size_t)nc * sizeof(double *));
        if (!n1 || !n2 || !n3 || !n4 || !n5 || !n6 || !n7 || !n8 || !n9 || !n10) {
            free(n1); free(n2); free(n3); free(n4); free(n5);
            free(n6); free(n7); free(n8); free(n9); free(n10);
            return PRIMAL_RES_ERR_ALLOC;
        }
        t->afe_nz = n1; t->afe_cap = n2; t->afe_sub = n3; t->afe_val = n4; t->afeg = n5;
        t->afe_barnz = n6; t->afe_barcap = n7;
        t->afe_baridx = n8; t->afe_barsym = n9; t->afe_barcoef = n10;
        t->afecap = nc;
    }
    for (int k = t->numafe; k < need; k++) {
        t->afe_nz[k] = 0; t->afe_cap[k] = 0;
        t->afe_sub[k] = NULL; t->afe_val[k] = NULL; t->afeg[k] = 0.0;
        t->afe_barnz[k] = 0; t->afe_barcap[k] = 0;
        t->afe_baridx[k] = NULL; t->afe_barsym[k] = NULL; t->afe_barcoef[k] = NULL;
    }
    t->numafe = need;
    return PRIMAL_RES_OK;
}

/* Report the number of affine expressions. */
PRIMALrescodee PRIMAL_getnumafe(PRIMALtask_t t, PRIMALint64t *numafe) {
    if (!t || !numafe) return PRIMAL_RES_ERR_NULL;
    *numafe = t->numafe;
    return PRIMAL_RES_OK;
}

/* Refresh the rows of every LINEAR-domain ACC from the current affine
 * expressions (issue #19): an accepted edit must be what the next solve uses. */
static void acc_sync_linear(PRIMALtask_t t);

/* replace F[i][j]; v == 0 removes the entry */
PRIMALrescodee PRIMAL_putafefentry(PRIMALtask_t t, PRIMALint64t i, int j, PRIMALrealt v) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe || j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (v != v) return PRIMAL_RES_ERR_ARG;
    int k = (int)i;
    int w = 0;
    for (int e = 0; e < t->afe_nz[k]; e++)
        if (t->afe_sub[k][e] != j) { t->afe_sub[k][w] = t->afe_sub[k][e]; t->afe_val[k][w] = t->afe_val[k][e]; w++; }
    t->afe_nz[k] = w;
    if (v != 0.0) {
        if (w == t->afe_cap[k]) {
            int nc = t->afe_cap[k] ? t->afe_cap[k] * 2 : 4;
            int *s2 = (int *)realloc(t->afe_sub[k], (size_t)nc * sizeof(int));
            double *v2 = (double *)realloc(t->afe_val[k], (size_t)nc * sizeof(double));
            if (!s2 || !v2) return PRIMAL_RES_ERR_ALLOC;
            t->afe_sub[k] = s2; t->afe_val[k] = v2; t->afe_cap[k] = nc;
        }
        t->afe_sub[k][w] = j; t->afe_val[k][w] = v; t->afe_nz[k] = w + 1;
    }
    acc_sync_linear(t);
    return PRIMAL_RES_OK;
}

/* Replace row i of F from (varidx,val); duplicates collapse, zeros are skipped. */
PRIMALrescodee PRIMAL_putafefrow(PRIMALtask_t t, PRIMALint64t i, int numnz,
                                 const int *varidx, const PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe || numnz < 0 || (numnz > 0 && (!varidx || !val)))
        return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < numnz; e++) {
        if (varidx[e] < 0 || varidx[e] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (val[e] != val[e]) return PRIMAL_RES_ERR_ARG;
    }
    int k = (int)i;
    t->afe_nz[k] = 0;
    for (int e = 0; e < numnz; e++) {
        int j = varidx[e]; double v = val[e];
        if (v == 0.0) continue;
        int w = 0;
        for (int q = 0; q < t->afe_nz[k]; q++) if (t->afe_sub[k][q] == j) { t->afe_val[k][q] = v; w = 1; break; }
        if (w) continue;
        if (t->afe_nz[k] == t->afe_cap[k]) {
            int nc = t->afe_cap[k] ? t->afe_cap[k] * 2 : 4;
            int *s2 = (int *)realloc(t->afe_sub[k], (size_t)nc * sizeof(int));
            double *v2 = (double *)realloc(t->afe_val[k], (size_t)nc * sizeof(double));
            if (!s2 || !v2) return PRIMAL_RES_ERR_ALLOC;
            t->afe_sub[k] = s2; t->afe_val[k] = v2; t->afe_cap[k] = nc;
        }
        t->afe_sub[k][t->afe_nz[k]] = j; t->afe_val[k][t->afe_nz[k]] = v; t->afe_nz[k]++;
    }
    return PRIMAL_RES_OK;
}

/* Set g[i] (NaN is refused). */
PRIMALrescodee PRIMAL_putafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt g) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe) return PRIMAL_RES_ERR_ARG;
    if (g != g) return PRIMAL_RES_ERR_ARG;
    t->afeg[i] = g;
    acc_sync_linear(t);
    return PRIMAL_RES_OK;
}

/* Return g[i]. */
PRIMALrescodee PRIMAL_getafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt *g) {
    if (!t || !g) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe) return PRIMAL_RES_ERR_ARG;
    *g = t->afeg[i];
    return PRIMAL_RES_OK;
}

/* Report the number of nonzeros in row i of F. */
PRIMALrescodee PRIMAL_getafefrownumnz(PRIMALtask_t t, PRIMALint64t i, int *numnz) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe) return PRIMAL_RES_ERR_ARG;
    *numnz = t->afe_nz[i];
    return PRIMAL_RES_OK;
}

/* Copy row i of F into varidx/val and report its length. */
PRIMALrescodee PRIMAL_getafefrow(PRIMALtask_t t, PRIMALint64t i, int *numnz,
                                 int *varidx, PRIMALrealt *val) {
    if (!t || !numnz) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numafe) return PRIMAL_RES_ERR_ARG;
    if (t->afe_nz[i] > 0 && (!varidx || !val)) return PRIMAL_RES_ERR_NULL;
    for (int e = 0; e < t->afe_nz[i]; e++) {
        if (varidx) varidx[e] = t->afe_sub[i][e];
        if (val) val[e] = t->afe_val[i][e];
    }
    *numnz = t->afe_nz[i];
    return PRIMAL_RES_OK;
}

/* ---- the AFE block read/write surface ----
 * The reference exposes `emptyafefrow`/`emptyafefcol`, `putafeglist`/
 * `putafegslice`/`getafegslice`, `putafefentrylist` and `getafeftrip`. They add
 * no rules: `putafeg*` and `putafefentrylist` loop over the scalar getters/
 * putters (which already validate), `getafegslice` is a slice of `g`, and
 * `getafeftrip` enumerates the store of F in triplets (row, column, value) in
 * the order it is stored. `getafefnumnz` is the same question as
 * `getafefrownumnz`, so it delegates instead of recounting. */

PRIMALrescodee PRIMAL_emptyafefrow(PRIMALtask_t t, PRIMALint64t afeidx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    t->afe_nz[afeidx] = 0;
    return PRIMAL_RES_OK;
}

/* Remove column varidx from every row of F. */
PRIMALrescodee PRIMAL_emptyafefcol(PRIMALtask_t t, int varidx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (varidx < 0 || varidx >= t->numvar) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numafe; i++) {
        int w = 0;
        for (int e = 0; e < t->afe_nz[i]; e++)
            if (t->afe_sub[i][e] != varidx) {
                t->afe_sub[i][w] = t->afe_sub[i][e];
                t->afe_val[i][w] = t->afe_val[i][e];
                w++;
            }
        t->afe_nz[i] = w;
    }
    return PRIMAL_RES_OK;
}

/* Set g at each listed AFE index. */
PRIMALrescodee PRIMAL_putafeglist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                  const PRIMALint64t *afeidx, const PRIMALrealt *g) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numafeidx < 0 || (numafeidx > 0 && (!afeidx || !g))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numafeidx; k++) {
        PRIMALrescodee rc = PRIMAL_putafeg(t, afeidx[k], g[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* Set g over the slice [first,last). */
PRIMALrescodee PRIMAL_putafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, const PRIMALrealt *slice) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numafe) return PRIMAL_RES_ERR_ARG;
    if (last > first && !slice) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t i = first; i < last; i++) {
        if (slice[i - first] != slice[i - first]) return PRIMAL_RES_ERR_ARG;
        t->afeg[i] = slice[i - first];
    }
    return PRIMAL_RES_OK;
}

/* Read g over the slice [first,last). */
PRIMALrescodee PRIMAL_getafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, PRIMALrealt *g) {
    if (!t || !g) return PRIMAL_RES_ERR_NULL;
    if (first < 0 || last < first || last > t->numafe) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t i = first; i < last; i++) g[i - first] = t->afeg[i];
    return PRIMAL_RES_OK;
}

/* Replace F entries from the list (afeidx,varidx,val). */
PRIMALrescodee PRIMAL_putafefentrylist(PRIMALtask_t t, PRIMALint64t numentr,
                                       const PRIMALint64t *afeidx,
                                       const int *varidx, const PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numentr < 0 || (numentr > 0 && (!afeidx || !varidx || !val)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numentr; k++) {
        PRIMALrescodee rc = PRIMAL_putafefentry(t, afeidx[k], varidx[k], val[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* Enumerate F as triplets (afeidx,varidx,val) in storage order. */
PRIMALrescodee PRIMAL_getafeftrip(PRIMALtask_t t, PRIMALint64t *afeidx,
                                  int *varidx, PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!afeidx || !varidx || !val) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t w = 0;
    for (int i = 0; i < t->numafe; i++)
        for (int e = 0; e < t->afe_nz[i]; e++) {
            afeidx[w] = i;
            varidx[w] = t->afe_sub[i][e];
            val[w] = t->afe_val[i][e];
            w++;
        }
    return PRIMAL_RES_OK;
}

/* Delegate of getafefrownumnz: nonzeros of one AFE row. */
PRIMALrescodee PRIMAL_getafefnumnz(PRIMALtask_t t, PRIMALint64t afeidx, int *numnz) {
    return PRIMAL_getafefrownumnz(t, afeidx, numnz);   /* same question */
}

/* ---- bar terms of an AFE (reference putafebarfentry and family) ----
 * Fbar[i][j] is a weighted combination of symmetric matrices from the store,
 * and <Fbar_ij, X_j> enters the i-th affine expression. Writing (i,j) replaces
 * the terms with the same barvaridx. `afe_add_bar_terms` is the point where the
 * terms enter the row that the AFE produces (linear or conic ACC). */
/* Ensure room for `extra` more bar terms in AFE k. */
static PRIMALrescodee afe_bar_reserve(PRIMALtask_t t, int k, int extra) {
    if (t->afe_barnz[k] + extra <= t->afe_barcap[k]) return PRIMAL_RES_OK;
    int nc = t->afe_barcap[k] ? t->afe_barcap[k] : 4;
    while (nc < t->afe_barnz[k] + extra) nc *= 2;
    int *a1 = (int *)realloc(t->afe_baridx[k], (size_t)nc * sizeof(int));
    int *a2 = (int *)realloc(t->afe_barsym[k], (size_t)nc * sizeof(int));
    double *a3 = (double *)realloc(t->afe_barcoef[k], (size_t)nc * sizeof(double));
    if (!a1 || !a2 || !a3) { free(a1); free(a2); free(a3); return PRIMAL_RES_ERR_ALLOC; }
    t->afe_baridx[k] = a1; t->afe_barsym[k] = a2; t->afe_barcoef[k] = a3;
    t->afe_barcap[k] = nc;
    return PRIMAL_RES_OK;
}
/* Replace the bar terms of Fbar[i][j] with the given combination. */
PRIMALrescodee PRIMAL_putafebarfentry(PRIMALtask_t t, PRIMALint64t afeidx, int barvaridx,
        PRIMALint64t numterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe || barvaridx < 0 || barvaridx >= t->numbarvar)
        return PRIMAL_RES_ERR_ARG;
    if (numterm < 0 || (numterm > 0 && (!termidx || !termweight))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t e = 0; e < numterm; e++) {
        if (termidx[e] < 0 || termidx[e] >= t->nsym) return PRIMAL_RES_ERR_ARG;
        if (t->sym_dim[termidx[e]] != t->barDim[barvaridx]) return PRIMAL_RES_ERR_ARG;
        if (termweight[e] != termweight[e]) return PRIMAL_RES_ERR_ARG;
    }
    int k = (int)afeidx;
    /* remove the terms with the same barvaridx (writing replaces) */
    int w = 0;
    for (int e = 0; e < t->afe_barnz[k]; e++)
        if (t->afe_baridx[k][e] != barvaridx) {
            t->afe_baridx[k][w] = t->afe_baridx[k][e];
            t->afe_barsym[k][w] = t->afe_barsym[k][e];
            t->afe_barcoef[k][w] = t->afe_barcoef[k][e];
            w++;
        }
    t->afe_barnz[k] = w;
    PRIMALrescodee rc = afe_bar_reserve(t, k, (int)numterm);
    if (rc != PRIMAL_RES_OK) return rc;
    for (PRIMALint64t e = 0; e < numterm; e++) {
        t->afe_baridx[k][t->afe_barnz[k]] = barvaridx;
        t->afe_barsym[k][t->afe_barnz[k]] = (int)termidx[e];
        t->afe_barcoef[k][t->afe_barnz[k]] = termweight[e];
        t->afe_barnz[k]++;
    }
    return PRIMAL_RES_OK;
}
/* Clear all bar terms of AFE afeidx. */
PRIMALrescodee PRIMAL_emptyafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    t->afe_barnz[afeidx] = 0;
    return PRIMAL_RES_OK;
}
/* Clear the bar terms of every listed AFE. */
PRIMALrescodee PRIMAL_emptyafebarfrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                          const PRIMALint64t *afeidxlist) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numafeidx < 0 || (numafeidx > 0 && !afeidxlist)) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numafeidx; k++) {
        PRIMALrescodee rc = PRIMAL_emptyafebarfrow(t, afeidxlist[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Report the number of distinct bar variables in AFE afeidx. */
PRIMALrescodee PRIMAL_getafebarfnumrowentries(PRIMALtask_t t, PRIMALint64t afeidx, int *numentr) {
    if (!t || !numentr) return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    int k = (int)afeidx, n = 0;
    for (int e = 0; e < t->afe_barnz[k]; e++) {
        int seen = 0;
        for (int q = 0; q < e; q++) if (t->afe_baridx[k][q] == t->afe_baridx[k][e]) { seen = 1; break; }
        if (!seen) n++;
    }
    *numentr = n;
    return PRIMAL_RES_OK;
}
/* Report the distinct bar count and the total term count of an AFE. */
PRIMALrescodee PRIMAL_getafebarfrowinfo(PRIMALtask_t t, PRIMALint64t afeidx,
                                        int *numentr, PRIMALint64t *numterm) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    if (numentr) {
        PRIMALrescodee rc = PRIMAL_getafebarfnumrowentries(t, afeidx, numentr);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    if (numterm) *numterm = t->afe_barnz[afeidx];
    return PRIMAL_RES_OK;
}
/* Copy an AFE's bar terms as (barvaridx,ptrterm,numterm,termidx,termweight). */
PRIMALrescodee PRIMAL_getafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int *barvaridx,
        PRIMALint64t *ptrterm, PRIMALint64t *numterm, PRIMALint64t *termidx,
        PRIMALrealt *termweight) {
    if (!t || !barvaridx || !ptrterm || !numterm || !termidx || !termweight)
        return PRIMAL_RES_ERR_NULL;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    int k = (int)afeidx, n = 0;
    for (int e = 0; e < t->afe_barnz[k]; e++) {
        int j = t->afe_baridx[k][e], seen = 0;
        for (int q = 0; q < e; q++) if (t->afe_baridx[k][q] == j) { seen = 1; break; }
        if (seen) continue;
        barvaridx[n] = j;
        ptrterm[n] = n;   /* one term per entry: ptrterm = starting index */
        numterm[n] = 1;
        termidx[n] = t->afe_barsym[k][e];
        termweight[n] = t->afe_barcoef[k][e];
        n++;
    }
    return PRIMAL_RES_OK;
}
/* Count the block triplets over all AFE bar terms. */
PRIMALrescodee PRIMAL_getafebarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip) {
    if (!t || !numtrip) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t n = 0;
    for (int i = 0; i < t->numafe; i++)
        for (int e = 0; e < t->afe_barnz[i]; e++) n += t->sym_nnz[t->afe_barsym[i][e]];
    *numtrip = n;
    return PRIMAL_RES_OK;
}
/* Write the AFE bar terms as block triplets; a short buffer is refused without
 * writing. */
PRIMALrescodee PRIMAL_getafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *afeidx, int *barvaridx, int *subk,
        int *subl, PRIMALrealt *valkl) {
    if (!t || !numtrip) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t need = 0;
    PRIMALrescodee rc = PRIMAL_getafebarfnumblocktriplets(t, &need);
    if (rc != PRIMAL_RES_OK) return rc;
    if (afeidx && barvaridx && subk && subl && valkl) {
        if (maxnumtrip < need) return PRIMAL_RES_ERR_ARG;   /* refusal without writing */
        PRIMALint64t w = 0;
        for (int i = 0; i < t->numafe; i++)
            for (int e = 0; e < t->afe_barnz[i]; e++) {
                int m = t->afe_barsym[i][e];
                for (int q = 0; q < t->sym_nnz[m]; q++) {
                    afeidx[w] = i;
                    barvaridx[w] = t->afe_baridx[i][e];
                    subk[w] = t->sym_subi[m][q];
                    subl[w] = t->sym_subj[m][q];
                    valkl[w] = t->afe_barcoef[i][e] * t->sym_val[m][q];
                    w++;
                }
            }
    }
    *numtrip = need;
    return PRIMAL_RES_OK;
}
/* Fbar in block triplets: every triplet (afeidx, barvaridx, k, l, val)
 * contributes one term. For each pair (i,j) the entries become a combination of
 * one-term matrices, and writing replaces Fbar[i][j]. */
PRIMALrescodee PRIMAL_putafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t numtrip,
        const PRIMALint64t *afeidx, const int *barvaridx, const int *subk,
        const int *subl, const PRIMALrealt *valkl) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numtrip < 0 || (numtrip > 0 && (!afeidx || !barvaridx || !subk || !subl || !valkl)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t a = 0; a < numtrip; a++) {
        int j = barvaridx[a];
        if (afeidx[a] < 0 || afeidx[a] >= t->numafe || j < 0 || j >= t->numbarvar)
            return PRIMAL_RES_ERR_ARG;
        int d = t->barDim[j];
        if (subk[a] < 0 || subk[a] >= d || subl[a] < 0 || subl[a] >= d || valkl[a] != valkl[a])
            return PRIMAL_RES_ERR_ARG;
    }
    /* empty Fbar and rebuild it per pair (i,j) */
    for (int i = 0; i < t->numafe; i++) t->afe_barnz[i] = 0;
    PRIMALint64t *syms = (PRIMALint64t *)malloc((size_t)(numtrip > 0 ? numtrip : 1) * sizeof(PRIMALint64t));
    double *ones = (double *)malloc((size_t)(numtrip > 0 ? numtrip : 1) * sizeof(double));
    if (!syms || !ones) { free(syms); free(ones); return PRIMAL_RES_ERR_ALLOC; }
    for (PRIMALint64t a = 0; a < numtrip; a++) {
        int i = (int)afeidx[a], j = barvaridx[a];
        int seen = 0;
        for (PRIMALint64t b = 0; b < a; b++)
            if (afeidx[b] == i && barvaridx[b] == j) { seen = 1; break; }
        if (seen) continue;
        int cnt = 0;
        for (PRIMALint64t b = a; b < numtrip; b++) {
            if (afeidx[b] != i || barvaridx[b] != j) continue;
            int p = subk[b], q = subl[b], idx = -1;
            PRIMALrescodee rc = PRIMAL_appendsparsesymmat(t, t->barDim[j], 1, &p, &q,
                                                          &valkl[b], &idx);
            if (rc != PRIMAL_RES_OK) { free(syms); free(ones); return rc; }
            syms[cnt] = idx;
            ones[cnt] = 1.0;
            cnt++;
        }
        PRIMALrescodee rc = PRIMAL_putafebarfentry(t, i, j, cnt, syms, ones);
        if (rc != PRIMAL_RES_OK) { free(syms); free(ones); return rc; }
    }
    free(syms); free(ones);
    return PRIMAL_RES_OK;
}

/* Set Fbar entries from packed lists (lenterm is unused). */
PRIMALrescodee PRIMAL_putafebarfentrylist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *barvaridx, const PRIMALint64t *numterm,
        const PRIMALint64t *ptrterm, PRIMALint64t lenterm, const PRIMALint64t *termidx,
        const PRIMALrealt *termweight) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)lenterm;
    if (numafeidx < 0 || (numafeidx > 0 && (!afeidx || !barvaridx || !numterm || !ptrterm)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numafeidx; k++) {
        PRIMALrescodee rc = PRIMAL_putafebarfentry(t, afeidx[k], barvaridx[k], numterm[k],
                termidx ? termidx + ptrterm[k] : NULL,
                termweight ? termweight + ptrterm[k] : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Replace all bar terms of one AFE from packed lists (lenterm is unused). */
PRIMALrescodee PRIMAL_putafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int numentr,
        const int *barvaridx, const PRIMALint64t *numterm, const PRIMALint64t *ptrterm,
        PRIMALint64t lenterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)lenterm;
    if (afeidx < 0 || afeidx >= t->numafe) return PRIMAL_RES_ERR_ARG;
    if (numentr < 0 || (numentr > 0 && (!barvaridx || !numterm || !ptrterm)))
        return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_emptyafebarfrow(t, afeidx);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < numentr; k++) {
        rc = PRIMAL_putafebarfentry(t, afeidx, barvaridx[k], numterm[k],
                termidx ? termidx + ptrterm[k] : NULL,
                termweight ? termweight + ptrterm[k] : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Fbar as implied by the ACCs: for each component (global index) the bar terms
 * of the AFE that names it. */
PRIMALrescodee PRIMAL_getaccbarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip) {
    if (!t || !numtrip) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t n = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++) {
            int afe = (int)t->acc_afe[i][e];
            for (int q = 0; q < t->afe_barnz[afe]; q++) n += t->sym_nnz[t->afe_barsym[afe][q]];
        }
    *numtrip = n;
    return PRIMAL_RES_OK;
}
/* Write the Fbar implied by the ACCs as block triplets; a short buffer is
 * refused without writing. */
PRIMALrescodee PRIMAL_getaccbarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *acc_afe, int *bar_var, int *blk_row,
        int *blk_col, PRIMALrealt *blk_val) {
    if (!t || !numtrip) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t need = 0;
    PRIMALrescodee rc = PRIMAL_getaccbarfnumblocktriplets(t, &need);
    if (rc != PRIMAL_RES_OK) return rc;
    if (acc_afe && bar_var && blk_row && blk_col && blk_val) {
        if (maxnumtrip < need) return PRIMAL_RES_ERR_ARG;
        PRIMALint64t w = 0, comp = 0;
        for (int i = 0; i < t->numacc; i++)
            for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++, comp++) {
                int afe = (int)t->acc_afe[i][e];
                for (int q = 0; q < t->afe_barnz[afe]; q++) {
                    int m = t->afe_barsym[afe][q];
                    for (int s = 0; s < t->sym_nnz[m]; s++) {
                        acc_afe[w] = comp;
                        bar_var[w] = t->afe_baridx[afe][q];
                        blk_row[w] = t->sym_subi[m][s];
                        blk_col[w] = t->sym_subj[m][s];
                        blk_val[w] = t->afe_barcoef[afe][q] * t->sym_val[m][s];
                        w++;
                    }
                }
            }
    }
    *numtrip = need;
    return PRIMAL_RES_OK;
}

/* Add the bar terms of AFE `afe` to row `row` (via putbaraij). */
static PRIMALrescodee afe_add_bar_terms(PRIMALtask_t t, int row, int afe) {
    for (int e = 0; e < t->afe_barnz[afe]; e++) {
        int j = t->afe_baridx[afe][e], m = t->afe_barsym[afe][e];
        double w = t->afe_barcoef[afe][e];
        PRIMALrescodee rc = PRIMAL_putbaraij(t, row, j, 1, &m, &w);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* Clear row i of F for every listed AFE. */
PRIMALrescodee PRIMAL_emptyafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                       const PRIMALint64t *afeidx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numafeidx < 0 || (numafeidx > 0 && !afeidx)) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numafeidx; k++) {
        PRIMALrescodee rc = PRIMAL_emptyafefrow(t, afeidx[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Clear column varidx of F for every listed variable. */
PRIMALrescodee PRIMAL_emptyafefcollist(PRIMALtask_t t, PRIMALint64t numvaridx,
                                       const int *varidx) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numvaridx < 0 || (numvaridx > 0 && !varidx)) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numvaridx; k++) {
        PRIMALrescodee rc = PRIMAL_emptyafefcol(t, varidx[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* putafefcol: clears column varidx of F and writes the given entries into it. */
PRIMALrescodee PRIMAL_putafefcol(PRIMALtask_t t, int varidx, PRIMALint64t numnz,
                                 const PRIMALint64t *afeidx, const PRIMALrealt *val) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (varidx < 0 || varidx >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (numnz < 0 || (numnz > 0 && (!afeidx || !val))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numnz; k++) {
        if (afeidx[k] < 0 || afeidx[k] >= t->numafe) return PRIMAL_RES_ERR_ARG;
        if (val[k] != val[k]) return PRIMAL_RES_ERR_ARG;
    }
    PRIMALrescodee rc = PRIMAL_emptyafefcol(t, varidx);
    if (rc != PRIMAL_RES_OK) return rc;
    for (PRIMALint64t k = 0; k < numnz; k++) {
        rc = PRIMAL_putafefentry(t, afeidx[k], varidx, val[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* ---- conic domains (reference append*domain / getdomaintype / getdomainn) ----
 * A domain is (type, dimension n, parameter). It is the shape an affine conic
 * constraint (ACC) is a member of. Domain type values are MSKdomaintypee's. */
static PRIMALrescodee append_domain(PRIMALtask_t t, int type, PRIMALint64t n,
                                    double param, PRIMALint64t *domidx) {
    if (!t || !domidx) return PRIMAL_RES_ERR_NULL;
    if (n < 0 || n > INT_MAX) return PRIMAL_RES_ERR_ARG;
    if (t->numdomain >= t->domcap) {
        int nc = t->domcap ? t->domcap * 2 : 4;
        int *a1 = (int *)realloc(t->dom_type, (size_t)nc * sizeof(int));
        PRIMALint64t *a2 = (PRIMALint64t *)realloc(t->dom_n, (size_t)nc * sizeof(PRIMALint64t));
        double *a3 = (double *)realloc(t->dom_param, (size_t)nc * sizeof(double));
        char **a4 = (char **)realloc(t->domname, (size_t)nc * sizeof(char *));
        if (!a1 || !a2 || !a3 || !a4) { free(a1); free(a2); free(a3); free(a4); return PRIMAL_RES_ERR_ALLOC; }
        t->dom_type = a1; t->dom_n = a2; t->dom_param = a3; t->domname = a4; t->domcap = nc;
    }
    int k = t->numdomain++;
    t->dom_type[k] = type; t->dom_n[k] = n; t->dom_param[k] = param;
    t->domname[k] = NULL;
    *domidx = k;
    return PRIMAL_RES_OK;
}

/* Append domain R of dimension n. */
PRIMALrescodee PRIMAL_appendrdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_R, n, 0.0, domidx);
}
/* Append domain RZERO of dimension n. */
PRIMALrescodee PRIMAL_appendrzerodomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_RZERO, n, 0.0, domidx);
}
/* Append domain RPLUS of dimension n. */
PRIMALrescodee PRIMAL_appendrplusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_RPLUS, n, 0.0, domidx);
}
/* Append domain RMINUS of dimension n. */
PRIMALrescodee PRIMAL_appendrminusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_RMINUS, n, 0.0, domidx);
}
/* Append a quadratic-cone domain of dimension n. */
PRIMALrescodee PRIMAL_appendquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_QUADRATIC_CONE, n, 0.0, domidx);
}
/* Append a rotated quadratic-cone domain of dimension n. */
PRIMALrescodee PRIMAL_appendrquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_RQUADRATIC_CONE, n, 0.0, domidx);
}
/* Append a primal exponential-cone domain (3 components). */
PRIMALrescodee PRIMAL_appendprimalexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_PRIMAL_EXP_CONE, 3, 0.0, domidx);
}
/* Append a dual exponential-cone domain (3 components). */
PRIMALrescodee PRIMAL_appenddualexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_DUAL_EXP_CONE, 3, 0.0, domidx);
}
/* Append a primal power-cone domain of dimension n and exponent alpha. */
PRIMALrescodee PRIMAL_appendprimalpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_PRIMAL_POWER_CONE, n, alpha, domidx);
}
/* Append a dual power-cone domain of dimension n and exponent alpha. */
PRIMALrescodee PRIMAL_appenddualpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx) {
    return append_domain(t, PRIMAL_DOMAIN_DUAL_POWER_CONE, n, alpha, domidx);
}
/* Append a svec PSD-cone domain of dimension dim(dim+1)/2. */
PRIMALrescodee PRIMAL_appendsvecpsdconedomain(PRIMALtask_t t, int dim, PRIMALint64t *domidx) {
    if (dim <= 0) return PRIMAL_RES_ERR_ARG;
    return append_domain(t, PRIMAL_DOMAIN_SVEC_PSD_CONE, (PRIMALint64t)dim * (dim + 1) / 2, 0.0, domidx);
}

/* Report the number of domains. */
PRIMALrescodee PRIMAL_getnumdomain(PRIMALtask_t t, PRIMALint64t *numdomain) {
    if (!t || !numdomain) return PRIMAL_RES_ERR_NULL;
    *numdomain = t->numdomain;
    return PRIMAL_RES_OK;
}
/* Return the type of domain domidx. */
PRIMALrescodee PRIMAL_getdomaintype(PRIMALtask_t t, PRIMALint64t domidx, PRIMALdomaintypee *domtype) {
    if (!t || !domtype) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    *domtype = (PRIMALdomaintypee)t->dom_type[domidx];
    return PRIMAL_RES_OK;
}
/* Return the dimension of domain domidx. */
PRIMALrescodee PRIMAL_getdomainn(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    *n = t->dom_n[domidx];
    return PRIMAL_RES_OK;
}
/* The geometric-mean cones (primal/dual): the type is stored, the solver does
 * not represent them (an ACC on such a domain is refused downstream like the
 * other non-linear domains). */
PRIMALrescodee PRIMAL_appendprimalgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    if (n < 2) return PRIMAL_RES_ERR_ARG;
    return append_domain(t, PRIMAL_DOMAIN_PRIMAL_GEO_MEAN_CONE, n, 0.0, domidx);
}
/* Append a dual geometric-mean-cone domain of dimension n >= 2. */
PRIMALrescodee PRIMAL_appenddualgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx) {
    if (n < 2) return PRIMAL_RES_ERR_ARG;
    return append_domain(t, PRIMAL_DOMAIN_DUAL_GEO_MEAN_CONE, n, 0.0, domidx);
}
/* domain names: the seventh table, the same name_put/name_find and the same
 * capacity domcap. `getpowerdomainalpha` reads back the power cone parameter;
 * `getpowerdomaininfo` gives (n, nleft): the dimension and how many components
 * sit on the left (2 for a three-component power). */
PRIMALrescodee PRIMAL_putdomainname(PRIMALtask_t t, PRIMALint64t domidx, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    return name_put(t->domname, t->numdomain, (int)domidx, name);
}
/* Report the length of the name of domain domidx. */
PRIMALrescodee PRIMAL_getdomainnamelen(PRIMALtask_t t, PRIMALint64t domidx, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->domname ? t->domname[domidx] : NULL);
    return PRIMAL_RES_OK;
}
/* Copy the name of domain domidx into name (sizename includes the terminator). */
PRIMALrescodee PRIMAL_getdomainname(PRIMALtask_t t, PRIMALint64t domidx, int sizename, char *name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    const char *nm = (t->domname && t->domname[domidx]) ? t->domname[domidx] : "";
    int len = (int)strlen(nm);
    if (sizename < len + 1) return PRIMAL_RES_ERR_ARG;
    memcpy(name, nm, (size_t)len + 1);
    return PRIMAL_RES_OK;
}
/* Return the exponent alpha of a power-cone domain. */
PRIMALrescodee PRIMAL_getpowerdomainalpha(PRIMALtask_t t, PRIMALint64t domidx, PRIMALrealt *alpha) {
    if (!t || !alpha) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    if (t->dom_type[domidx] != PRIMAL_DOMAIN_PRIMAL_POWER_CONE &&
        t->dom_type[domidx] != PRIMAL_DOMAIN_DUAL_POWER_CONE) return PRIMAL_RES_ERR_ARG;
    *alpha = t->dom_param[domidx];
    return PRIMAL_RES_OK;
}
/* Return the dimension and left-component count of a power-cone domain. */
PRIMALrescodee PRIMAL_getpowerdomaininfo(PRIMALtask_t t, PRIMALint64t domidx,
                                         PRIMALint64t *n, PRIMALint64t *nleft) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    if (t->dom_type[domidx] != PRIMAL_DOMAIN_PRIMAL_POWER_CONE &&
        t->dom_type[domidx] != PRIMAL_DOMAIN_DUAL_POWER_CONE) return PRIMAL_RES_ERR_ARG;
    *n = t->dom_n[domidx];
    if (nleft) *nleft = 2;   /* x0^a x1^(1-a) >= |x2|: two components on the left */
    return PRIMAL_RES_OK;
}
/* No-op capacity hint for domains. */
PRIMALrescodee PRIMAL_putmaxnumdomain(PRIMALtask_t t, PRIMALint64t maxnumdomain) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumdomain < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;   /* suggested capacity */
}

/* Append a cone with the given type, parameter and members; validates type and
 * sizes. */
PRIMALrescodee PRIMAL_appendcone(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt coneparam, int nummem, const int *submem) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!submem || nummem <= 0) return PRIMAL_RES_ERR_ARG;
    if (ct != PRIMAL_CT_QUAD && ct != PRIMAL_CT_RQUAD &&
        ct != PRIMAL_CT_PEXP && ct != PRIMAL_CT_DEXP &&
        ct != PRIMAL_CT_PPOW && ct != PRIMAL_CT_RPOW) return PRIMAL_RES_ERR_ARG;
    if ((ct == PRIMAL_CT_PEXP || ct == PRIMAL_CT_DEXP ||
         ct == PRIMAL_CT_PPOW || ct == PRIMAL_CT_RPOW) && nummem != 3)
        return PRIMAL_RES_ERR_ARG;  /* exp/power: 3 members */
    if ((ct == PRIMAL_CT_PPOW || ct == PRIMAL_CT_RPOW) &&
        !(coneparam > 0.0 && coneparam < 1.0)) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < nummem; i++)
        if (submem[i] < 0 || submem[i] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    int oldcap = t->cone_cap;
    if (t->numcones >= t->cone_cap) {
        int nc = t->cone_cap ? t->cone_cap * 2 : 4;
        int *nt = (int *)realloc(t->cone_type, (size_t)nc * sizeof(int));
        int *nm = (int *)realloc(t->cone_nmem, (size_t)nc * sizeof(int));
        int **cm = (int **)realloc(t->cone_mem, (size_t)nc * sizeof(int *));
        double *cp = (double *)realloc(t->cone_param, (size_t)nc * sizeof(double));
        if (!nt || !nm || !cm || !cp) { free(nt); free(nm); free(cm); free(cp); return PRIMAL_RES_ERR_ALLOC; }
        t->cone_type = nt; t->cone_nmem = nm; t->cone_mem = cm; t->cone_param = cp; t->cone_cap = nc;
    }
    /* The name table shares cone_cap with the other four: one capacity only,
     * not five to keep in step. The table moves, not the strings: a name taken
     * on loan before this point keeps the same address after. */
    if (t->cone_cap && (t->cone_cap != oldcap || !t->conename)) {
        int had = t->conename != NULL;
        char **cn = (char **)realloc(t->conename, (size_t)t->cone_cap * sizeof(char *));
        if (!cn) return PRIMAL_RES_ERR_ALLOC;   /* numcones not moved yet */
        t->conename = cn;
        for (int k = had ? oldcap : 0; k < t->cone_cap; k++) t->conename[k] = NULL;
    }
    int *mem = (int *)malloc((size_t)nummem * sizeof(int));
    if (!mem) return PRIMAL_RES_ERR_ALLOC;
    for (int i = 0; i < nummem; i++) mem[i] = submem[i];
    int k = t->numcones++;
    t->cone_type[k] = (int)ct;
    t->cone_param[k] = (ct == PRIMAL_CT_PPOW || ct == PRIMAL_CT_RPOW) ? coneparam : 0.0;
    t->cone_nmem[k] = nummem;
    t->cone_mem[k] = mem;
    return PRIMAL_RES_OK;
}

/* Reference appendconeseq/appendconesseq: a cone whose members are the
 * CONTIGUOUS variables j..j+nummem-1 (one cone, or several). */
PRIMALrescodee PRIMAL_appendconeseq(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt conepar,
                                    int nummem, int j) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (nummem < 0) return PRIMAL_RES_ERR_ARG;
    if (nummem == 0) return PRIMAL_appendcone(t, ct, conepar, 0, NULL);
    if (j < 0 || j + nummem > t->numvar) return PRIMAL_RES_ERR_ARG;
    int *mem = (int *)malloc((size_t)nummem * sizeof(int));
    if (!mem) return PRIMAL_RES_ERR_ALLOC;
    for (int k = 0; k < nummem; k++) mem[k] = j + k;
    PRIMALrescodee rc = PRIMAL_appendcone(t, ct, conepar, nummem, mem);
    free(mem);
    return rc;
}

/* Append several cones whose members are contiguous variables. */
PRIMALrescodee PRIMAL_appendconesseq(PRIMALtask_t t, int num, const PRIMALconetypee *ct,
    const PRIMALrealt *conepar, const int *nummem, const int *j) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!ct || !conepar || !nummem || !j))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (nummem[k] < 0) return PRIMAL_RES_ERR_ARG;
        if (nummem[k] > 0 && (j[k] < 0 || j[k] + nummem[k] > t->numvar)) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++) {
        PRIMALrescodee rc = PRIMAL_appendconeseq(t, ct[k], conepar[k], nummem[k], j[k]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

/* Report the number of cones. */
PRIMALrescodee PRIMAL_getnumcone(PRIMALtask_t t, int *numcone) {
    if (!t || !numcone) return PRIMAL_RES_ERR_NULL;
    *numcone = t->numcones;
    return PRIMAL_RES_OK;
}

/* Reference removevars: remove the variables at the given indices. c, bounds,
 * columns, type and names are compacted; the dense qcon blocks are reshaped
 * (rows/columns of removed variables dropped, stride rebuilt); the qobj triplets
 * and the cone member lists are remapped (a cone left empty is removed). */
PRIMALrescodee PRIMAL_removevars(PRIMALtask_t t, int num, const int *subset) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subset)) return PRIMAL_RES_ERR_ARG;
    for (int a = 0; a < num; a++) {
        if (subset[a] < 0 || subset[a] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        for (int b = a + 1; b < num; b++) if (subset[a] == subset[b]) return PRIMAL_RES_ERR_ARG;
    }
    int oldn = t->numvar;
    char *del = (char *)calloc((size_t)(oldn > 0 ? oldn : 1), 1);
    int *remap = (int *)malloc((size_t)(oldn > 0 ? oldn : 1) * sizeof(int));
    if (!del || !remap) { free(del); free(remap); return PRIMAL_RES_ERR_ALLOC; }
    for (int a = 0; a < num; a++) del[subset[a]] = 1;
    int w = 0;
    for (int j = 0; j < oldn; j++) {
        if (del[j]) {
            remap[j] = -1;
            free(t->varname[j]);
            if (t->cols) { free(t->cols[j].sub); free(t->cols[j].val); }
            continue;
        }
        remap[j] = w;
        t->c[w] = t->c[j];
        t->bkx[w] = t->bkx[j]; t->blx[w] = t->blx[j]; t->bux[w] = t->bux[j];
        t->vartype[w] = t->vartype[j];
        t->varname[w] = t->varname[j];
        if (t->cols) t->cols[w] = t->cols[j];
        w++;
    }
    for (int j = w; j < oldn; j++) {
        t->varname[j] = NULL;
        if (t->cols) { t->cols[j].sub = NULL; t->cols[j].val = NULL; t->cols[j].nz = 0; t->cols[j].cap = 0; }
    }
    int newn = w;
    t->numvar = newn;
    /* qcon: rebuild each row's dense block at the new stride */
    if (t->qcon) {
        for (int i = 0; i < t->qcon_cap; i++) {
            if (!t->qcon[i]) continue;
            double *old = t->qcon[i];
            size_t ns = (size_t)(newn > 0 ? newn : 1);
            double *nq = (double *)calloc(ns * ns, sizeof(double));
            if (!nq) { free(del); free(remap); return PRIMAL_RES_ERR_ALLOC; }
            for (int a = 0; a < oldn; a++) {
                if (del[a]) continue;
                for (int b = 0; b < oldn; b++) {
                    if (del[b]) continue;
                    nq[(size_t)remap[a] * newn + remap[b]] = old[(size_t)a * oldn + b];
                }
            }
            free(old); t->qcon[i] = nq;
        }
    }
    /* qobj triplets */
    if (t->qt_n > 0) {
        int wq = 0;
        for (int e = 0; e < t->qt_n; e++) {
            int a = t->qt_i[e], b = t->qt_j[e];
            if (a < 0 || a >= oldn || b < 0 || b >= oldn || del[a] || del[b]) continue;
            t->qt_i[wq] = remap[a]; t->qt_j[wq] = remap[b]; t->qt_v[wq] = t->qt_v[e]; wq++;
        }
        t->qt_n = wq;
        free(t->qobj); t->qobj = NULL;
        t->has_qobj = wq > 0;
    }
    /* cones: remap members, drop removed; an empty cone is removed */
    int wc = 0;
    for (int k = 0; k < t->numcones; k++) {
        int nk = t->cone_nmem[k], ww = 0;
        for (int m = 0; m < nk; m++) {
            int v = t->cone_mem[k][m];
            if (v >= 0 && v < oldn && !del[v]) t->cone_mem[k][ww++] = remap[v];
        }
        if (ww == 0) { free(t->cone_mem[k]); free(t->conename[k]); continue; }
        t->cone_nmem[k] = ww;
        if (wc != k) {
            t->cone_type[wc] = t->cone_type[k];
            t->cone_nmem[wc] = t->cone_nmem[k];
            t->cone_mem[wc] = t->cone_mem[k];
            t->cone_param[wc] = t->cone_param[k];
            t->conename[wc] = t->conename[k];
        }
        wc++;
    }
    for (int k = wc; k < t->numcones; k++) { t->cone_mem[k] = NULL; t->conename[k] = NULL; }
    t->numcones = wc;
    if (t->has_qcon > 0 && t->qcon) {
        t->has_qcon = 0;
        for (int i = 0; i < newn; i++) if (t->qcon[i]) {
            for (int e = 0; e < newn * newn; e++) if (t->qcon[i][e] != 0.0) { t->has_qcon++; break; }
        }
    }
    free(del); free(remap);
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/* Reference removecons: remove the constraints at the given indices and compact.
 * The A entries of removed rows are dropped from every column and the remaining
 * row indices are remapped; the qcon blocks follow the same remap. */
PRIMALrescodee PRIMAL_removecons(PRIMALtask_t t, int num, const int *subset) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subset)) return PRIMAL_RES_ERR_ARG;
    for (int a = 0; a < num; a++) {
        if (subset[a] < 0 || subset[a] >= t->numcon) return PRIMAL_RES_ERR_ARG;
        for (int b = a + 1; b < num; b++) if (subset[a] == subset[b]) return PRIMAL_RES_ERR_ARG;
    }
    int oldn = t->numcon;
    char *del = (char *)calloc((size_t)(oldn > 0 ? oldn : 1), 1);
    int *remap = (int *)malloc((size_t)(oldn > 0 ? oldn : 1) * sizeof(int));
    if (!del || !remap) { free(del); free(remap); return PRIMAL_RES_ERR_ALLOC; }
    for (int a = 0; a < num; a++) del[subset[a]] = 1;
    int w = 0;
    for (int i = 0; i < oldn; i++) {
        if (del[i]) { remap[i] = -1; free(t->conname[i]); continue; }
        remap[i] = w;
        t->bkc[w] = t->bkc[i]; t->blc[w] = t->blc[i]; t->buc[w] = t->buc[i];
        t->conname[w] = t->conname[i];
        w++;
    }
    for (int i = w; i < oldn; i++) t->conname[i] = NULL;
    t->numcon = w;
    /* A: drop/remap the row index of every column entry */
    if (t->cols) for (int j = 0; j < t->numvar; j++) {
        Col *c = &t->cols[j];
        int ww = 0;
        for (int k = 0; k < c->nz; k++) {
            int r = c->sub[k];
            if (r >= 0 && r < oldn && remap[r] >= 0) {
                c->sub[ww] = remap[r]; c->val[ww] = c->val[k]; ww++;
            }
        }
        c->nz = ww;
    }
    /* qcon rows follow the same remap */
    if (t->qcon) {
        int cap = t->qcon_cap;
        double **nq = (double **)calloc((size_t)(cap > 0 ? cap : 1), sizeof(double *));
        if (!nq) { free(del); free(remap); return PRIMAL_RES_ERR_ALLOC; }
        for (int i = 0; i < cap; i++) {
            if (i < oldn && !del[i]) nq[remap[i]] = t->qcon[i];
            else free(t->qcon[i]);
        }
        free(t->qcon); t->qcon = nq;
    }
    /* barA terms on removed constraints are dropped, the rest remapped */
    int wa = 0;
    for (int k = 0; k < t->nbarA; k++) {
        int i = t->barA_con[k];
        if (i < 0 || i >= oldn || remap[i] < 0) continue;
        t->barA_con[wa] = remap[i];
        t->barA_bar[wa] = t->barA_bar[k];
        t->barA_sym[wa] = t->barA_sym[k];
        t->barA_coef[wa] = t->barA_coef[k];
        wa++;
    }
    t->nbarA = wa;
    if (t->has_qcon > 0 && t->qcon) {
        t->has_qcon = 0;
        for (int i = 0; i < t->numcon; i++) if (t->qcon[i]) {
            for (int e = 0; e < t->numvar * t->numvar; e++)
                if (t->qcon[i][e] != 0.0) { t->has_qcon++; break; }
        }
    }
    free(del); free(remap);
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/* Reference removecones: remove the cones at the given indices and compact. */
PRIMALrescodee PRIMAL_removecones(PRIMALtask_t t, int num, const int *subset) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subset)) return PRIMAL_RES_ERR_ARG;
    for (int a = 0; a < num; a++) {
        if (subset[a] < 0 || subset[a] >= t->numcones) return PRIMAL_RES_ERR_ARG;
        for (int b = a + 1; b < num; b++) if (subset[a] == subset[b]) return PRIMAL_RES_ERR_ARG;
    }
    char *del = (char *)calloc((size_t)(t->numcones > 0 ? t->numcones : 1), 1);
    if (!del) return PRIMAL_RES_ERR_ALLOC;
    for (int a = 0; a < num; a++) del[subset[a]] = 1;
    int oldn = t->numcones, w = 0;
    for (int k = 0; k < oldn; k++) {
        if (del[k]) { free(t->cone_mem[k]); free(t->conename[k]); continue; }
        t->cone_type[w] = t->cone_type[k];
        t->cone_nmem[w] = t->cone_nmem[k];
        t->cone_mem[w] = t->cone_mem[k];
        t->cone_param[w] = t->cone_param[k];
        t->conename[w] = t->conename[k];
        w++;
    }
    for (int k = w; k < oldn; k++) { t->cone_mem[k] = NULL; t->conename[k] = NULL; }
    t->numcones = w;
    free(del);
    return PRIMAL_RES_OK;
}

/* Return the type, member count and members of cone k (each output optional). */
PRIMALrescodee PRIMAL_getcone(PRIMALtask_t t, int k, PRIMALconetypee *ct, int *nummem, int *submem) {
    if (!t || !nummem) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcones) return PRIMAL_RES_ERR_ARG;
    if (ct) *ct = (PRIMALconetypee)t->cone_type[k];
    *nummem = t->cone_nmem[k];
    if (submem)
        for (int i = 0; i < t->cone_nmem[k]; i++) submem[i] = t->cone_mem[k][i];
    return PRIMAL_RES_OK;
}
/* Return the parameter of cone k. */
PRIMALrescodee PRIMAL_getconeparam(PRIMALtask_t t, int k, PRIMALrealt *param) {
    if (!t || !param) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcones) return PRIMAL_RES_ERR_ARG;
    *param = t->cone_param[k];
    return PRIMAL_RES_OK;
}

/* Report the member count of cone k. */
PRIMALrescodee PRIMAL_getnumconemem(PRIMALtask_t t, int k, int *nummem) {
    if (!t || !nummem) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcones) return PRIMAL_RES_ERR_ARG;
    *nummem = t->cone_nmem[k];
    return PRIMAL_RES_OK;
}

/* ---- affine conic constraints: v = A'x + b in K ----
 * Realized with auxiliary variables v (free) + one equality row per term
 * (v_i - a_i'x = b_i) + the cone on the v (PRIMAL-style ACC).
 * appendaccseq (internal encoder): nz[i] entries for term i from the flat
 * vectors. */
static PRIMALrescodee accseq_encode(PRIMALtask_t t, PRIMALconetypee domtype, double domparam,
                             int numterms, const int *nz,
                             const int *aidx, const double *aval,
                             const double *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numterms <= 0 || !nz || !aidx || !aval || !b) return PRIMAL_RES_ERR_NULL;
    if (domtype != PRIMAL_CT_QUAD && domtype != PRIMAL_CT_RQUAD &&
        domtype != PRIMAL_CT_PEXP && domtype != PRIMAL_CT_DEXP &&
        domtype != PRIMAL_CT_PPOW && domtype != PRIMAL_CT_RPOW) return PRIMAL_RES_ERR_ARG;
    if ((domtype == PRIMAL_CT_PEXP || domtype == PRIMAL_CT_DEXP ||
         domtype == PRIMAL_CT_PPOW || domtype == PRIMAL_CT_RPOW) && numterms != 3)
        return PRIMAL_RES_ERR_ARG;
    if ((domtype == PRIMAL_CT_PPOW || domtype == PRIMAL_CT_RPOW) &&
        !(domparam > 0.0 && domparam < 1.0)) return PRIMAL_RES_ERR_ARG;
    int nzent = 0;
    for (int i = 0; i < numterms; i++) {
        if (nz[i] < 0) return PRIMAL_RES_ERR_ARG;
        nzent += nz[i];
    }
    /* validate the flat entries */
    for (int k = 0; k < nzent; k++) {
        if (aidx[k] < 0 || aidx[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (aval[k] != aval[k]) return PRIMAL_RES_ERR_ARG;   /* NaN */
    }
    int vbase = t->numvar;
    int rbase = t->numcon;
    PRIMALrescodee rc = PRIMAL_appendvars(t, numterms);
    if (rc != PRIMAL_RES_OK) return rc;
    rc = PRIMAL_appendcons(t, numterms);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int i = 0; i < numterms; i++)
        PRIMAL_putvarbound(t, vbase + i, PRIMAL_BK_FR, -INF, INF);
    int k = 0;
    for (int i = 0; i < numterms; i++) {
        /* row rbase+i: v_i - sum_k a_k x_{aidx_k} = b_i */
        int cap = nz[i] + 1;
        int *sub = (int *)malloc((size_t)cap * sizeof(int));
        double *val = (double *)malloc((size_t)cap * sizeof(double));
        if (!sub || !val) { free(sub); free(val); return PRIMAL_RES_ERR_ALLOC; }
        int w = 0;
        for (int q = 0; q < nz[i]; q++) {
            sub[w] = aidx[k]; val[w] = -aval[k]; w++; k++;
        }
        sub[w] = vbase + i; val[w] = 1.0; w++;
        rc = PRIMAL_putarow(t, rbase + i, w, sub, val);
        free(sub); free(val);
        if (rc != PRIMAL_RES_OK) return rc;
        PRIMAL_putconbound(t, rbase + i, PRIMAL_BK_FX, b[i], b[i]);
    }
    /* cone on the aux variables */
    int *mem = (int *)malloc((size_t)numterms * sizeof(int));
    if (!mem) return PRIMAL_RES_ERR_ALLOC;
    for (int i = 0; i < numterms; i++) mem[i] = vbase + i;
    rc = PRIMAL_appendcone(t, domtype, domparam, numterms, mem);
    free(mem);
    return rc;
}

/* ---- reference-style affine conic constraints (ACC) ----
 * appendacc(domidx, numafeidx, afeidxlist, b): the e-th component is the AFE
 * afeidxlist[e] (F_e x + g_e) plus the constant b[e], and the vector of the
 * numafeidx components belongs to domain domidx (the dimensions must match).
 * The model extends immediately: conic domains go through the internal encoder
 * accseq_encode, linear domains (R/RZERO/RPLUS/RMINUS) through rows. */
static PRIMALrescodee acc_store(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t numafeidx,
                                const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
                                PRIMALint64t rowbase) {
    if (t->numacc >= t->acccap) {
        int nc = t->acccap ? t->acccap * 2 : 4;
        PRIMALint64t *a1 = (PRIMALint64t *)realloc(t->acc_dom, (size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t *a2 = (PRIMALint64t *)realloc(t->acc_nafe, (size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t **a3 = (PRIMALint64t **)realloc(t->acc_afe, (size_t)nc * sizeof(PRIMALint64t *));
        double **a4 = (double **)realloc(t->acc_b, (size_t)nc * sizeof(double *));
        PRIMALint64t *a6 = (PRIMALint64t *)realloc(t->acc_rowbase, (size_t)nc * sizeof(PRIMALint64t));
        char **a5 = (char **)realloc(t->accname, (size_t)nc * sizeof(char *));
        if (!a1 || !a2 || !a3 || !a4 || !a5 || !a6) {
            free(a1); free(a2); free(a3); free(a4); free(a5); free(a6);
            return PRIMAL_RES_ERR_ALLOC;
        }
        t->acc_dom = a1; t->acc_nafe = a2; t->acc_afe = a3; t->acc_b = a4;
        t->acc_rowbase = a6; t->accname = a5; t->acccap = nc;
    }
    int k = t->numacc;
    t->acc_dom[k] = domidx; t->acc_nafe[k] = numafeidx;
    t->acc_rowbase[k] = rowbase;
    t->accname[k] = NULL;
    size_t nb = (size_t)(numafeidx > 0 ? numafeidx : 1);
    t->acc_afe[k] = (PRIMALint64t *)malloc(nb * sizeof(PRIMALint64t));
    t->acc_b[k] = (double *)malloc(nb * sizeof(double));
    if (!t->acc_afe[k] || !t->acc_b[k]) return PRIMAL_RES_ERR_ALLOC;
    for (int e = 0; e < (int)numafeidx; e++) {
        t->acc_afe[k][e] = afeidxlist[e];
        t->acc_b[k][e] = b ? b[e] : 0.0;
    }
    t->numacc++;
    return PRIMAL_RES_OK;
}

/* Append an affine conic constraint: the listed AFEs must lie in domain domidx. */
static void acc_sync_linear(PRIMALtask_t t) {
    for (int k = 0; k < t->numacc; k++) {
        int type = t->dom_type[t->acc_dom[k]];
        if (!(type == PRIMAL_DOMAIN_R || type == PRIMAL_DOMAIN_RZERO ||
              type == PRIMAL_DOMAIN_RPLUS || type == PRIMAL_DOMAIN_RMINUS)) continue;
        int rbase = (int)t->acc_rowbase[k];
        for (int e = 0; e < (int)t->acc_nafe[k]; e++) {
            int afe = (int)t->acc_afe[k][e];
            double g = t->afeg[afe] - t->acc_b[k][e];   /* F x + g - b */
            if (PRIMAL_putarow(t, rbase + e, t->afe_nz[afe], t->afe_sub[afe],
                               t->afe_val[afe]) != PRIMAL_RES_OK) return;
            afe_add_bar_terms(t, rbase + e, afe);
            if      (type == PRIMAL_DOMAIN_R)     PRIMAL_putconbound(t, rbase+e, PRIMAL_BK_FR, -INFINITY, INFINITY);
            else if (type == PRIMAL_DOMAIN_RZERO) PRIMAL_putconbound(t, rbase+e, PRIMAL_BK_FX, -g, -g);
            else if (type == PRIMAL_DOMAIN_RPLUS) PRIMAL_putconbound(t, rbase+e, PRIMAL_BK_LO, -g, INFINITY);
            else                                  PRIMAL_putconbound(t, rbase+e, PRIMAL_BK_UP, -INFINITY, -g);
        }
    }
}

PRIMALrescodee PRIMAL_appendacc(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t numafeidx,
                                const PRIMALint64t *afeidxlist, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (domidx < 0 || domidx >= t->numdomain) return PRIMAL_RES_ERR_ARG;
    if (numafeidx < 0 || numafeidx > INT_MAX) return PRIMAL_RES_ERR_ARG;
    if (numafeidx > 0 && !afeidxlist) return PRIMAL_RES_ERR_ARG;
    if (numafeidx != t->dom_n[domidx]) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < (int)numafeidx; e++) {
        if (afeidxlist[e] < 0 || afeidxlist[e] >= t->numafe) return PRIMAL_RES_ERR_ARG;
        if (b && b[e] != b[e]) return PRIMAL_RES_ERR_ARG;
    }
    int type = t->dom_type[domidx];
    double param = t->dom_param[domidx];
    int n = (int)numafeidx;
    /* linear domains: one row per component, no auxiliary */
    if (type == PRIMAL_DOMAIN_R || type == PRIMAL_DOMAIN_RZERO ||
        type == PRIMAL_DOMAIN_RPLUS || type == PRIMAL_DOMAIN_RMINUS) {
        int rbase = t->numcon;
        PRIMALrescodee rc = PRIMAL_appendcons(t, n);
        if (rc != PRIMAL_RES_OK) return rc;
        for (int e = 0; e < n; e++) {
            int afe = (int)afeidxlist[e];
            /* reference convention: the expression is F x + g - b */
            double g = t->afeg[afe] - (b ? b[e] : 0.0);
            rc = PRIMAL_putarow(t, rbase + e, t->afe_nz[afe], t->afe_sub[afe], t->afe_val[afe]);
            if (rc != PRIMAL_RES_OK) return rc;
            rc = afe_add_bar_terms(t, rbase + e, afe);   /* <Fbar, X> in the row */
            if (rc != PRIMAL_RES_OK) return rc;
            if (type == PRIMAL_DOMAIN_R)
                PRIMAL_putconbound(t, rbase + e, PRIMAL_BK_FR, -INFINITY, INFINITY);
            else if (type == PRIMAL_DOMAIN_RZERO)
                PRIMAL_putconbound(t, rbase + e, PRIMAL_BK_FX, -g, -g);
            else if (type == PRIMAL_DOMAIN_RPLUS)
                PRIMAL_putconbound(t, rbase + e, PRIMAL_BK_LO, -g, INFINITY);
            else
                PRIMAL_putconbound(t, rbase + e, PRIMAL_BK_UP, -INFINITY, -g);
        }
        return acc_store(t, domidx, numafeidx, afeidxlist, b, rbase);
    }
    /* conic domains: map to the internal cone type */
    PRIMALconetypee ct;
    switch (type) {
    case PRIMAL_DOMAIN_QUADRATIC_CONE:    ct = PRIMAL_CT_QUAD;  break;
    case PRIMAL_DOMAIN_RQUADRATIC_CONE:   ct = PRIMAL_CT_RQUAD; break;
    case PRIMAL_DOMAIN_PRIMAL_EXP_CONE:   ct = PRIMAL_CT_PEXP;  break;
    case PRIMAL_DOMAIN_DUAL_EXP_CONE:     ct = PRIMAL_CT_DEXP;  break;
    case PRIMAL_DOMAIN_PRIMAL_POWER_CONE: ct = PRIMAL_CT_PPOW;  break;
    default: return PRIMAL_RES_ERR_ARG;   /* dual-power, geo-mean, PSD: not represented */
    }
    int tot = 0;
    for (int e = 0; e < n; e++) tot += t->afe_nz[(int)afeidxlist[e]];
    int *nz = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    int *aidx = (int *)malloc((size_t)(tot > 0 ? tot : 1) * sizeof(int));
    double *aval = (double *)malloc((size_t)(tot > 0 ? tot : 1) * sizeof(double));
    double *bb = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!nz || !aidx || !aval || !bb) { free(nz); free(aidx); free(aval); free(bb); return PRIMAL_RES_ERR_ALLOC; }
    int w = 0;
    for (int e = 0; e < n; e++) {
        int afe = (int)afeidxlist[e];
        nz[e] = t->afe_nz[afe];
        for (int q = 0; q < nz[e]; q++) { aidx[w] = t->afe_sub[afe][q]; aval[w] = t->afe_val[afe][q]; w++; }
        bb[e] = t->afeg[afe] - (b ? b[e] : 0.0);   /* F x + g - b */
    }
    int rbase = t->numcon;
    PRIMALrescodee rc = accseq_encode(t, ct, param, n, nz, aidx, aval, bb);
    free(nz); free(aidx); free(aval); free(bb);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int e = 0; e < n; e++) {   /* <Fbar, X> in the auxiliary row */
        rc = afe_add_bar_terms(t, rbase + e, (int)afeidxlist[e]);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return acc_store(t, domidx, numafeidx, afeidxlist, b, rbase);
}

/* Report the number of ACCs. */
PRIMALrescodee PRIMAL_getnumacc(PRIMALtask_t t, PRIMALint64t *numacc) {
    if (!t || !numacc) return PRIMAL_RES_ERR_NULL;
    *numacc = t->numacc;
    return PRIMAL_RES_OK;
}
/* Report the number of components of ACC accidx. */
PRIMALrescodee PRIMAL_getaccn(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    *n = t->acc_nafe[accidx];
    return PRIMAL_RES_OK;
}
/* Return the domain of ACC accidx. */
PRIMALrescodee PRIMAL_getaccdomain(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *domidx) {
    if (!t || !domidx) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    *domidx = t->acc_dom[accidx];
    return PRIMAL_RES_OK;
}
/* Copy the AFE index list of ACC accidx. */
PRIMALrescodee PRIMAL_getaccafeidxlist(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *afeidxlist) {
    if (!t || !afeidxlist) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < (int)t->acc_nafe[accidx]; e++) afeidxlist[e] = t->acc_afe[accidx][e];
    return PRIMAL_RES_OK;
}
/* Copy the b vector of ACC accidx. */
PRIMALrescodee PRIMAL_getaccb(PRIMALtask_t t, PRIMALint64t accidx, PRIMALrealt *b) {
    if (!t || !b) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    for (int e = 0; e < (int)t->acc_nafe[accidx]; e++) b[e] = t->acc_b[accidx][e];
    return PRIMAL_RES_OK;
}

/* ---- ACC block surface and names (reference appendaccs/getaccs/
 * getaccntot/putaccb, and the sixth name table) ----
 * `appendaccs` loops over appendacc (which already validates); `getaccs`
 * concatenates the per-ACC lists; `getaccntot` sums the dimensions; `putaccb`
 * rewrites the b vector of an existing ACC. The names use the same
 * `name_put`/`name_find` and the same capacity `acccap`. */

PRIMALrescodee PRIMAL_appendaccs(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numaccs < 0 || numafeidx < 0) return PRIMAL_RES_ERR_ARG;
    if (numaccs == 0) return PRIMAL_RES_OK;
    if (!domidxs || (numafeidx > 0 && !afeidxlist)) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t ac = 0;
    for (PRIMALint64t i = 0; i < numaccs; i++) {
        PRIMALint64t dom = domidxs[i];
        if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
        PRIMALint64t n = t->dom_n[dom];
        if (ac + n > numafeidx) return PRIMAL_RES_ERR_ARG;
        PRIMALrescodee rc = PRIMAL_appendacc(t, dom, n, afeidxlist + ac,
                                             b ? b + ac : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
        ac += n;
    }
    if (ac != numafeidx) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/* Report the total number of ACC components. */
PRIMALrescodee PRIMAL_getaccntot(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t s = 0;
    for (int i = 0; i < t->numacc; i++) s += t->acc_nafe[i];
    *n = s;
    return PRIMAL_RES_OK;
}

/* Concatenate the per-ACC domain, AFE-index and b lists. */
PRIMALrescodee PRIMAL_getaccs(PRIMALtask_t t, PRIMALint64t *domidxlist,
                              PRIMALint64t *afeidxlist, PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t dc = 0, ac = 0;
    for (int i = 0; i < t->numacc; i++) {
        if (domidxlist) domidxlist[dc++] = t->acc_dom[i];
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++) {
            if (afeidxlist) afeidxlist[ac] = t->acc_afe[i][e];
            if (b) b[ac] = t->acc_b[i][e];
            ac++;
        }
    }
    return PRIMAL_RES_OK;
}

/* The duals of an ACC (reference getaccdoty/getaccdotys/putaccdoty): the
 * multipliers of the rows the ACC produced, read from `t->y`. The reference's
 * convention for `doty` was not read; here it is that of our `y` (declared
 * deviation), and `getaccdotys` concatenates them. */
PRIMALrescodee PRIMAL_getaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 PRIMALrealt *doty) {
    (void)which;
    if (!t || !doty) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t base = t->acc_rowbase[accidx];
    for (PRIMALint64t e = 0; e < t->acc_nafe[accidx]; e++) doty[e] = t->y[base + e];
    return PRIMAL_RES_OK;
}
/* Concatenate the ACC row multipliers from the published y. */
PRIMALrescodee PRIMAL_getaccdotys(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *doty) {
    (void)which;
    if (!t || !doty) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t w = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++) doty[w++] = t->y[t->acc_rowbase[i] + e];
    return PRIMAL_RES_OK;
}
/* Write the row multipliers of ACC accidx into y. */
PRIMALrescodee PRIMAL_putaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 const PRIMALrealt *doty) {
    (void)which;
    if (!t || !doty) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    if (!t->y) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t base = t->acc_rowbase[accidx];
    for (PRIMALint64t e = 0; e < t->acc_nafe[accidx]; e++) t->y[base + e] = doty[e];
    return PRIMAL_RES_OK;
}

/* Rewrite the b vector of ACC accidx. */
PRIMALrescodee PRIMAL_putaccb(PRIMALtask_t t, PRIMALint64t accidx,
                              PRIMALint64t lengthb, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    if (lengthb != t->acc_nafe[accidx]) return PRIMAL_RES_ERR_ARG;
    if (lengthb > 0 && !b) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t e = 0; e < lengthb; e++) {
        if (b[e] != b[e]) return PRIMAL_RES_ERR_ARG;
        t->acc_b[accidx][e] = b[e];
    }
    return PRIMAL_RES_OK;
}

/* Set the name of ACC accidx (shared name table). */
PRIMALrescodee PRIMAL_putaccname(PRIMALtask_t t, PRIMALint64t accidx, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    return name_put(t->accname, t->numacc, (int)accidx, name);
}

/* Report the length of the name of ACC accidx. */
PRIMALrescodee PRIMAL_getaccnamelen(PRIMALtask_t t, PRIMALint64t accidx, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->accname ? t->accname[accidx] : NULL);
    return PRIMAL_RES_OK;
}

/* Copy the name of ACC accidx into name (sizename includes the terminator). */
PRIMALrescodee PRIMAL_getaccname(PRIMALtask_t t, PRIMALint64t accidx,
                                 int sizename, char *name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    const char *nm = (t->accname && t->accname[accidx]) ? t->accname[accidx] : "";
    int len = (int)strlen(nm);
    if (sizename < len + 1) return PRIMAL_RES_ERR_ARG;
    memcpy(name, nm, (size_t)len + 1);
    return PRIMAL_RES_OK;
}

/* ---- ACCs with contiguous AFEs, activity and capacity hints ----
 * `appendaccseq(domidx, numafeidx, afeidxfirst, b)` is `appendacc` with the
 * AFEs consecutive starting from `afeidxfirst`; `appendaccsseq` appends
 * numaccs of them. `evaluateacc` gives the activity `F x + g - b` of an ACC at
 * the published point (component by component); `evaluateaccs` concatenates
 * them. The `putmaxnum*` are no-ops. */
PRIMALrescodee PRIMAL_appendaccseq(PRIMALtask_t t, PRIMALint64t domidx,
                                   PRIMALint64t numafeidx, PRIMALint64t afeidxfirst,
                                   const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numafeidx < 0 || afeidxfirst < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t *list = (PRIMALint64t *)malloc((size_t)(numafeidx > 0 ? numafeidx : 1) * sizeof(PRIMALint64t));
    if (!list) return PRIMAL_RES_ERR_ALLOC;
    for (PRIMALint64t e = 0; e < numafeidx; e++) list[e] = afeidxfirst + e;
    PRIMALrescodee rc = PRIMAL_appendacc(t, domidx, numafeidx, list, b);
    free(list);
    return rc;
}
/* Append several contiguous-AFE ACCs in one call. */
PRIMALrescodee PRIMAL_appendaccsseq(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        PRIMALint64t afeidxfirst, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numaccs < 0 || numafeidx < 0 || afeidxfirst < 0) return PRIMAL_RES_ERR_ARG;
    if (numaccs == 0) return PRIMAL_RES_OK;
    if (!domidxs) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t ac = 0;
    for (PRIMALint64t i = 0; i < numaccs; i++) {
        PRIMALint64t dom = domidxs[i];
        if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
        PRIMALint64t n = t->dom_n[dom];
        if (ac + n > numafeidx) return PRIMAL_RES_ERR_ARG;
        PRIMALrescodee rc = PRIMAL_appendaccseq(t, dom, n, afeidxfirst + ac,
                                                b ? b + ac : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
        ac += n;
    }
    if (ac != numafeidx) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/* Evaluate one ACC's activity F x + g - b at the published point. */
PRIMALrescodee PRIMAL_evaluateacc(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                  PRIMALrealt *activity) {
    (void)which;
    if (!t || !activity) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < t->acc_nafe[accidx]; e++) {
        PRIMALint64t afe = t->acc_afe[accidx][e];
        double v = t->afeg[afe] - t->acc_b[accidx][e];
        for (int q = 0; q < t->afe_nz[afe]; q++)
            v += t->afe_val[afe][q] * t->x[t->afe_sub[afe][q]];
        activity[e] = v;
    }
    return PRIMAL_RES_OK;
}
/* Evaluate every ACC's activity, concatenated. */
PRIMALrescodee PRIMAL_evaluateaccs(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *activity) {
    (void)which;
    if (!t || !activity) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t w = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++) {
            PRIMALint64t afe = t->acc_afe[i][e];
            double v = t->afeg[afe] - t->acc_b[i][e];
            for (int q = 0; q < t->afe_nz[afe]; q++)
                v += t->afe_val[afe][q] * t->x[t->afe_sub[afe][q]];
            activity[w++] = v;
        }
    return PRIMAL_RES_OK;
}
/* No-op capacity hint for ACCs. */
PRIMALrescodee PRIMAL_putmaxnumacc(PRIMALtask_t t, PRIMALint64t maxnumacc) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumacc < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/* No-op capacity hint for AFEs. */
PRIMALrescodee PRIMAL_putmaxnumafe(PRIMALtask_t t, PRIMALint64t maxnumafe) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumafe < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/* No-op capacity hint for DJCs. */
PRIMALrescodee PRIMAL_putmaxnumdjc(PRIMALtask_t t, PRIMALint64t maxnumdjc) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumdjc < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/* One component of the b vector of an existing ACC. */
PRIMALrescodee PRIMAL_putaccbj(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t j, PRIMALrealt bj) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (accidx < 0 || accidx >= t->numacc) return PRIMAL_RES_ERR_ARG;
    if (j < 0 || j >= t->acc_nafe[accidx]) return PRIMAL_RES_ERR_ARG;
    if (bj != bj) return PRIMAL_RES_ERR_ARG;
    t->acc_b[accidx][j] = bj;
    acc_sync_linear(t);
    return PRIMAL_RES_OK;
}

/* The internal cone type of a domain, or -1 if not representable as a cone
 * (R and the linear domains are handled by the caller; dual-power/geo-mean/PSD
 * have no internal cone). */
static int domain_cone_kind(int domtype) {
    switch (domtype) {
    case PRIMAL_DOMAIN_QUADRATIC_CONE:    return PRIMAL_CT_QUAD;
    case PRIMAL_DOMAIN_RQUADRATIC_CONE:   return PRIMAL_CT_RQUAD;
    case PRIMAL_DOMAIN_PRIMAL_EXP_CONE:   return PRIMAL_CT_PEXP;
    case PRIMAL_DOMAIN_DUAL_EXP_CONE:     return PRIMAL_CT_DEXP;
    case PRIMAL_DOMAIN_PRIMAL_POWER_CONE: return PRIMAL_CT_PPOW;
    default: return -1;
    }
}

/* Primal violation of a set of ACCs (reference getpviolacc): the activity
 * v = F x + g - b must lie in the domain; for a linear domain the violation is
 * R 0, RZERO max|v|, RPLUS max(0,-v), RMINUS max(0,v); for a conic domain it is
 * max(0, -cone_signed_slack). It reads the published point and the current
 * model. */
PRIMALrescodee PRIMAL_getpviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol) {
    (void)which;
    if (!t || !viol) return PRIMAL_RES_ERR_NULL;
    if (numaccidx < 0 || (numaccidx > 0 && !accidxlist)) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t maxn = 1;
    for (int i = 0; i < t->numacc; i++) if (t->acc_nafe[i] > maxn) maxn = t->acc_nafe[i];
    double *v = (double *)malloc((size_t)maxn * sizeof(double));
    if (!v) return PRIMAL_RES_ERR_ALLOC;
    for (PRIMALint64t k = 0; k < numaccidx; k++) {
        PRIMALint64t a = accidxlist[k];
        if (a < 0 || a >= t->numacc) { free(v); return PRIMAL_RES_ERR_ARG; }
        PRIMALint64t n = t->acc_nafe[a];
        for (PRIMALint64t e = 0; e < n; e++) {
            PRIMALint64t afe = t->acc_afe[a][e];
            double val = t->afeg[afe] - t->acc_b[a][e];
            for (int q = 0; q < t->afe_nz[afe]; q++)
                val += t->afe_val[afe][q] * t->x[t->afe_sub[afe][q]];
            v[e] = val;
        }
        int ty = t->dom_type[t->acc_dom[a]];
        double worst = 0.0;
        if (ty == PRIMAL_DOMAIN_R) {
            worst = 0.0;
        } else if (ty == PRIMAL_DOMAIN_RZERO || ty == PRIMAL_DOMAIN_RPLUS ||
                   ty == PRIMAL_DOMAIN_RMINUS) {
            for (PRIMALint64t e = 0; e < n; e++) {
                double vv = 0.0;
                if (ty == PRIMAL_DOMAIN_RZERO) vv = fabs(v[e]);
                else if (ty == PRIMAL_DOMAIN_RPLUS) vv = v[e] < 0.0 ? -v[e] : 0.0;
                else vv = v[e] > 0.0 ? v[e] : 0.0;
                if (vv > worst) worst = vv;
            }
        } else {
            int ct = domain_cone_kind(ty);
            if (ct < 0) { free(v); return PRIMAL_RES_ERR_ARG; }   /* domain not representable */
            double sl = cone_signed_slack(ct, t->dom_param[t->acc_dom[a]], v, (int)n);
            if (isfinite(sl) && sl < 0.0) worst = -sl;
        }
        viol[k] = worst;
    }
    free(v);
    return PRIMAL_RES_OK;
}

/* Dual violation of a set of ACCs (reference getdviolacc): the vector `doty`
 * must lie in the dual of the domain. For a linear domain: R has dual {0}
 * (violation max|doty|), RZERO has dual R (none), RPLUS max(0,-doty), RMINUS
 * max(0,doty); for a conic domain it is max(0,-cone_dual_signed_slack). */
PRIMALrescodee PRIMAL_getdviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol) {
    (void)which;
    if (!t || !viol) return PRIMAL_RES_ERR_NULL;
    if (numaccidx < 0 || (numaccidx > 0 && !accidxlist)) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t maxn = 1;
    for (int i = 0; i < t->numacc; i++) if (t->acc_nafe[i] > maxn) maxn = t->acc_nafe[i];
    double *d = (double *)malloc((size_t)maxn * sizeof(double));
    if (!d) return PRIMAL_RES_ERR_ALLOC;
    for (PRIMALint64t k = 0; k < numaccidx; k++) {
        PRIMALint64t a = accidxlist[k];
        if (a < 0 || a >= t->numacc) { free(d); return PRIMAL_RES_ERR_ARG; }
        PRIMALint64t n = t->acc_nafe[a];
        PRIMALint64t base = t->acc_rowbase[a];
        for (PRIMALint64t e = 0; e < n; e++) d[e] = t->y[base + e];
        int ty = t->dom_type[t->acc_dom[a]];
        double worst = 0.0;
        if (ty == PRIMAL_DOMAIN_R) {
            for (PRIMALint64t e = 0; e < n; e++) if (fabs(d[e]) > worst) worst = fabs(d[e]);
        } else if (ty == PRIMAL_DOMAIN_RZERO) {
            worst = 0.0;
        } else if (ty == PRIMAL_DOMAIN_RPLUS) {
            for (PRIMALint64t e = 0; e < n; e++) if (-d[e] > worst) worst = -d[e];
        } else if (ty == PRIMAL_DOMAIN_RMINUS) {
            for (PRIMALint64t e = 0; e < n; e++) if (d[e] > worst) worst = d[e];
        } else {
            int ct = domain_cone_kind(ty);
            if (ct < 0) { free(d); return PRIMAL_RES_ERR_ARG; }
            double sl = cone_dual_signed_slack(ct, t->dom_param[t->acc_dom[a]], d, (int)n);
            if (isfinite(sl) && sl < 0.0) worst = -sl;
        }
        viol[k] = worst;
    }
    free(d);
    return PRIMAL_RES_OK;
}

/* Power-domain sequences (appendprimal/dualpowerconedomainseq): num domains
 * with dimensions n[k], nleft[k] and exponents alpha[k]. `nleft` is not stored
 * (this solver represents the three-component power). */
PRIMALrescodee PRIMAL_appendprimalpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)nleft;
    if (num < 0 || (num > 0 && (!n || !alpha))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < num; k++) {
        PRIMALint64t idx = -1;
        PRIMALrescodee rc = PRIMAL_appendprimalpowerconedomain(t, n[k], alpha[k], &idx);
        if (rc != PRIMAL_RES_OK) return rc;
        if (domidxlist) domidxlist[k] = idx;
    }
    return PRIMAL_RES_OK;
}
/* Append a sequence of dual power-cone domains. */
PRIMALrescodee PRIMAL_appenddualpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)nleft;
    if (num < 0 || (num > 0 && (!n || !alpha))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < num; k++) {
        PRIMALint64t idx = -1;
        PRIMALrescodee rc = PRIMAL_appenddualpowerconedomain(t, n[k], alpha[k], &idx);
        if (rc != PRIMAL_RES_OK) return rc;
        if (domidxlist) domidxlist[k] = idx;
    }
    return PRIMAL_RES_OK;
}

/* The reference's "implicit" ACC reads, i.e. the F and g that the order of the
 * AFEs inside the ACCs implies (the AFEs are a separate store, the ACCs name
 * them). `getaccfnumnz` counts the nonzeros, `getaccftrip` gives F in triplets
 * (component, variable, value) and `getaccgvector` the vector of the constants
 * `g_afe - b` used inside the ACCs, in order. */
PRIMALrescodee PRIMAL_getaccfnumnz(PRIMALtask_t t, PRIMALint64t *accfnnz) {
    if (!t || !accfnnz) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t s = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++) s += t->afe_nz[t->acc_afe[i][e]];
    *accfnnz = s;
    return PRIMAL_RES_OK;
}
/* Write the constants g_afe - b used inside the ACCs, in order. */
PRIMALrescodee PRIMAL_getaccgvector(PRIMALtask_t t, PRIMALrealt *g) {
    if (!t || !g) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t w = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++)
            g[w++] = t->afeg[t->acc_afe[i][e]] - t->acc_b[i][e];
    return PRIMAL_RES_OK;
}
/* Write the implied F in triplets (component, variable, value). */
PRIMALrescodee PRIMAL_getaccftrip(PRIMALtask_t t, PRIMALint64t *frow,
                                  int *fcol, PRIMALrealt *fval) {
    if (!t || !frow || !fcol || !fval) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t w = 0, comp = 0;
    for (int i = 0; i < t->numacc; i++)
        for (PRIMALint64t e = 0; e < t->acc_nafe[i]; e++, comp++) {
            PRIMALint64t afe = t->acc_afe[i][e];
            for (int q = 0; q < t->afe_nz[afe]; q++) {
                frow[w] = comp;
                fcol[w] = t->afe_sub[afe][q];
                fval[w] = t->afe_val[afe][q];
                w++;
            }
        }
    return PRIMAL_RES_OK;
}

