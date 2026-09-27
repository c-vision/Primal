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
/* primal_djc.c - disjunctive constraints (reference-style DJC over big-M).
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* ---- disjunctive constraints: OR of linear systems, via big-M MIP ----
 * Internal encoder: for each disjunction d (nrow_d rows sum_j a_ij x_j <= b_i)
 * the binary z_d is introduced together with the rows a_ij'x + M*z_d <= b_i + M,
 * plus the selection sum_d z_d >= 1. A disjunction with 0 rows is always
 * satisfied (a clause with no constrained component): its binary stays in the
 * selection sum, which is satisfiable, so the constraint adds nothing.
 * With free x the big-M holds only with finite user bounds (documented
 * deviation). It is no longer public API: it is the backend of PRIMAL_putdjc. */
#define DJC_BIGM 1e6
/* Backend of PRIMAL_putdjc: extends the model with a selection binary per
 * clause and the big-M rows that enforce each clause's linear domains. */
static PRIMALrescodee djc_encode(PRIMALtask_t t, int ndis, const int *disj_start,
                          const int *rows_per_disj, const int *ncoef,
                          const int *varidx, const double *rowcoefs,
                          const double *rhs) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (ndis <= 0 || !disj_start || !rows_per_disj || !ncoef) return PRIMAL_RES_ERR_NULL;
    int nz_all = 0, nrow_all = 0;
    for (int d = 0; d < ndis; d++) {
        if (rows_per_disj[d] < 0) return PRIMAL_RES_ERR_ARG;
        if (disj_start[d] < 0) return PRIMAL_RES_ERR_ARG;
        nrow_all += rows_per_disj[d];
    }
    /* ncoef/rhs are flat per global row (nrow_all entries); disj_start[d]
     * points at the first row of disjunction d and must be consistent with
     * the sequential layout (checked at the end) */
    {
        int acc = 0;
        for (int d = 0; d < ndis; d++) {
            if (disj_start[d] != acc) return PRIMAL_RES_ERR_ARG;
            acc += rows_per_disj[d];
        }
    }
    for (int i = 0; i < nrow_all; i++) {
        if (ncoef[i] < 0) return PRIMAL_RES_ERR_ARG;
        nz_all += ncoef[i];
    }
    if ((nz_all > 0 && (!varidx || !rowcoefs)) ||
        (nrow_all > 0 && !rhs)) return PRIMAL_RES_ERR_NULL;
    /* flat layout: row i of disjunction d uses ncoef[row_global] entries
     * from the flat arrays varidx/rowcoefs, sequentially across ALL rows */
    int ridx = 0;
    int cidx = 0;
    for (int d = 0; d < ndis; d++)
        for (int i = 0; i < rows_per_disj[d]; i++) {
            int n = ncoef[ridx];
            for (int k = 0; k < n; k++) {
                if (varidx[cidx] < 0 || varidx[cidx] >= t->numvar)
                    return PRIMAL_RES_ERR_ARG;
                if (rowcoefs[cidx] != rowcoefs[cidx]) return PRIMAL_RES_ERR_ARG;
                cidx++;
            }
            ridx++;
        }
    /* 1) binaries z_d */
    int zbase = t->numvar;
    PRIMALrescodee rc = PRIMAL_appendvars(t, ndis);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int d = 0; d < ndis; d++) {
        PRIMAL_putvarbound(t, zbase + d, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, zbase + d, PRIMAL_VAR_TYPE_INT_BIN);
    }
    /* 2) big-M rows: one per (d, i) */
    int rbase = t->numcon;
    rc = PRIMAL_appendcons(t, nrow_all + 1);
    if (rc != PRIMAL_RES_OK) return rc;
    ridx = 0; cidx = 0;
    int row = rbase;
    for (int d = 0; d < ndis; d++)
        for (int i = 0; i < rows_per_disj[d]; i++) {
            int n = ncoef[ridx];
            /* cap: max(1, n)+1 entries (n coefs + z_d) */
            int cap = n + 1;
            int *sub = (int *)malloc((size_t)cap * sizeof(int));
            double *val = (double *)malloc((size_t)cap * sizeof(double));
            if (!sub || !val) { free(sub); free(val); return PRIMAL_RES_ERR_ALLOC; }
            int w = 0;
            for (int k = 0; k < n; k++) {
                sub[w] = varidx[cidx];
                val[w] = rowcoefs[cidx];
                w++; cidx++;
            }
            sub[w] = zbase + d;
            /* Tight big-M from the user bounds when they are finite: a
             * constant 1e6 on a row whose variables live in [0,1] makes the
             * matrix coefficients 1e6 and destroys the conic IPM's scaling
             * (it then loses the relaxation).  M = max over the box of the
             * row's left-hand side minus its rhs, or the global fallback when
             * some variable is unbounded in the growing direction. */
            {
                double lhsmax = 0.0; int finite = 1;
                for (int k = 0; k < n; k++) {
                    int j = varidx[cidx - n + k];
                    double a = rowcoefs[cidx - n + k];
                    double lb = t->blx[j], ub = t->bux[j];
                    double tmax = (a >= 0.0) ? a * ub : a * lb;
                    if (!isfinite(tmax)) { finite = 0; break; }
                    lhsmax += tmax;
                }
                double M;
                if (!finite) M = DJC_BIGM;
                else { M = lhsmax - rhs[ridx]; if (M < 0.0) M = 0.0; }
                val[w] = M;
                rc = PRIMAL_putarow(t, row, w + 1, sub, val);
                free(sub); free(val);
                if (rc != PRIMAL_RES_OK) return rc;
                PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INF, rhs[ridx] + M);
            }
            row++;
            ridx++;
        }
    /* 3) selection: sum z >= 1 */
    {
        int *sub = (int *)malloc((size_t)ndis * sizeof(int));
        double *val = (double *)malloc((size_t)ndis * sizeof(double));
        if (!sub || !val) { free(sub); free(val); return PRIMAL_RES_ERR_ALLOC; }
        for (int d = 0; d < ndis; d++) { sub[d] = zbase + d; val[d] = 1.0; }
        rc = PRIMAL_putarow(t, row, ndis, sub, val);
        free(sub); free(val);
        if (rc != PRIMAL_RES_OK) return rc;
        PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 1.0, INF);
    }
    return PRIMAL_RES_OK;
}

/* ---- reference-style disjunctive constraints (DJC) ----
 * putdjc(djcidx, numdomidx, domidxlist, numafeidx, afeidxlist, b, numterms,
 * termsizelist): the djcidx-th clause is the OR of numterms terms, and term i is
 * the conjunction of termsizelist[i] domains applied to affine expressions.
 * domidxlist concatenates the domains of all terms (its length is
 * sum termsizelist); afeidxlist concatenates the expressions, one per domain
 * component (length = sum of the domain dimensions). b, optional, is the
 * constant SUBTRACTED from every expression, the reference convention:
 * F x + g - b.
 * The model extends immediately, as for an ACC: the terms become big-M clauses
 * (a selection binary per term + rows <= relaxed by M), so a djcidx already
 * written is not rewritable (ERR_ARG): emitted rows cannot be withdrawn.
 * appenddjcs pre-allocates EMPTY slots (numterm == 0) and putdjc fills them.
 * Declared deviation: the backend represents only LINEAR domains
 * (R/RZERO/RPLUS/RMINUS); a conic domain in a DJC is refused with ERR_ARG (no
 * conic MIP in this solver), and the big-M M = 1e6 holds only with finite user
 * bounds, as for every MIP of this solver. */

/* Complete validation with no side effects. */
static PRIMALrescodee djc_validate(PRIMALtask_t t, PRIMALint64t numdomidx,
        const PRIMALint64t *domidxlist, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
        PRIMALint64t numterms, const PRIMALint64t *termsizelist) {
    if (numdomidx < 0 || numafeidx < 0 || numterms < 1) return PRIMAL_RES_ERR_ARG;
    if ((numdomidx > 0 && !domidxlist) || (numafeidx > 0 && !afeidxlist) ||
        !termsizelist) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t tsum = 0;
    for (PRIMALint64t i = 0; i < numterms; i++) {
        if (termsizelist[i] < 0) return PRIMAL_RES_ERR_ARG;
        tsum += termsizelist[i];
    }
    if (tsum != numdomidx) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t dimsum = 0;
    for (PRIMALint64t d = 0; d < numdomidx; d++) {
        PRIMALint64t dom = domidxlist[d];
        if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
        int ty = t->dom_type[dom];
        if (ty != PRIMAL_DOMAIN_R && ty != PRIMAL_DOMAIN_RZERO &&
            ty != PRIMAL_DOMAIN_RPLUS && ty != PRIMAL_DOMAIN_RMINUS)
            return PRIMAL_RES_ERR_ARG;   /* deviation: linear domains only */
        dimsum += t->dom_n[dom];
    }
    if (dimsum != numafeidx) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < numafeidx; e++) {
        if (afeidxlist[e] < 0 || afeidxlist[e] >= t->numafe) return PRIMAL_RES_ERR_ARG;
        if (b && b[e] != b[e]) return PRIMAL_RES_ERR_ARG;
    }
    return PRIMAL_RES_OK;
}

/* Translation of the linear domains into the backend's <= rows and storage of
 * the metadata. Assumes djc_validate has already passed. */
static PRIMALrescodee djc_apply(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t numdomidx, const PRIMALint64t *domidxlist,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist,
        const PRIMALrealt *b, PRIMALint64t numterms,
        const PRIMALint64t *termsizelist) {
    int ndis = (int)numterms;
    int *rpd = (int *)calloc((size_t)ndis, sizeof(int));
    int *disj_start = (int *)malloc((size_t)ndis * sizeof(int));
    if (!rpd || !disj_start) { free(rpd); free(disj_start); return PRIMAL_RES_ERR_ALLOC; }
    /* first pass: how many rows per term and how many coefficients in total */
    int afe_cur = 0, dom_cur = 0, nrow_all = 0, nz_all = 0, st = 0;
    for (int i = 0; i < ndis; i++) {
        int nrows = 0;
        for (int q = 0; q < (int)termsizelist[i]; q++) {
            int dom = (int)domidxlist[dom_cur++];
            int ty = t->dom_type[dom];
            int n = (int)t->dom_n[dom];
            for (int c = 0; c < n; c++) {
                int a = (int)afeidxlist[afe_cur++];
                int nz = t->afe_nz[a];
                if (ty == PRIMAL_DOMAIN_R) continue;
                if (ty == PRIMAL_DOMAIN_RZERO) { nrows += 2; nz_all += 2 * nz; }
                else { nrows += 1; nz_all += nz; }
            }
        }
        disj_start[i] = st; rpd[i] = nrows; st += nrows; nrow_all += nrows;
    }
    int *ncoef = (int *)malloc((size_t)(nrow_all > 0 ? nrow_all : 1) * sizeof(int));
    double *rhs = (double *)malloc((size_t)(nrow_all > 0 ? nrow_all : 1) * sizeof(double));
    int *varidx = (int *)malloc((size_t)(nz_all > 0 ? nz_all : 1) * sizeof(int));
    double *rowcoefs = (double *)malloc((size_t)(nz_all > 0 ? nz_all : 1) * sizeof(double));
    if (!ncoef || !rhs || !varidx || !rowcoefs) {
        free(rpd); free(disj_start); free(ncoef); free(rhs); free(varidx); free(rowcoefs);
        return PRIMAL_RES_ERR_ALLOC;
    }
    /* second pass: fill. expr = F x + g - bv; RPLUS -> -expr <= 0,
     * RMINUS -> expr <= 0, RZERO -> both, R -> none. */
    afe_cur = 0; dom_cur = 0;
    int r = 0, w = 0;
    for (int i = 0; i < ndis; i++) {
        for (int q = 0; q < (int)termsizelist[i]; q++) {
            int dom = (int)domidxlist[dom_cur++];
            int ty = t->dom_type[dom];
            int n = (int)t->dom_n[dom];
            for (int c = 0; c < n; c++) {
                int a = (int)afeidxlist[afe_cur];
                double g = t->afeg[a];
                double bv = b ? b[afe_cur] : 0.0;
                afe_cur++;
                if (ty == PRIMAL_DOMAIN_R) continue;
                if (ty == PRIMAL_DOMAIN_RZERO) {
                    ncoef[r] = t->afe_nz[a];
                    for (int z = 0; z < t->afe_nz[a]; z++) {
                        varidx[w] = t->afe_sub[a][z]; rowcoefs[w] = t->afe_val[a][z]; w++;
                    }
                    rhs[r++] = bv - g;          /*  F x <= bv - g */
                    ncoef[r] = t->afe_nz[a];
                    for (int z = 0; z < t->afe_nz[a]; z++) {
                        varidx[w] = t->afe_sub[a][z]; rowcoefs[w] = -t->afe_val[a][z]; w++;
                    }
                    rhs[r++] = g - bv;          /* -F x <= g - bv */
                } else if (ty == PRIMAL_DOMAIN_RPLUS) {
                    ncoef[r] = t->afe_nz[a];
                    for (int z = 0; z < t->afe_nz[a]; z++) {
                        varidx[w] = t->afe_sub[a][z]; rowcoefs[w] = -t->afe_val[a][z]; w++;
                    }
                    rhs[r++] = g - bv;          /* -expr <= 0 */
                } else {                        /* RMINUS */
                    ncoef[r] = t->afe_nz[a];
                    for (int z = 0; z < t->afe_nz[a]; z++) {
                        varidx[w] = t->afe_sub[a][z]; rowcoefs[w] = t->afe_val[a][z]; w++;
                    }
                    rhs[r++] = bv - g;          /*  expr <= 0 */
                }
            }
        }
    }
    /* metadata: the exact description, for the getters. Allocated locally and
     * published only on success, so a failed allocation does not leave the slot
     * marked as written. */
    int k = (int)djcidx;
    size_t nbd = (size_t)(numdomidx > 0 ? numdomidx : 1);
    size_t nba = (size_t)(numafeidx > 0 ? numafeidx : 1);
    PRIMALint64t *md = (PRIMALint64t *)malloc(nbd * sizeof(PRIMALint64t));
    PRIMALint64t *ma = (PRIMALint64t *)malloc(nba * sizeof(PRIMALint64t));
    double *mb = (double *)malloc(nba * sizeof(double));
    PRIMALint64t *mt = (PRIMALint64t *)malloc((size_t)ndis * sizeof(PRIMALint64t));
    if (!md || !ma || !mb || !mt) {
        free(md); free(ma); free(mb); free(mt);
        free(rpd); free(disj_start); free(ncoef); free(rhs); free(varidx); free(rowcoefs);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (PRIMALint64t e = 0; e < numdomidx; e++) md[e] = domidxlist[e];
    for (PRIMALint64t e = 0; e < numafeidx; e++) {
        ma[e] = afeidxlist[e];
        mb[e] = b ? b[e] : 0.0;
    }
    for (int i = 0; i < ndis; i++) mt[i] = termsizelist[i];
    t->djc_ndom[k] = numdomidx;
    t->djc_nafe[k] = numafeidx;
    t->djc_dom[k] = md;
    t->djc_afe[k] = ma;
    t->djc_b[k] = mb;
    t->djc_termsize[k] = mt;
    t->djc_numterm[k] = numterms;   /* last: it is the "written" marker */
    PRIMALrescodee rc = djc_encode(t, ndis, disj_start, rpd, ncoef, varidx, rowcoefs, rhs);
    free(rpd); free(disj_start); free(ncoef); free(rhs); free(varidx); free(rowcoefs);
    return rc;
}

/* Append `num` empty DJC slots for later putdjc calls. */
PRIMALrescodee PRIMAL_appenddjcs(PRIMALtask_t t, PRIMALint64t num) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX) return PRIMAL_RES_ERR_ARG;
    if (num == 0) return PRIMAL_RES_OK;
    int want = t->numdjc + (int)num;
    if (want > t->djccap) {
        int nc = t->djccap ? t->djccap : 4;
        while (nc < want) nc *= 2;
        PRIMALint64t *n1 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t *n2 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t *n3 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t **n4 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        PRIMALint64t **n5 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        double **n6 = (double **)malloc((size_t)nc * sizeof(double *));
        PRIMALint64t **n7 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        char **n8 = (char **)malloc((size_t)nc * sizeof(char *));
        if (!n1 || !n2 || !n3 || !n4 || !n5 || !n6 || !n7 || !n8) {
            free(n1); free(n2); free(n3); free(n4); free(n5); free(n6); free(n7); free(n8);
            return PRIMAL_RES_ERR_ALLOC;
        }
        for (int i = 0; i < t->numdjc; i++) {
            n1[i] = t->djc_ndom[i]; n2[i] = t->djc_nafe[i]; n3[i] = t->djc_numterm[i];
            n4[i] = t->djc_dom[i]; n5[i] = t->djc_afe[i]; n6[i] = t->djc_b[i];
            n7[i] = t->djc_termsize[i]; n8[i] = t->djcname[i];
        }
        free(t->djc_ndom); free(t->djc_nafe); free(t->djc_numterm);
        free(t->djc_dom); free(t->djc_afe); free(t->djc_b); free(t->djc_termsize);
        free(t->djcname);
        t->djc_ndom = n1; t->djc_nafe = n2; t->djc_numterm = n3;
        t->djc_dom = n4; t->djc_afe = n5; t->djc_b = n6; t->djc_termsize = n7;
        t->djcname = n8; t->djccap = nc;
    }
    for (int i = t->numdjc; i < want; i++) {
        t->djc_ndom[i] = 0; t->djc_nafe[i] = 0; t->djc_numterm[i] = 0;
        t->djc_dom[i] = NULL; t->djc_afe[i] = NULL; t->djc_b[i] = NULL;
        t->djc_termsize[i] = NULL; t->djcname[i] = NULL;
    }
    t->numdjc = want;
    return PRIMAL_RES_OK;
}

/* Fill one DJC from its domain/AFE/term description (validated first). */
PRIMALrescodee PRIMAL_putdjc(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t numdomidx, const PRIMALint64t *domidxlist,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist,
        const PRIMALrealt *b, PRIMALint64t numterms,
        const PRIMALint64t *termsizelist) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    if (t->djc_numterm[djcidx] != 0) return PRIMAL_RES_ERR_ARG;   /* already written */
    PRIMALrescodee rc = djc_validate(t, numdomidx, domidxlist, numafeidx,
                                     afeidxlist, b, numterms, termsizelist);
    if (rc != PRIMAL_RES_OK) return rc;
    return djc_apply(t, djcidx, numdomidx, domidxlist, numafeidx, afeidxlist,
                     b, numterms, termsizelist);
}

/* putdjcslice: idxlast-idxfirst consecutive DJCs, termsindjc[i] = number of
 * terms of DJC idxfirst+i; the rest is the concatenation of the descriptions
 * (as in putdjc). Everything is validated before applying, so a refused slice
 * does not leave the model half-written. */
PRIMALrescodee PRIMAL_putdjcslice(PRIMALtask_t t, PRIMALint64t idxfirst,
        PRIMALint64t idxlast, PRIMALint64t numdomidx,
        const PRIMALint64t *domidxlist, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
        PRIMALint64t numterms, const PRIMALint64t *termsizelist,
        const PRIMALint64t *termsindjc) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (idxfirst < 0 || idxlast < idxfirst || idxlast > t->numdjc) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t L = idxlast - idxfirst;
    if (L == 0) return PRIMAL_RES_OK;
    if (!termsindjc) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t tsum = 0;
    for (PRIMALint64t i = 0; i < L; i++) {
        if (termsindjc[i] < 1) return PRIMAL_RES_ERR_ARG;
        tsum += termsindjc[i];
    }
    if (tsum != numterms) return PRIMAL_RES_ERR_ARG;
    if ((numdomidx > 0 && !domidxlist) || (numafeidx > 0 && !afeidxlist))
        return PRIMAL_RES_ERR_NULL;
    /* first half: no DJC of the slice may already be written */
    for (PRIMALint64t i = 0; i < L; i++)
        if (t->djc_numterm[idxfirst + i] != 0) return PRIMAL_RES_ERR_ARG;
    /* second half: validate every sub-DJC without applying. The sub-pointers
     * are passed NULL when the sub-list is empty, to avoid arithmetic on a NULL
     * pointer. */
    PRIMALint64t tc = 0, dc = 0, ac = 0;
    for (PRIMALint64t i = 0; i < L; i++) {
        PRIMALint64t nt = termsindjc[i];
        PRIMALint64t nd = 0, na = 0;
        for (PRIMALint64t q = 0; q < nt; q++) {
            if (termsizelist[tc + q] < 0) return PRIMAL_RES_ERR_ARG;
            nd += termsizelist[tc + q];
        }
        for (PRIMALint64t d = 0; d < nd; d++) {
            PRIMALint64t dom = domidxlist[dc + d];
            if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
            int ty = t->dom_type[dom];
            if (ty != PRIMAL_DOMAIN_R && ty != PRIMAL_DOMAIN_RZERO &&
                ty != PRIMAL_DOMAIN_RPLUS && ty != PRIMAL_DOMAIN_RMINUS)
                return PRIMAL_RES_ERR_ARG;
            na += t->dom_n[dom];
        }
        const PRIMALint64t *dsub = nd > 0 ? domidxlist + dc : NULL;
        const PRIMALint64t *asub = na > 0 ? afeidxlist + ac : NULL;
        const PRIMALrealt *bsub = (b && na > 0) ? b + ac : NULL;
        PRIMALrescodee rc = djc_validate(t, nd, dsub, na, asub, bsub, nt, termsizelist + tc);
        if (rc != PRIMAL_RES_OK) return rc;
        tc += nt; dc += nd; ac += na;
    }
    if (tc != numdomidx) return PRIMAL_RES_ERR_ARG;
    if (ac != numafeidx) return PRIMAL_RES_ERR_ARG;
    /* apply */
    tc = 0; dc = 0; ac = 0;
    for (PRIMALint64t i = 0; i < L; i++) {
        PRIMALint64t nt = termsindjc[i];
        PRIMALint64t nd = 0, na = 0;
        for (PRIMALint64t q = 0; q < nt; q++) nd += termsizelist[tc + q];
        for (PRIMALint64t d = 0; d < nd; d++) na += t->dom_n[domidxlist[dc + d]];
        const PRIMALint64t *dsub = nd > 0 ? domidxlist + dc : NULL;
        const PRIMALint64t *asub = na > 0 ? afeidxlist + ac : NULL;
        const PRIMALrealt *bsub = (b && na > 0) ? b + ac : NULL;
        PRIMALrescodee rc = djc_apply(t, idxfirst + i, nd, dsub, na, asub, bsub, nt,
                                      termsizelist + tc);
        if (rc != PRIMAL_RES_OK) return rc;
        tc += nt; dc += nd; ac += na;
    }
    return PRIMAL_RES_OK;
}

/* Primal violation of a DJC (reference getpvioldjc). The violation of a
 * disjunction is min_i(max_j viol(T_ij)): the minimum over the terms of the
 * maximum over the components. For a linear domain on an affine expression
 * expr = F x + g - b: R none, RZERO |expr|, RPLUS max(0,-expr),
 * RMINUS max(0,expr). The measure reads the PUBLISHED point and the CURRENT
 * model, as getpviolcon/getpviolvar do (T113). */
PRIMALrescodee PRIMAL_getpvioldjc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numdjcidx, const PRIMALint64t *djcidxlist, PRIMALrealt *viol) {
    (void)which;
    if (!t || !viol) return PRIMAL_RES_ERR_NULL;
    if (numdjcidx < 0 || (numdjcidx > 0 && !djcidxlist)) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t k = 0; k < numdjcidx; k++) {
        PRIMALint64t d = djcidxlist[k];
        if (d < 0 || d >= t->numdjc) return PRIMAL_RES_ERR_ARG;
        PRIMALint64t domc = 0, afec = 0;
        double best = HUGE_VAL;
        for (PRIMALint64t term = 0; term < t->djc_numterm[d]; term++) {
            double worst = 0.0;
            for (PRIMALint64t q = 0; q < t->djc_termsize[d][term]; q++) {
                PRIMALint64t dom = t->djc_dom[d][domc++];
                int ty = t->dom_type[dom];
                PRIMALint64t n = t->dom_n[dom];
                for (PRIMALint64t c = 0; c < n; c++) {
                    PRIMALint64t afe = t->djc_afe[d][afec];
                    double bv = t->djc_b[d][afec];
                    afec++;
                    double expr = t->afeg[afe] - bv;
                    for (int e = 0; e < t->afe_nz[afe]; e++)
                        expr += t->afe_val[afe][e] * t->x[t->afe_sub[afe][e]];
                    double vv = 0.0;
                    if (ty == PRIMAL_DOMAIN_RZERO) vv = fabs(expr);
                    else if (ty == PRIMAL_DOMAIN_RPLUS) vv = expr < 0.0 ? -expr : 0.0;
                    else if (ty == PRIMAL_DOMAIN_RMINUS) vv = expr > 0.0 ? expr : 0.0;
                    if (vv > worst) worst = vv;
                }
            }
            if (worst < best) best = worst;
        }
        viol[k] = isfinite(best) ? best : 0.0;
    }
    return PRIMAL_RES_OK;
}

/* Report the number of DJCs. */
PRIMALrescodee PRIMAL_getnumdjc(PRIMALtask_t t, PRIMALint64t *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    *num = t->numdjc;
    return PRIMAL_RES_OK;
}
/* Report the number of domains in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumdomain(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    *n = t->djc_ndom[djcidx];
    return PRIMAL_RES_OK;
}
/* Report the number of AFEs in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumafe(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    *n = t->djc_nafe[djcidx];
    return PRIMAL_RES_OK;
}
/* Report the number of terms (clauses) in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumterm(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    *n = t->djc_numterm[djcidx];
    return PRIMAL_RES_OK;
}
/* Copy the domain index list of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcdomainidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
                                          PRIMALint64t *domidxlist) {
    if (!t || !domidxlist) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < t->djc_ndom[djcidx]; e++)
        domidxlist[e] = t->djc_dom[djcidx][e];
    return PRIMAL_RES_OK;
}
/* Copy the AFE index list of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcafeidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
                                       PRIMALint64t *afeidxlist) {
    if (!t || !afeidxlist) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < t->djc_nafe[djcidx]; e++)
        afeidxlist[e] = t->djc_afe[djcidx][e];
    return PRIMAL_RES_OK;
}
/* Copy the b vector of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcb(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALrealt *b) {
    if (!t || !b) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < t->djc_nafe[djcidx]; e++) b[e] = t->djc_b[djcidx][e];
    return PRIMAL_RES_OK;
}
/* Copy the per-term domain counts of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjctermsizelist(PRIMALtask_t t, PRIMALint64t djcidx,
                                         PRIMALint64t *termsizelist) {
    if (!t || !termsizelist) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t i = 0; i < t->djc_numterm[djcidx]; i++)
        termsizelist[i] = t->djc_termsize[djcidx][i];
    return PRIMAL_RES_OK;
}
/* Report the total number of domains over all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumdomaintot(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t s = 0;
    for (int i = 0; i < t->numdjc; i++) s += t->djc_ndom[i];
    *n = s;
    return PRIMAL_RES_OK;
}
/* Report the total number of AFEs over all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumafetot(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t s = 0;
    for (int i = 0; i < t->numdjc; i++) s += t->djc_nafe[i];
    *n = s;
    return PRIMAL_RES_OK;
}
/* Report the total number of terms over all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumtermtot(PRIMALtask_t t, PRIMALint64t *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t s = 0;
    for (int i = 0; i < t->numdjc; i++) s += t->djc_numterm[i];
    *n = s;
    return PRIMAL_RES_OK;
}
/* Bulk read of every DJC (reference getdjcs): the per-DJC lists concatenated,
 * with one `numterms` entry per DJC. A NULL buffer is skipped. */
PRIMALrescodee PRIMAL_getdjcs(PRIMALtask_t t, PRIMALint64t *domidxlist,
        PRIMALint64t *afeidxlist, PRIMALrealt *b, PRIMALint64t *termsizelist,
        PRIMALint64t *numterms) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t dc = 0, ac = 0, tc = 0;
    for (int i = 0; i < t->numdjc; i++) {
        if (numterms) numterms[i] = t->djc_numterm[i];
        if (domidxlist)
            for (PRIMALint64t e = 0; e < t->djc_ndom[i]; e++) domidxlist[dc++] = t->djc_dom[i][e];
        PRIMALint64t abase = ac;
        if (afeidxlist)
            for (PRIMALint64t e = 0; e < t->djc_nafe[i]; e++) afeidxlist[ac++] = t->djc_afe[i][e];
        else ac += t->djc_nafe[i];
        if (b)
            for (PRIMALint64t e = 0; e < t->djc_nafe[i]; e++) b[abase + e] = t->djc_b[i][e];
        if (termsizelist)
            for (PRIMALint64t e = 0; e < t->djc_numterm[i]; e++) termsizelist[tc++] = t->djc_termsize[i][e];
    }
    return PRIMAL_RES_OK;
}

/* DJC names: the fifth table, on the same two sites name_put/name_find and on
 * the same capacity djccap. Separate namespace, as for the other nameable
 * entities; the reference offers putdjcname/getdjcname/getdjcnamelen (not a
 * search by name, which indeed does not exist for DJCs). */
PRIMALrescodee PRIMAL_putdjcname(PRIMALtask_t t, PRIMALint64t djcidx, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || djcidx < 0 || djcidx >= t->numdjc || !t->djcname) return PRIMAL_RES_ERR_ARG;
    return name_put(t->djcname, t->numdjc, (int)djcidx, name);
}
/* Report the length of the name of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnamelen(PRIMALtask_t t, PRIMALint64t djcidx, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->djcname ? t->djcname[djcidx] : NULL);
    return PRIMAL_RES_OK;
}
/* Copy the name of DJC djcidx into name (sizename includes the terminator). */
PRIMALrescodee PRIMAL_getdjcname(PRIMALtask_t t, PRIMALint64t djcidx,
                                 int sizename, char *name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    const char *nm = (t->djcname && t->djcname[djcidx]) ? t->djcname[djcidx] : "";
    int len = (int)strlen(nm);
    if (sizename < len + 1) return PRIMAL_RES_ERR_ARG;   /* no room for the zero */
    memcpy(name, nm, (size_t)len + 1);
    return PRIMAL_RES_OK;
}

