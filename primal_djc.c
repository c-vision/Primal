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
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* Generated rows are refreshed before solving. Big-M is derived only from
 * explicit variable bounds and external scalar singleton rows. An unsupported
 * unbounded relaxation is an argument error, never a guessed finite box. */
static int djc_generated_row(PRIMALtask_t t, int row) {
    for (int k = 0; k < t->numdjc; k++)
        if (t->djc_numterm[k] && row >= t->djc_rowbase[k] &&
            row < t->djc_rowbase[k] + t->djc_nrow[k]) return 1;
    return 0;
}

static PRIMALrescodee djc_box(PRIMALtask_t t, double *lo, double *up) {
    int m = t->numcon;
    int *count = calloc((size_t)(m ? m : 1), sizeof(int));
    int *col = calloc((size_t)(m ? m : 1), sizeof(int));
    double *coef = calloc((size_t)(m ? m : 1), sizeof(double));
    if (!count || !col || !coef) { free(count); free(col); free(coef); return PRIMAL_RES_ERR_ALLOC; }
    for (int j = 0; j < t->numvar; j++) {
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo[j], &up[j]);
        if (t->vartype[j] == PRIMAL_VAR_TYPE_SEMI_CONT || t->vartype[j] == PRIMAL_VAR_TYPE_SEMI_INT) {
            lo[j] = fmin(lo[j], 0); up[j] = fmax(up[j], 0);
        }
        for (int e = 0; e < t->cols[j].nz; e++) {
            int r = t->cols[j].sub[e]; double a = t->cols[j].val[e];
            if (a == 0) continue;
            count[r]++; col[r] = j; coef[r] = a;
        }
    }
    for (int e = 0; e < t->nbarA; e++) count[t->barA_con[e]] = 2;
    for (int r = 0; r < m; r++) {
        if (count[r] != 1 || djc_generated_row(t,r) || (t->qcon && t->qcon[r])) continue;
        int j = col[r]; double a = coef[r], l, u;
        if (!isfinite(a)) continue;
        bound_range(t->bkc[r],t->blc[r],t->buc[r],&l,&u);
        double nl = nextafter((a > 0 ? l : u)/a, -INFINITY);
        double nu = nextafter((a > 0 ? u : l)/a, INFINITY);
        lo[j] = fmax(lo[j],nl); up[j] = fmin(up[j],nu);
    }
    free(count); free(col); free(coef);
    return PRIMAL_RES_OK;
}

static PRIMALrescodee djc_encode(PRIMALtask_t t, int k, int ndis,
        const int *rows_per_disj, const int *ncoef, const int *varidx,
        const double *rowcoefs, const double *rhs, int create) {
    int nr = 0, always = 0;
    for (int d = 0; d < ndis; d++) { nr += rows_per_disj[d]; if (!rows_per_disj[d]) always = 1; }
    if (create) {
        if (ndis > INT_MAX-t->numvar || nr >= INT_MAX-t->numcon) return PRIMAL_RES_ERR_ARG;
        int rb = t->numcon, vb = t->numvar;
        PRIMALrescodee rc = PRIMAL_appendcons(t,nr+1);
        if (rc != PRIMAL_RES_OK) return rc;
        rc = PRIMAL_appendvars(t,ndis);
        if (rc != PRIMAL_RES_OK) { t->numcon = rb; return rc; }
        for (int d = 0; d < ndis; d++) {
            PRIMAL_putvarbound(t,vb+d,PRIMAL_BK_RA,0,1);
            PRIMAL_putvartype(t,vb+d,PRIMAL_VAR_TYPE_INT_BIN);
        }
        t->djc_rowbase[k] = rb; t->djc_varbase[k] = vb; t->djc_nrow[k] = nr+1;
        return PRIMAL_RES_OK;
    }
    int rb = t->djc_rowbase[k], vb = t->djc_varbase[k];
    if (rb < 0 || nr+1 != t->djc_nrow[k] || rb > t->numcon-nr-1 || vb < 0 || vb > t->numvar-ndis)
        return PRIMAL_RES_ERR_ARG;
    for (int d = 0; d < ndis; d++) {
        double l, u; bound_range(t->bkx[vb+d],t->blx[vb+d],t->bux[vb+d],&l,&u);
        if (t->vartype[vb+d] != PRIMAL_VAR_TYPE_INT_BIN || l < 0 || u > 1) return PRIMAL_RES_ERR_ARG;
    }
    double *lo = malloc((size_t)t->numvar*sizeof(double));
    double *up = malloc((size_t)t->numvar*sizeof(double));
    double *M = calloc((size_t)(nr ? nr : 1),sizeof(double));
    int cap = ndis;
    for (int r = 0; r < nr; r++) if (ncoef[r]+1 > cap) cap = ncoef[r]+1;
    int *sub = malloc((size_t)cap*sizeof(int));
    double *val = malloc((size_t)cap*sizeof(double));
    PRIMALrescodee rc = PRIMAL_RES_ERR_ALLOC;
    if (!lo || !up || !M || !sub || !val) goto done;
    rc = djc_box(t,lo,up);
    if (rc != PRIMAL_RES_OK) goto done;
    int pos = 0;
    for (int r = 0; r < nr; r++) {
        long double lhs = 0;
        for (int e = 0; e < ncoef[r]; e++) {
            int j = varidx[pos]; double a = rowcoefs[pos++];
            if (a != 0) lhs = nextafterl(lhs + nextafterl((long double)a*(a > 0 ? up[j] : lo[j]),INFINITY), INFINITY);
        }
        if (!always && ndis > 1) {
            if (!isfinite(lhs)) { rc = PRIMAL_RES_ERR_ARG; goto done; }
            M[r] = fmax(0,nextafter((double)(lhs-(long double)rhs[r]),INFINITY));
            /* Round upward to a power of two. This keeps the relaxation
             * valid and avoids tiny coefficients/right-hand sides caused by
             * cancellation at a tight box boundary. */
            if (M[r] > 0 && isfinite(M[r])) {
                int exponent;
                frexp(fmax(1,M[r]),&exponent);
                M[r] = ldexp(1,exponent);
            }
            if (!isfinite(M[r]) || !isfinite(rhs[r]+M[r])) {
                tlog(t,"DJC requires finite relaxation bounds; supply finite variable bounds or scalar singleton rows.\n");
                rc = PRIMAL_RES_ERR_ARG; goto done;
            }
        }
    }
    pos = 0;
    int r = 0;
    for (int d = 0; d < ndis; d++) for (int e = 0; e < rows_per_disj[d]; e++, r++) {
        int n = ncoef[r];
        for (int q = 0; q < n; q++, pos++) { sub[q] = varidx[pos]; val[q] = rowcoefs[pos]; }
        sub[n] = vb+d; val[n] = M[r];
        rc = PRIMAL_putarow(t,rb+r,always ? 0 : n+1,sub,val);
        if (rc != PRIMAL_RES_OK) goto done;
        rc = PRIMAL_putconbound(t,rb+r,always ? PRIMAL_BK_FR : PRIMAL_BK_UP,-INF,rhs[r]+M[r]);
        if (rc != PRIMAL_RES_OK) goto done;
    }
    for (int d = 0; d < ndis; d++) { sub[d] = vb+d; val[d] = 1; }
    rc = PRIMAL_putarow(t,rb+nr,ndis,sub,val);
    if (rc == PRIMAL_RES_OK) rc = PRIMAL_putconbound(t,rb+nr,PRIMAL_BK_LO,1,INF);
 done:
    free(lo); free(up); free(M); free(sub); free(val);
    return rc;
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
 * conic big-M encoding here). Finite bounds are checked before solving. */

/* Complete validation with no side effects. */
static PRIMALrescodee djc_validate(PRIMALtask_t t, PRIMALint64t numdomidx,
        const PRIMALint64t *domidxlist, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
        PRIMALint64t numterms, const PRIMALint64t *termsizelist) {
    if (numdomidx < 0 || numdomidx > INT_MAX || numafeidx < 0 || numafeidx > INT_MAX/2 || numterms < 1 || numterms > INT_MAX) return PRIMAL_RES_ERR_ARG;
    if ((numdomidx > 0 && !domidxlist) || (numafeidx > 0 && !afeidxlist) ||
        !termsizelist) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t tsum = 0;
    for (PRIMALint64t i = 0; i < numterms; i++) {
        if (termsizelist[i] < 0 || termsizelist[i] > numdomidx-tsum) return PRIMAL_RES_ERR_ARG;
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
        if (t->dom_n[dom] > numafeidx-dimsum) return PRIMAL_RES_ERR_ARG;
        dimsum += t->dom_n[dom];
    }
    if (dimsum != numafeidx) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t e = 0; e < numafeidx; e++) {
        if (afeidxlist[e] < 0 || afeidxlist[e] >= t->numafe) return PRIMAL_RES_ERR_ARG;
        if (b && !isfinite(b[e])) return PRIMAL_RES_ERR_ARG;
        int a = (int)afeidxlist[e];
        if (t->afe_barnz[a] || !isfinite(t->afeg[a]) || !isfinite(t->afeg[a]-(b ? b[e] : 0))) return PRIMAL_RES_ERR_ARG;
        for (int q = 0; q < t->afe_nz[a]; q++) {
            if (!isfinite(t->afe_val[a][q])) return PRIMAL_RES_ERR_ARG;
            int j = t->afe_sub[a][q];
            for (int k = 0; k < t->numdjc; k++)
                if (t->djc_numterm[k] && j >= t->djc_varbase[k] && j < t->djc_varbase[k]+t->djc_numterm[k])
                    return PRIMAL_RES_ERR_ARG;
        }
    }
    return PRIMAL_RES_OK;
}

/* Translation of the linear domains into the backend's <= rows and storage of
 * the metadata. Assumes djc_validate has already passed. */
static PRIMALrescodee djc_apply(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t numdomidx, const PRIMALint64t *domidxlist,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist,
        const PRIMALrealt *b, PRIMALint64t numterms,
        const PRIMALint64t *termsizelist, int create) {
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
                int mult = ty == PRIMAL_DOMAIN_RZERO ? 2 : ty == PRIMAL_DOMAIN_R ? 0 : 1;
                if (mult && nz > (INT_MAX-nz_all)/mult) {
                    free(rpd); free(disj_start); return PRIMAL_RES_ERR_ARG;
                }
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
    PRIMALrescodee rc;
    if (create) {
        /* Allocate metadata before extending the physical model. */
        size_t nd = (size_t)(numdomidx ? numdomidx : 1), na = (size_t)(numafeidx ? numafeidx : 1);
        PRIMALint64t *md = malloc(nd*sizeof(*md)), *ma = malloc(na*sizeof(*ma));
        double *mb = malloc(na*sizeof(*mb));
        PRIMALint64t *mt = malloc((size_t)ndis*sizeof(*mt));
        if (!md || !ma || !mb || !mt) { free(md); free(ma); free(mb); free(mt); rc = PRIMAL_RES_ERR_ALLOC; }
        else {
            for (PRIMALint64t e = 0; e < numdomidx; e++) md[e] = domidxlist[e];
            for (PRIMALint64t e = 0; e < numafeidx; e++) { ma[e] = afeidxlist[e]; mb[e] = b ? b[e] : 0; }
            for (int e = 0; e < ndis; e++) mt[e] = termsizelist[e];
            rc = djc_encode(t,(int)djcidx,ndis,rpd,ncoef,varidx,rowcoefs,rhs,1);
            if (rc == PRIMAL_RES_OK) {
                t->djc_ndom[djcidx] = numdomidx; t->djc_nafe[djcidx] = numafeidx;
                t->djc_dom[djcidx] = md; t->djc_afe[djcidx] = ma; t->djc_b[djcidx] = mb;
                t->djc_termsize[djcidx] = mt; t->djc_numterm[djcidx] = numterms;
            } else { free(md); free(ma); free(mb); free(mt); }
        }
    } else rc = djc_encode(t,(int)djcidx,ndis,rpd,ncoef,varidx,rowcoefs,rhs,0);
    free(rpd); free(disj_start); free(ncoef); free(rhs); free(varidx); free(rowcoefs);
    return rc;
}

PRIMALrescodee djc_sync(PRIMALtask_t t) {
    for (int k = 0; k < t->numdjc; k++) if (t->djc_numterm[k]) {
        PRIMALrescodee rc = djc_validate(t,t->djc_ndom[k],t->djc_dom[k],t->djc_nafe[k],
            t->djc_afe[k],t->djc_b[k],t->djc_numterm[k],t->djc_termsize[k]);
        if (rc != PRIMAL_RES_OK) return rc;
        rc = djc_apply(t,k,t->djc_ndom[k],t->djc_dom[k],t->djc_nafe[k],t->djc_afe[k],
            t->djc_b[k],t->djc_numterm[k],t->djc_termsize[k],0);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}

PRIMALrescodee djc_copy(PRIMALtask_t s, PRIMALtask_t d) {
    PRIMALrescodee rc = PRIMAL_appenddjcs(d,s->numdjc);
    if (rc != PRIMAL_RES_OK) return rc;
    for (int k = 0; k < s->numdjc; k++) {
        if (s->djcname[k]) { rc = PRIMAL_putdjcname(d,k,s->djcname[k]); if (rc != PRIMAL_RES_OK) return rc; }
        if (!s->djc_numterm[k]) continue;
        size_t nd = (size_t)(s->djc_ndom[k] ? s->djc_ndom[k] : 1);
        size_t na = (size_t)(s->djc_nafe[k] ? s->djc_nafe[k] : 1), nt = (size_t)s->djc_numterm[k];
        d->djc_dom[k] = malloc(nd*sizeof(PRIMALint64t));
        d->djc_afe[k] = malloc(na*sizeof(PRIMALint64t));
        d->djc_b[k] = malloc(na*sizeof(double));
        d->djc_termsize[k] = malloc(nt*sizeof(PRIMALint64t));
        if (!d->djc_dom[k] || !d->djc_afe[k] || !d->djc_b[k] || !d->djc_termsize[k]) return PRIMAL_RES_ERR_ALLOC;
        d->djc_ndom[k] = s->djc_ndom[k]; d->djc_nafe[k] = s->djc_nafe[k]; d->djc_numterm[k] = s->djc_numterm[k];
        memcpy(d->djc_dom[k],s->djc_dom[k],(size_t)s->djc_ndom[k]*sizeof(PRIMALint64t));
        memcpy(d->djc_afe[k],s->djc_afe[k],(size_t)s->djc_nafe[k]*sizeof(PRIMALint64t));
        memcpy(d->djc_b[k],s->djc_b[k],(size_t)s->djc_nafe[k]*sizeof(double));
        memcpy(d->djc_termsize[k],s->djc_termsize[k],nt*sizeof(PRIMALint64t));
        d->djc_rowbase[k] = s->djc_rowbase[k]; d->djc_varbase[k] = s->djc_varbase[k]; d->djc_nrow[k] = s->djc_nrow[k];
    }
    return PRIMAL_RES_OK;
}

/* Append `num` empty DJC slots for later putdjc calls. */
PRIMALrescodee PRIMAL_appenddjcs(PRIMALtask_t t, PRIMALint64t num) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX-t->numdjc) return PRIMAL_RES_ERR_ARG;
    if (num == 0) return PRIMAL_RES_OK;
    int want = t->numdjc + (int)num;
    if (want > t->djccap) {
        int nc = t->djccap ? t->djccap : 4;
        while (nc < want) { if (nc > INT_MAX/2) { nc = want; break; } nc *= 2; }
        PRIMALint64t *n1 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t *n2 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t *n3 = (PRIMALint64t *)malloc((size_t)nc * sizeof(PRIMALint64t));
        PRIMALint64t **n4 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        PRIMALint64t **n5 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        double **n6 = (double **)malloc((size_t)nc * sizeof(double *));
        PRIMALint64t **n7 = (PRIMALint64t **)malloc((size_t)nc * sizeof(PRIMALint64t *));
        char **n8 = (char **)malloc((size_t)nc * sizeof(char *));
        int *n9 = malloc((size_t)nc*sizeof(int)), *n10 = malloc((size_t)nc*sizeof(int)), *n11 = malloc((size_t)nc*sizeof(int));
        if (!n9 || !n10 || !n11 || !n1 || !n2 || !n3 || !n4 || !n5 || !n6 || !n7 || !n8) {
            free(n9); free(n10); free(n11); free(n1); free(n2); free(n3); free(n4); free(n5); free(n6); free(n7); free(n8);
            return PRIMAL_RES_ERR_ALLOC;
        }
        for (int i = 0; i < t->numdjc; i++) {
            n1[i] = t->djc_ndom[i]; n2[i] = t->djc_nafe[i]; n3[i] = t->djc_numterm[i];
            n4[i] = t->djc_dom[i]; n5[i] = t->djc_afe[i]; n6[i] = t->djc_b[i];
            n7[i] = t->djc_termsize[i]; n8[i] = t->djcname[i];
            n9[i] = t->djc_rowbase[i]; n10[i] = t->djc_varbase[i]; n11[i] = t->djc_nrow[i];
        }
        free(t->djc_ndom); free(t->djc_nafe); free(t->djc_numterm);
        free(t->djc_dom); free(t->djc_afe); free(t->djc_b); free(t->djc_termsize);
        free(t->djcname); free(t->djc_rowbase); free(t->djc_varbase); free(t->djc_nrow);
        t->djc_ndom = n1; t->djc_nafe = n2; t->djc_numterm = n3;
        t->djc_dom = n4; t->djc_afe = n5; t->djc_b = n6; t->djc_termsize = n7;
        t->djcname = n8; t->djccap = nc;
        t->djc_rowbase = n9; t->djc_varbase = n10; t->djc_nrow = n11;
    }
    for (int i = t->numdjc; i < want; i++) {
        t->djc_ndom[i] = 0; t->djc_nafe[i] = 0; t->djc_numterm[i] = 0;
        t->djc_dom[i] = NULL; t->djc_afe[i] = NULL; t->djc_b[i] = NULL;
        t->djc_termsize[i] = NULL; t->djcname[i] = NULL;
        t->djc_rowbase[i] = t->djc_varbase[i] = -1; t->djc_nrow[i] = 0;
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
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (djcidx < 0 || djcidx >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    if (t->djc_numterm[djcidx] != 0) return PRIMAL_RES_ERR_ARG;   /* already written */
    PRIMALrescodee rc = djc_validate(t, numdomidx, domidxlist, numafeidx,
                                     afeidxlist, b, numterms, termsizelist);
    if (rc != PRIMAL_RES_OK) return rc;
    return djc_apply(t, djcidx, numdomidx, domidxlist, numafeidx, afeidxlist,
                     b, numterms, termsizelist, 1);
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
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (idxfirst < 0 || idxlast < idxfirst || idxlast > t->numdjc) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t L = idxlast - idxfirst;
    if (L == 0) return PRIMAL_RES_OK;
    if (!termsindjc || !termsizelist) return PRIMAL_RES_ERR_NULL;
    if (numdomidx < 0 || numafeidx < 0 || numterms < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALint64t tsum = 0;
    for (PRIMALint64t i = 0; i < L; i++) {
        if (termsindjc[i] < 1 || termsindjc[i] > numterms-tsum) return PRIMAL_RES_ERR_ARG;
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
            if (termsizelist[tc + q] < 0 || termsizelist[tc + q] > numdomidx-dc-nd) return PRIMAL_RES_ERR_ARG;
            nd += termsizelist[tc + q];
        }
        for (PRIMALint64t d = 0; d < nd; d++) {
            PRIMALint64t dom = domidxlist[dc + d];
            if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
            int ty = t->dom_type[dom];
            if (ty != PRIMAL_DOMAIN_R && ty != PRIMAL_DOMAIN_RZERO &&
                ty != PRIMAL_DOMAIN_RPLUS && ty != PRIMAL_DOMAIN_RMINUS)
                return PRIMAL_RES_ERR_ARG;
            if (t->dom_n[dom] > numafeidx-ac-na) return PRIMAL_RES_ERR_ARG;
            na += t->dom_n[dom];
        }
        const PRIMALint64t *dsub = nd > 0 ? domidxlist + dc : NULL;
        const PRIMALint64t *asub = na > 0 ? afeidxlist + ac : NULL;
        const PRIMALrealt *bsub = (b && na > 0) ? b + ac : NULL;
        PRIMALrescodee rc = djc_validate(t, nd, dsub, na, asub, bsub, nt, termsizelist + tc);
        if (rc != PRIMAL_RES_OK) return rc;
        tc += nt; dc += nd; ac += na;
    }
    if (dc != numdomidx) return PRIMAL_RES_ERR_ARG;
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
                                      termsizelist + tc, 1);
        if (rc != PRIMAL_RES_OK) return rc;
        tc += nt; dc += nd; ac += na;
    }
    return PRIMAL_RES_OK;
}

/* Measure the original OR-of-ANDs independently of the generated big-M rows. */
double djc_violation(PRIMALtask_t t, int d, const double *x) {
    PRIMALint64t domc = 0, afec = 0;
    double best = HUGE_VAL;
    if (!t->djc_numterm[d]) return 0;
    for (PRIMALint64t term = 0; term < t->djc_numterm[d]; term++) {
        double worst = 0;
        for (PRIMALint64t q = 0; q < t->djc_termsize[d][term]; q++) {
            PRIMALint64t dom = t->djc_dom[d][domc++];
            int ty = t->dom_type[dom];
            for (PRIMALint64t c = 0; c < t->dom_n[dom]; c++, afec++) {
                int afe = (int)t->djc_afe[d][afec];
                if (ty == PRIMAL_DOMAIN_R) continue;
                if (t->afe_barnz[afe]) return HUGE_VAL;
                double expr = t->afeg[afe]-t->djc_b[d][afec];
                for (int e = 0; e < t->afe_nz[afe]; e++) expr += t->afe_val[afe][e]*x[t->afe_sub[afe][e]];
                if (!isfinite(expr)) return HUGE_VAL;
                double v = ty == PRIMAL_DOMAIN_RZERO ? fabs(expr) :
                    ty == PRIMAL_DOMAIN_RPLUS ? fmax(0,-expr) : fmax(0,expr);
                worst = fmax(worst,v);
            }
        }
        best = fmin(best,worst);
    }
    return best;
}

PRIMALrescodee PRIMAL_getpvioldjc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numdjcidx, const PRIMALint64t *djcidxlist, PRIMALrealt *viol) {
    if (!t || !viol || (numdjcidx > 0 && !djcidxlist)) return PRIMAL_RES_ERR_NULL;
    if (numdjcidx < 0 || !sol_key_ok(which) || !t->has_sol) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t k = 0; k < numdjcidx; k++)
        if (djcidxlist[k] < 0 || djcidxlist[k] >= t->numdjc) return PRIMAL_RES_ERR_ARG;
    for (PRIMALint64t k = 0; k < numdjcidx; k++) viol[k] = djc_violation(t,(int)djcidxlist[k],t->x);
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

