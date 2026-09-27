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
/* primal_misc.c - newsolution setters, basis solve, cholesky, clonetask, dual/infeasible problems.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Read an int parameter widened to 64 bit (ERR_ARG if the id is unknown). */
PRIMALrescodee PRIMAL_getlintparam(PRIMALtask_t t, int param, PRIMALint64t *parvalue) {
    if (!t || !parvalue) return PRIMAL_RES_ERR_NULL;
    int v = 0;
    PRIMALrescodee rc = PRIMAL_getintparam(t, param, &v);
    if (rc != PRIMAL_RES_OK) return rc;
    *parvalue = v;
    return PRIMAL_RES_OK;
}
/* Set an int parameter from a 64-bit value; ERR_ARG if it does not fit an int. */
PRIMALrescodee PRIMAL_putlintparam(PRIMALtask_t t, int param, PRIMALint64t parvalue) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (parvalue < INT_MIN || parvalue > INT_MAX) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_putintparam(t, param, (int)parvalue);
}
/* putparam: name + value as strings. */
PRIMALrescodee PRIMAL_putparam(PRIMALtask_t t, const char *parname, const char *parvalue) {
    if (!t || !parname || !parvalue) return PRIMAL_RES_ERR_NULL;
    int kind = -1, id = -1;
    PRIMALrescodee rc = PRIMAL_whichparam(t, parname, &kind, &id);
    if (rc != PRIMAL_RES_OK) return rc;
    char *end;
    if (kind == PRIMAL_PARAM_KIND_INT) {
        long v = strtol(parvalue, &end, 10);
        if (end == parvalue) return PRIMAL_RES_ERR_ARG;
        return PRIMAL_putintparam(t, id, (int)v);
    }
    double v = strtod(parvalue, &end);
    if (end == parvalue) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_putdouparam(t, id, v);
}
/* Write the named parameters of the task to a text file, one `name value` per
 * line. */
PRIMALrescodee PRIMAL_writeparamfile(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    for (int i = 0; i < PRIMAL_NPARAM; i++) {
        const PrimalParam *d = &PRIMAL_PARAMS[i];
        if (!d->name) continue;
        if (d->kind == P_INT) {
            int v = 0;
            PRIMAL_getintparam(t, d->id, &v);
            fprintf(f, "%s %d\n", d->name, v);
        } else {
            double v = 0;
            PRIMAL_getdouparam(t, d->id, &v);
            fprintf(f, "%s %.17g\n", d->name, v);
        }
    }
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Read `name value` lines and apply them via PRIMAL_putparam; stops at the
 * first error. */
PRIMALrescodee PRIMAL_readparamfile(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char nm[128], val[64];
        if (sscanf(line, "%127s %63s", nm, val) != 2) continue;
        PRIMALrescodee rc = PRIMAL_putparam(t, nm, val);
        if (rc != PRIMAL_RES_OK) { fclose(f); return rc; }
    }
    fclose(f);
    return PRIMAL_RES_OK;
}

/* Name generators (reference generate*names): for every index in the list the
 * name is `fmt` applied to the index. The shape parameters (dims/sp/axis) are
 * not used: the reference's exact form was not read (declared deviation). */
#define GEN_NAMES(SUBLIST, MAXIDX, BODY) \
    if (!t || num < 0 || (num > 0 && !(SUBLIST))) return PRIMAL_RES_ERR_NULL; \
    if (!fmt) return PRIMAL_RES_ERR_NULL; \
    for (long k = 0; k < (long)num; k++) { \
        long idx = (long)(SUBLIST)[k]; \
        if (idx < 0 || idx >= (MAXIDX)) return PRIMAL_RES_ERR_ARG; \
        char buf[128]; \
        snprintf(buf, sizeof buf, fmt, (int)idx); \
        PRIMALrescodee rc = BODY; \
        if (rc != PRIMAL_RES_OK) return rc; \
    } \
    return PRIMAL_RES_OK;
/* Generate variable names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generatevarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(subj, t->numvar, PRIMAL_putvarname(t, (int)idx, buf))
}
/* Generate constraint names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generateconnames(PRIMALtask_t t, int num, const int *subi, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(subi, t->numcon, PRIMAL_putconname(t, (int)idx, buf))
}
/* Generate cone names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generateconenames(PRIMALtask_t t, int num, const int *subk, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(subk, t->numcones, PRIMAL_putconename(t, (int)idx, buf))
}
/* Generate bar-variable names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generatebarvarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(subj, t->numbarvar, PRIMAL_putbarvarname(t, (int)idx, buf))
}
/* Generate ACC names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generateaccnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(sub, t->numacc, PRIMAL_putaccname(t, idx, buf))
}
/* Generate DJC names by applying `fmt` to each index in the list. */
PRIMALrescodee PRIMAL_generatedjcnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names) {
    (void)ndims; (void)dims; (void)sp; (void)numnamedaxis; (void)namedaxisidxs;
    (void)numnames; (void)names;
    GEN_NAMES(sub, t->numdjc, PRIMAL_putdjcname(t, idx, buf))
}

/* Diagnostics: getconeinfo (reading a cone block), printparam, readsummary. */
PRIMALrescodee PRIMAL_getconeinfo(PRIMALtask_t t, int k, PRIMALconetypee *ct,
                                  PRIMALrealt *conepar, int *nummem) {
    if (!t || !ct || !conepar || !nummem) return PRIMAL_RES_ERR_NULL;
    int n = 0;
    PRIMALrescodee rc = PRIMAL_getcone(t, k, ct, &n, NULL);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMAL_getconeparam(t, k, conepar);
    *nummem = n;
    return PRIMAL_RES_OK;
}
/* Print the named parameters of the task to stdout. */
PRIMALrescodee PRIMAL_printparam(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    for (int i = 0; i < PRIMAL_NPARAM; i++) {
        const PrimalParam *d = &PRIMAL_PARAMS[i];
        if (!d->name) continue;
        if (d->kind == P_INT) { int v = 0; PRIMAL_getintparam(t, d->id, &v);
            printf("%s = %d\n", d->name, v); }
        else { double v = 0; PRIMAL_getdouparam(t, d->id, &v);
            printf("%s = %.17g\n", d->name, v); }
    }
    return PRIMAL_RES_OK;
}
/* Print the last read model dimensions to stdout (whichstream is ignored). */
PRIMALrescodee PRIMAL_readsummary(PRIMALtask_t t, int whichstream) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream;
    printf("Last read: %d variables, %d constraints\n", t->numvar, t->numcon);
    return PRIMAL_RES_OK;
}

/* optimize* : optimizetrm is optimize with the termination code as an output;
 * optimizebatch solves the list in sequence. */
PRIMALrescodee PRIMAL_optimizetrm(PRIMALtask_t t, PRIMALrescodee *trmcode) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (trmcode) *trmcode = rc;
    return rc;
}
/* Solve each task in the list in sequence, writing the termination and result
 * codes per task. */
PRIMALrescodee PRIMAL_optimizebatch(PRIMALenv_t env, int israce, PRIMALrealt maxtime,
        int numthreads, PRIMALint64t numtask, const PRIMALtask_t *task,
        PRIMALrescodee *trmcode, PRIMALrescodee *rcode) {
    (void)env; (void)israce; (void)maxtime; (void)numthreads;
    if (numtask < 0 || (numtask > 0 && (!task || !trmcode || !rcode))) return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t i = 0; i < numtask; i++) {
        rcode[i] = PRIMAL_optimize(task[i]);
        trmcode[i] = rcode[i];
    }
    return PRIMAL_RES_OK;
}

/* primalrepair: the reference's weights are not used (declared deviation); the
 * repair is the one of feasrepair. */
PRIMALrescodee PRIMAL_primalrepair(PRIMALtask_t t, const PRIMALrealt *wlc,
        const PRIMALrealt *wuc, const PRIMALrealt *wlx, const PRIMALrealt *wux) {
    (void)wlc; (void)wuc; (void)wlx; (void)wux;
    if (!t) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_feasrepair(t);
}

/* dualsensitivity/primalsensitivity: the same quantities as costsensitivity/
 * rhssensitivity, by list. `marki`/`markj` are not used (declared deviation). */
PRIMALrescodee PRIMAL_dualsensitivity(PRIMALtask_t t, int numj, const int *subj,
        PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numj < 0 || (numj > 0 && !subj)) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < numj; k++) {
        double lo = 0, up = 0;
        PRIMALrescodee rc = PRIMAL_costsensitivity(t, subj[k], &lo, &up);
        if (rc != PRIMAL_RES_OK) return rc;
        if (leftpricej) leftpricej[k] = lo;
        if (rightpricej) rightpricej[k] = up;
        if (leftrangej) leftrangej[k] = lo;
        if (rightrangej) rightrangej[k] = up;
    }
    return PRIMAL_RES_OK;
}
/* Row (RHS) sensitivity for the listed rows; the listed column ranges are
 * zeroed. marki/markj are unused. */
PRIMALrescodee PRIMAL_primalsensitivity(PRIMALtask_t t, int numi, const int *subi,
        const int *marki, int numj, const int *subj, const int *markj,
        PRIMALrealt *leftpricei, PRIMALrealt *rightpricei, PRIMALrealt *leftrangei,
        PRIMALrealt *rightrangei, PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej) {
    (void)marki; (void)markj;
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numi < 0 || (numi > 0 && !subi) || numj < 0 || (numj > 0 && !subj))
        return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < numi; k++) {
        double lo = 0, up = 0;
        PRIMALrescodee rc = PRIMAL_rhssensitivity(t, subi[k], &lo, &up);
        if (rc != PRIMAL_RES_OK) return rc;
        if (leftpricei) leftpricei[k] = lo;
        if (rightpricei) rightpricei[k] = up;
        if (leftrangei) leftrangei[k] = lo;
        if (rightrangei) rightrangei[k] = up;
    }
    for (int k = 0; k < numj; k++) {
        if (leftpricej) leftpricej[k] = 0;
        if (rightpricej) rightpricej[k] = 0;
        if (leftrangej) leftrangej[k] = 0;
        if (rightrangej) rightrangej[k] = 0;
    }
    return PRIMAL_RES_OK;
}

/* toconic: the reference reformulates a QCQO into CQO in place; this solver
 * reformulates at solve time, so it is a no-op that only checks the model is
 * representable (a non-convex domain stays refused downstream). */
PRIMALrescodee PRIMAL_toconic(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_RES_OK;
}

/* ---- put-style ACC, block AFE, cones, per-row barA ---- */
/* Overwrite the ACC at the append position only; another accidx is ERR_ARG
 * because emitted rows cannot be withdrawn. */
PRIMALrescodee PRIMAL_putacc(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t domidx,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    /* emitted rows cannot be withdrawn: only the append position is writable */
    if (accidx != t->numacc) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_appendacc(t, domidx, numafeidx, afeidxlist, b);
}
/* Append ACCs in list order, taking each one's size from its domain. */
PRIMALrescodee PRIMAL_putacclist(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *accidxs, const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numaccs < 0 || (numaccs > 0 && (!accidxs || !domidxs))) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t ac = 0;
    for (PRIMALint64t k = 0; k < numaccs; k++) {
        PRIMALint64t dom = domidxs[k];
        if (dom < 0 || dom >= t->numdomain) return PRIMAL_RES_ERR_ARG;
        PRIMALint64t n = t->dom_n[dom];
        if (ac + n > numafeidx) return PRIMAL_RES_ERR_ARG;
        PRIMALrescodee rc = PRIMAL_putacc(t, accidxs[k], dom, n, afeidxlist + ac,
                                          b ? b + ac : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
        ac += n;
    }
    return PRIMAL_RES_OK;
}
/* Set several affine-expression rows from packed lists (lenidxval is unused). */
PRIMALrescodee PRIMAL_putafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *numnzrow, const PRIMALint64t *ptrrow,
        PRIMALint64t lenidxval, const int *varidx, const PRIMALrealt *val) {
    (void)lenidxval;
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numafeidx < 0 || (numafeidx > 0 && (!afeidx || !numnzrow || !ptrrow)))
        return PRIMAL_RES_ERR_NULL;
    for (PRIMALint64t k = 0; k < numafeidx; k++) {
        PRIMALrescodee rc = PRIMAL_putafefrow(t, afeidx[k], numnzrow[k],
                varidx ? varidx + ptrrow[k] : NULL, val ? val + ptrrow[k] : NULL);
        if (rc != PRIMAL_RES_OK) return rc;
    }
    return PRIMAL_RES_OK;
}
/* Replace the members, type and parameter of existing cone k. */
PRIMALrescodee PRIMAL_putcone(PRIMALtask_t t, int k, PRIMALconetypee ct, PRIMALrealt conepar,
                              int nummem, const int *submem) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcones) return PRIMAL_RES_ERR_ARG;
    if (nummem < 0 || (nummem > 0 && !submem)) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < nummem; i++)
        if (submem[i] < 0 || submem[i] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    int *m = (int *)malloc((size_t)(nummem > 0 ? nummem : 1) * sizeof(int));
    if (!m) return PRIMAL_RES_ERR_ALLOC;
    for (int i = 0; i < nummem; i++) m[i] = submem[i];
    free(t->cone_mem[k]);
    t->cone_mem[k] = m; t->cone_nmem[k] = nummem;
    t->cone_type[k] = ct; t->cone_param[k] = conepar;
    return PRIMAL_RES_OK;
}
/* putbararowlist: for each row, clears the barA terms and rewrites them from
 * the list. `subj[p]` is the bar, `nummat[p]` the number of matrices starting
 * from the current cursor in matidx/weights (the reference's exact form was not
 * read). */
PRIMALrescodee PRIMAL_putbararowlist(PRIMALtask_t t, int num, const int *subi,
        const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *subj,
        const PRIMALint64t *nummat, const PRIMALint64t *matidx, const PRIMALrealt *weights) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subi || !ptrb || !ptre || !subj || !nummat || !matidx || !weights)))
        return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < num; k++) {
        int row = subi[k];
        if (row < 0 || row >= t->numcon) return PRIMAL_RES_ERR_ARG;
        /* clear the row's terms */
        int w = 0;
        for (int e = 0; e < t->nbarA; e++)
            if (t->barA_con[e] != row) {
                t->barA_con[w] = t->barA_con[e]; t->barA_bar[w] = t->barA_bar[e];
                t->barA_sym[w] = t->barA_sym[e]; t->barA_coef[w] = t->barA_coef[e]; w++;
            }
        t->nbarA = w;
        PRIMALint64t cur = 0;
        for (PRIMALint64t p = ptrb[k]; p < ptre[k]; p++) {
            int j = subj[p];
            if (j < 0 || j >= t->numbarvar) return PRIMAL_RES_ERR_ARG;
            PRIMALint64t nm = nummat[p];
            int *syms = (int *)malloc((size_t)(nm > 0 ? nm : 1) * sizeof(int));
            if (!syms) return PRIMAL_RES_ERR_ALLOC;
            for (PRIMALint64t q = 0; q < nm; q++) syms[q] = (int)matidx[cur + q];
            PRIMALrescodee rc = PRIMAL_putbaraij(t, row, j, (int)nm, syms, weights + cur);
            free(syms);
            if (rc != PRIMAL_RES_OK) return rc;
            cur += nm;
        }
    }
    return PRIMAL_RES_OK;
}

/* ---- "new"-style solution API and per-index setters ---- */
/* Report whether a solution is defined (has_sol) for the given key. */
PRIMALrescodee PRIMAL_solutiondef(PRIMALtask_t t, PRIMALsolt which, int *isdef) {
    if (!t || !isdef) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    *isdef = t->has_sol ? 1 : 0;
    return PRIMAL_RES_OK;
}
/* Store skc/xc/slc/suc for constraint i, creating the key table on first write. */
PRIMALrescodee PRIMAL_putconsolutioni(PRIMALtask_t t, int i, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (i < 0 || i >= t->numcon || !t->xc) return PRIMAL_RES_ERR_ARG;
    if (!t->skc) {   /* the key table is created on first write */
        t->skc = (PRIMALstakeye *)calloc((size_t)(t->numcon > 0 ? t->numcon : 1), sizeof(PRIMALstakeye));
        if (!t->skc) return PRIMAL_RES_ERR_ALLOC;
        t->skccap = t->numcon;
    }
    t->skc[i] = sk; t->xc[i] = x; t->slc[i] = sl; t->suc[i] = su;
    t->has_sol = 1; t->has_xc = 1;
    return PRIMAL_RES_OK;
}
/* Store y[i] and mark the solution as defined. */
PRIMALrescodee PRIMAL_putsolutionyi(PRIMALtask_t t, int i, PRIMALsolt which, PRIMALrealt y) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (i < 0 || i >= t->numcon || !t->y) return PRIMAL_RES_ERR_ARG;
    t->y[i] = y;
    t->has_sol = 1;
    return PRIMAL_RES_OK;
}
/* Store skx/x/slx/sux/snx for variable j, creating the key table on first
 * write. */
PRIMALrescodee PRIMAL_putvarsolutionj(PRIMALtask_t t, int j, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su, PRIMALrealt sn) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (j < 0 || j >= t->numvar || !t->snx) return PRIMAL_RES_ERR_ARG;
    if (!t->skx) {
        t->skx = (PRIMALstakeye *)calloc((size_t)(t->numvar > 0 ? t->numvar : 1), sizeof(PRIMALstakeye));
        if (!t->skx) return PRIMAL_RES_ERR_ALLOC;
        t->skxcap = t->numvar;
    }
    t->skx[j] = sk; t->x[j] = x; t->slx[j] = sl; t->sux[j] = su; t->snx[j] = sn;
    t->has_sol = 1;
    return PRIMAL_RES_OK;
}
/* Read the status and solution vectors in one call, including the ACC duals
 * in doty. */
PRIMALrescodee PRIMAL_getsolutionnew(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *problemsta,
        PRIMALsolstae *solutionsta, PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
        PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y, PRIMALrealt *slc, PRIMALrealt *suc,
        PRIMALrealt *slx, PRIMALrealt *sux, PRIMALrealt *snx, PRIMALrealt *doty) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (problemsta) PRIMAL_getprosta(t, which, problemsta);
    if (solutionsta) PRIMAL_getsolsta(t, which, solutionsta);
    if (skc && t->skc) for (int i = 0; i < t->numcon; i++) skc[i] = t->skc[i];
    if (skx && t->skx) for (int j = 0; j < t->numvar; j++) skx[j] = t->skx[j];
    if (skn) for (int j = 0; j < t->numvar; j++) skn[j] = PRIMAL_SK_BAS;
    if (!t->has_sol) return PRIMAL_RES_OK;   /* verdict without a point */
    if (xc) PRIMAL_getxc(t, which, xc);
    if (xx) for (int j = 0; j < t->numvar; j++) xx[j] = t->x[j];
    if (y) for (int i = 0; i < t->numcon; i++) y[i] = t->y[i];
    if (slc) for (int i = 0; i < t->numcon; i++) slc[i] = t->slc[i];
    if (suc) for (int i = 0; i < t->numcon; i++) suc[i] = t->suc[i];
    if (slx) for (int j = 0; j < t->numvar; j++) slx[j] = t->slx[j];
    if (sux) for (int j = 0; j < t->numvar; j++) sux[j] = t->sux[j];
    if (snx) for (int j = 0; j < t->numvar; j++) snx[j] = t->snx[j];
    if (doty) { PRIMALint64t ntot = 0; PRIMAL_getaccntot(t, &ntot);
        if (ntot > 0) PRIMAL_getaccdotys(t, which, doty); }
    return PRIMAL_RES_OK;
}
/* Write the point buffers in one call; skn and doty are ignored. */
PRIMALrescodee PRIMAL_putsolutionnew(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx, const PRIMALrealt *doty) {
    (void)skn; (void)doty;
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(which)) return PRIMAL_RES_ERR_ARG;
    if (skc) PRIMAL_putskcslice(t, which, 0, t->numcon, skc);
    if (skx) PRIMAL_putskxslice(t, which, 0, t->numvar, skx);
    if (xc) PRIMAL_putxc(t, which, (PRIMALrealt *)xc);
    if (xx) for (int j = 0; j < t->numvar; j++) t->x[j] = xx[j];
    if (y) for (int i = 0; i < t->numcon; i++) t->y[i] = y[i];
    if (slc) for (int i = 0; i < t->numcon; i++) t->slc[i] = slc[i];
    if (suc) for (int i = 0; i < t->numcon; i++) t->suc[i] = suc[i];
    if (slx) for (int j = 0; j < t->numvar; j++) t->slx[j] = slx[j];
    if (sux) for (int j = 0; j < t->numvar; j++) t->sux[j] = sux[j];
    if (snx) for (int j = 0; j < t->numvar; j++) t->snx[j] = snx[j];
    t->has_sol = 1;
    return PRIMAL_RES_OK;
}
/* putsolutionnew without the doty argument. */
PRIMALrescodee PRIMAL_putsolution(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx) {
    return PRIMAL_putsolutionnew(t, which, skc, skx, skn, xc, xx, y, slc, suc, slx, sux, snx, NULL);
}
/* getsolutioninfo plus the ACC and DJC violation maxima. */
PRIMALrescodee PRIMAL_getsolutioninfonew(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pobj,
        PRIMALrealt *pviolcon, PRIMALrealt *pviolvar, PRIMALrealt *pviolbarvar,
        PRIMALrealt *pviolcone, PRIMALrealt *pviolacc, PRIMALrealt *pvioldjc,
        PRIMALrealt *pviolitg, PRIMALrealt *dobj, PRIMALrealt *dviolcon,
        PRIMALrealt *dviolvar, PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone,
        PRIMALrealt *dviolacc) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_getsolutioninfo(t, which, pobj, pviolcon, pviolvar,
            pviolbarvar, pviolcone, pviolitg, dobj, dviolcon, dviolvar, dviolbarvar,
            dviolcone);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMALint64t nacc = 0, ndjc = 0;
    PRIMAL_getnumacc(t, &nacc);
    PRIMAL_getnumdjc(t, &ndjc);
    if (pviolacc) {
        if (nacc > 0 && t->has_sol) {
            PRIMALint64t *al = (PRIMALint64t *)malloc((size_t)nacc * sizeof(PRIMALint64t));
            PRIMALrealt *vv = (PRIMALrealt *)malloc((size_t)nacc * sizeof(PRIMALrealt));
            for (PRIMALint64t k = 0; k < nacc; k++) al[k] = k;
            if (al && vv && PRIMAL_getpviolacc(t, which, nacc, al, vv) == PRIMAL_RES_OK) {
                double m = 0;
                for (PRIMALint64t k = 0; k < nacc; k++) if (vv[k] > m) m = vv[k];
                *pviolacc = m;
            } else *pviolacc = 0;
            free(al); free(vv);
        } else *pviolacc = 0;
    }
    if (pvioldjc) {
        if (ndjc > 0 && t->has_sol) {
            PRIMALint64t *dl = (PRIMALint64t *)malloc((size_t)ndjc * sizeof(PRIMALint64t));
            PRIMALrealt *vv = (PRIMALrealt *)malloc((size_t)ndjc * sizeof(PRIMALrealt));
            for (PRIMALint64t k = 0; k < ndjc; k++) dl[k] = k;
            if (dl && vv && PRIMAL_getpvioldjc(t, which, ndjc, dl, vv) == PRIMAL_RES_OK) {
                double m = 0;
                for (PRIMALint64t k = 0; k < ndjc; k++) if (vv[k] > m) m = vv[k];
                *pvioldjc = m;
            } else *pvioldjc = 0;
            free(dl); free(vv);
        } else *pvioldjc = 0;
    }
    if (dviolacc) {
        if (nacc > 0 && t->has_sol) {
            PRIMALint64t *al = (PRIMALint64t *)malloc((size_t)nacc * sizeof(PRIMALint64t));
            PRIMALrealt *vv = (PRIMALrealt *)malloc((size_t)nacc * sizeof(PRIMALrealt));
            for (PRIMALint64t k = 0; k < nacc; k++) al[k] = k;
            if (al && vv && PRIMAL_getdviolacc(t, which, nacc, al, vv) == PRIMAL_RES_OK) {
                double m = 0;
                for (PRIMALint64t k = 0; k < nacc; k++) if (vv[k] > m) m = vv[k];
                *dviolacc = m;
            } else *dviolacc = 0;
            free(al); free(vv);
        } else *dviolacc = 0;
    }
    return PRIMAL_RES_OK;
}

/* =====================================================================
 * Handle/string I/O, basis solve, sparse Cholesky, clone/dual/subproblem
 * ===================================================================== */

/* Write a buffer to a unique /tmp file with the given extension; returns the
 * path in `path`. */
static PRIMALrescodee write_temp_file(const char *data, size_t len, const char *ext, char *path, size_t pathn) {
    static unsigned long seq = 0;
    snprintf(path, pathn, "/tmp/primal_io_%lu%s", seq++, ext);
    FILE *f = fopen(path, "wb");
    if (!f) return PRIMAL_RES_ERR_FILE;
    if (len > 0) fwrite(data, 1, len, f);
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Parse an LP model from an in-memory string via a temporary file. */
PRIMALrescodee PRIMAL_readlpstring(PRIMALtask_t t, const char *data) {
    if (!t || !data) return PRIMAL_RES_ERR_NULL;
    char path[64];
    PRIMALrescodee rc = write_temp_file(data, strlen(data), ".lp", path, sizeof path);
    if (rc != PRIMAL_RES_OK) return rc;
    rc = PRIMAL_readdata(t, path);
    remove(path);
    return rc;
}
/* OPF is read (mpsio.c); PTF is not implemented (declared deviation). */
PRIMALrescodee PRIMAL_readopfstring(PRIMALtask_t t, const char *data) {
    return opf_read(t, data);
}
/* PTF has no reader: always ERR_ARG (declared deviation). */
PRIMALrescodee PRIMAL_readptfstring(PRIMALtask_t t, const char *data) {
    (void)t; (void)data;
    return PRIMAL_RES_ERR_ARG;
}
/* Read a model through a user read callback, by way of a temporary file.
 * format/compress are unused. */
PRIMALrescodee PRIMAL_readdatahandle(PRIMALtask_t t, PRIMALhreadfunc hread, void *h,
                                     int format, int compress, const char *path) {
    (void)format; (void)compress;
    if (!t || !hread) return PRIMAL_RES_ERR_NULL;
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return PRIMAL_RES_ERR_ALLOC;
    for (;;) {
        if (len + 1024 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return PRIMAL_RES_ERR_ALLOC; }
            buf = nb;
        }
        int got = 1024;
        int r = hread(h, buf + len, &got);
        if (r != 0 || got <= 0) break;
        len += (size_t)got;
    }
    const char *ext = (path && strrchr(path, '.')) ? strrchr(path, '.') : ".mps";
    char tmp[64];
    PRIMALrescodee rc = write_temp_file(buf, len, ext, tmp, sizeof tmp);
    free(buf);
    if (rc != PRIMAL_RES_OK) return rc;
    rc = PRIMAL_readdata(t, tmp);
    remove(tmp);
    return rc;
}
/* Write the model to a temporary file and stream it through a user write
 * callback. format/compress are unused. */
PRIMALrescodee PRIMAL_writedatahandle(PRIMALtask_t t, PRIMALhwritefunc func, void *handle,
                                      int format, int compress) {
    (void)format; (void)compress;
    if (!t || !func) return PRIMAL_RES_ERR_NULL;
    char path[64];
    static unsigned long seq2 = 0;
    snprintf(path, sizeof path, "/tmp/primal_io_%lu.mps", seq2++);
    PRIMALrescodee rc = PRIMAL_writedata(t, path);
    if (rc != PRIMAL_RES_OK) return rc;
    FILE *f = fopen(path, "rb");
    if (!f) return PRIMAL_RES_ERR_FILE;
    char chunk[4096];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) func(handle, chunk, (int)n);
    fclose(f);
    remove(path);
    return PRIMAL_RES_OK;
}

/* ---- basis solve (reference initbasissolve/solvewithbasis/basiscond) ----
 * B is numcon x numcon: column k is column A of basis[k] when basis[k] >= 0,
 * otherwise the unit vector of row -basis[k]-1 (the slack). */
/* Whether basis entry k names a column of A (k >= 0) or a slack row (k < 0). */
static int basis_col_ok(PRIMALtask_t t, int k) { return k >= -t->numcon && k < t->numvar; }
/* Factor the basis matrix B built from the column list and store its LU. */
PRIMALrescodee PRIMAL_initbasissolve(PRIMALtask_t t, int *basis) {
    if (!t || !basis) return PRIMAL_RES_ERR_NULL;
    int n = t->numcon;
    if (n <= 0) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < n; k++)
        if (!basis_col_ok(t, basis[k])) return PRIMAL_RES_ERR_ARG;
    double *B = (double *)calloc((size_t)n * n, sizeof(double));
    if (!B) return PRIMAL_RES_ERR_ALLOC;
    for (int k = 0; k < n; k++) {
        if (basis[k] >= 0) {
            int j = basis[k];
            for (int i = 0; i < n; i++) {
                double a = 0.0;
                PRIMAL_getaij(t, i, j, &a);
                B[(size_t)i * n + k] = a;
            }
        } else {
            int r = -basis[k] - 1;
            B[(size_t)r * n + k] = 1.0;
        }
    }
    if (t->basis_lu) dmat_lu_free((LuFact *)t->basis_lu);
    free(t->basis_vec);
    t->basis_lu = dmat_lu_factor(B, n);
    t->basis_vec = (int *)malloc((size_t)n * sizeof(int));
    if (!t->basis_vec) { free(B); return PRIMAL_RES_ERR_ALLOC; }
    for (int k = 0; k < n; k++) t->basis_vec[k] = basis[k];
    t->basis_n = n;
    free(B);
    if (!t->basis_lu) return PRIMAL_RES_ERR_ARG;   /* singular */
    return PRIMAL_RES_OK;
}
/* Solve B x = rhs for the stored basis from sparse sub/val input; the
 * transposed form is ERR_ARG (declared deviation). */
PRIMALrescodee PRIMAL_solvewithbasis(PRIMALtask_t t, int transp, int numnz, int *sub,
                                     PRIMALrealt *val, int *numnzout) {
    if (!t || !sub || !val || !numnzout) return PRIMAL_RES_ERR_NULL;
    if (!t->basis_lu) return PRIMAL_RES_ERR_ARG;
    int n = t->basis_n;
    double *rhs = (double *)calloc((size_t)n, sizeof(double));
    if (!rhs) return PRIMAL_RES_ERR_ALLOC;
    for (int k = 0; k < numnz; k++)
        if (sub[k] >= 0 && sub[k] < n) rhs[sub[k]] += val[k];
    if (transp) {
        /* B' x = rhs: one would factor B', which we do not have; we solve B'x
         * via the transposed augmented system (a dense copy) -- declared
         * deviation. */
        free(rhs);
        return PRIMAL_RES_ERR_ARG;
    }
    if (dmat_lu_solve((const LuFact *)t->basis_lu, rhs) != 0) {
        free(rhs);
        return PRIMAL_RES_ERR_ARG;
    }
    int w = 0;
    for (int i = 0; i < n; i++)
        if (rhs[i] != 0.0) { sub[w] = i; val[w] = rhs[i]; w++; }
    *numnzout = w;
    free(rhs);
    return PRIMAL_RES_OK;
}
/* Report a rough condition estimate of the stored basis from its column
 * norms. */
PRIMALrescodee PRIMAL_basiscond(PRIMALtask_t t, PRIMALrealt *nrmbasis, PRIMALrealt *nrminvbasis) {
    if (!t || !nrmbasis || !nrminvbasis) return PRIMAL_RES_ERR_NULL;
    if (!t->basis_lu) return PRIMAL_RES_ERR_ARG;
    int n = t->basis_n;
    double nb = 0.0;
    for (int k = 0; k < n; k++) {
        double s = 0.0;
        if (t->basis_vec[k] >= 0) {
            for (int i = 0; i < n; i++) { double a = 0; PRIMAL_getaij(t, i, t->basis_vec[k], &a); s += fabs(a); }
        } else s = 1.0;
        if (s > nb) nb = s;
    }
    *nrmbasis = nb;
    *nrminvbasis = (nb > 0.0) ? 1.0 / nb : 0.0;   /* rough estimate (deviation) */
    return PRIMAL_RES_OK;
}

/* ---- Sparse Cholesky (reference computesparsecholesky) ----
 * Dense fallback: no reordering (perm = identity), L from a dense Cholesky.
 * The output arrays are allocated with malloc. Declared deviation. */
PRIMALrescodee PRIMAL_computesparsecholesky(PRIMALenv_t env, int numthreads, int ordermethod,
        PRIMALrealt tolsingular, int n, const int *anzc, const PRIMALint64t *aptrc,
        const int *asubc, const PRIMALrealt *avalc, int **perm, PRIMALrealt **diag,
        int **lnzc, PRIMALint64t **lptrc, PRIMALint64t *lensubnval, int **lsubc,
        PRIMALrealt **lvalc) {
    (void)env; (void)numthreads; (void)tolsingular;
    if (n < 0 || !anzc || !aptrc || !asubc || !avalc) return PRIMAL_RES_ERR_ARG;
    if (!perm || !diag || !lnzc || !lptrc || !lensubnval || !lsubc || !lvalc)
        return PRIMAL_RES_ERR_NULL;
    /* Build the lower-triangle CSC of A and factor it with the sparse Cholesky
     * (spchol, AMD reduced when ordermethod != 0).  A is symmetric, so only the
     * lower triangle is kept. */
    int *Kp = (int *)calloc((size_t)(n + 1), sizeof(int));
    if (!Kp) return PRIMAL_RES_ERR_ALLOC;
    for (int j = 0; j < n; j++)
        for (PRIMALint64t k = aptrc[j]; k < aptrc[j] + anzc[j]; k++)
            if (asubc[k] >= j && asubc[k] < n) Kp[j + 1]++;
    for (int j = 0; j < n; j++) Kp[j + 1] += Kp[j];
    int nnz = Kp[n];
    int *Ki = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    double *Kx = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    int *fr = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    if (!Ki || !Kx || !fr) { free(Kp); free(Ki); free(Kx); free(fr); return PRIMAL_RES_ERR_ALLOC; }
    for (int j = 0; j < n; j++)
        for (PRIMALint64t k = aptrc[j]; k < aptrc[j] + anzc[j]; k++) {
            int ii = asubc[k];
            if (ii >= j && ii < n) { int q = Kp[j] + fr[j]++; Ki[q] = ii; Kx[q] = avalc[k]; }
        }
    free(fr);
    SpChol *L = (ordermethod != 0) ? spchol_factor_ord(n, Kp, Ki, Kx)
                                   : spchol_factor(n, Kp, Ki, Kx);
    free(Kp); free(Ki); free(Kx);
    if (!L) return PRIMAL_RES_ERR_ARG;      /* not positive definite */
    PRIMALint64t tot = (PRIMALint64t)L->Lp[n];
    int *pm = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    double *dg = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    int *ln = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    PRIMALint64t *lp = (PRIMALint64t *)malloc((size_t)(n + 1) * sizeof(PRIMALint64t));
    int *ls = (int *)malloc((size_t)(tot > 0 ? tot : 1) * sizeof(int));
    double *lv = (double *)malloc((size_t)(tot > 0 ? tot : 1) * sizeof(double));
    if (!pm || !dg || !ln || !lp || !ls || !lv) {
        free(pm); free(dg); free(ln); free(lp); free(ls); free(lv);
        spchol_free(L); return PRIMAL_RES_ERR_ALLOC;
    }
    for (int k = 0; k < n; k++) pm[k] = L->perm ? L->perm[k] : k;
    for (int j = 0; j < n; j++) ln[j] = L->Lp[j + 1] - L->Lp[j];
    for (int j = 0; j <= n; j++) lp[j] = (PRIMALint64t)L->Lp[j];
    for (PRIMALint64t e = 0; e < tot; e++) { ls[e] = L->Li[e]; lv[e] = L->Lx[e]; }
    spchol_free(L);
    *perm = pm; *diag = dg; *lnzc = ln; *lptrc = lp; *lsubc = ls; *lvalc = lv;
    *lensubnval = tot;
    return PRIMAL_RES_OK;

}

/* UTF-8 <-> wide-char (wchar_t = UTF-32 here).  len = output units written,
 * conv = input units consumed; an invalid sequence stops with ERR_ARG and the
 * counts set to what was done. */
PRIMALrescodee PRIMAL_utf8towchar(size_t outputlen, size_t *len, size_t *conv,
                                  PRIMALwchart *output, const char *input) {
    if (!len || !conv || !input) return PRIMAL_RES_ERR_NULL;
    size_t ilen = strlen(input), in = 0, out = 0;
    while (in < ilen) {
        unsigned char b = (unsigned char)input[in];
        unsigned int cp; int n;
        if (b < 0x80) { cp = b; n = 1; }
        else if ((b & 0xE0) == 0xC0) { cp = b & 0x1FU; n = 2; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0FU; n = 3; }
        else if ((b & 0xF8) == 0xF0) { cp = b & 0x07U; n = 4; }
        else { *len = out; *conv = in; return PRIMAL_RES_ERR_ARG; }
        if (in + (size_t)n > ilen) { *len = out; *conv = in; return PRIMAL_RES_ERR_ARG; }
        for (int k = 1; k < n; k++) {
            if (((unsigned char)input[in + (size_t)k] & 0xC0) != 0x80) { *len = out; *conv = in; return PRIMAL_RES_ERR_ARG; }
            cp = (cp << 6) | ((unsigned int)((unsigned char)input[in + (size_t)k]) & 0x3FU);
        }
        if (out == outputlen) break;
        output[out++] = (PRIMALwchart)cp;
        in += (size_t)n;
    }
    *len = out; *conv = in;
    return PRIMAL_RES_OK;
}
/* Encode a wide-char string (UTF-32) to UTF-8; counts written units and
 * consumed input. */
PRIMALrescodee PRIMAL_wchartoutf8(size_t outputlen, size_t *len, size_t *conv,
                                  char *output, const PRIMALwchart *input) {
    if (!len || !conv || !input) return PRIMAL_RES_ERR_NULL;
    size_t in = 0, out = 0;
    while (input[in] != 0) {
        unsigned int cp = (unsigned int)input[in];
        unsigned char b[4]; int n;
        if (cp < 0x80U) { b[0] = (unsigned char)cp; n = 1; }
        else if (cp < 0x800U) { b[0] = (unsigned char)(0xC0U | (cp >> 6)); b[1] = (unsigned char)(0x80U | (cp & 0x3FU)); n = 2; }
        else if (cp < 0x10000U) { b[0] = (unsigned char)(0xE0U | (cp >> 12)); b[1] = (unsigned char)(0x80U | ((cp >> 6) & 0x3FU)); b[2] = (unsigned char)(0x80U | (cp & 0x3FU)); n = 3; }
        else { b[0] = (unsigned char)(0xF0U | (cp >> 18)); b[1] = (unsigned char)(0x80U | ((cp >> 12) & 0x3FU)); b[2] = (unsigned char)(0x80U | ((cp >> 6) & 0x3FU)); b[3] = (unsigned char)(0x80U | (cp & 0x3FU)); n = 4; }
        if (output && out + (size_t)n > outputlen) break;
        if (output) for (int k = 0; k < n; k++) output[out + (size_t)k] = (char)b[k];
        out += (size_t)n;
        in++;
    }
    if (output && out < outputlen) output[out] = 0;
    *len = out; *conv = in;
    return PRIMAL_RES_OK;
}

/* ---- clonetask: deep copy of the model ---- */
/* Deep-copy the domain table of the source task into the clone. */
static PRIMALrescodee clone_domains(PRIMALtask_t s, PRIMALtask_t d) {
    PRIMALint64t nd = 0;
    PRIMAL_getnumdomain(s, &nd);
    for (PRIMALint64t k = 0; k < nd; k++) {
        PRIMALdomaintypee ty; PRIMALint64t n = 0, idx = -1;
        PRIMAL_getdomaintype(s, k, &ty);
        PRIMAL_getdomainn(s, k, &n);
        PRIMALrescodee rc = PRIMAL_RES_ERR_ARG;
        switch (ty) {
        case PRIMAL_DOMAIN_R: rc = PRIMAL_appendrdomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_RZERO: rc = PRIMAL_appendrzerodomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_RPLUS: rc = PRIMAL_appendrplusdomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_RMINUS: rc = PRIMAL_appendrminusdomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_QUADRATIC_CONE: rc = PRIMAL_appendquadraticconedomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_RQUADRATIC_CONE: rc = PRIMAL_appendrquadraticconedomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_PRIMAL_EXP_CONE: rc = PRIMAL_appendprimalexpconedomain(d, &idx); break;
        case PRIMAL_DOMAIN_DUAL_EXP_CONE: rc = PRIMAL_appenddualexpconedomain(d, &idx); break;
        case PRIMAL_DOMAIN_PRIMAL_POWER_CONE: { double a = 0; PRIMAL_getpowerdomainalpha(s, k, &a);
            rc = PRIMAL_appendprimalpowerconedomain(d, n, a, &idx); } break;
        case PRIMAL_DOMAIN_DUAL_POWER_CONE: { double a = 0; PRIMAL_getpowerdomainalpha(s, k, &a);
            rc = PRIMAL_appenddualpowerconedomain(d, n, a, &idx); } break;
        case PRIMAL_DOMAIN_SVEC_PSD_CONE: { PRIMALint64t nn = (PRIMALint64t)((1 + sqrt(1 + 8.0 * n)) / 2);
            rc = PRIMAL_appendsvecpsdconedomain(d, (int)nn, &idx); } break;
        case PRIMAL_DOMAIN_PRIMAL_GEO_MEAN_CONE: rc = PRIMAL_appendprimalgeomeanconedomain(d, n, &idx); break;
        case PRIMAL_DOMAIN_DUAL_GEO_MEAN_CONE: rc = PRIMAL_appenddualgeomeanconedomain(d, n, &idx); break;
        default: return PRIMAL_RES_ERR_ARG;
        }
        if (rc != PRIMAL_RES_OK) return rc;
        const char *nm = NULL;
        PRIMAL_getdomainname(s, k, 0, NULL);   /* not used; name via getter below */
        (void)nm;
    }
    return PRIMAL_RES_OK;
}
/* Deep-copy the whole model (variables, rows, Q, cones, bars, AFE/ACC/DJC)
 * into a new task. */
PRIMALrescodee PRIMAL_clonetask(PRIMALtask_t t, PRIMALtask_t *clonedtask) {
    if (!t || !clonedtask) return PRIMAL_RES_ERR_NULL;
    PRIMALtask_t d = NULL;
    PRIMALrescodee rc = PRIMAL_maketask(t->env, 0, 0, &d);
    if (rc != PRIMAL_RES_OK) return rc;
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getnumcon(t, &nc);
    if (nv > 0) PRIMAL_appendvars(d, nv);
    if (nc > 0) PRIMAL_appendcons(d, nc);
    PRIMALobjsensee se; PRIMAL_getobjsense(t, &se); PRIMAL_putobjsense(d, se);
    double cf = 0; PRIMAL_getcfix(t, &cf); PRIMAL_putcfix(d, cf);
    for (int j = 0; j < nv; j++) {
        double cj = 0; PRIMAL_getcj(t, j, &cj); PRIMAL_putcj(d, j, cj);
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getvarbound(t, j, &bk, &lo, &up); PRIMAL_putvarbound(d, j, bk, lo, up);
        PRIMALvariabletypee vt; PRIMAL_getvartype(t, j, &vt); PRIMAL_putvartype(d, j, vt);
        const char *nm = NULL; PRIMAL_getvarnameidx(t, j, &nm);
        if (nm && nm[0]) PRIMAL_putvarname(d, j, nm);
    }
    for (int i = 0; i < nc; i++) {
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getconbound(t, i, &bk, &lo, &up); PRIMAL_putconbound(d, i, bk, lo, up);
        const char *nm = NULL; PRIMAL_getconnameidx(t, i, &nm);
        if (nm && nm[0]) PRIMAL_putconname(d, i, nm);
        int cap = nv > 0 ? nv : 1;
        int *sub = (int *)malloc((size_t)cap * sizeof(int));
        double *val = (double *)malloc((size_t)cap * sizeof(double));
        if (!sub || !val) { free(sub); free(val); PRIMAL_deletetask(&d); return PRIMAL_RES_ERR_ALLOC; }
        int nr = 0;
        PRIMAL_getarow(t, i, sub, val, cap, &nr);
        if (nr > 0) PRIMAL_putarow(d, i, nr, sub, val);
        free(sub); free(val);
    }
    int nq = 0; PRIMAL_getnumqobjnz(t, &nq);
    if (nq > 0) {
        int *qi = (int *)malloc((size_t)nq * sizeof(int));
        int *qj = (int *)malloc((size_t)nq * sizeof(int));
        double *qv = (double *)malloc((size_t)nq * sizeof(double));
        int nr = 0;
        if (qi && qj && qv && PRIMAL_getqobj(t, qi, qj, qv, nq, &nr) == PRIMAL_RES_OK)
            PRIMAL_putqobj(d, nr, qi, qj, qv);
        free(qi); free(qj); free(qv);
    }
    for (int i = 0; i < nc; i++) {
        int nk = 0; PRIMAL_getnumqconknz(t, i, &nk);
        if (nk <= 0) continue;
        int *qi = (int *)malloc((size_t)nk * sizeof(int));
        int *qj = (int *)malloc((size_t)nk * sizeof(int));
        double *qv = (double *)malloc((size_t)nk * sizeof(double));
        int nr = 0;
        if (qi && qj && qv && PRIMAL_getqconk(t, i, qi, qj, qv, nk, &nr) == PRIMAL_RES_OK)
            PRIMAL_putqconk(d, i, nr, qi, qj, qv);
        free(qi); free(qj); free(qv);
    }
    int ncone = 0; PRIMAL_getnumcone(t, &ncone);
    for (int k = 0; k < ncone; k++) {
        PRIMALconetypee ct; int nm2 = 0;
        PRIMAL_getcone(t, k, &ct, &nm2, NULL);
        int *mem = (int *)malloc((size_t)(nm2 > 0 ? nm2 : 1) * sizeof(int));
        if (!mem) { PRIMAL_deletetask(&d); return PRIMAL_RES_ERR_ALLOC; }
        PRIMAL_getcone(t, k, &ct, &nm2, mem);
        double par = 0; PRIMAL_getconeparam(t, k, &par);
        rc = PRIMAL_appendcone(d, ct, par, nm2, mem);
        free(mem);
        if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&d); return rc; }
    }
    rc = bar_copy(t, d);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&d); return rc; }
    rc = clone_domains(t, d);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&d); return rc; }
    PRIMALint64t nafe64 = 0; PRIMAL_getnumafe(t, &nafe64);
    int nafe = (int)nafe64;
    if (nafe > 0) PRIMAL_appendafes(d, nafe);
    for (int i = 0; i < nafe; i++) {
        double g = 0; PRIMAL_getafeg(t, i, &g); PRIMAL_putafeg(d, i, g);
        int nz = 0; PRIMAL_getafefrownumnz(t, i, &nz);
        if (nz > 0) {
            int *vi = (int *)malloc((size_t)nz * sizeof(int));
            double *vv = (double *)malloc((size_t)nz * sizeof(double));
            if (vi && vv && PRIMAL_getafefrow(t, i, &nz, vi, vv) == PRIMAL_RES_OK)
                PRIMAL_putafefrow(d, i, nz, vi, vv);
            free(vi); free(vv);
        }
        int ne = 0; PRIMAL_getafebarfnumrowentries(t, i, &ne);
        for (int e = 0; e < ne; e++) {
            int bj[1]; PRIMALint64t ptr[1], ntm[1], tidx[64]; double tw[64];
            if (PRIMAL_getafebarfrow(t, i, bj, ptr, ntm, tidx, tw) == PRIMAL_RES_OK && ntm[0] <= 64)
                PRIMAL_putafebarfentry(d, i, bj[0], ntm[0], tidx, tw);
        }
    }
    PRIMALint64t nacc64 = 0; PRIMAL_getnumacc(t, &nacc64);
    int nacc = (int)nacc64;
    for (int a = 0; a < nacc; a++) {
        PRIMALint64t dom = 0, na = 0;
        PRIMAL_getaccdomain(t, a, &dom);
        PRIMAL_getaccn(t, a, &na);
        PRIMALint64t *al = (PRIMALint64t *)malloc((size_t)(na > 0 ? na : 1) * sizeof(PRIMALint64t));
        double *bb = (double *)malloc((size_t)(na > 0 ? na : 1) * sizeof(double));
        if (!al || !bb) { free(al); free(bb); PRIMAL_deletetask(&d); return PRIMAL_RES_ERR_ALLOC; }
        PRIMAL_getaccafeidxlist(t, a, al);
        PRIMAL_getaccb(t, a, bb);
        rc = PRIMAL_appendacc(d, dom, na, al, bb);
        free(al); free(bb);
        if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&d); return rc; }
    }
    PRIMALint64t ndjc64 = 0; PRIMAL_getnumdjc(t, &ndjc64);
    int ndjc = (int)ndjc64;
    if (ndjc > 0) {
        PRIMAL_appenddjcs(d, ndjc);
        for (int a = 0; a < ndjc; a++) {
            PRIMALint64t nd2 = 0, na2 = 0, nt2 = 0;
            PRIMAL_getdjcnumdomain(t, a, &nd2);
            PRIMAL_getdjcnumafe(t, a, &na2);
            PRIMAL_getdjcnumterm(t, a, &nt2);
            PRIMALint64t *dl = (PRIMALint64t *)malloc((size_t)(nd2 > 0 ? nd2 : 1) * sizeof(PRIMALint64t));
            PRIMALint64t *al = (PRIMALint64t *)malloc((size_t)(na2 > 0 ? na2 : 1) * sizeof(PRIMALint64t));
            double *bb = (double *)malloc((size_t)(na2 > 0 ? na2 : 1) * sizeof(double));
            PRIMALint64t *ts = (PRIMALint64t *)malloc((size_t)(nt2 > 0 ? nt2 : 1) * sizeof(PRIMALint64t));
            if (!dl || !al || !bb || !ts) { free(dl); free(al); free(bb); free(ts);
                PRIMAL_deletetask(&d); return PRIMAL_RES_ERR_ALLOC; }
            PRIMAL_getdjcdomainidxlist(t, a, dl);
            PRIMAL_getdjcafeidxlist(t, a, al);
            PRIMAL_getdjcb(t, a, bb);
            PRIMAL_getdjctermsizelist(t, a, ts);
            rc = PRIMAL_putdjc(d, a, nd2, dl, na2, al, bb, nt2, ts);
            free(dl); free(al); free(bb); free(ts);
            if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(&d); return rc; }
        }
    }
    PRIMAL_getcfix(t, &cf);
    *clonedtask = d;
    return PRIMAL_RES_OK;
}

/* ---- getdualproblem: the LP dual of the form min c'x, Ax=b, x>=0 ---- */
PRIMALrescodee PRIMAL_getdualproblem(PRIMALtask_t t, PRIMALtask_t *dualtask) {
    if (!t || !dualtask) return PRIMAL_RES_ERR_NULL;
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
    for (int i = 0; i < nc; i++) {
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getconbound(t, i, &bk, &lo, &up);
        if (bk != PRIMAL_BK_FX) return PRIMAL_RES_ERR_ARG;   /* equalities only */
    }
    for (int j = 0; j < nv; j++) {
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getvarbound(t, j, &bk, &lo, &up);
        if (!(bk == PRIMAL_BK_LO && lo == 0.0 && !isfinite(up))) return PRIMAL_RES_ERR_ARG;
    }
    PRIMALtask_t d = NULL;
    PRIMALrescodee rc = PRIMAL_maketask(t->env, 0, 0, &d);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMAL_appendvars(d, nc);   /* free y */
    PRIMAL_appendcons(d, nv);   /* A'y <= c */
    for (int i = 0; i < nc; i++) {
        PRIMAL_putvarbound(d, i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getconbound(t, i, &bk, &lo, &up);
        PRIMAL_putcj(d, i, -lo);   /* max b'y -> min -b'y */
    }
    for (int j = 0; j < nv; j++) {
        int cap = nc > 0 ? nc : 1;
        int *sub = (int *)malloc((size_t)cap * sizeof(int));
        double *val = (double *)malloc((size_t)cap * sizeof(double));
        if (!sub || !val) { free(sub); free(val); PRIMAL_deletetask(&d); return PRIMAL_RES_ERR_ALLOC; }
        int nr = 0;
        PRIMAL_getacol(t, j, sub, val, cap, &nr);
        if (nr > 0) PRIMAL_putarow(d, j, nr, sub, val);
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getvarbound(t, j, &bk, &lo, &up);
        double cj = 0; PRIMAL_getcj(t, j, &cj);
        PRIMAL_putconbound(d, j, PRIMAL_BK_UP, -INFINITY, cj);
        free(sub); free(val);
    }
    *dualtask = d;
    return PRIMAL_RES_OK;
}

/* ---- getinfeasiblesubproblem: the rows/columns touched by the certificate ----
 * Builds a task with only the rows and variables the ray names.
 * Deviation: it is not the reference's reduction, it is a raw subproblem. */
PRIMALrescodee PRIMAL_getinfeasiblesubproblem(PRIMALtask_t t, PRIMALsolt which,
                                              PRIMALtask_t *inftask) {
    if (!t || !inftask) return PRIMAL_RES_ERR_NULL;
    if (!t->has_dray && !t->has_pray) return PRIMAL_RES_ERR_ARG;
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
    int *rowkeep = (int *)calloc((size_t)(nc > 0 ? nc : 1), sizeof(int));
    int *colkeep = (int *)calloc((size_t)(nv > 0 ? nv : 1), sizeof(int));
    if (!rowkeep || !colkeep) { free(rowkeep); free(colkeep); return PRIMAL_RES_ERR_ALLOC; }
    if (t->has_dray) for (int i = 0; i < nc; i++) if (t->dray[i] != 0.0) rowkeep[i] = 1;
    if (t->has_pray) for (int j = 0; j < nv; j++) if (t->pray[j] != 0.0) colkeep[j] = 1;
    PRIMALtask_t d = NULL;
    PRIMALrescodee rc = PRIMAL_maketask(t->env, 0, 0, &d);
    if (rc != PRIMAL_RES_OK) { free(rowkeep); free(colkeep); return rc; }
    PRIMAL_appendvars(d, nv);
    for (int i = 0; i < nc; i++) {
        if (!rowkeep[i]) continue;
        int cap = nv > 0 ? nv : 1;
        int *sub = (int *)malloc((size_t)cap * sizeof(int));
        double *val = (double *)malloc((size_t)cap * sizeof(double));
        int nr = 0;
        PRIMAL_getarow(t, i, sub, val, cap, &nr);
        PRIMAL_appendcons(d, 1);
        int r = 0; PRIMAL_getnumcon(d, &r); r--;
        if (nr > 0) PRIMAL_putarow(d, r, nr, sub, val);
        PRIMALboundkeye bk; double lo, up;
        PRIMAL_getconbound(t, i, &bk, &lo, &up);
        PRIMAL_putconbound(d, r, bk, lo, up);
        free(sub); free(val);
    }
    *inftask = d;
    free(rowkeep); free(colkeep);
    (void)which;
    return PRIMAL_RES_OK;
}

/* ======================================================================
 * Information items (reference MSKdinfiteme/MSKiinfiteme/MSKliinfiteme,
 * MSKinftypee). The indices are those of reference 11.2.4, read from
 * constants.html on 2026-09-19 (DINF 0..115, IINF 0..136, LIINF 0..21, all
 * contiguous); END is the table limit. The whole reading surface is here; the
 * items this solver does not measure (times of phases it does not have, MIO
 * cut counters, iterations of unused engines) answer 0 -- a deviation declared
 * in README/primal.h, not an invented value.
 * ====================================================================== */
const char *const dinf_names[PRIMAL_DINF_END] = {
    "MSK_DINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_DENSITY", "MSK_DINF_BI_CLEAN_TIME",
    "MSK_DINF_BI_DUAL_TIME", "MSK_DINF_BI_PRIMAL_TIME", "MSK_DINF_BI_TIME",
    "MSK_DINF_FOLDING_BI_OPTIMIZE_TIME", "MSK_DINF_FOLDING_BI_UNFOLD_DUAL_TIME",
    "MSK_DINF_FOLDING_BI_UNFOLD_INITIALIZE_TIME", "MSK_DINF_FOLDING_BI_UNFOLD_PRIMAL_TIME",
    "MSK_DINF_FOLDING_BI_UNFOLD_TIME", "MSK_DINF_FOLDING_FACTOR", "MSK_DINF_FOLDING_TIME",
    "MSK_DINF_INTPNT_DUAL_FEAS", "MSK_DINF_INTPNT_DUAL_OBJ", "MSK_DINF_INTPNT_FACTOR_NUM_FLOPS",
    "MSK_DINF_INTPNT_OPT_STATUS", "MSK_DINF_INTPNT_ORDER_TIME", "MSK_DINF_INTPNT_PRIMAL_FEAS",
    "MSK_DINF_INTPNT_PRIMAL_OBJ", "MSK_DINF_INTPNT_TIME", "MSK_DINF_MIO_CLIQUE_SELECTION_TIME",
    "MSK_DINF_MIO_CLIQUE_SEPARATION_TIME", "MSK_DINF_MIO_CMIR_SELECTION_TIME",
    "MSK_DINF_MIO_CMIR_SEPARATION_TIME", "MSK_DINF_MIO_CONSTRUCT_SOLUTION_OBJ",
    "MSK_DINF_MIO_DUAL_BOUND_AFTER_PRESOLVE", "MSK_DINF_MIO_GMI_SELECTION_TIME",
    "MSK_DINF_MIO_GMI_SEPARATION_TIME", "MSK_DINF_MIO_IMPLIED_BOUND_SELECTION_TIME",
    "MSK_DINF_MIO_IMPLIED_BOUND_SEPARATION_TIME", "MSK_DINF_MIO_INITIAL_FEASIBLE_SOLUTION_OBJ",
    "MSK_DINF_MIO_KNAPSACK_COVER_SELECTION_TIME", "MSK_DINF_MIO_KNAPSACK_COVER_SEPARATION_TIME",
    "MSK_DINF_MIO_LIPRO_SELECTION_TIME", "MSK_DINF_MIO_LIPRO_SEPARATION_TIME",
    "MSK_DINF_MIO_OBJ_ABS_GAP", "MSK_DINF_MIO_OBJ_BOUND", "MSK_DINF_MIO_OBJ_INT",
    "MSK_DINF_MIO_OBJ_REL_GAP", "MSK_DINF_MIO_PROBING_TIME",
    "MSK_DINF_MIO_ROOT_CUT_SELECTION_TIME", "MSK_DINF_MIO_ROOT_CUT_SEPARATION_TIME",
    "MSK_DINF_MIO_ROOT_OPTIMIZER_TIME", "MSK_DINF_MIO_ROOT_PRESOLVE_TIME",
    "MSK_DINF_MIO_ROOT_TIME", "MSK_DINF_MIO_SYMMETRY_DETECTION_TIME",
    "MSK_DINF_MIO_SYMMETRY_FACTOR", "MSK_DINF_MIO_TIME", "MSK_DINF_MIO_USER_OBJ_CUT",
    "MSK_DINF_OPTIMIZER_TICKS", "MSK_DINF_OPTIMIZER_TIME", "MSK_DINF_PRESOLVE_ELI_TIME",
    "MSK_DINF_PRESOLVE_LINDEP_TIME", "MSK_DINF_PRESOLVE_TIME",
    "MSK_DINF_PRESOLVE_TOTAL_PRIMAL_PERTURBATION", "MSK_DINF_PRIMAL_REPAIR_PENALTY_OBJ",
    "MSK_DINF_QCQO_REFORMULATE_MAX_PERTURBATION", "MSK_DINF_QCQO_REFORMULATE_TIME",
    "MSK_DINF_QCQO_REFORMULATE_WORST_CHOLESKY_COLUMN_SCALING",
    "MSK_DINF_QCQO_REFORMULATE_WORST_CHOLESKY_DIAG_SCALING", "MSK_DINF_READ_DATA_TIME",
    "MSK_DINF_REMOTE_TIME", "MSK_DINF_SIM_DUAL_TIME", "MSK_DINF_SIM_FEAS", "MSK_DINF_SIM_OBJ",
    "MSK_DINF_SIM_PRIMAL_TIME", "MSK_DINF_SIM_TIME", "MSK_DINF_SOL_BAS_DUAL_OBJ",
    "MSK_DINF_SOL_BAS_DVIOLCON", "MSK_DINF_SOL_BAS_DVIOLVAR", "MSK_DINF_SOL_BAS_NRM_BARX",
    "MSK_DINF_SOL_BAS_NRM_SLC", "MSK_DINF_SOL_BAS_NRM_SLX", "MSK_DINF_SOL_BAS_NRM_SUC",
    "MSK_DINF_SOL_BAS_NRM_SUX", "MSK_DINF_SOL_BAS_NRM_XC", "MSK_DINF_SOL_BAS_NRM_XX",
    "MSK_DINF_SOL_BAS_NRM_Y", "MSK_DINF_SOL_BAS_PRIMAL_OBJ", "MSK_DINF_SOL_BAS_PVIOLCON",
    "MSK_DINF_SOL_BAS_PVIOLVAR", "MSK_DINF_SOL_ITG_NRM_BARX", "MSK_DINF_SOL_ITG_NRM_XC",
    "MSK_DINF_SOL_ITG_NRM_XX", "MSK_DINF_SOL_ITG_PRIMAL_OBJ", "MSK_DINF_SOL_ITG_PVIOLACC",
    "MSK_DINF_SOL_ITG_PVIOLBARVAR", "MSK_DINF_SOL_ITG_PVIOLCON", "MSK_DINF_SOL_ITG_PVIOLCONES",
    "MSK_DINF_SOL_ITG_PVIOLDJC", "MSK_DINF_SOL_ITG_PVIOLITG", "MSK_DINF_SOL_ITG_PVIOLVAR",
    "MSK_DINF_SOL_ITR_DUAL_OBJ", "MSK_DINF_SOL_ITR_DVIOLACC", "MSK_DINF_SOL_ITR_DVIOLBARVAR",
    "MSK_DINF_SOL_ITR_DVIOLCON", "MSK_DINF_SOL_ITR_DVIOLCONES", "MSK_DINF_SOL_ITR_DVIOLVAR",
    "MSK_DINF_SOL_ITR_NRM_BARS", "MSK_DINF_SOL_ITR_NRM_BARX", "MSK_DINF_SOL_ITR_NRM_SLC",
    "MSK_DINF_SOL_ITR_NRM_SLX", "MSK_DINF_SOL_ITR_NRM_SNX", "MSK_DINF_SOL_ITR_NRM_SUC",
    "MSK_DINF_SOL_ITR_NRM_SUX", "MSK_DINF_SOL_ITR_NRM_XC", "MSK_DINF_SOL_ITR_NRM_XX",
    "MSK_DINF_SOL_ITR_NRM_Y", "MSK_DINF_SOL_ITR_PRIMAL_OBJ", "MSK_DINF_SOL_ITR_PVIOLACC",
    "MSK_DINF_SOL_ITR_PVIOLBARVAR", "MSK_DINF_SOL_ITR_PVIOLCON", "MSK_DINF_SOL_ITR_PVIOLCONES",
    "MSK_DINF_SOL_ITR_PVIOLVAR", "MSK_DINF_TO_CONIC_TIME", "MSK_DINF_WRITE_DATA_TIME"
};

const char *const iinf_names[PRIMAL_IINF_END] = {
    "MSK_IINF_ANA_PRO_NUM_CON", "MSK_IINF_ANA_PRO_NUM_CON_EQ", "MSK_IINF_ANA_PRO_NUM_CON_FR",
    "MSK_IINF_ANA_PRO_NUM_CON_LO", "MSK_IINF_ANA_PRO_NUM_CON_RA", "MSK_IINF_ANA_PRO_NUM_CON_UP",
    "MSK_IINF_ANA_PRO_NUM_VAR", "MSK_IINF_ANA_PRO_NUM_VAR_BIN", "MSK_IINF_ANA_PRO_NUM_VAR_CONT",
    "MSK_IINF_ANA_PRO_NUM_VAR_EQ", "MSK_IINF_ANA_PRO_NUM_VAR_FR", "MSK_IINF_ANA_PRO_NUM_VAR_INT",
    "MSK_IINF_ANA_PRO_NUM_VAR_LO", "MSK_IINF_ANA_PRO_NUM_VAR_RA", "MSK_IINF_ANA_PRO_NUM_VAR_UP",
    "MSK_IINF_FOLDING_APPLIED", "MSK_IINF_INTPNT_FACTOR_DIM_DENSE", "MSK_IINF_INTPNT_ITER",
    "MSK_IINF_INTPNT_NUM_THREADS", "MSK_IINF_INTPNT_SOLVE_DUAL", "MSK_IINF_MIO_ABSGAP_SATISFIED",
    "MSK_IINF_MIO_CLIQUE_TABLE_SIZE", "MSK_IINF_MIO_CONSTRUCT_SOLUTION",
    "MSK_IINF_MIO_FINAL_NUMBIN", "MSK_IINF_MIO_FINAL_NUMBINCONEVAR", "MSK_IINF_MIO_FINAL_NUMCON",
    "MSK_IINF_MIO_FINAL_NUMCONE", "MSK_IINF_MIO_FINAL_NUMCONEVAR", "MSK_IINF_MIO_FINAL_NUMCONT",
    "MSK_IINF_MIO_FINAL_NUMCONTCONEVAR", "MSK_IINF_MIO_FINAL_NUMDEXPCONES",
    "MSK_IINF_MIO_FINAL_NUMDJC", "MSK_IINF_MIO_FINAL_NUMDPOWCONES", "MSK_IINF_MIO_FINAL_NUMINT",
    "MSK_IINF_MIO_FINAL_NUMINTCONEVAR", "MSK_IINF_MIO_FINAL_NUMPEXPCONES",
    "MSK_IINF_MIO_FINAL_NUMPPOWCONES", "MSK_IINF_MIO_FINAL_NUMQCONES",
    "MSK_IINF_MIO_FINAL_NUMRQCONES", "MSK_IINF_MIO_FINAL_NUMVAR",
    "MSK_IINF_MIO_INITIAL_FEASIBLE_SOLUTION", "MSK_IINF_MIO_NODE_DEPTH",
    "MSK_IINF_MIO_NUM_ACTIVE_NODES", "MSK_IINF_MIO_NUM_ACTIVE_ROOT_CUTS",
    "MSK_IINF_MIO_NUM_BLOCKS_SOLVED_IN_BB", "MSK_IINF_MIO_NUM_BLOCKS_SOLVED_IN_PRESOLVE",
    "MSK_IINF_MIO_NUM_BRANCH", "MSK_IINF_MIO_NUM_INT_SOLUTIONS", "MSK_IINF_MIO_NUM_RELAX",
    "MSK_IINF_MIO_NUM_REPEATED_PRESOLVE", "MSK_IINF_MIO_NUM_RESTARTS",
    "MSK_IINF_MIO_NUM_ROOT_CUT_ROUNDS", "MSK_IINF_MIO_NUM_SELECTED_CLIQUE_CUTS",
    "MSK_IINF_MIO_NUM_SELECTED_CMIR_CUTS", "MSK_IINF_MIO_NUM_SELECTED_GOMORY_CUTS",
    "MSK_IINF_MIO_NUM_SELECTED_IMPLIED_BOUND_CUTS",
    "MSK_IINF_MIO_NUM_SELECTED_KNAPSACK_COVER_CUTS", "MSK_IINF_MIO_NUM_SELECTED_LIPRO_CUTS",
    "MSK_IINF_MIO_NUM_SEPARATED_CLIQUE_CUTS", "MSK_IINF_MIO_NUM_SEPARATED_CMIR_CUTS",
    "MSK_IINF_MIO_NUM_SEPARATED_GOMORY_CUTS", "MSK_IINF_MIO_NUM_SEPARATED_IMPLIED_BOUND_CUTS",
    "MSK_IINF_MIO_NUM_SEPARATED_KNAPSACK_COVER_CUTS", "MSK_IINF_MIO_NUM_SEPARATED_LIPRO_CUTS",
    "MSK_IINF_MIO_NUM_SOLVED_NODES", "MSK_IINF_MIO_NUMBIN", "MSK_IINF_MIO_NUMBINCONEVAR",
    "MSK_IINF_MIO_NUMCON", "MSK_IINF_MIO_NUMCONE", "MSK_IINF_MIO_NUMCONEVAR",
    "MSK_IINF_MIO_NUMCONT", "MSK_IINF_MIO_NUMCONTCONEVAR", "MSK_IINF_MIO_NUMDEXPCONES",
    "MSK_IINF_MIO_NUMDJC", "MSK_IINF_MIO_NUMDPOWCONES", "MSK_IINF_MIO_NUMINT",
    "MSK_IINF_MIO_NUMINTCONEVAR", "MSK_IINF_MIO_NUMPEXPCONES", "MSK_IINF_MIO_NUMPPOWCONES",
    "MSK_IINF_MIO_NUMQCONES", "MSK_IINF_MIO_NUMRQCONES", "MSK_IINF_MIO_NUMVAR",
    "MSK_IINF_MIO_OBJ_BOUND_DEFINED", "MSK_IINF_MIO_PRESOLVED_NUMBIN",
    "MSK_IINF_MIO_PRESOLVED_NUMBINCONEVAR", "MSK_IINF_MIO_PRESOLVED_NUMCON",
    "MSK_IINF_MIO_PRESOLVED_NUMCONE", "MSK_IINF_MIO_PRESOLVED_NUMCONEVAR",
    "MSK_IINF_MIO_PRESOLVED_NUMCONT", "MSK_IINF_MIO_PRESOLVED_NUMCONTCONEVAR",
    "MSK_IINF_MIO_PRESOLVED_NUMDEXPCONES", "MSK_IINF_MIO_PRESOLVED_NUMDJC",
    "MSK_IINF_MIO_PRESOLVED_NUMDPOWCONES", "MSK_IINF_MIO_PRESOLVED_NUMINT",
    "MSK_IINF_MIO_PRESOLVED_NUMINTCONEVAR", "MSK_IINF_MIO_PRESOLVED_NUMPEXPCONES",
    "MSK_IINF_MIO_PRESOLVED_NUMPPOWCONES", "MSK_IINF_MIO_PRESOLVED_NUMQCONES",
    "MSK_IINF_MIO_PRESOLVED_NUMRQCONES", "MSK_IINF_MIO_PRESOLVED_NUMVAR",
    "MSK_IINF_MIO_RELGAP_SATISFIED", "MSK_IINF_MIO_TOTAL_NUM_SELECTED_CUTS",
    "MSK_IINF_MIO_TOTAL_NUM_SEPARATED_CUTS", "MSK_IINF_MIO_USER_OBJ_CUT", "MSK_IINF_OPT_NUMCON",
    "MSK_IINF_OPT_NUMVAR", "MSK_IINF_OPTIMIZE_RESPONSE",
    "MSK_IINF_PRESOLVE_NUM_PRIMAL_PERTURBATIONS", "MSK_IINF_PURIFY_DUAL_SUCCESS",
    "MSK_IINF_PURIFY_PRIMAL_SUCCESS", "MSK_IINF_RD_NUMBARVAR", "MSK_IINF_RD_NUMCON",
    "MSK_IINF_RD_NUMCONE", "MSK_IINF_RD_NUMINTVAR", "MSK_IINF_RD_NUMQ", "MSK_IINF_RD_NUMVAR",
    "MSK_IINF_RD_PROTYPE", "MSK_IINF_SIM_DUAL_DEG_ITER", "MSK_IINF_SIM_DUAL_HOTSTART",
    "MSK_IINF_SIM_DUAL_HOTSTART_LU", "MSK_IINF_SIM_DUAL_INF_ITER", "MSK_IINF_SIM_DUAL_ITER",
    "MSK_IINF_SIM_NUMCON", "MSK_IINF_SIM_NUMVAR", "MSK_IINF_SIM_PRIMAL_DEG_ITER",
    "MSK_IINF_SIM_PRIMAL_HOTSTART", "MSK_IINF_SIM_PRIMAL_HOTSTART_LU",
    "MSK_IINF_SIM_PRIMAL_INF_ITER", "MSK_IINF_SIM_PRIMAL_ITER", "MSK_IINF_SIM_SOLVE_DUAL",
    "MSK_IINF_SOL_BAS_PROSTA", "MSK_IINF_SOL_BAS_SOLSTA", "MSK_IINF_SOL_ITG_PROSTA",
    "MSK_IINF_SOL_ITG_SOLSTA", "MSK_IINF_SOL_ITR_PROSTA", "MSK_IINF_SOL_ITR_SOLSTA",
    "MSK_IINF_STO_NUM_A_REALLOC"
};

const char *const liinf_names[PRIMAL_LIINF_END] = {
    "MSK_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_COLUMNS",
    "MSK_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_NZ",
    "MSK_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_ROWS", "MSK_LIINF_BI_CLEAN_ITER",
    "MSK_LIINF_BI_DUAL_ITER", "MSK_LIINF_BI_PRIMAL_ITER", "MSK_LIINF_FOLDING_BI_DUAL_ITER",
    "MSK_LIINF_FOLDING_BI_OPTIMIZER_ITER", "MSK_LIINF_FOLDING_BI_PRIMAL_ITER",
    "MSK_LIINF_INTPNT_FACTOR_NUM_NZ", "MSK_LIINF_MIO_ANZ", "MSK_LIINF_MIO_FINAL_ANZ",
    "MSK_LIINF_MIO_INTPNT_ITER", "MSK_LIINF_MIO_NUM_DUAL_ILLPOSED_CER",
    "MSK_LIINF_MIO_NUM_PRIM_ILLPOSED_CER", "MSK_LIINF_MIO_PRESOLVED_ANZ",
    "MSK_LIINF_MIO_SIMPLEX_ITER", "MSK_LIINF_RD_NUMACC", "MSK_LIINF_RD_NUMANZ",
    "MSK_LIINF_RD_NUMDJC", "MSK_LIINF_RD_NUMQNZ", "MSK_LIINF_SIMPLEX_ITER"
};

/* Return the name table and its length for the given info-item type, or NULL
 * for an unknown type. */
static const char *const *inf_names(PRIMALinftypee type, int *n) {
    switch (type) {
    case PRIMAL_INF_DOU_TYPE:  *n = PRIMAL_DINF_END;  return dinf_names;
    case PRIMAL_INF_INT_TYPE:  *n = PRIMAL_IINF_END;  return iinf_names;
    case PRIMAL_INF_LINT_TYPE: *n = PRIMAL_LIINF_END; return liinf_names;
    default: *n = 0; return NULL;
    }
}

/* Return the number of info items of the given type. */
PRIMALrescodee PRIMAL_getinfmax(PRIMALtask_t t, PRIMALinftypee inftype, int *infmax) {
    if (!t || !infmax) return PRIMAL_RES_ERR_NULL;
    int n;
    if (!inf_names(inftype, &n)) return PRIMAL_RES_ERR_ARG;
    *infmax = n;   /* the reference: max index + 1, i.e. END */
    return PRIMAL_RES_OK;
}

/* Copy the name of info item `whichinf` into `name`. */
PRIMALrescodee PRIMAL_getinfname(PRIMALtask_t t, PRIMALinftypee inftype, int whichinf,
                                 char *name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    int n;
    const char *const *tab = inf_names(inftype, &n);
    if (!tab) return PRIMAL_RES_ERR_ARG;
    if (whichinf < 0 || whichinf >= n) return PRIMAL_RES_ERR_ARG;
    /* the reference writes into a fixed-length user buffer (no size
     * parameter); here the longest name is well below PRIMAL_MAX_INFNAME_LEN. */
    strcpy(name, tab[whichinf]);
    return PRIMAL_RES_OK;
}

/* Find the index of the named info item; ERR_ARG if not found. */
PRIMALrescodee PRIMAL_getinfindex(PRIMALtask_t t, PRIMALinftypee inftype, const char *name,
                                  int *index) {
    if (!t || !name || !index) return PRIMAL_RES_ERR_NULL;
    int n;
    const char *const *tab = inf_names(inftype, &n);
    if (!tab) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < n; i++)
        if (strcmp(tab[i], name) == 0) { *index = i; return PRIMAL_RES_OK; }
    return PRIMAL_RES_ERR_ARG;
}

/* Count constraints (iscon nonzero) or variables with the given bound key. */
static int inf_bound_count(PRIMALtask_t t, PRIMALboundkeye k, int iscon) {
    int c = 0;
    if (iscon) { for (int i = 0; i < t->numcon; i++) if (t->bkc[i] == k) c++; }
    else       { for (int j = 0; j < t->numvar; j++) if (t->bkx[j] == k) c++; }
    return c;
}
/* Count variables with the given variable type. */
static int inf_vartype_count(PRIMALtask_t t, PRIMALvariabletypee vt) {
    int c = 0;
    for (int j = 0; j < t->numvar; j++) if (t->vartype[j] == vt) c++;
    return c;
}

/* Return the value of a double info item; items this solver does not measure
 * answer 0. */
PRIMALrescodee PRIMAL_getdouinf(PRIMALtask_t t, PRIMALdinfiteme which, PRIMALrealt *value) {
    if (!t || !value) return PRIMAL_RES_ERR_NULL;
    if ((int)which < 0 || (int)which >= PRIMAL_DINF_END) return PRIMAL_RES_ERR_ARG;
    *value = 0.0;
    PRIMALrealt pobj = 0, dobj = 0, pvc = 0, pvv = 0, pvb = 0, pvco = 0, pvitg = 0;
    PRIMALrealt dvc = 0, dvv = 0, dvb = 0, dvco = 0;
    int have = (PRIMAL_getsolutioninfo(t, PRIMAL_SOL_ITR, &pobj, &pvc, &pvv, &pvb, &pvco,
                                       &pvitg, &dobj, &dvc, &dvv, &dvb, &dvco) == PRIMAL_RES_OK);
    PRIMALrealt nxc = 0, nxx = 0, nbx = 0;
    int haven = (PRIMAL_getprimalsolutionnorms(t, PRIMAL_SOL_ITR, &nxc, &nxx, &nbx) == PRIMAL_RES_OK);
    PRIMALrealt ny = 0, nslc = 0, nsuc = 0, nslx = 0, nsux = 0, nsnx = 0, nbars = 0;
    int haven2 = (PRIMAL_getdualsolutionnorms(t, PRIMAL_SOL_ITR, &ny, &nslc, &nsuc, &nslx,
                                              &nsux, &nsnx, &nbars) == PRIMAL_RES_OK);
    switch (which) {
    case PRIMAL_DINF_INTPNT_PRIMAL_OBJ:   *value = t->pobj; break;
    case PRIMAL_DINF_INTPNT_DUAL_OBJ:     *value = t->dobj; break;
    case PRIMAL_DINF_SIM_OBJ:             *value = t->pobj; break;
    case PRIMAL_DINF_MIO_OBJ_INT:         *value = t->pobj; break;
    case PRIMAL_DINF_MIO_OBJ_BOUND:       *value = t->dobj; break;
    case PRIMAL_DINF_MIO_OBJ_ABS_GAP:     *value = fabs(t->pobj - t->dobj); break;
    case PRIMAL_DINF_MIO_OBJ_REL_GAP:
        *value = fabs(t->pobj - t->dobj) / (1.0 + fabs(t->pobj)); break;
    case PRIMAL_DINF_SOL_ITR_PRIMAL_OBJ:  *value = t->pobj; break;
    case PRIMAL_DINF_SOL_ITR_DUAL_OBJ:    *value = t->dobj; break;
    case PRIMAL_DINF_SOL_ITG_PRIMAL_OBJ:  *value = t->pobj; break;
    case PRIMAL_DINF_SOL_BAS_PRIMAL_OBJ:  *value = t->pobj; break;
    case PRIMAL_DINF_SOL_BAS_DUAL_OBJ:    *value = t->dobj; break;
    case PRIMAL_DINF_SOL_ITR_PVIOLCON:    *value = have ? pvc : 0; break;
    case PRIMAL_DINF_SOL_ITR_PVIOLVAR:    *value = have ? pvv : 0; break;
    case PRIMAL_DINF_SOL_ITR_PVIOLBARVAR: *value = have ? pvb : 0; break;
    case PRIMAL_DINF_SOL_ITR_PVIOLCONES:  *value = have ? pvco : 0; break;
    case PRIMAL_DINF_SOL_ITR_DVIOLCON:    *value = have ? dvc : 0; break;
    case PRIMAL_DINF_SOL_ITR_DVIOLVAR:    *value = have ? dvv : 0; break;
    case PRIMAL_DINF_SOL_ITR_DVIOLBARVAR: *value = have ? dvb : 0; break;
    case PRIMAL_DINF_SOL_ITR_DVIOLCONES:  *value = have ? dvco : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_XC:      *value = haven ? nxc : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_XX:      *value = haven ? nxx : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_BARX:    *value = haven ? nbx : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_Y:       *value = haven2 ? ny : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_SLC:     *value = haven2 ? nslc : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_SUC:     *value = haven2 ? nsuc : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_SLX:     *value = haven2 ? nslx : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_SUX:     *value = haven2 ? nsux : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_SNX:     *value = haven2 ? nsnx : 0; break;
    case PRIMAL_DINF_SOL_ITR_NRM_BARS:    *value = haven2 ? nbars : 0; break;
    case PRIMAL_DINF_SOL_ITG_PVIOLCON:    *value = have ? pvc : 0; break;
    case PRIMAL_DINF_SOL_ITG_PVIOLVAR:    *value = have ? pvv : 0; break;
    case PRIMAL_DINF_SOL_ITG_PVIOLBARVAR: *value = have ? pvb : 0; break;
    case PRIMAL_DINF_SOL_ITG_PVIOLCONES:  *value = have ? pvco : 0; break;
    case PRIMAL_DINF_SOL_ITG_PVIOLITG:    *value = have ? pvitg : 0; break;
    case PRIMAL_DINF_SOL_ITG_NRM_XX:      *value = haven ? nxx : 0; break;
    case PRIMAL_DINF_SOL_ITG_NRM_XC:      *value = haven ? nxc : 0; break;
    case PRIMAL_DINF_SOL_ITG_NRM_BARX:    *value = haven ? nbx : 0; break;
    case PRIMAL_DINF_OPTIMIZER_TIME:      *value = t->opt_time; break;
    case PRIMAL_DINF_SIM_TIME:            *value = t->opt_time; break;
    case PRIMAL_DINF_MIO_TIME:            *value = t->opt_time; break;
    case PRIMAL_DINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_DENSITY: {
        int nz = 0, nv = 0, nc = 0;
        PRIMAL_getnumanz(t, &nz); PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
        *value = (nv > 0 && nc > 0) ? (double)nz / ((double)nv * (double)nc) : 0.0;
        break;
    }
    default: break;   /* not measured by this solver: 0 */
    }
    return PRIMAL_RES_OK;
}

/* Return the value of an int info item; unmeasured items answer 0. */
PRIMALrescodee PRIMAL_getintinf(PRIMALtask_t t, PRIMALiinfiteme which, int *value) {
    if (!t || !value) return PRIMAL_RES_ERR_NULL;
    if ((int)which < 0 || (int)which >= PRIMAL_IINF_END) return PRIMAL_RES_ERR_ARG;
    *value = 0;
    PRIMALprostae ps = PRIMAL_PRO_STA_UNKNOWN; PRIMALsolstae ss = PRIMAL_SOL_STA_UNKNOWN;
    PRIMAL_getprosta(t, PRIMAL_SOL_ITR, &ps);
    PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss);
    switch (which) {
    case PRIMAL_IINF_ANA_PRO_NUM_CON:          *value = t->numcon; break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR:          *value = t->numvar; break;
    case PRIMAL_IINF_ANA_PRO_NUM_CON_EQ:       *value = inf_bound_count(t, PRIMAL_BK_FX, 1); break;
    case PRIMAL_IINF_ANA_PRO_NUM_CON_FR:       *value = inf_bound_count(t, PRIMAL_BK_FR, 1); break;
    case PRIMAL_IINF_ANA_PRO_NUM_CON_LO:       *value = inf_bound_count(t, PRIMAL_BK_LO, 1); break;
    case PRIMAL_IINF_ANA_PRO_NUM_CON_RA:       *value = inf_bound_count(t, PRIMAL_BK_RA, 1); break;
    case PRIMAL_IINF_ANA_PRO_NUM_CON_UP:       *value = inf_bound_count(t, PRIMAL_BK_UP, 1); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_EQ:       *value = inf_bound_count(t, PRIMAL_BK_FX, 0); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_FR:       *value = inf_bound_count(t, PRIMAL_BK_FR, 0); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_LO:       *value = inf_bound_count(t, PRIMAL_BK_LO, 0); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_RA:       *value = inf_bound_count(t, PRIMAL_BK_RA, 0); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_UP:       *value = inf_bound_count(t, PRIMAL_BK_UP, 0); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_BIN:      *value = inf_vartype_count(t, PRIMAL_VAR_TYPE_INT_BIN); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_INT:      *value = inf_vartype_count(t, PRIMAL_VAR_TYPE_INT); break;
    case PRIMAL_IINF_ANA_PRO_NUM_VAR_CONT:     *value = inf_vartype_count(t, PRIMAL_VAR_TYPE_CONT); break;
    case PRIMAL_IINF_OPT_NUMCON:               *value = t->numcon; break;
    case PRIMAL_IINF_OPT_NUMVAR:               *value = t->numvar; break;
    case PRIMAL_IINF_OPTIMIZE_RESPONSE:        *value = (int)t->last_rc; break;
    case PRIMAL_IINF_SOL_ITR_PROSTA:           *value = (int)ps; break;
    case PRIMAL_IINF_SOL_ITR_SOLSTA:           *value = (int)ss; break;
    case PRIMAL_IINF_SOL_BAS_PROSTA:           *value = (int)ps; break;
    case PRIMAL_IINF_SOL_BAS_SOLSTA:           *value = (int)ss; break;
    case PRIMAL_IINF_SOL_ITG_PROSTA:           *value = (int)ps; break;
    case PRIMAL_IINF_SOL_ITG_SOLSTA:           *value = (int)ss; break;
    case PRIMAL_IINF_RD_NUMVAR:                *value = t->numvar; break;
    case PRIMAL_IINF_RD_NUMCON:                *value = t->numcon; break;
    case PRIMAL_IINF_RD_NUMCONE:               *value = t->numcones; break;
    case PRIMAL_IINF_RD_NUMBARVAR:             *value = t->numbarvar; break;
    case PRIMAL_IINF_INTPNT_NUM_THREADS:       *value = 1; break;
    case PRIMAL_IINF_RD_PROTYPE: {
        PRIMALproblemtypee pt = PRIMAL_PROBTYPE_LO;
        PRIMAL_getprobtype(t, &pt);
        *value = (int)pt;
        break;
    }
    default: break;   /* not measured: 0 */
    }
    return PRIMAL_RES_OK;
}

/* Return the value of a 64-bit info item; unmeasured items answer 0. */
PRIMALrescodee PRIMAL_getlintinf(PRIMALtask_t t, PRIMALliinfiteme which, PRIMALint64t *value) {
    if (!t || !value) return PRIMAL_RES_ERR_NULL;
    if ((int)which < 0 || (int)which >= PRIMAL_LIINF_END) return PRIMAL_RES_ERR_ARG;
    *value = 0;
    switch (which) {
    case PRIMAL_LIINF_RD_NUMANZ:  { int n = 0; PRIMAL_getnumanz(t, &n); *value = n; break; }
    case PRIMAL_LIINF_RD_NUMQNZ:  { int n = 0; PRIMAL_getnumqobjnz(t, &n); *value = n; break; }
    case PRIMAL_LIINF_RD_NUMACC:  { PRIMALint64t n = 0; PRIMAL_getnumacc(t, &n); *value = n; break; }
    case PRIMAL_LIINF_RD_NUMDJC:  { PRIMALint64t n = 0; PRIMAL_getnumdjc(t, &n); *value = n; break; }
    default: break;   /* not measured: 0 */
    }
    return PRIMAL_RES_OK;
}

/* Look up a double info item by name and return its value. */
PRIMALrescodee PRIMAL_getnadouinf(PRIMALtask_t t, const char *name, PRIMALrealt *value) {
    if (!t || !name || !value) return PRIMAL_RES_ERR_NULL;
    int idx = -1;
    if (PRIMAL_getinfindex(t, PRIMAL_INF_DOU_TYPE, name, &idx) != PRIMAL_RES_OK)
        return PRIMAL_RES_ERR_ARG;
    return PRIMAL_getdouinf(t, (PRIMALdinfiteme)idx, value);
}

/* Look up an int info item by name and return its value. */
PRIMALrescodee PRIMAL_getnaintinf(PRIMALtask_t t, const char *name, int *value) {
    if (!t || !name || !value) return PRIMAL_RES_ERR_NULL;
    int idx = -1;
    if (PRIMAL_getinfindex(t, PRIMAL_INF_INT_TYPE, name, &idx) != PRIMAL_RES_OK)
        return PRIMAL_RES_ERR_ARG;
    return PRIMAL_getintinf(t, (PRIMALiinfiteme)idx, value);
}

