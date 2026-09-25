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

/* stdform.c - general problem -> standard form transformation (SPARSE)
 *
 * Standard form:  min cs'xt (+1/2 xt'Qs xt)  s.t. As xt = bs, xt >= 0, bs >= 0.
 *
 * Transformations applied:
 *  - fixed variables (lx==ux) are substituted out (folded into ceff/cfix and
 *    constraint right-hand sides)
 *  - finite lower bound:  x = lx + xt            (shift)
 *  - only upper bound:    x = ux - xt            (negated column)
 *  - free variable:       x = xp - xq            (two columns)
 *  - finite upper bound on a shifted variable: extra row xt + s = ux - lx
 *  - constraints: FR dropped; FX -> equality; LO -> A x - s = lc;
 *    UP -> A x + s = uc; ranged -> LO row + UP row
 *  - rows with negative rhs are negated (sigma = -1) so bs >= 0
 *
 * As and Qs are built SPARSE (CSC); Qs is lower-triangular with an explicit
 * diagonal in every column (so the sparse QP IPM can add Z/X to it directly). */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "stdform.h"

static int is_fin(double v) { return !isinf(v) && !isnan(v); }

/* ---- growable triplet list ---- */
typedef struct { int *r, *c; double *v; int n, cap; } Tri;
static int tri_add(Tri *t, int r, int c, double v) {
    if (t->n == t->cap) {
        int nc = t->cap ? t->cap * 2 : 256;
        int *nr = (int *)realloc(t->r, (size_t)nc * sizeof(int));
        int *ncol = (int *)realloc(t->c, (size_t)nc * sizeof(int));
        double *nv = (double *)realloc(t->v, (size_t)nc * sizeof(double));
        if (!nr || !ncol || !nv) { free(nr); free(ncol); free(nv); return -1; }
        t->r = nr; t->c = ncol; t->v = nv; t->cap = nc;
    }
    t->r[t->n] = r; t->c[t->n] = c; t->v[t->n] = v; t->n++;
    return 0;
}
static void tri_free(Tri *t) { free(t->r); free(t->c); free(t->v); t->r = t->c = NULL; t->v = NULL; t->n = t->cap = 0; }

/* build CSC from triplets over ncol columns, summing duplicate (row,col) pairs
 * and sorting rows ascending within each column. Returns 0 ok / -1 alloc fail. */
static int tri_to_csc(int ncol, const Tri *t, int **optr, int **orow, double **oval) {
    int nt = t->n;
    int *cnt = (int *)calloc((size_t)(ncol > 0 ? ncol : 1), sizeof(int));
    int *ptr = (int *)malloc((size_t)(ncol + 1) * sizeof(int));
    int *row = (int *)malloc((size_t)(nt > 0 ? nt : 1) * sizeof(int));
    double *val = (double *)malloc((size_t)(nt > 0 ? nt : 1) * sizeof(double));
    int *cur = (int *)malloc((size_t)(ncol > 0 ? ncol : 1) * sizeof(int));
    if (!cnt || !ptr || !row || !val || !cur) { free(cnt); free(ptr); free(row); free(val); free(cur); return -1; }
    for (int e = 0; e < nt; e++) cnt[t->c[e]]++;
    ptr[0] = 0;
    for (int j = 0; j < ncol; j++) { ptr[j + 1] = ptr[j] + cnt[j]; cur[j] = ptr[j]; }
    for (int e = 0; e < nt; e++) { int j = t->c[e]; row[cur[j]] = t->r[e]; val[cur[j]] = t->v[e]; cur[j]++; }
    free(cnt); free(cur);
    int *nptr = (int *)malloc((size_t)(ncol + 1) * sizeof(int));
    if (!nptr) { free(ptr); free(row); free(val); return -1; }
    int w = 0; nptr[0] = 0;
    for (int j = 0; j < ncol; j++) {
        int s = ptr[j], en = ptr[j + 1];
        for (int a = s + 1; a < en; a++) {        /* insertion sort by row */
            int rr = row[a]; double vv = val[a]; int bp = a - 1;
            while (bp >= s && row[bp] > rr) { row[bp + 1] = row[bp]; val[bp + 1] = val[bp]; bp--; }
            row[bp + 1] = rr; val[bp + 1] = vv;
        }
        for (int a = s; a < en; a++) {             /* dedup-sum adjacent */
            if (a > s && row[a] == row[a - 1]) val[w - 1] += val[a];
            else { row[w] = row[a]; val[w] = val[a]; w++; }
        }
        nptr[j + 1] = w;
    }
    free(ptr);
    *optr = nptr; *orow = row; *oval = val;
    return 0;
}

StdForm *stdform_build(int nvar, int ncon,
                       const double *c_int,
                       const int *qi, const int *qj, const double *qv, int nq,
                       const double *lx, const double *ux,
                       const double *lc, const double *uc,
                       const int *col_ptr, const int *sub, const double *val) {
    if (nvar < 0 || ncon < 0) return NULL;
    int hasQ = (nq > 0 && qi && qj && qv);

    for (int j = 0; j < nvar; j++)
        if (is_fin(lx[j]) && is_fin(ux[j]) && lx[j] > ux[j] + 1e-12 * (1.0 + fabs(lx[j]))) return NULL;
    for (int i = 0; i < ncon; i++)
        if (is_fin(lc[i]) && is_fin(uc[i]) && lc[i] > uc[i] + 1e-12 * (1.0 + fabs(lc[i]))) return NULL;

    StdForm *sf = (StdForm *)calloc(1, sizeof(StdForm));
    if (!sf) return NULL;
    sf->nvar = nvar; sf->ncon = ncon;
    sf->vars = (SfrVar *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(SfrVar));
    if (!sf->vars) { stdform_free(sf); return NULL; }

    int *fixed = (int *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(int));
    double *vfix = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    if (!fixed || !vfix) { free(fixed); free(vfix); stdform_free(sf); return NULL; }
    for (int j = 0; j < nvar; j++)
        if (is_fin(lx[j]) && is_fin(ux[j]) && ux[j] - lx[j] <= 1e-12 * (1.0 + fabs(lx[j]))) {
            fixed[j] = 1; vfix[j] = lx[j];
            sf->vars[j].fixed = 1; sf->vars[j].fixval = lx[j];
        }

    /* ---- fold fixed variables (sparse Q) ---- */
    double *ceff = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc2 = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc2 = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    if (!ceff || !lc2 || !uc2) { free(fixed); free(vfix); free(ceff); free(lc2); free(uc2); stdform_free(sf); return NULL; }
    for (int j = 0; j < nvar; j++) ceff[j] = c_int[j];
    double cfix = 0.0;
    for (int j = 0; j < nvar; j++) if (fixed[j]) cfix += c_int[j] * vfix[j];
    if (hasQ) {
        for (int e = 0; e < nq; e++) {
            int a = qi[e], b = qj[e]; double v = qv[e];
            if (a == b) {
                if (fixed[a]) cfix += 0.5 * v * vfix[a] * vfix[a];
            } else {
                if (fixed[b] && !fixed[a]) ceff[a] += v * vfix[b];
                if (fixed[a] && !fixed[b]) ceff[b] += v * vfix[a];
                if (fixed[a] && fixed[b]) cfix += v * vfix[a] * vfix[b];
            }
        }
    }
    double *frow = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    if (!frow) { free(fixed); free(vfix); free(ceff); free(lc2); free(uc2); stdform_free(sf); return NULL; }
    for (int j = 0; j < nvar; j++)
        if (fixed[j])
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) frow[sub[k]] += val[k] * vfix[j];
    for (int i = 0; i < ncon; i++) { lc2[i] = lc[i] - frow[i]; uc2[i] = uc[i] - frow[i]; }
    free(frow); free(fixed); free(vfix);

    /* ---- column assignment for non-fixed variables ---- */
    int ncol = 0;
    for (int j = 0; j < nvar; j++) {
        if (sf->vars[j].fixed) { sf->vars[j].ncols = 0; continue; }
        if (!is_fin(lx[j]) && !is_fin(ux[j])) {
            sf->vars[j].ncols = 2;
            sf->vars[j].col[0] = ncol;     sf->vars[j].tau[0] = 1.0;
            sf->vars[j].col[1] = ncol + 1; sf->vars[j].tau[1] = -1.0;
            sf->vars[j].shift = 0.0; ncol += 2;
        } else if (is_fin(lx[j])) {
            sf->vars[j].ncols = 1; sf->vars[j].col[0] = ncol; sf->vars[j].tau[0] = 1.0;
            sf->vars[j].shift = lx[j]; ncol += 1;
        } else {
            sf->vars[j].ncols = 1; sf->vars[j].col[0] = ncol; sf->vars[j].tau[0] = -1.0;
            sf->vars[j].shift = ux[j]; ncol += 1;
        }
    }
    int nstruct = ncol;

    /* ---- row bounds after shifts: lc3 = lc2 - sum_j A_ij shift_j ---- */
    double *sh = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    if (!sh) { free(ceff); free(lc2); free(uc2); stdform_free(sf); return NULL; }
    for (int j = 0; j < nvar; j++)
        if (!sf->vars[j].fixed)
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) sh[sub[k]] += val[k] * sf->vars[j].shift;
    double *lc3 = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc3 = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    if (!lc3 || !uc3) { free(sh); free(ceff); free(lc2); free(uc2); free(lc3); free(uc3); stdform_free(sf); return NULL; }
    for (int i = 0; i < ncon; i++) { lc3[i] = lc2[i] - sh[i]; uc3[i] = uc2[i] - sh[i]; }
    free(sh); free(lc2); free(uc2);

    int nrow = 0;
    for (int i = 0; i < ncon; i++) {
        int flo = is_fin(lc3[i]), fup = is_fin(uc3[i]);
        if (!flo && !fup) continue;
        if (flo && fup && fabs(lc3[i] - uc3[i]) <= 1e-12 * (1.0 + fabs(lc3[i]))) nrow += 1;
        else if (!flo) nrow += 1;
        else if (!fup) nrow += 1;
        else nrow += 2;
    }
    for (int j = 0; j < nvar; j++)
        if (!sf->vars[j].fixed && is_fin(lx[j]) && is_fin(ux[j])) nrow += 1;

    sf->m = nrow;
    sf->n = nstruct + nrow;
    sf->b = (double *)calloc((size_t)(nrow > 0 ? nrow : 1), sizeof(double));
    sf->c = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    sf->rows = (SfrRow *)calloc((size_t)(nrow > 0 ? nrow : 1), sizeof(SfrRow));
    sf->cols = (SfrCol *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(SfrCol));
    if (!sf->b || !sf->c || !sf->rows || !sf->cols) { free(ceff); free(lc3); free(uc3); stdform_free(sf); return NULL; }
    for (int rr = 0; rr < nrow; rr++) { sf->rows[rr].sigma = 1.0; sf->rows[rr].orig = -1; sf->rows[rr].kind = SFRK_EQ; }
    for (int k = 0; k < sf->n; k++) { sf->cols[k].kind = SFCK_VAR; sf->cols[k].idx = -1; sf->cols[k].tau = 0.0; }
    for (int j = 0; j < nvar; j++)
        for (int t = 0; t < sf->vars[j].ncols; t++) {
            int k = sf->vars[j].col[t];
            sf->cols[k].kind = SFCK_VAR; sf->cols[k].idx = j; sf->cols[k].tau = sf->vars[j].tau[t];
        }

    /* ---- c_std (sparse Q shift) ---- */
    double *qshift = hasQ ? (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double)) : NULL;
    if (hasQ && !qshift) { free(ceff); free(lc3); free(uc3); stdform_free(sf); return NULL; }
    if (hasQ)
        for (int e = 0; e < nq; e++) {
            int a = qi[e], b = qj[e]; double v = qv[e];
            if (sf->vars[a].fixed) { if (a != b && !sf->vars[b].fixed) qshift[b] += v * sf->vars[a].shift; continue; }
            if (a == b) { qshift[a] += v * sf->vars[a].shift; }
            else {
                if (!sf->vars[b].fixed) qshift[a] += v * sf->vars[b].shift;
                if (!sf->vars[a].fixed) qshift[b] += v * sf->vars[a].shift;
            }
        }
    for (int j = 0; j < nvar; j++) {
        if (sf->vars[j].fixed) continue;
        double base = ceff[j] + (qshift ? qshift[j] : 0.0);
        if (qshift) cfix += 0.5 * sf->vars[j].shift * qshift[j];
        cfix += ceff[j] * sf->vars[j].shift;
        for (int t = 0; t < sf->vars[j].ncols; t++) sf->c[sf->vars[j].col[t]] = sf->vars[j].tau[t] * base;
    }
    free(qshift); free(ceff);

    /* ---- input A as CSR (by constraint) for fast row scatter ---- */
    int nnzA = col_ptr[nvar];
    int *csr_ptr = (int *)calloc((size_t)(ncon + 1), sizeof(int));
    int *csr_j = (int *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(int));
    double *csr_v = (double *)malloc((size_t)(nnzA > 0 ? nnzA : 1) * sizeof(double));
    if (!csr_ptr || !csr_j || !csr_v) { free(csr_ptr); free(csr_j); free(csr_v); free(lc3); free(uc3); stdform_free(sf); return NULL; }
    for (int j = 0; j < nvar; j++)
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) csr_ptr[sub[k] + 1]++;
    for (int i = 0; i < ncon; i++) csr_ptr[i + 1] += csr_ptr[i];
    { int *fill = (int *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(int));
      if (!fill) { free(csr_ptr); free(csr_j); free(csr_v); free(lc3); free(uc3); stdform_free(sf); return NULL; }
      for (int j = 0; j < nvar; j++)
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
            int i = sub[k]; int p = csr_ptr[i] + fill[i]++; csr_j[p] = j; csr_v[p] = val[k];
        }
      free(fill); }

    /* ---- emit A triplets ---- */
    Tri ta = {0};
    #define EMIT_SCATTER(i, rr, sg) do { \
        for (int kk = csr_ptr[i]; kk < csr_ptr[i + 1]; kk++) { \
            int j = csr_j[kk]; if (sf->vars[j].fixed) continue; \
            for (int t = 0; t < sf->vars[j].ncols; t++) \
                if (tri_add(&ta, (rr), sf->vars[j].col[t], csr_v[kk] * sf->vars[j].tau[t] * (sg))) goto a_fail; \
        } } while (0)
    int r = 0;
    for (int i = 0; i < ncon; i++) {
        int flo = is_fin(lc3[i]), fup = is_fin(uc3[i]);
        if (!flo && !fup) continue;
        if (flo && fup && fabs(lc3[i] - uc3[i]) <= 1e-12 * (1.0 + fabs(lc3[i]))) {
            double br = lc3[i]; double sg = br < 0.0 ? -1.0 : 1.0; sf->b[r] = sg * br;
            EMIT_SCATTER(i, r, sg);
            sf->rows[r].orig = i; sf->rows[r].kind = SFRK_EQ; sf->rows[r].sigma = sg; r++;
        } else if (!flo) {
            double br = uc3[i]; double sg = br < 0.0 ? -1.0 : 1.0; sf->b[r] = sg * br;
            EMIT_SCATTER(i, r, sg);
            if (tri_add(&ta, r, nstruct + r, 1.0 * sg)) goto a_fail;
            sf->cols[nstruct + r].kind = SFCK_SLACK; sf->cols[nstruct + r].idx = r;
            sf->rows[r].orig = i; sf->rows[r].kind = SFRK_UP; sf->rows[r].sigma = sg; r++;
        } else if (!fup) {
            double br = lc3[i]; double sg = br < 0.0 ? -1.0 : 1.0; sf->b[r] = sg * br;
            EMIT_SCATTER(i, r, sg);
            if (tri_add(&ta, r, nstruct + r, -1.0 * sg)) goto a_fail;
            sf->cols[nstruct + r].kind = SFCK_SLACK; sf->cols[nstruct + r].idx = r;
            sf->rows[r].orig = i; sf->rows[r].kind = SFRK_LO; sf->rows[r].sigma = sg; r++;
        } else {
            double br = lc3[i]; double sg = br < 0.0 ? -1.0 : 1.0; sf->b[r] = sg * br;
            EMIT_SCATTER(i, r, sg);
            if (tri_add(&ta, r, nstruct + r, -1.0 * sg)) goto a_fail;
            sf->cols[nstruct + r].kind = SFCK_SLACK; sf->cols[nstruct + r].idx = r;
            sf->rows[r].orig = i; sf->rows[r].kind = SFRK_LO; sf->rows[r].sigma = sg; r++;
            br = uc3[i]; sg = br < 0.0 ? -1.0 : 1.0; sf->b[r] = sg * br;
            EMIT_SCATTER(i, r, sg);
            if (tri_add(&ta, r, nstruct + r, 1.0 * sg)) goto a_fail;
            sf->cols[nstruct + r].kind = SFCK_SLACK; sf->cols[nstruct + r].idx = r;
            sf->rows[r].orig = i; sf->rows[r].kind = SFRK_UP; sf->rows[r].sigma = sg; r++;
        }
    }
    for (int j = 0; j < nvar; j++) {
        if (sf->vars[j].fixed || !is_fin(lx[j]) || !is_fin(ux[j])) continue;
        int c0 = sf->vars[j].col[0];
        if (tri_add(&ta, r, c0, 1.0)) goto a_fail;
        if (tri_add(&ta, r, nstruct + r, 1.0)) goto a_fail;
        sf->cols[nstruct + r].kind = SFCK_SLACK; sf->cols[nstruct + r].idx = r;
        sf->b[r] = ux[j] - lx[j];
        sf->rows[r].orig = j; sf->rows[r].kind = SFRK_VARUB; sf->rows[r].sigma = 1.0; r++;
    }
    sf->n = nstruct + r;
    if (tri_to_csc(sf->n, &ta, &sf->Aptr, &sf->Arow, &sf->Aval)) goto a_fail;
    tri_free(&ta);
    free(csr_ptr); free(csr_j); free(csr_v); free(lc3); free(uc3);

    /* ---- Q_std (sparse, lower CSC with explicit diagonal) ---- */
    if (hasQ) {
        Tri tq = {0};
        for (int j = 0; j < sf->n; j++) if (tri_add(&tq, j, j, 0.0)) { tri_free(&tq); goto qfail; }
        for (int e = 0; e < nq; e++) {
            int a = qi[e], b = qj[e]; double v = qv[e];
            if (sf->vars[a].fixed || sf->vars[b].fixed) continue;
            for (int ta2 = 0; ta2 < sf->vars[a].ncols; ta2++)
                for (int tb2 = 0; tb2 < sf->vars[b].ncols; tb2++) {
                    int P = sf->vars[a].col[ta2], R = sf->vars[b].col[tb2];
                    double w = sf->vars[a].tau[ta2] * sf->vars[b].tau[tb2] * v;
                    if (a == b) { if (P >= R) { if (tri_add(&tq, P, R, w)) { tri_free(&tq); goto qfail; } } }
                    else { int hi = P >= R ? P : R, lo = P >= R ? R : P;
                           if (tri_add(&tq, hi, lo, w)) { tri_free(&tq); goto qfail; } }
                }
        }
        if (tri_to_csc(sf->n, &tq, &sf->Qptr, &sf->Qrow, &sf->Qval)) { tri_free(&tq); goto qfail; }
        tri_free(&tq);
    }

    sf->cfix = cfix;
    return sf;

a_fail:
    tri_free(&ta);
    free(csr_ptr); free(csr_j); free(csr_v); free(lc3); free(uc3);
    stdform_free(sf);
    return NULL;
qfail:
    free(lc3); free(uc3);
    stdform_free(sf);
    return NULL;
}

/**
 * Frees a standard-form problem and all its associated memory.
 *
 * @param sf [in] StdForm object to free.
 *
 * @note Frees A (CSC), Q (CSC), b, c, rows, cols, vars arrays, and the struct itself.
 *       Safe to call with NULL.
 *
 * @example
 * stdform_free(sf);
 */
void stdform_free(StdForm *sf) {
    if (!sf) return;
    free(sf->Aptr); free(sf->Arow); free(sf->Aval);
    free(sf->Qptr); free(sf->Qrow); free(sf->Qval);
    free(sf->b); free(sf->c);
    free(sf->rows); free(sf->cols); free(sf->vars);
    free(sf);
}

/**
 * Maps standard-form solution x back to the original model's variables.
 *
 * @param sf  [in]  Standard-form problem.
 * @param xt  [in]  Solution in standard-form space (size sf->n).
 * @param x   [out] Original model variables (size sf->nvar). Must not be NULL.
 *
 * @note For fixed variables, returns the fixed value. For free variables,
 *       x = xp - xq (two standard-form columns). For lower-bounded,
 *       x = lx + x'. For upper-bounded, x = ux - x'. For ranged,
 *       x = lx + x'.
 *
 * @example
 * double *xt = malloc(sf->n * sizeof(double));
 * double *x = malloc(sf->nvar * sizeof(double));
 * // ... solve standard form ...
 * stdform_map_x(sf, xt, x);
 * // x now holds original model solution
 */
void stdform_map_x(const StdForm *sf, const double *xt, double *x) {
    for (int j = 0; j < sf->nvar; j++) {
        if (sf->vars[j].fixed) { x[j] = sf->vars[j].fixval; continue; }
        double v = sf->vars[j].shift;
        for (int t = 0; t < sf->vars[j].ncols; t++) v += sf->vars[j].tau[t] * xt[sf->vars[j].col[t]];
        x[j] = v;
    }
}

/**
 * Maps a standard-form direction to the original model's variable space.
 *
 * @param sf  [in]  Standard-form problem.
 * @param dxt [in]  Direction in standard-form space (size sf->n).
 * @param dx  [out] Direction in original model space (size sf->nvar). Must not be NULL.
 *
 * @note Unlike stdform_map_x, this does NOT include shifts. A direction
 *       lives on the homogeneous part of the substitution:
 *       - Fixed: dx = 0
 *       - Free (xp - xq): dx = dxp - dxq
 *       - Lower-bounded: dx = dxt
 *       - Upper-bounded: dx = -dxt
 *       - Ranged: dx = dxt
 *
 * @example
 * double *dxt = malloc(sf->n * sizeof(double));
 * double *dx = malloc(sf->nvar * sizeof(double));
 * stdform_map_dir(sf, dxt, dx);
 * // dx is the direction in original model space
 */
void stdform_map_dir(const StdForm *sf, const double *dxt, double *dx) {
    for (int j = 0; j < sf->nvar; j++) {
        if (sf->vars[j].fixed) { dx[j] = 0.0; continue; }
        double v = 0.0;
        for (int t = 0; t < sf->vars[j].ncols; t++) v += sf->vars[j].tau[t] * dxt[sf->vars[j].col[t]];
        dx[j] = v;
    }
}

/**
 * Maps standard-form dual variables y back to the original model's constraints.
 *
 * @param sf   [in]  Standard-form problem.
 * @param ystd [in]  Dual solution for standard-form constraints (size sf->m).
 * @param ymin [out] Dual variables for original model constraints (size sf->ncon).
 *                     Must not be NULL.
 *
 * @note Sums sigma * ystd[r] over all standard-form rows r that map to the
 *       same original constraint i. Skips variable-bound rows (SFRK_VARUB)
 *       which don't correspond to original constraints.
 *
 * @example
 * double *ystd = malloc(sf->m * sizeof(double));
 * double *y = malloc(sf->ncon * sizeof(double));
 * stdform_map_y(sf, ystd, y);
 * // y now holds dual variables for original constraints
 */
void stdform_map_y(const StdForm *sf, const double *ystd, double *ymin) {
    for (int i = 0; i < sf->ncon; i++) ymin[i] = 0.0;
    for (int r = 0; r < sf->m; r++) {
        if (sf->rows[r].kind == SFRK_VARUB) continue;
        if (sf->rows[r].orig < 0 || sf->rows[r].orig >= sf->ncon) continue;
        ymin[sf->rows[r].orig] -= sf->rows[r].sigma * ystd[r];
    }
}

/**
 * Extracts the dense A matrix from a standard-form problem.
 *
 * @param sf [in]  Standard-form problem.
 *
 * @return Pointer to dense matrix (m x n, row-major), or NULL on allocation failure.
 *         Caller must free the returned pointer.
 *
 * @note Converts CSC A to dense row-major: A[i,j] = A[row, col].
 *       Returns NULL if sf is NULL.
 *
 * @example
 * double *A = stdform_dense_A(sf);
 * if (A) { // A[i * n + j] holds A[i,j] ... free(A); }
 */
double *stdform_dense_A(const StdForm *sf) {
    if (!sf) return NULL;
    double *A = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1) * (size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    if (!A) return NULL;
    for (int j = 0; j < sf->n; j++)
        for (int p = sf->Aptr[j]; p < sf->Aptr[j + 1]; p++) A[(size_t)sf->Arow[p] * sf->n + j] = sf->Aval[p];
    return A;
}

/**
 * Extracts the dense Q matrix from a standard-form problem.
 *
 * @param sf [in]  Standard-form problem.
 *
 * @return Pointer to dense matrix (n x n, symmetric, row-major), or NULL on allocation failure or if no Q.
 *         Caller must free the returned pointer.
 *
 * @note Converts CSC Q to dense symmetric row-major: Q[i,j] and Q[j,i] both set.
 *       Only lower triangle stored in CSC; upper triangle filled symmetrically.
 *       Returns NULL if sf is NULL or no Q matrix exists.
 *
 * @example
 * double *Q = stdform_dense_Q(sf);
 * if (Q) { // Q[i * n + j] holds Q[i,j] ... free(Q); }
 */
double *stdform_dense_Q(const StdForm *sf) {
    if (!sf || !sf->Qptr) return NULL;
    double *Q = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1) * (size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    if (!Q) return NULL;
    for (int j = 0; j < sf->n; j++)
        for (int p = sf->Qptr[j]; p < sf->Qptr[j + 1]; p++) {
            int i = sf->Qrow[p];
            Q[(size_t)i * sf->n + j] = sf->Qval[p];
            if (i != j) Q[(size_t)j * sf->n + i] = sf->Qval[p];
        }
    return Q;
}
