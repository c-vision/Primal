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

/* presolve.c - LP presolve reductions on the standard form
 *   min c'x  s.t. A x = b, x >= 0   (A SPARSE, CSC)
 *
 * Reductions (triggered by the exact zero STRUCTURE of A; only the scalar value
 * checks use tol):
 *  - empty row    (no alive col, |b_i| <= tol): redundant, removed; dual y_i = 0.
 *  - empty column (no alive row, c_j >= -tol): x_j = 0 optimal, removed.
 *  - singleton row (one alive nonzero A_ij): fixes x_j = b_i/A_ij, substitutes it
 *    into the other rows, removes row i and column j.
 *  - duplicate rows (identical alive pattern + equal RHS): keep one, drop the
 *    rest (dropped dual = 0; the kept row carries the combined dual).
 *  - duplicate columns (identical alive pattern): keep the cheapest, drop the
 *    rest with x = 0 (the kept min-cost column reproduces the original optimum).
 * Ambiguous cases (|b_i| > tol, x_j < -tol, c_j < -tol, tiny pivot) are LEFT to
 * the solver, so presolve never mis-declares infeasibility/unboundedness.
 *
 * Postsolve replays the reduction log in REVERSE. Primal: kept columns take the
 * reduced value, removed columns their recorded fixed value (0 for empty/dup).
 * Dual: kept rows take the reduced dual; a removed empty/duplicate row gets 0; a
 * removed singleton row i (which fixed column j) is recovered from KKT
 * stationarity of column j with reduced cost z_j = 0:
 *     y_i = (c_j - sum_{k != i} A_kj y_k) / A_ij.
 * Every row k != i touching column j is still alive when row i is reduced (two
 * singleton rows cannot share a live column), so the reverse pass has y_k. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "presolve.h"

enum { OP_EMPTY_ROW = 0, OP_SINGLETON = 1 };

typedef struct { int type, i, j; double Aij, cj; int nk; int *ks; double *Akj; } POp;

struct Presolve {
    int m, n, mr, nr, changed;
    int *Arptr, *Arrow; double *Arval;   /* reduced A (CSC) */
    double *br, *cr;
    int *row_map, *col_map;
    double *col_fix;
    POp *ops; int nops, opcap;
};

static unsigned long long hfold(unsigned long long h, int idx, double v) {
    unsigned long long vb; memcpy(&vb, &v, sizeof(vb));
    h = (h ^ (unsigned long long)idx) * 1099511628211ULL;
    h = (h ^ vb) * 1099511628211ULL;
    return h;
}

static int push_op(Presolve *p, int type, int i, int j, double Aij, double cj,
                   int nk, int *ks, double *Akj) {
    if (p->nops == p->opcap) {
        int nc = p->opcap ? p->opcap * 2 : 64;
        POp *no = (POp *)realloc(p->ops, (size_t)nc * sizeof(POp));
        if (!no) { free(ks); free(Akj); return 0; }
        p->ops = no; p->opcap = nc;
    }
    POp *o = &p->ops[p->nops++];
    o->type = type; o->i = i; o->j = j; o->Aij = Aij; o->cj = cj; o->nk = nk; o->ks = ks; o->Akj = Akj;
    return 1;
}

/* compare two alive rows over alive columns (CSR) + RHS, using a stamp marker */
/* Riga i2 proporzionale a i1 (stesso pattern, i2 = c*i1) con RHS coerente
 * (bw[i2] = c*bw[i1]): ridondante. Se il RHS non e' coerente la riga e'
 * inconsistente, ma NON e' una riduzione sicura qui (il presolve non dichiara
 * infeasibility), quindi si salta. */
static int rows_prop_sp(const int *rptr, const int *ridx, const double *rval,
                        const char *alive_col, const double *bw,
                        int *mstamp, double *mval, int stamp, int i1, int i2) {
    double a1 = 0.0, a2 = 0.0; int c1 = 0;
    for (int p = rptr[i1]; p < rptr[i1 + 1]; p++) {
        int j = ridx[p]; if (!alive_col[j]) continue;
        if (c1 == 0) a1 = rval[p];
        mstamp[j] = stamp; mval[j] = rval[p]; c1++;
    }
    int c2 = 0;
    for (int p = rptr[i2]; p < rptr[i2 + 1]; p++) {
        int j = ridx[p]; if (!alive_col[j]) continue;
        if (c2 == 0) a2 = rval[p];
        c2++;
        if (mstamp[j] != stamp) return 0;
    }
    if (c1 != c2 || c1 == 0 || a1 == 0.0) return 0;
    double c = a2 / a1;
    for (int p = rptr[i2]; p < rptr[i2 + 1]; p++) {
        int j = ridx[p]; if (!alive_col[j]) continue;
        if (mval[j] * c != rval[p]) return 0;
    }
    return bw[i2] == c * bw[i1];
}
static int cols_equal_sp(const int *Aptr, const int *Arow, const double *Aval,
                         const char *alive_row, int *mstamp, double *mval, int stamp,
                         int j1, int j2) {
    int c1 = 0;
    for (int p = Aptr[j1]; p < Aptr[j1 + 1]; p++) {
        int i = Arow[p]; if (!alive_row[i]) continue;
        mstamp[i] = stamp; mval[i] = Aval[p]; c1++;
    }
    int c2 = 0;
    for (int p = Aptr[j2]; p < Aptr[j2 + 1]; p++) {
        int i = Arow[p]; if (!alive_row[i]) continue;
        c2++;
        if (mstamp[i] != stamp || mval[i] != Aval[p]) return 0;
    }
    return c1 == c2;
}

/**
 * Runs LP presolve reductions on a standard-form problem (CSC format).
 *
 * @param Aptr    [in]  Column pointers for A (size n+1, CSC).
 * @param Arow    [in]  Row indices for A (size Aptr[n]).
 * @param Aval    [in]  Values for A (size Aptr[n]).
 * @param b       [in]  RHS vector (size m).
 * @param c       [in]  Objective coefficients (size n).
 * @param m       [in]  Number of constraints.
 * @param n       [in]  Number of variables.
 * @param tol     [in]  Numerical tolerance for zero.
 * @param level   [in]  Presolve level: 0 = off, 1 = empty/singleton, 2 = +duplicates.
 * @param out     [out] Pointer to Presolve* receiving the presolve object.
 *
 * @return 0 on success, -1 on error/allocation failure.
 *
 * @note Performs the following reductions (based on exact zero structure):
 *       - Empty row:   |b_i| <= tol, row removed; dual = 0.
 *       - Empty column: c_j >= -tol, x_j = 0, column removed.
 *       - Singleton row: One alive nonzero A_ij, fixes x_j = b_i/A_ij,
 *         substitutes into other rows, removes row i and column j.
 *       - Duplicate rows (level 2): Identical pattern + proportional RHS,
 *         keeps one, drops rest (dual of dropped = 0).
 *       - Duplicate columns (level 2): Identical pattern, keeps cheapest,
 *         drops others with x = 0.
 *
 *       Ambiguous cases (|b_i| > tol, x_j < -tol, etc.) are LEFT to the
 *       solver -- presolve never declares infeasibility/unboundedness.
 *
 * @example
 * Presolve *p;
 * int rc = lp_presolve(Aptr, Arow, Aval, b, c, m, n, 1e-12, 2, &p);
 * if (rc == 0) {
 *     // Use p->Arptr, p->Arrow, p->Arval, p->br, p->cr for reduced problem
 * }
 */
int lp_presolve(const int *Aptr, const int *Arow, const double *Aval,
                const double *b, const double *c, int m, int n, double tol,
                int level, Presolve **out) {
    if (out) *out = NULL;
    if (m < 0 || n < 0) return -1;
    int nnz = Aptr[n];
    Presolve *p = (Presolve *)calloc(1, sizeof(Presolve));
    if (!p) return -1;
    p->m = m; p->n = n; p->mr = m; p->nr = n;
    p->row_map = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    p->col_map = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    p->col_fix = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    char *alive_row = (char *)malloc((size_t)(m > 0 ? m : 1));
    char *alive_col = (char *)malloc((size_t)(n > 0 ? n : 1));
    double *bw = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *rptr = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *ridx = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    double *rval = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    int *mstamp = (int *)calloc((size_t)(m > n ? m : n) + 1, sizeof(int));
    double *mval = (double *)calloc((size_t)(m > n ? m : n) + 1, sizeof(double));
    if (!p->row_map || !p->col_map || !p->col_fix || !alive_row || !alive_col ||
        !bw || !rptr || !ridx || !rval || !mstamp || !mval) {
        free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval);
        free(mstamp); free(mval); presolve_free(p); return -1;
    }
    for (int i = 0; i < m; i++) { alive_row[i] = 1; bw[i] = b[i]; }
    for (int j = 0; j < n; j++) alive_col[j] = 1;
    /* CSR of A */
    for (int j = 0; j < n; j++)
        for (int q = Aptr[j]; q < Aptr[j + 1]; q++) rptr[Arow[q] + 1]++;
    for (int i = 0; i < m; i++) rptr[i + 1] += rptr[i];
    { int *fill = (int *)calloc((size_t)(m > 0 ? m : 1), sizeof(int));
      if (!fill) { free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval); presolve_free(p); return -1; }
      for (int j = 0; j < n; j++)
        for (int q = Aptr[j]; q < Aptr[j + 1]; q++) { int i = Arow[q]; int pos = rptr[i] + fill[i]++; ridx[pos] = j; rval[pos] = Aval[q]; }
      free(fill); }

    int changed = 0, progress = 1, stamp = 1;
    while (progress) {
        progress = 0;
        /* Livello 0: nessuna riduzione. Livello 1: empty/singleton. Livello 2:
         * anche i duplicati (aggressive), MSK_IPAR_PRESOLVE_LEVEL. */
        if (level < 1) break;
        /* ---- rows: empty + singleton ---- */
        for (int i = 0; i < m; i++) {
            if (!alive_row[i]) continue;
            int cnt = 0, sj = -1; double sv = 0.0;
            for (int q = rptr[i]; q < rptr[i + 1]; q++) {
                int j = ridx[q]; if (!alive_col[j]) continue;
                cnt++; sj = j; sv = rval[q]; if (cnt > 1) break;
            }
            if (cnt == 0) {
                if (fabs(bw[i]) <= tol) {
                    alive_row[i] = 0;
                    if (!push_op(p, OP_EMPTY_ROW, i, -1, 0, 0, 0, NULL, NULL)) goto fail;
                    changed = progress = 1;
                }
            } else if (cnt == 1) {
                int j = sj; double v = sv;
                if (fabs(v) < 1e-12) continue;
                double xj = bw[i] / v;
                if (xj < -tol) continue;
                if (xj < 0.0) xj = 0.0;
                int nk = 0;
                for (int q = Aptr[j]; q < Aptr[j + 1]; q++) { int k = Arow[q]; if (k != i && alive_row[k]) nk++; }
                int *ks = nk ? (int *)malloc((size_t)nk * sizeof(int)) : NULL;
                double *Akj = nk ? (double *)malloc((size_t)nk * sizeof(double)) : NULL;
                if (nk && (!ks || !Akj)) { free(ks); free(Akj); goto fail; }
                int t = 0;
                for (int q = Aptr[j]; q < Aptr[j + 1]; q++) {
                    int k = Arow[q]; if (k == i || !alive_row[k]) continue;
                    ks[t] = k; Akj[t] = Aval[q]; t++; bw[k] -= Aval[q] * xj;
                }
                p->col_fix[j] = xj;
                alive_row[i] = 0; alive_col[j] = 0;
                if (!push_op(p, OP_SINGLETON, i, j, v, c[j], nk, ks, Akj)) goto fail;
                changed = progress = 1;
            }
        }
        /* ---- columns: empty ---- */
        for (int j = 0; j < n; j++) {
            if (!alive_col[j]) continue;
            int cnt = 0;
            for (int q = Aptr[j]; q < Aptr[j + 1]; q++) if (alive_row[Arow[q]]) { cnt = 1; break; }
            if (cnt == 0 && c[j] >= -tol) {
                p->col_fix[j] = 0.0; alive_col[j] = 0;
                changed = progress = 1;
            }
        }
        if (level >= 2) {
        /* ---- duplicate rows ---- */
        {
            int TS = 1; while (TS < 2 * (m + 2)) TS <<= 1;
            int *tab = (int *)malloc((size_t)TS * sizeof(int));
            unsigned long long *rh = (unsigned long long *)malloc((size_t)(m > 0 ? m : 1) * sizeof(unsigned long long));
            if (!tab || !rh) { free(tab); free(rh); goto fail; }
            for (int s = 0; s < TS; s++) tab[s] = -1;
            for (int i = 0; i < m; i++) {
                if (!alive_row[i]) continue;
                unsigned long long h = 1469598103934665603ULL; int nz = 0; double a0 = 0.0;
                for (int q = rptr[i]; q < rptr[i + 1]; q++)
                    if (alive_col[ridx[q]]) { if (nz == 0) a0 = rval[q]; nz++; }
                if (nz == 0 || a0 == 0.0) { rh[i] = 0; continue; }
                /* hash NORMALIZZATO per il primo coefficiente: cosi' righe
                 * proporzionali (LINDEP-lite) collidono e vengono confrontate. */
                for (int q = rptr[i]; q < rptr[i + 1]; q++)
                    if (alive_col[ridx[q]]) h = hfold(h, ridx[q], rval[q] / a0);
                rh[i] = h;
                int pos = (int)(h & (unsigned long long)(TS - 1)), probes = 0, placed = 0;
                while (tab[pos] != -1 && probes < TS) {
                    int i0 = tab[pos];
                    if (rh[i0] == h && alive_row[i0] &&
                        rows_prop_sp(rptr, ridx, rval, alive_col, bw, mstamp, mval, ++stamp, i0, i)) {
                        alive_row[i] = 0; changed = progress = 1; placed = 1; break;
                    }
                    pos = (pos + 1) & (TS - 1); probes++;
                }
                if (!placed && tab[pos] == -1) tab[pos] = i;
            }
            free(tab); free(rh);
        }
        /* ---- duplicate columns (keep cheapest) ---- */
        {
            int TS = 1; while (TS < 2 * (n + 2)) TS <<= 1;
            int *tab = (int *)malloc((size_t)TS * sizeof(int));
            unsigned long long *ch = (unsigned long long *)malloc((size_t)(n > 0 ? n : 1) * sizeof(unsigned long long));
            if (!tab || !ch) { free(tab); free(ch); goto fail; }
            for (int s = 0; s < TS; s++) tab[s] = -1;
            for (int j = 0; j < n; j++) {
                if (!alive_col[j]) continue;
                unsigned long long h = 1469598103934665603ULL; int nz = 0;
                for (int q = Aptr[j]; q < Aptr[j + 1]; q++) if (alive_row[Arow[q]]) { nz++; h = hfold(h, Arow[q], Aval[q]); }
                ch[j] = h;
                if (nz == 0) continue;
                int pos = (int)(h & (unsigned long long)(TS - 1)), probes = 0, done = 0;
                while (tab[pos] != -1 && probes < TS) {
                    int j0 = tab[pos];
                    if (ch[j0] == h && alive_col[j0] && cols_equal_sp(Aptr, Arow, Aval, alive_row, mstamp, mval, ++stamp, j0, j)) {
                        int keep = (c[j0] <= c[j]) ? j0 : j;
                        int drop = (keep == j0) ? j : j0;
                        alive_col[drop] = 0; p->col_fix[drop] = 0.0;
                        tab[pos] = keep; changed = progress = 1; done = 1; break;
                    }
                    pos = (pos + 1) & (TS - 1); probes++;
                }
                if (!done && tab[pos] == -1) tab[pos] = j;
            }
            free(tab); free(ch);
        }
        }
    }

    p->changed = changed;
    if (!changed) {
        for (int i = 0; i < m; i++) p->row_map[i] = i;
        for (int j = 0; j < n; j++) p->col_map[j] = j;
        free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval);
        if (out) *out = p;
        return 0;
    }
    int mr = 0, nr = 0;
    for (int i = 0; i < m; i++) p->row_map[i] = alive_row[i] ? mr++ : -1;
    for (int j = 0; j < n; j++) p->col_map[j] = alive_col[j] ? nr++ : -1;
    p->mr = mr; p->nr = nr;
    p->br = (double *)malloc((size_t)(mr > 0 ? mr : 1) * sizeof(double));
    p->cr = (double *)malloc((size_t)(nr > 0 ? nr : 1) * sizeof(double));
    p->Arptr = (int *)calloc((size_t)(nr + 1), sizeof(int));
    if (!p->br || !p->cr || !p->Arptr) { free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval); presolve_free(p); return -1; }
    for (int i = 0; i < m; i++) if (alive_row[i]) p->br[p->row_map[i]] = bw[i];
    for (int j = 0; j < n; j++) if (alive_col[j]) p->cr[p->col_map[j]] = c[j];
    /* reduced CSC: count then fill */
    for (int j = 0; j < n; j++) {
        if (!alive_col[j]) continue;
        for (int q = Aptr[j]; q < Aptr[j + 1]; q++) if (alive_row[Arow[q]]) p->Arptr[p->col_map[j] + 1]++;
    }
    for (int jj = 0; jj < nr; jj++) p->Arptr[jj + 1] += p->Arptr[jj];
    int rtot = p->Arptr[nr];
    p->Arrow = (int *)malloc((size_t)(rtot > 0 ? rtot : 1) * sizeof(int));
    p->Arval = (double *)malloc((size_t)(rtot > 0 ? rtot : 1) * sizeof(double));
    if (!p->Arrow || !p->Arval) { free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval); presolve_free(p); return -1; }
    { int *w = (int *)malloc((size_t)(nr > 0 ? nr : 1) * sizeof(int));
      if (!w) { free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval); presolve_free(p); return -1; }
      for (int jj = 0; jj < nr; jj++) w[jj] = p->Arptr[jj];
      for (int j = 0; j < n; j++) {
          if (!alive_col[j]) continue;
          int jj = p->col_map[j];
          for (int q = Aptr[j]; q < Aptr[j + 1]; q++) {
              int i = Arow[q]; if (!alive_row[i]) continue;
              p->Arrow[w[jj]] = p->row_map[i]; p->Arval[w[jj]] = Aval[q]; w[jj]++;
          }
      }
      free(w); }
    free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval);
    if (out) *out = p;
    return 0;

fail:
    free(alive_row); free(alive_col); free(bw); free(rptr); free(ridx); free(rval); free(mstamp); free(mval);
    presolve_free(p);
    return -1;
}

/**
 * Checks if presolve made any changes.
 *
 * @param p [in] Presolve object.
 *
 * @return 1 if changes were made, 0 otherwise (or NULL p).
 */
int presolve_changed(const Presolve *p) { return p ? p->changed : 0; }

/**
 * Retrieves the reduced problem data from a Presolve object.
 *
 * @param p     [in]  Presolve object.
 * @param Arptr [out] Optional pointer to reduced column pointers (CSC).
 * @param Arrow [out] Optional pointer to reduced row indices.
 * @param Arval [out] Optional pointer to reduced values.
 * @param br    [out] Optional pointer to reduced RHS.
 * @param cr    [out] Optional pointer to reduced objective.
 * @param mr    [out] Optional pointer to reduced constraint count.
 * @param nr    [out] Optional pointer to reduced variable count.
 *
 * @note All output pointers are optional (can be NULL). The returned pointers
 *       are owned by the Presolve object and are freed by presolve_free().
 *
 * @example
 * int mr, nr;
 * const int *Arptr, *Arrow;
 * const double *Arval, *br, *cr;
 * presolve_reduced(p, &Arptr, &Arrow, &Arval, &br, &cr, &mr, &nr);
 * // Solve reduced problem using Arptr, Arrow, Arval, br, cr
 */
void presolve_reduced(const Presolve *p, const int **Arptr, const int **Arrow,
                      const double **Arval, const double **br, const double **cr,
                      int *mr, int *nr) {
    if (!p) return;
    if (Arptr) *Arptr = p->Arptr;
    if (Arrow) *Arrow = p->Arrow;
    if (Arval) *Arval = p->Arval;
    if (br) *br = p->br;
    if (cr) *cr = p->cr;
    if (mr) *mr = p->mr;
    if (nr) *nr = p->nr;
}

/**
 * Reconstructs the full primal and dual solution from a reduced solution.
 *
 * @param p     [in]  Presolve object.
 * @param xred  [in]  Primal solution of reduced problem (size nr).
 * @param yred  [in]  Dual solution of reduced problem (size mr).
 * @param xfull [out] Full primal solution (size n). Must not be NULL.
 * @param yfull [out] Full dual solution (size m). Must not be NULL.
 *
 * @note Maps reduced variables back using col_map/row_map.
 *       Fixed columns get their stored values. Removed rows get dual = 0.
 *       For singleton rows, recovers dual from KKT: y_i = (c_j - sum_k A_kj y_k) / A_ij.
 *
 * @example
 * double *xred = malloc(nr * sizeof(double));
 * double *yred = malloc(mr * sizeof(double));
 * double *xfull = malloc(n * sizeof(double));
 * double *yfull = malloc(m * sizeof(double));
 * // ... solve reduced problem ...
 * presolve_postsolve(p, xred, yred, xfull, yfull);
 * // xfull, yfull now hold the full solution
 */
void presolve_postsolve(const Presolve *p, const double *xred, const double *yred,
                        double *xfull, double *yfull) {
    if (!p) return;
    for (int j = 0; j < p->n; j++)
        xfull[j] = (p->col_map[j] >= 0) ? xred[p->col_map[j]] : p->col_fix[j];
    for (int i = 0; i < p->m; i++)
        yfull[i] = (p->row_map[i] >= 0) ? yred[p->row_map[i]] : 0.0;
    for (int o = p->nops - 1; o >= 0; o--) {
        const POp *op = &p->ops[o];
        if (op->type == OP_EMPTY_ROW) yfull[op->i] = 0.0;
        else if (op->type == OP_SINGLETON) {
            double sum = 0.0;
            for (int q = 0; q < op->nk; q++) sum += op->Akj[q] * yfull[op->ks[q]];
            yfull[op->i] = (op->cj - sum) / op->Aij;
        }
    }
}

/**
 * Reconstructs a Farkas direction (primal or dual) from a reduced direction.
 *
 * @param p     [in]  Presolve object.
 * @param rred  [in]  Direction in reduced space (size nr for primal, mr for dual).
 * @param yred  [in]  Dual multipliers from reduced problem.
 * @param rfull [out] Full direction in original space (size n or m).
 * @param yfull [out] Full dual multipliers (size m).
 *
 * @note Replays the reduction log in REVERSE with HOMOGENEOUS lifts:
 *       y_i = -(sum_k A_kj y_k) / A_ij (no c_j term -- direction, not point).
 *       Empty/duplicate rows get 0. Singleton rows are lifted.
 *       This is the dual of the primal postsolve, ensuring b'y is invariant.
 *
 * @note This is the key function that allows Farkas certificates to survive
 *       presolve. The homogeneous lift ensures b'y is preserved exactly.
 *
 * @example
 * double *rred = malloc(nr * sizeof(double));
 * double *yred = malloc(mr * sizeof(double));
 * double *rfull = malloc(n * sizeof(double));
 * double *yfull = malloc(m * sizeof(double));
 * presolve_postsolve_dir(p, rred, yred, rfull, yfull);
 * // rfull now holds the Farkas direction in the original space
 */
void presolve_postsolve_dir(const Presolve *p, const double *rred, const double *yred,
                            double *rfull, double *yfull) {
    if (!p) return;
    for (int j = 0; j < p->n; j++)
        rfull[j] = (rred && p->col_map[j] >= 0) ? rred[p->col_map[j]] : 0.0;
    for (int i = 0; i < p->m; i++)
        yfull[i] = (yred && p->row_map[i] >= 0) ? yred[p->row_map[i]] : 0.0;
    for (int o = p->nops - 1; o >= 0; o--) {
        const POp *op = &p->ops[o];
        if (op->type != OP_SINGLETON) continue;   /* empty/duplicate row: stays 0 */
        double sum = 0.0;
        for (int q = 0; q < op->nk; q++) sum += op->Akj[q] * yfull[op->ks[q]];
        yfull[op->i] = -sum / op->Aij;
    }
}

/**
 * Frees a Presolve object and all its associated memory.
 *
 * @param p [in] Presolve object to free.
 *
 * @note Frees all internal arrays: row_map, col_map, col_fix, ops (with their
 *       ks/Akj), Arptr, Arrow, Arval, br, cr, and the struct itself.
 *       Safe to call with NULL.
 */
void presolve_free(Presolve *p) {
    if (!p) return;
    for (int o = 0; o < p->nops; o++) { free(p->ops[o].ks); free(p->ops[o].Akj); }
    free(p->ops);
    free(p->Arptr); free(p->Arrow); free(p->Arval); free(p->br); free(p->cr);
    free(p->row_map); free(p->col_map); free(p->col_fix);
    free(p);
}
