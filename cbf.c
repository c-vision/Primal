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

/* cbf.c - Conic Benchmark Format (CBF v4) I/O (sottoinsieme, vedi cbf.h)
 *
 * Semantica della mappatura (inner product <A,X> = somma_ij A_ij X_ij,
 * identica alla convenzione PRIMAL del clone):
 *  - vincolo scalare CBF i:  g_i = sum_j a_ij x_j + sum_j <F_ij, X_j> + b_i,
 *    con g_i nel cone del gruppo. Nel clone: riga i con coefficienti a_ij,
 *    termini barA (coef=1, F_ij) e bound derivato da b_i e dal cone.
 *  - cone lineari L+/L-/L=: bound della riga = -b_i (LO/UP/FX).
 *  - cone non lineari (Q/QR/EXP/EXPx/POW): variabile ausiliaria v_i (FR),
 *    riga FX: expr_i - v_i = -b_i, cone sui v_i (nello stesso ordine).
 *  - PSDCON G_i >= 0: variabile bar aggiuntiva (dopo le PSDVAR) con righe
 *    X[p,q] = sum_j x_j H_ij[p,q] + D_i[p,q] per le posizioni non nulle.
 * Coordinate ripetute sulla stessa posizione si accumulano (il formato le
 * dichiara un errore; il lettore e' tollerante per semplicita').
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "cbf.h"

#define CBF_MAXL     8192
#define CBF_INF      1e30
#define CBF_MAXPOW   256

/* ---------------- lettura righe ---------------- */
/**
 * Converts a string to double, skipping whitespace.
 *
 * @param s [in] Input string.
 *
 * @return Converted double value.
 */
static double cbf_atof(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return atof(s);
}

/**
 * Reads a non-empty, non-comment line from a CBF file.
 *
 * @param f   [in]  Open FILE* stream.
 * @param buf [out] Buffer to receive the line (size CBF_MAXL).
 *
 * @return 1 on success, 0 on EOF or error.
 *
 * @note Skips comment lines (starting with '#') and blank lines.
 *       Strips trailing whitespace and newline characters.
 */
static int rd_line(FILE *f, char *buf) {
    for (;;) {
        if (!fgets(buf, CBF_MAXL, f)) return 0;
        size_t n = strlen(buf);
        while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#') continue;
        char *e = p + strlen(p);
        while (e > p && (e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
        if (*p == 0) continue;
        memmove(buf, p, (size_t)(e - p) + 1);
        return 1;
    }
}

/**
 * Builds a dense symmetric matrix from the sparse symmetric matrix store.
 *
 * @param t   [in]  Task handle.
 * @param m   [in]  Symmetric matrix index.
 * @param dim [in]  Matrix dimension.
 * @param M   [out] Dense matrix (dim x dim, row-major). Must be pre-allocated.
 *
 * @note Reads the sparse matrix entries and builds a dense symmetric matrix
 *       by accumulating values for both (i,j) and (j,i).
 */
static void sym_dense(PRIMALtask_t t, int m, int dim, double *M) {
    int nnz = 0, dd = 0;
    PRIMAL_getsymmatinfo(t, m, &dd, &nnz);
    (void)dd;
    for (int e = 0; e < nnz; e++) {
        int si, sj; double v;
        PRIMAL_getsymmatentry(t, m, e, &si, &sj, &v);
        M[si * dim + sj] += v;
        if (si != sj) M[sj * dim + si] += v;
    }
}

/* ================================================================
 *                       SCRITTURA
 * ================================================================ */
typedef struct { int kind; int row; int cone; } CLine;
/* kind: 0=L=, 1=L+, 2=L-, 3=membro di cone (row=offset nel cone) */

/**
 * Writes a double value to a file in CBF format (%.17g).
 *
 * @param f [in] Open FILE* stream.
 * @param v [in] Double value to write.
 */
static void w_num(FILE *f, double v) { fprintf(f, "%.17g\n", v); }

/**
 * Writes a task in CBF (Conic Benchmark Format) v4.
 *
 * @param t [in] Task handle.
 * @param f [in] Open FILE* stream (must be writable).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or f is NULL,
 *         PRIMAL_RES_ERR_ALLOC on memory allocation failure,
 *         PRIMAL_RES_ERR_ARG on unsupported cone (e.g., RPOW).
 *
 * @note Writes CBF v4 format including:
 *       - VER 4
 *       - POWCONES (for PPOW, no RPOW)
 *       - OBJSENSE (MIN/MAX)
 *       - PSDVAR (PSD variables with dimensions)
 *       - VAR (with INT/BIN markers)
 *       - CON (linear + cone constraints, grouped)
 *       - OBJACOORD, OBJBCOORD (objective)
 *       - OBJFCOORD, FCOORD (PSD objective/constraint matrices)
 *       - ACOORD (linear constraint matrix + cone incidence)
 *       - BCOORD (bounds for linear/cone/variable bounds)
 *
 * @note Deviations from CBF reference:
 *       - RPOW cones are rejected (writer returns ERR_ARG).
 *       - POW*CONES (dual power cones) are not supported.
 *
 * @example
 * FILE *f = fopen("model.cbf", "w");
 * PRIMALrescodee rc = cbf_write(task, f);
 * fclose(f);
 */
PRIMALrescodee cbf_write(PRIMALtask_t t, FILE *f) {
    if (!t || !f) return PRIMAL_RES_ERR_NULL;
    int nv, nc, nb, nk;
    PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
    PRIMAL_getnumbarvar(t, &nb); PRIMAL_getnumcone(t, &nk);
    double cfix; PRIMAL_getcfix(t, &cfix);
    PRIMALobjsensee sense; PRIMAL_getobjsense(t, &sense);

    /* tavola POWCONES (solo PPOW, alpha = (a, 1-a)) */
    double powa[CBF_MAXPOW]; int npow = 0;
    for (int k = 0; k < nk; k++) {
        PRIMALconetypee ct; int nmem; int mem[64];
        PRIMAL_getcone(t, k, &ct, &nmem, mem);
        if (ct == PRIMAL_CT_PPOW) {
            if (npow >= CBF_MAXPOW) return PRIMAL_RES_ERR_ARG;
            PRIMAL_getconeparam(t, k, &powa[npow]);
            npow++;
        }
    }

    /* righe logiche: vincoli, coni, bounds variabili */
    int nlines = 0, capl = 64;
    CLine *lines = (CLine *)malloc((size_t)capl * sizeof(CLine));
    if (!lines) return PRIMAL_RES_ERR_ALLOC;
#define PUSH_L(kind_, row_, cone_) do { \
        if (nlines == capl) { capl *= 2; \
            CLine *lt_ = (CLine *)realloc(lines, (size_t)capl * sizeof(CLine)); \
            if (!lt_) { free(lines); return PRIMAL_RES_ERR_ALLOC; } lines = lt_; } \
        lines[nlines].kind = (kind_); lines[nlines].row = (row_); \
        lines[nlines].cone = (cone_); nlines++; } while (0)

    for (int i = 0; i < nc; i++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        int has = (bk != PRIMAL_BK_FR);
        for (int j = 0; j < nv && !has; j++) {
            double v; PRIMAL_getaij(t, i, j, &v);
            if (v != 0.0) has = 1;
        }
        int ntA = 0; PRIMAL_getnumbaraterm(t, &ntA);
        for (int k = 0; k < ntA && !has; k++) {
            int ci, jb, ms; double cf;
            PRIMAL_getbaraitem(t, k, &ci, &jb, &ms, &cf);
            if (ci == i) has = 1;
        }
        if (!has) continue;
        if (bk == PRIMAL_BK_FX) PUSH_L(0, i, -1);
        else {
            if (bk == PRIMAL_BK_LO || bk == PRIMAL_BK_RA) PUSH_L(1, i, -1);
            if (bk == PRIMAL_BK_UP || bk == PRIMAL_BK_RA) PUSH_L(2, i, -1);
        }
    }
    int nlin = nlines;
    for (int k = 0; k < nk; k++) {
        PRIMALconetypee ct; int nmem; int mem[64];
        PRIMAL_getcone(t, k, &ct, &nmem, mem);
        if (ct == PRIMAL_CT_RPOW) { free(lines); return PRIMAL_RES_ERR_ARG; }
        for (int q = 0; q < nmem; q++) PUSH_L(3, q, k);
    }
    int nconl = nlines - nlin;
    int nvb0 = nlines;
    for (int j = 0; j < nv; j++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
        if (bk == PRIMAL_BK_FR) continue;
        if (bk == PRIMAL_BK_LO) PUSH_L(1, j, -2);
        else if (bk == PRIMAL_BK_UP) PUSH_L(2, j, -2);
        else if (bk == PRIMAL_BK_FX) PUSH_L(0, j, -2);
        else { PUSH_L(1, j, -2); PUSH_L(2, j, -2); }
    }
    int nrow = nlines;

    /* --- header --- */
    fprintf(f, "VER\n4\n");
    if (npow > 0) {
        fprintf(f, "POWCONES\n%d %d\n", npow, 2 * npow);
        for (int k = 0; k < npow; k++) {
            fprintf(f, "2\n"); w_num(f, powa[k]); w_num(f, 1.0 - powa[k]);
        }
    }
    fprintf(f, "OBJSENSE\n%s\n", sense == PRIMAL_OPTIMIZE_MAXIMIZE ? "MAX" : "MIN");
    if (nb > 0) {
        fprintf(f, "PSDVAR\n%d\n", nb);
        for (int j = 0; j < nb; j++) {
            int d; PRIMAL_getbarsize(t, j, &d);
            fprintf(f, "%d\n", d);
        }
    }
    fprintf(f, "VAR\n%d 1\nF %d\n", nv, nv);
    {
        int nint = 0; PRIMAL_getnumintvar(t, &nint);
        if (nint > 0) {
            fprintf(f, "INT\n%d\n", nint);
            for (int j = 0; j < nv; j++) {
                PRIMALvariabletypee vt; PRIMAL_getvartype(t, j, &vt);
                if (vt == PRIMAL_VAR_TYPE_INT) fprintf(f, "%d\n", j);
            }
        }
    }
    /* CON: i gruppi lineari hanno size 1; ogni cone e' un gruppo */
    {
        int grps = 0, r = 0;
        while (r < nlines) {
            if (lines[r].kind == 3) {
                PRIMALconetypee ct; int nmem; int mem[64];
                PRIMAL_getcone(t, lines[r].cone, &ct, &nmem, mem);
                grps++; r += nmem;
            } else { grps++; r++; }
        }
        fprintf(f, "CON\n%d %d\n", nrow, grps);
        r = 0;
        while (r < nlines) {
            if (lines[r].kind == 3) {
                PRIMALconetypee ct; int nmem; int mem[64];
                PRIMAL_getcone(t, lines[r].cone, &ct, &nmem, mem);
                char nm[32];
                switch (ct) {
                    case PRIMAL_CT_QUAD:  snprintf(nm, sizeof nm, "Q"); break;
                    case PRIMAL_CT_RQUAD: snprintf(nm, sizeof nm, "QR"); break;
                    case PRIMAL_CT_PEXP:  snprintf(nm, sizeof nm, "EXP"); break;
                    case PRIMAL_CT_DEXP:  snprintf(nm, sizeof nm, "EXP*"); break;
                    default: {   /* PPOW */
                        int pidx = 0;
                        for (int k2 = 0; k2 < nk; k2++) {
                            PRIMALconetypee c2; int n2; int m2[64];
                            PRIMAL_getcone(t, k2, &c2, &n2, m2);
                            if (c2 != PRIMAL_CT_PPOW) continue;
                            if (k2 == lines[r].cone) break;
                            pidx++;
                        }
                        snprintf(nm, sizeof nm, "@%d:POW", pidx);
                        break;
                    }
                }
                fprintf(f, "%s %d\n", nm, nmem);
                r += nmem;
            } else {
                fprintf(f, "%s 1\n", lines[r].kind == 0 ? "L=" :
                                    lines[r].kind == 1 ? "L+" : "L-");
                r++;
            }
        }
    }

    /* --- dati --- */
    int n_objA = 0;
    for (int j = 0; j < nv; j++) { double v; PRIMAL_getcj(t, j, &v); if (v != 0.0) n_objA++; }
    fprintf(f, "OBJACOORD\n%d\n", n_objA);
    for (int j = 0; j < nv; j++) {
        double v; PRIMAL_getcj(t, j, &v);
        if (v != 0.0) { fprintf(f, "%d ", j); w_num(f, v); }
    }
    if (cfix != 0.0) { fprintf(f, "OBJBCOORD\n"); w_num(f, cfix); }

    int ntC = 0, ntA = 0, nsym = 0;
    PRIMAL_getnumbarcterm(t, &ntC); PRIMAL_getnumbaraterm(t, &ntA);
    PRIMAL_getnumsymmat(t, &nsym);

    /* cache densa per sym matrix */
    double **dense = (double **)calloc((size_t)(nsym > 0 ? nsym : 1), sizeof(double *));
    if (!dense) { free(lines); return PRIMAL_RES_ERR_ALLOC; }
    for (int k = 0; k < ntC; k++) {
        int jb, ms; double cf; PRIMAL_getbarcitem(t, k, &jb, &ms, &cf);
        int d; PRIMAL_getbarsize(t, jb, &d);
        if (!dense[ms]) {
            dense[ms] = (double *)calloc((size_t)d * (size_t)d, sizeof(double));
            if (!dense[ms]) goto oom;
            sym_dense(t, ms, d, dense[ms]);
        }
    }
    for (int k = 0; k < ntA; k++) {
        int ci, jb, ms; double cf; PRIMAL_getbaraitem(t, k, &ci, &jb, &ms, &cf);
        int d; PRIMAL_getbarsize(t, jb, &d);
        if (!dense[ms]) {
            dense[ms] = (double *)calloc((size_t)d * (size_t)d, sizeof(double));
            if (!dense[ms]) goto oom;
            sym_dense(t, ms, d, dense[ms]);
        }
    }

    /* OBJFCOORD */
    {
        int nfc = 0;
        for (int k = 0; k < ntC; k++) {
            int jb, ms; double cf; PRIMAL_getbarcitem(t, k, &jb, &ms, &cf);
            int d; PRIMAL_getbarsize(t, jb, &d);
            for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                if (cf * dense[ms][p * d + q] != 0.0) nfc++;
        }
        fprintf(f, "OBJFCOORD\n%d\n", nfc);
        for (int k = 0; k < ntC; k++) {
            int jb, ms; double cf; PRIMAL_getbarcitem(t, k, &jb, &ms, &cf);
            int d; PRIMAL_getbarsize(t, jb, &d);
            double *M = dense[ms];
            for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                if (cf * M[p * d + q] != 0.0) {
                    fprintf(f, "%d %d %d ", jb, p, q); w_num(f, cf * M[p * d + q]);
                }
        }
    }
    /* FCOORD: la riga logica e' la PRIMA copia del vincolo (RA spezzato) */
    {
        int nfc = 0;
        for (int k = 0; k < ntA; k++) {
            int ci, jb, ms; double cf; PRIMAL_getbaraitem(t, k, &ci, &jb, &ms, &cf);
            int d; PRIMAL_getbarsize(t, jb, &d);
            int first = -1;
            for (int r = 0; r < nlin; r++)
                if (lines[r].row == ci) { first = r; break; }
            if (first < 0) continue;
            double *M = dense[ms];
            for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                if (cf * M[p * d + q] != 0.0) nfc++;
        }
        fprintf(f, "FCOORD\n%d\n", nfc);
        for (int k = 0; k < ntA; k++) {
            int ci, jb, ms; double cf; PRIMAL_getbaraitem(t, k, &ci, &jb, &ms, &cf);
            int d; PRIMAL_getbarsize(t, jb, &d);
            int first = -1;
            for (int r = 0; r < nlin; r++)
                if (lines[r].row == ci) { first = r; break; }
            if (first < 0) continue;
            double *M = dense[ms];
            for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                if (cf * M[p * d + q] != 0.0) {
                    fprintf(f, "%d %d %d %d ", first, jb, p, q);
                    w_num(f, cf * M[p * d + q]);
                }
        }
    }
    /* ACOORD */
    {
        int na = 0;
        for (int r = 0; r < nlin; r++) {
            int i = lines[r].row;
            for (int j = 0; j < nv; j++) {
                double v; PRIMAL_getaij(t, i, j, &v);
                if (v != 0.0) na++;
            }
        }
        for (int r = nlin; r < nlin + nconl; r++) na++;
        for (int r = nvb0; r < nrow; r++) na++;
        fprintf(f, "ACOORD\n%d\n", na);
        for (int r = 0; r < nlin; r++) {
            int i = lines[r].row;
            for (int j = 0; j < nv; j++) {
                double v; PRIMAL_getaij(t, i, j, &v);
                if (v != 0.0) { fprintf(f, "%d %d ", r, j); w_num(f, v); }
            }
        }
        for (int r = nlin; r < nlin + nconl; r++) {
            int k = lines[r].cone, q = lines[r].row;
            PRIMALconetypee ct; int nmem; int mem[64];
            PRIMAL_getcone(t, k, &ct, &nmem, mem);
            fprintf(f, "%d %d 1\n", r, mem[q]);
        }
        for (int r = nvb0; r < nrow; r++)
            fprintf(f, "%d %d 1\n", r, lines[r].row);
    }
    /* BCOORD */
    {
        int nbc = 0;
        for (int r = 0; r < nlin; r++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getconbound(t, lines[r].row, &bk, &bl, &bu);
            if ((lines[r].kind == 2 ? -bu : -bl) != 0.0) nbc++;
        }
        for (int r = nvb0; r < nrow; r++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getvarbound(t, lines[r].row, &bk, &bl, &bu);
            double b = (lines[r].kind == 2) ? -bu : -bl;
            if (b != 0.0) nbc++;
        }
        fprintf(f, "BCOORD\n%d\n", nbc);
        for (int r = 0; r < nlin; r++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getconbound(t, lines[r].row, &bk, &bl, &bu);
            double b = (lines[r].kind == 2) ? -bu : -bl;
            if (b != 0.0) { fprintf(f, "%d ", r); w_num(f, b); }
        }
        for (int r = nvb0; r < nrow; r++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getvarbound(t, lines[r].row, &bk, &bl, &bu);
            double b = (lines[r].kind == 2) ? -bu : -bl;
            if (b != 0.0) { fprintf(f, "%d ", r); w_num(f, b); }
        }
    }
    for (int mm = 0; mm < nsym; mm++) free(dense[mm]);
    free(dense); free(lines);
    return PRIMAL_RES_OK;

oom:
    for (int mm = 0; mm < nsym; mm++) free(dense[mm]);
    free(dense); free(lines);
    return PRIMAL_RES_ERR_ALLOC;
}

/* ================================================================
 *                       LETTURA
 * ================================================================ */
typedef struct { double v[2]; int len; } PowEnt;
typedef struct { char name[16]; int start, size; } CbGroup;
typedef struct { int i, j; double v; } T2;
typedef struct { int i, j, p, q; double v; } T4;

typedef struct { T2 *d; int n, cap; } V2;
typedef struct { T4 *d; int n, cap; } V4;

static int push2(V2 *v, T2 e) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        T2 *nd = (T2 *)realloc(v->d, (size_t)v->cap * sizeof(T2));
        if (!nd) return 0;
        v->d = nd;
    }
    v->d[v->n++] = e; return 1;
}
static int push4(V4 *v, T4 e) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        T4 *nd = (T4 *)realloc(v->d, (size_t)v->cap * sizeof(T4));
        if (!nd) return 0;
        v->d = nd;
    }
    v->d[v->n++] = e; return 1;
}

/* classificazione dominio: 100=F, 101=L+, 102=L-, 103=L=, altrimenti ct clone */
static int cone_dom(const char *nm, int size, const PowEnt *pows, int npows,
                    int *ct, double *param) {
    *ct = -1; *param = 0.0;
    if (!strcmp(nm, "F")) *ct = 100;
    else if (!strcmp(nm, "L+")) *ct = 101;
    else if (!strcmp(nm, "L-")) *ct = 102;
    else if (!strcmp(nm, "L=")) *ct = 103;
    else if (!strcmp(nm, "Q")) *ct = PRIMAL_CT_QUAD;
    else if (!strcmp(nm, "QR")) *ct = PRIMAL_CT_RQUAD;
    else if (!strcmp(nm, "EXP")) *ct = PRIMAL_CT_PEXP;
    else if (!strcmp(nm, "EXP*")) *ct = PRIMAL_CT_DEXP;
    else if (nm[0] == '@') {
        int ix = -1;
        if (sscanf(nm, "@%d:POW", &ix) != 1) return 0;
        if (ix < 0 || ix >= npows || pows[ix].len != 2) return 0;
        double a0 = pows[ix].v[0], a1 = pows[ix].v[1];
        if (!(a0 + a1 > 0.0)) return 0;
        *ct = PRIMAL_CT_PPOW; *param = a0 / (a0 + a1);
    } else return 0;
    if (*ct == PRIMAL_CT_RQUAD && size < 2) return 0;
    if ((*ct == PRIMAL_CT_PEXP || *ct == PRIMAL_CT_DEXP || *ct == PRIMAL_CT_PPOW) &&
        size != 3) return 0;
    return 1;
}

static void bounds_for_dom(int ct, PRIMALboundkeye *bk, double *bl, double *bu) {
    switch (ct) {
        case 101: *bk = PRIMAL_BK_LO; *bl = 0.0; *bu = CBF_INF; break;
        case 102: *bk = PRIMAL_BK_UP; *bl = -CBF_INF; *bu = 0.0; break;
        case 103: *bk = PRIMAL_BK_FX; *bl = 0.0; *bu = 0.0; break;
        default:  *bk = PRIMAL_BK_FR; *bl = -CBF_INF; *bu = CBF_INF; break;
    }
}

/* merge di una lista di (j,v) con get/set sul row corrente:
 * combina le coordinate ripetute sommandole */
typedef struct { int *sub; double *val; int nz, cap; } RowList;
static void rl_add(RowList *r, int j, double v) {
    for (int k = 0; k < r->nz; k++)
        if (r->sub[k] == j) { r->val[k] += v; return; }
    if (r->nz == r->cap) {
        r->cap = r->cap ? r->cap * 2 : 8;
        r->sub = (int *)realloc(r->sub, (size_t)r->cap * sizeof(int));
        r->val = (double *)realloc(r->val, (size_t)r->cap * sizeof(double));
    }
    r->sub[r->nz] = j; r->val[r->nz] = v; r->nz++;
}

static PRIMALrescodee build_task(PRIMALtask_t t, int sense_max,
        int nv, CbGroup *vgrp, int nvgrp, const PowEnt *pows, int npows,
        int *ints, int nint, int *psdvar, int npsdvar,
        int m, CbGroup *cgrp, int ncgrp,
        T2 *objA, int nobjA, double bobj,
        T4 *objF, int nobjF,
        T2 *ac, int nac, T2 *bc, int nbc, T4 *fc, int nfc,
        T4 *hc, int nhc, T4 *dc, int ndc, int *psdcon, int npsdcon) {
    /* --- barvars --- */
    if (npsdvar > 0 && PRIMAL_appendbarvars(t, npsdvar, psdvar) != PRIMAL_RES_OK)
        return PRIMAL_RES_ERR_ARG;
    for (int c = 0; c < npsdcon; c++) {
        int dim = psdcon[c];
        if (PRIMAL_appendbarvars(t, 1, &dim) != PRIMAL_RES_OK) return PRIMAL_RES_ERR_ARG;
    }
    if (PRIMAL_appendvars(t, nv) != PRIMAL_RES_OK) return PRIMAL_RES_ERR_ARG;

    /* --- VAR: bounds + cone (membri contigui) --- */
    for (int g = 0; g < nvgrp; g++) {
        int ct; double pr = 0.0;
        if (!cone_dom(vgrp[g].name, vgrp[g].size, pows, npows, &ct, &pr))
            return PRIMAL_RES_ERR_ARG;
        PRIMALboundkeye bk; double bl, bu;
        bounds_for_dom(ct, &bk, &bl, &bu);
        for (int q = 0; q < vgrp[g].size; q++)
            PRIMAL_putvarbound(t, vgrp[g].start + q, bk, bl, bu);
        if (ct < 100) {
            int mem[3];
            if (vgrp[g].size > 3) return PRIMAL_RES_ERR_ARG;
            for (int q = 0; q < vgrp[g].size; q++) mem[q] = vgrp[g].start + q;
            if (PRIMAL_appendcone(t, (PRIMALconetypee)ct, pr, vgrp[g].size, mem)
                != PRIMAL_RES_OK) return PRIMAL_RES_ERR_ARG;
        }
    }
    for (int k = 0; k < nint; k++)
        PRIMAL_putvartype(t, ints[k], PRIMAL_VAR_TYPE_INT);

    /* --- CON: righe scalari --- */
    PRIMAL_appendcons(t, m);

    /* --- ausiliarie per cone CON non lineari --- */
    int naux = 0;
    for (int g = 0; g < ncgrp; g++) {
        int ct; double pr;
        if (!cone_dom(cgrp[g].name, cgrp[g].size, pows, npows, &ct, &pr))
            return PRIMAL_RES_ERR_ARG;
        if (ct < 100) naux += cgrp[g].size;
    }
    if (naux > 0 && PRIMAL_appendvars(t, naux) != PRIMAL_RES_OK)
        return PRIMAL_RES_ERR_ALLOC;
    int aux0 = 0; PRIMAL_getnumvar(t, &aux0); aux0 -= naux;

    /* --- righe scalari: coefficienti + bounds --- */
    for (int i = 0; i < m; i++) {
        int gidx = -1;
        for (int g = 0; g < ncgrp; g++)
            if (i >= cgrp[g].start && i < cgrp[g].start + cgrp[g].size)
                { gidx = g; break; }
        if (gidx < 0) return PRIMAL_RES_ERR_ARG;
        int ct; double pr;
        if (!cone_dom(cgrp[gidx].name, cgrp[gidx].size, pows, npows, &ct, &pr))
            return PRIMAL_RES_ERR_ARG;
        /* b_i */
        double b = 0.0;
        for (int k = 0; k < nbc; k++) if (bc[k].i == i) b += bc[k].v;
        /* coefficienti */
        RowList rl; memset(&rl, 0, sizeof(rl));
        for (int k = 0; k < nac; k++)
            if (ac[k].i == i) rl_add(&rl, ac[k].j, ac[k].v);
        if (ct < 100) {
            /* ausiliaria: expr - v = -b */
            int off = 0;
            for (int g2 = 0; g2 < gidx; g2++) {
                int ct2; double pr2;
                cone_dom(cgrp[g2].name, cgrp[g2].size, pows, npows, &ct2, &pr2);
                if (ct2 < 100) off += cgrp[g2].size;
            }
            int vi = aux0 + off + (i - cgrp[gidx].start);
            rl_add(&rl, vi, -1.0);
            PRIMAL_putvarbound(t, vi, PRIMAL_BK_FR, -CBF_INF, CBF_INF);
            PRIMAL_putarow(t, i, rl.nz, rl.sub, rl.val);
            PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -b, -b);
        } else {
            PRIMAL_putarow(t, i, rl.nz, rl.sub, rl.val);
            PRIMALboundkeye bk; double bl, bu;
            bounds_for_dom(ct, &bk, &bl, &bu);
            if (bk == PRIMAL_BK_FX)      PRIMAL_putconbound(t, i, PRIMAL_BK_FX, -b, -b);
            else if (bk == PRIMAL_BK_LO) PRIMAL_putconbound(t, i, PRIMAL_BK_LO, -b, CBF_INF);
            else if (bk == PRIMAL_BK_UP) PRIMAL_putconbound(t, i, PRIMAL_BK_UP, -CBF_INF, -b);
            else PRIMAL_putconbound(t, i, PRIMAL_BK_FR, -CBF_INF, CBF_INF);
        }
        free(rl.sub); free(rl.val);
    }
    /* --- cone sui vincoli non lineari (sui v_i) --- */
    for (int g = 0; g < ncgrp; g++) {
        int ct; double pr;
        if (!cone_dom(cgrp[g].name, cgrp[g].size, pows, npows, &ct, &pr))
            return PRIMAL_RES_ERR_ARG;
        if (ct >= 100) continue;
        int off = 0;
        for (int g2 = 0; g2 < g; g2++) {
            int ct2; double pr2;
            cone_dom(cgrp[g2].name, cgrp[g2].size, pows, npows, &ct2, &pr2);
            if (ct2 < 100) off += cgrp[g2].size;
        }
        int mem[3];
        if (cgrp[g].size > 3) return PRIMAL_RES_ERR_ARG;
        for (int q = 0; q < cgrp[g].size; q++)
            mem[q] = aux0 + off + q;
        if (PRIMAL_appendcone(t, (PRIMALconetypee)ct, pr, cgrp[g].size, mem)
            != PRIMAL_RES_OK) return PRIMAL_RES_ERR_ARG;
    }

    /* --- PSDCON: righe per le posizioni non nulle --- */
    {
        int nposTot = 0;
        for (int c = 0; c < npsdcon; c++) {
            for (int k = 0; k < nhc + ndc; k++) {
                int p, q;
                if (k < nhc) { if (hc[k].i != c) continue; p = hc[k].p; q = hc[k].q; }
                else { int k2 = k - nhc; if (dc[k2].i != c) continue; p = dc[k2].p; q = dc[k2].q; }
                int dup = 0;
                for (int u = 0; u < k; u++) {
                    int up, uq, uok;
                    if (u < nhc) uok = (hc[u].i == c), up = hc[u].p, uq = hc[u].q;
                    else { int u2 = u - nhc; uok = (dc[u2].i == c); up = dc[u2].p; uq = dc[u2].q; }
                    if (uok && up == p && uq == q) { dup = 1; break; }
                }
                if (!dup) nposTot++;
            }
        }
        if (nposTot > 0) PRIMAL_appendcons(t, nposTot);
        int r = m;
        for (int c = 0; c < npsdcon; c++) {
            int baridx = npsdvar + c, mc = psdcon[c];
            for (int p = 0; p < mc; p++) {
                for (int q = p; q < mc; q++) {
                    /* la posizione (p,q) ha dati? */
                    int hasH = 0, hasD = 0;
                    for (int k = 0; k < nhc; k++)
                        if (hc[k].i == c && hc[k].p == p && hc[k].q == q) { hasH = 1; break; }
                    for (int k = 0; k < ndc; k++)
                        if (dc[k].i == c && dc[k].p == p && dc[k].q == q) { hasD = 1; break; }
                    if (!hasH && !hasD) continue;
                    /* baraij: <E_pq, X> = expr */
                    int sm;
                    if (PRIMAL_appendsparsesymmat(t, mc, 1, (int[]){p}, (int[]){q},
                                               (double[]){1.0}, &sm) != PRIMAL_RES_OK)
                        return PRIMAL_RES_ERR_ARG;
                    double coef = (p == q) ? 1.0 : 0.5;
                    if (PRIMAL_putbaraij(t, r, baridx, 1, &sm, &coef) != PRIMAL_RES_OK)
                        return PRIMAL_RES_ERR_ARG;
                    /* coefficienti lineari */
                    RowList rl; memset(&rl, 0, sizeof(rl));
                    for (int k = 0; k < nhc; k++)
                        if (hc[k].i == c && hc[k].p == p && hc[k].q == q)
                            rl_add(&rl, hc[k].j, hc[k].v);
                    if (rl.nz > 0)
                        PRIMAL_putarow(t, r, rl.nz, rl.sub, rl.val);
                    free(rl.sub); free(rl.val);
                    /* bound FX = -D[p][q] */
                    double dsum = 0.0;
                    for (int k = 0; k < ndc; k++)
                        if (dc[k].i == c && dc[k].p == p && dc[k].q == q)
                            dsum += dc[k].v;
                    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, -dsum, -dsum);
                    r++;
                }
            }
        }
    }

    /* --- FCOORD / OBJFCOORD: matrici F dai coordinate --- */
    for (int pass = 0; pass < 2; pass++) {
        int cnt = (pass == 0) ? nobjF : nfc;
        T4 *dat = (pass == 0) ? objF : fc;
        for (int k = 0; k < cnt; k++) {
            int ci = dat[k].i, jb = dat[k].j;
            int seen = 0;
            for (int u = 0; u < k; u++)
                if (dat[u].i == ci && dat[u].j == jb) { seen = 1; break; }
            if (seen) continue;
            int d;
            PRIMAL_getbarsize(t, jb, &d);
            double *F = (double *)calloc((size_t)d * (size_t)d, sizeof(double));
            if (!F) return PRIMAL_RES_ERR_ALLOC;
            for (int u = 0; u < cnt; u++)
                if (dat[u].i == ci && dat[u].j == jb) {
                    F[dat[u].p * d + dat[u].q] += dat[u].v;
                    if (dat[u].p != dat[u].q) F[dat[u].q * d + dat[u].p] += dat[u].v;
                }
            /* upper triangle */
            int nnz = 0;
            for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                if (F[p * d + q] != 0.0) nnz++;
            if (nnz > 0) {
                int *si = (int *)malloc((size_t)nnz * sizeof(int));
                int *sj = (int *)malloc((size_t)nnz * sizeof(int));
                double *sv = (double *)malloc((size_t)nnz * sizeof(double));
                if (!si || !sj || !sv) { free(si); free(sj); free(sv); free(F);
                                         return PRIMAL_RES_ERR_ALLOC; }
                int e = 0;
                for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                    if (F[p * d + q] != 0.0) {
                        si[e] = p; sj[e] = q; sv[e] = F[p * d + q]; e++;
                    }
                int sm;
                if (PRIMAL_appendsparsesymmat(t, d, nnz, si, sj, sv, &sm) != PRIMAL_RES_OK) {
                    free(si); free(sj); free(sv); free(F);
                    return PRIMAL_RES_ERR_ARG;
                }
                if (pass == 0) {
                    double one = 1.0;
                    PRIMAL_putbarcj(t, jb, 1, &sm, &one);
                } else {
                    double one = 1.0;
                    PRIMAL_putbaraij(t, ci, jb, 1, &sm, &one);
                }
                free(si); free(sj); free(sv);
            }
            free(F);
        }
    }

    /* --- OBJACOORD / OBJBCOORD --- */
    for (int k = 0; k < nobjA; k++) {
        double v; PRIMAL_getcj(t, objA[k].i, &v);
        v += objA[k].v;
        PRIMAL_putcj(t, objA[k].i, v);
    }
    if (bobj != 0.0) PRIMAL_putcfix(t, bobj);

    /* --- sense --- */
    if (sense_max) PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    return PRIMAL_RES_OK;
}

/* ---- CHANGE section (CBF v4): post-build modifications ----
 * Supported blocks (coordinates refer to the ADDED dimensions):
 *   VAR   <n> <groups>     : append variables (F/L+/L-/L= per gruppo)
 *   CON   <m> <groups>      : append rows (L+/L-/L=; default free)
 *   PSDVAR <k> <dims>       : append bar variables
 *   OBJACOORD <n> (i,v)     : cj of the new variables (i relative to new)
 *   ACOORD  <n> (i,j,v)     : a_ij (i relative to new rows, j absolute)
 *   BCOORD  <n> (i,l,u)     : bounds of the new rows (i relative) */
static PRIMALrescodee read_change(PRIMALtask_t t, char **lines, int nl, int *idxp) {
#define CNEXT() (*idxp < nl ? lines[(*idxp)++] : NULL)
#define CPEEK() (*idxp < nl ? lines[*idxp] : NULL)
    const char *l = CNEXT();   /* consume "CHANGE" */
    (void)l;
    int nvar0 = 0, ncon0 = 0;
    PRIMAL_getnumvar(t, &nvar0);
    PRIMAL_getnumcon(t, &ncon0);
    int nvar_add = 0, ncon_add = 0;
    PRIMALrescodee rc = PRIMAL_RES_OK;
    for (;;) {
        const char *kw = CPEEK();
        if (!kw) break;
        if (!strcmp(kw, "VAR")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int n = 0, kg = 0;
            if (sscanf(l, "%d %d", &n, &kg) != 2) return PRIMAL_RES_ERR_ARG;
            if (n < 0 || kg < 0) return PRIMAL_RES_ERR_ARG;
            /* read group descriptors first (sizes sum must be <= n) */
            char names[16][16]; int sizes[16];
            if (kg > 16) return PRIMAL_RES_ERR_ARG;
            for (int g = 0; g < kg; g++) {
                l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                if (sscanf(l, "%15s %d", names[g], &sizes[g]) != 2)
                    return PRIMAL_RES_ERR_ARG;
            }
            rc = PRIMAL_appendvars(t, n);
            if (rc) return rc;
            int base = nvar0 + nvar_add;
            int off = 0;
            for (int g = 0; g < kg; g++) {
                for (int q = 0; q < sizes[g]; q++) {
                    int j = base + off + q;
                    if (!strcmp(names[g], "L+"))
                        rc = PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
                    else if (!strcmp(names[g], "L-"))
                        rc = PRIMAL_putvarbound(t, j, PRIMAL_BK_UP, -INFINITY, 0.0);
                    else if (!strcmp(names[g], "L="))
                        rc = PRIMAL_putvarbound(t, j, PRIMAL_BK_FX, 0.0, 0.0);
                    /* F: free default */
                    if (rc) return rc;
                }
                off += sizes[g];
            }
            nvar_add += n;
            continue;
        }
        if (!strcmp(kw, "CON")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int m = 0, kg = 0;
            if (sscanf(l, "%d %d", &m, &kg) != 2) return PRIMAL_RES_ERR_ARG;
            if (m < 0 || kg < 0) return PRIMAL_RES_ERR_ARG;
            char names[16][16]; int sizes[16];
            if (kg > 16) return PRIMAL_RES_ERR_ARG;
            for (int g = 0; g < kg; g++) {
                l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                if (sscanf(l, "%15s %d", names[g], &sizes[g]) != 2)
                    return PRIMAL_RES_ERR_ARG;
            }
            rc = PRIMAL_appendcons(t, m);
            if (rc) return rc;
            /* CBF rows default: L+ group -> lower 0; L- -> upper 0; L= -> FX 0 */
            int base = ncon0 + ncon_add;
            int off = 0;
            for (int g = 0; g < kg; g++) {
                for (int q = 0; q < sizes[g]; q++) {
                    int i = base + off + q;
                    if (!strcmp(names[g], "L+"))
                        rc = PRIMAL_putconbound(t, i, PRIMAL_BK_LO, 0.0, INFINITY);
                    else if (!strcmp(names[g], "L-"))
                        rc = PRIMAL_putconbound(t, i, PRIMAL_BK_UP, -INFINITY, 0.0);
                    else if (!strcmp(names[g], "L="))
                        rc = PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 0.0, 0.0);
                    if (rc) return rc;
                }
                off += sizes[g];
            }
            ncon_add += m;
            continue;
        }
        if (!strcmp(kw, "PSDVAR")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int k = atoi(l);
            if (k < 0) return PRIMAL_RES_ERR_ARG;
            int *dims = (int *)calloc((size_t)(k > 0 ? k : 1), sizeof(int));
            if (!dims) return PRIMAL_RES_ERR_ALLOC;
            for (int q = 0; q < k; q++) {
                l = CNEXT(); if (!l) { free(dims); return PRIMAL_RES_ERR_ARG; }
                dims[q] = atoi(l);
            }
            rc = PRIMAL_appendbarvars(t, k, dims);
            free(dims);
            if (rc) return rc;
            continue;
        }
        if (!strcmp(kw, "OBJACOORD")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int q = 0; q < cnt; q++) {
                l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                int i; double v;
                if (sscanf(l, "%d %lf", &i, &v) != 2) return PRIMAL_RES_ERR_ARG;
                if (i < 0 || i >= nvar_add) return PRIMAL_RES_ERR_ARG;
                rc = PRIMAL_putcj(t, nvar0 + i, v);
                if (rc) return rc;
            }
            continue;
        }
        if (!strcmp(kw, "ACOORD")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            /* accumulate per new row, then one putarow each (rows are new
             * and empty: CHANGE entries fully define them) */
            int *rrow = (int *)calloc((size_t)(cnt > 0 ? cnt : 1), sizeof(int));
            int *rcol = (int *)calloc((size_t)(cnt > 0 ? cnt : 1), sizeof(int));
            double *rval = (double *)calloc((size_t)(cnt > 0 ? cnt : 1), sizeof(double));
            if (!rrow || !rcol || !rval) { free(rrow); free(rcol); free(rval); return PRIMAL_RES_ERR_ALLOC; }
            int ok = 1;
            for (int q = 0; q < cnt && ok; q++) {
                l = CNEXT(); if (!l) { ok = 0; break; }
                int i, j; double v;
                if (sscanf(l, "%d %d %lf", &i, &j, &v) != 3) { ok = 0; break; }
                if (i < 0 || i >= ncon_add || j < 0 || j >= nvar0 + nvar_add) { ok = 0; break; }
                rrow[q] = i; rcol[q] = j; rval[q] = v;
            }
            if (ok) {
                for (int i = 0; i < ncon_add && ok; i++) {
                    /* collect entries of new row i */
                    int n = 0;
                    for (int q = 0; q < cnt; q++) if (rrow[q] == i) n++;
                    if (n == 0) continue;
                    int *sub = (int *)malloc((size_t)n * sizeof(int));
                    double *val = (double *)malloc((size_t)n * sizeof(double));
                    if (!sub || !val) { free(sub); free(val); ok = 0; break; }
                    int w = 0;
                    for (int q = 0; q < cnt; q++)
                        if (rrow[q] == i) { sub[w] = rcol[q]; val[w] = rval[q]; w++; }
                    rc = PRIMAL_putarow(t, ncon0 + i, n, sub, val);
                    free(sub); free(val);
                    if (rc) { ok = 0; break; }
                }
            }
            free(rrow); free(rcol); free(rval);
            if (!ok) return PRIMAL_RES_ERR_ARG;
            continue;
        }
        if (!strcmp(kw, "BCOORD")) {
            CNEXT();
            l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int q = 0; q < cnt; q++) {
                l = CNEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                int i; double v;
                if (sscanf(l, "%d %lf", &i, &v) != 2) return PRIMAL_RES_ERR_ARG;
                if (i < 0 || i >= ncon_add) return PRIMAL_RES_ERR_ARG;
                int j = ncon0 + i;
                PRIMALboundkeye bk; double bl, bu;
                PRIMAL_getconbound(t, j, &bk, &bl, &bu);
                /* CBF convention (as the main reader): A x + b in K,
                 * i.e. the clone bound is -v on the active side */
                if (bk == PRIMAL_BK_FX)
                    rc = PRIMAL_putconbound(t, j, PRIMAL_BK_FX, -v, -v);
                else if (bk == PRIMAL_BK_UP)
                    rc = PRIMAL_putconbound(t, j, PRIMAL_BK_UP, -CBF_INF, -v);
                else
                    rc = PRIMAL_putconbound(t, j, PRIMAL_BK_LO, -v, CBF_INF);
                if (rc) return rc;
            }
            continue;
        }
        return PRIMAL_RES_ERR_ARG;   /* CHANGE block non supportato */
    }
    return PRIMAL_RES_OK;
#undef CNEXT
#undef CPEEK
}

/**
 * Reads a task from a CBF (Conic Benchmark Format) v4 file.
 *
 * @param t [in] Task handle (must be empty: numvar=0, numcon=0).
 * @param f [in] Open FILE* stream (readable).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or f is NULL,
 *         PRIMAL_RES_ERR_ARG on invalid format,
 *         PRIMAL_RES_ERR_ALLOC on memory allocation failure.
 *
 * @note Parses CBF v4 sections:
 *       - VER, POWCONES, OBJSENSE, PSDVAR, PSDCON
 *       - VAR (variable groups with cone types)
 *       - INT (integer variables)
 *       - CON (constraint groups with cone types)
 *       - OBJACOORD, OBJBCOORD (objective)
 *       - ACOORD, BCOORD (linear constraints)
 *       - OBJFCOORD, FCOORD, HCOORD, DCOORD (PSD matrices)
 *       - CHANGE (post-build modifications, optional)
 *
 *       Supported cone types: Q, QR, EXP, EXP*, POW (via @idx:POW).
 *       PPOW parameter alpha = a/(a+b) from POWCONES definition.
 *
 *       Deviations from CBF reference:
 *       - POW*CONES (dual power cones) are REJECTED (returns ERR_ARG).
 *       - The reference dual power cone is not representable with this
 *         solver's primal PPOW cone. A file using dual power cones is
 *         not representable and is rejected with ERR_ARG.
 *
 * @example
 * FILE *f = fopen("model.cbf", "r");
 * PRIMALrescodee rc = cbf_read(task, f);
 * fclose(f);
 */
PRIMALrescodee cbf_read(PRIMALtask_t t, FILE *f) {
    if (!t || !f) return PRIMAL_RES_ERR_NULL;
    char *lines[CBF_MAXL];
    int nl = 0;
    char buf[CBF_MAXL];
    while (nl < CBF_MAXL && rd_line(f, buf)) {
        lines[nl] = strdup(buf);
        if (!lines[nl]) return PRIMAL_RES_ERR_ALLOC;
        nl++;
    }
    int idx = 0;
#define NEXT() (idx < nl ? lines[idx++] : NULL)
#define PEEK() (idx < nl ? lines[idx] : NULL)

    {
        const char *l = NEXT();
        if (!l || strcmp(l, "VER") != 0) return PRIMAL_RES_ERR_ARG;
        l = NEXT();
        if (!l) return PRIMAL_RES_ERR_ARG;
        int ver = atoi(l);
        if (ver < 1 || ver > 4) return PRIMAL_RES_ERR_ARG;
    }

    int sense_max = 0;
    PowEnt pows[CBF_MAXPOW]; int npows = 0;
    int nv = 0, m = 0, nint = 0, npsdvar = 0, npsdcon = 0;
    CbGroup *vgrp = NULL, *cgrp = NULL; int nvgrp = 0, ncgrp = 0;
    int *ints = NULL, *psdvar = NULL, *psdcon = NULL;
    V2 objA, ac, bc; memset(&objA, 0, sizeof objA);
    memset(&ac, 0, sizeof ac); memset(&bc, 0, sizeof bc);
    V4 objF, fc, hc, dc; memset(&objF, 0, sizeof objF);
    memset(&fc, 0, sizeof fc); memset(&hc, 0, sizeof hc); memset(&dc, 0, sizeof dc);
    double bobj = 0.0;
    int have_var = 0, have_con = 0;
    PRIMALrescodee rc;

    for (;;) {
        const char *l = PEEK();
        if (!l) break;
        if (!strcmp(l, "CHANGE")) break;
        if (!strcmp(l, "POWCONES") || !strcmp(l, "POW*CONES")) {
            int dual = (l[3] == '*');
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int nent = 0, totlen = 0;
            sscanf(l, "%d %d", &nent, &totlen);
            (void)totlen;
            for (int k = 0; k < nent; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                int len = atoi(l);
                PowEnt pe; pe.len = len;
                for (int q = 0; q < len && q < 2; q++) {
                    l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                    pe.v[q] = cbf_atof(l);
                }
                if (dual) {
                    /* POW*CONES e' il cono di potenza DUALE: questo solver ha il
                     * cono primale (PRIMAL_CT_PPOW), quindi un file che vincola
                     * nel duale non e' rappresentabile. Prima la sezione veniva
                     * scartata in silenzio e il task che ne usciva era un modello
                     * a cui mancavano dei vincoli; ora e' un rifiuto dichiarato
                     * (deviazione), non un modello incompleto. */
                    return PRIMAL_RES_ERR_ARG;
                }
                if (npows >= CBF_MAXPOW) return PRIMAL_RES_ERR_ARG;
                pows[npows++] = pe;
            }
            continue;
        }
        if (!strcmp(l, "OBJSENSE")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            sense_max = !strcmp(l, "MAX");
            continue;
        }
        if (!strcmp(l, "PSDVAR")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            npsdvar = atoi(l);
            psdvar = (int *)calloc((size_t)(npsdvar > 0 ? npsdvar : 1), sizeof(int));
            if (!psdvar) return PRIMAL_RES_ERR_ALLOC;
            for (int k = 0; k < npsdvar; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                psdvar[k] = atoi(l);
            }
            continue;
        }
        if (!strcmp(l, "PSDCON")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            npsdcon = atoi(l);
            psdcon = (int *)calloc((size_t)(npsdcon > 0 ? npsdcon : 1), sizeof(int));
            if (!psdcon) return PRIMAL_RES_ERR_ALLOC;
            for (int k = 0; k < npsdcon; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                psdcon[k] = atoi(l);
            }
            continue;
        }
        if (!strcmp(l, "VAR")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int kg = 0;
            sscanf(l, "%d %d", &nv, &kg);
            have_var = 1;
            for (int g = 0; g < kg; g++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                char nm[16]; int sz;
                if (sscanf(l, "%15s %d", nm, &sz) != 2) return PRIMAL_RES_ERR_ARG;
                CbGroup G;
                snprintf(G.name, sizeof G.name, "%s", nm);
                G.size = sz;
                G.start = nvgrp == 0 ? 0 : vgrp[nvgrp - 1].start + vgrp[nvgrp - 1].size;
                nvgrp++;
                vgrp = (CbGroup *)realloc(vgrp, (size_t)nvgrp * sizeof(CbGroup));
                if (!vgrp) return PRIMAL_RES_ERR_ALLOC;
                vgrp[nvgrp - 1] = G;
            }
            continue;
        }
        if (!strcmp(l, "INT")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            nint = atoi(l);
            ints = (int *)calloc((size_t)(nint > 0 ? nint : 1), sizeof(int));
            if (!ints) return PRIMAL_RES_ERR_ALLOC;
            for (int k = 0; k < nint; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                ints[k] = atoi(l);
            }
            continue;
        }
        if (!strcmp(l, "CON")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int kg = 0;
            sscanf(l, "%d %d", &m, &kg);
            have_con = 1;
            for (int g = 0; g < kg; g++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                char nm[16]; int sz;
                if (sscanf(l, "%15s %d", nm, &sz) != 2) return PRIMAL_RES_ERR_ARG;
                CbGroup G;
                snprintf(G.name, sizeof G.name, "%s", nm);
                G.size = sz;
                G.start = ncgrp == 0 ? 0 : cgrp[ncgrp - 1].start + cgrp[ncgrp - 1].size;
                ncgrp++;
                cgrp = (CbGroup *)realloc(cgrp, (size_t)ncgrp * sizeof(CbGroup));
                if (!cgrp) return PRIMAL_RES_ERR_ALLOC;
                cgrp[ncgrp - 1] = G;
            }
            continue;
        }
        if (!strcmp(l, "OBJACOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T2 e; e.j = 0;
                if (sscanf(l, "%d %lf", &e.i, &e.v) != 2) return PRIMAL_RES_ERR_ARG;
                if (!push2(&objA, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "OBJBCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            bobj = cbf_atof(l);
            continue;
        }
        if (!strcmp(l, "ACOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T2 e;
                if (sscanf(l, "%d %d %lf", &e.i, &e.j, &e.v) != 3) return PRIMAL_RES_ERR_ARG;
                if (!push2(&ac, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "BCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T2 e; e.j = 0;
                if (sscanf(l, "%d %lf", &e.i, &e.v) != 2) return PRIMAL_RES_ERR_ARG;
                if (!push2(&bc, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "OBJFCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T4 e;
                if (sscanf(l, "%d %d %d %lf", &e.j, &e.p, &e.q, &e.v) != 4)
                    return PRIMAL_RES_ERR_ARG;
                e.i = -1;
                if (!push4(&objF, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "FCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T4 e;
                if (sscanf(l, "%d %d %d %d %lf", &e.i, &e.j, &e.p, &e.q, &e.v) != 5)
                    return PRIMAL_RES_ERR_ARG;
                if (!push4(&fc, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "HCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T4 e;
                if (sscanf(l, "%d %d %d %d %lf", &e.i, &e.j, &e.p, &e.q, &e.v) != 5)
                    return PRIMAL_RES_ERR_ARG;
                if (!push4(&hc, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        if (!strcmp(l, "DCOORD")) {
            NEXT();
            l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
            int cnt = atoi(l);
            for (int k = 0; k < cnt; k++) {
                l = NEXT(); if (!l) return PRIMAL_RES_ERR_ARG;
                T4 e; e.j = 0;
                if (sscanf(l, "%d %d %d %lf", &e.i, &e.p, &e.q, &e.v) != 4)
                    return PRIMAL_RES_ERR_ARG;
                if (!push4(&dc, e)) return PRIMAL_RES_ERR_ALLOC;
            }
            continue;
        }
        return PRIMAL_RES_ERR_ARG;   /* keyword non riconosciuta */
    }

    if (!have_var || !have_con) return PRIMAL_RES_ERR_ARG;
    rc = build_task(t, sense_max, nv, vgrp, nvgrp, pows, npows,
                    ints, nint, psdvar, npsdvar, m, cgrp, ncgrp,
                    objA.d, objA.n, bobj, objF.d, objF.n,
                    ac.d, ac.n, bc.d, bc.n, fc.d, fc.n,
                    hc.d, hc.n, dc.d, dc.n, psdcon, npsdcon);
    if (rc == PRIMAL_RES_OK && idx < nl && PEEK() && !strcmp(PEEK(), "CHANGE"))
        rc = read_change(t, lines, nl, &idx);
    for (int i = 0; i < nl; i++) free(lines[i]);
    free(vgrp); free(cgrp); free(ints); free(psdvar); free(psdcon);
    free(objA.d); free(ac.d); free(bc.d); free(objF.d);
    free(fc.d); free(hc.d); free(dc.d);
    return rc;
}