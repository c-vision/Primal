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

/* mpsio.c - free-format MPS reader/writer and CPLEX LP reader/writer.
 *
 * MPS conventions implemented:
 *  - OBJSENSE (MAX/MIN); first N row = objective; other N rows = free rows
 *    (entries on them are ignored)
 *  - RHS on the objective row = MINUS the objective constant (standard
 *    solver convention)
 *  - RANGES: L row -> [b-r, b]; G row -> [b, b+r];
 *    E row r>0 -> [b, b+r], r<0 -> [b+r, b]
 *  - BOUNDS: LO/UP/FX/FR/MI/PL/BV; default 0 <= x < inf;
 *    UP with negative value and lower still at default 0 -> lower = -inf
 *  - MARKER INTORG/INTEND for integrality; BV -> binary
 *  - QSECTION/QMATRIX: entries accumulated in Q (obj adds 0.5 x'Qx)
 *
 * LP conventions (subset):
 *  - sections: min/max, subject to, bounds, general, binaries, end
 *  - terms: [+-] [coef] var; numeric constant in objective allowed
 *  - relations <= >= = (also < > == =< =>)
 *  - default bounds 0..inf; negative upper with default lower 0 -> lower -inf
 *  - Q terms are NOT supported in LP (write MPS instead)
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpsio.h"
#include "cbf.h"

#define INF INFINITY

/* ================= token stream ================= */
typedef struct {
    char **tok;
    int *line;          /* source line index of each token (0-based) */
    int curline;        /* line currently being tokenized */
    int n, cap;
} Toks;

/* Appends one token.  There is no error return: an allocation that fails
 * writes to stderr and exits the process, where the rest of the library hands
 * back PRIMAL_RES_ERR_ALLOC.  A caller of PRIMAL_readdata therefore cannot
 * recover from an out-of-memory here -- a deviation, not an oversight to be
 * argued away. */
static void toks_push(Toks *T, const char *s, int len) {
    if (T->n == T->cap) {
        T->cap = T->cap ? T->cap * 2 : 256;
        char **nt = (char **)realloc(T->tok, (size_t)T->cap * sizeof(char *));
        int *nl = (int *)realloc(T->line, (size_t)T->cap * sizeof(int));
        if (!nt || !nl) { fprintf(stderr, "out of memory\n"); exit(1); }
        T->tok = nt; T->line = nl;
    }
    char *c = (char *)malloc((size_t)len + 1);
    if (!c) { fprintf(stderr, "out of memory\n"); exit(1); }
    memcpy(c, s, (size_t)len); c[len] = 0;
    T->line[T->n] = T->curline;
    T->tok[T->n++] = c;
}

/* Loads the WHOLE file into memory before anything is parsed: the reader works
 * on a token array it can look ahead in, which is why a section can be
 * finished by any recognised keyword or by end of stream, and why a line longer
 * than the 8192-byte buffer is silently split into two.  Returns 0 only when
 * the file cannot be opened; the dispatcher turns that into ERR_FILE.
 * Grammar: a '*' is a comment only in column 1 (MPS), a '\' truncates the rest
 * of the line even inside MPS text (LP), and the characters : < > = are tokens
 * of their own -- with a following '=' glued to them -- so a reader can
 * recognise a relation without looking ahead. */
static int toks_load(const char *filename, Toks *T) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        T->curline++;
        char *p = line;
        if (*p == '*') continue;                      /* MPS comment line */
        char *q = strchr(line, '\\');
        if (q) *q = 0;                                /* LP comment */
        p = line;
        while (*p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            if (*p == ':') { toks_push(T, p, 1); p++; continue; }
            if (*p == '<' || *p == '>') {
                int len = (p[1] == '=') ? 2 : 1;
                toks_push(T, p, len); p += len; continue;
            }
            if (*p == '=') {
                int len = (p[1] == '=') ? 2 : 1;
                toks_push(T, p, len); p += len; continue;
            }
            int len = 0;
            while (p[len] && !isspace((unsigned char)p[len]) &&
                   !strchr(":<>=", p[len])) len++;
            toks_push(T, p, len);
            p += len;
        }
    }
    fclose(f);
    return 1;
}

static void toks_free(Toks *T) {
    for (int i = 0; i < T->n; i++) free(T->tok[i]);
    free(T->tok);
    free(T->line);
}

/* tok_is is bounds-safe: an index past the end answers no, which is how every
 * section here can end either on its keyword or on the end of the stream.
 * teq is exact, ieq is case-insensitive, and WHICH ONE a matcher uses is a rule
 * of the format, not a style choice: the MPS section keywords and the relations
 * are matched case-exactly -- so they must be spelled as the standard writes
 * them, in uppercase -- while the LP section keywords are accepted in any case.
 * rel_char folds the eight accepted spellings onto the three MPS row types and
 * answers E for anything else, so a caller must test is_rel before trusting it. */
static int tok_is(const Toks *T, int i, const char *s) {
    return i < T->n && strcmp(T->tok[i], s) == 0;
}
static int teq(const char *a, const char *b) { return strcmp(a, b) == 0; }
static int ieq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}
static int is_mps_section(const char *s) {
    return teq(s, "NAME") || teq(s, "OBJSENSE") || teq(s, "OBJNAME") ||
           teq(s, "ROWS") || teq(s, "COLUMNS") || teq(s, "RHS") ||
           teq(s, "RANGES") || teq(s, "BOUNDS") || teq(s, "QSECTION") ||
           teq(s, "QUADOBJ") || teq(s, "QMATRIX") || teq(s, "QCMATRIX") ||
           teq(s, "CSECTION") || teq(s, "ENDATA");
}
static int is_lp_section(const char *s) {
    return ieq(s, "subject") || ieq(s, "such") || ieq(s, "st") || ieq(s, "s.t.") ||
           ieq(s, "bounds") || ieq(s, "bound") || ieq(s, "general") ||
           ieq(s, "generals") || ieq(s, "gen") || ieq(s, "integer") ||
           ieq(s, "integers") || ieq(s, "binaries") || ieq(s, "binary") ||
           ieq(s, "bin") || ieq(s, "end");
}
static int is_rel(const char *s) {
    return ieq(s, "<=") || ieq(s, "<") || ieq(s, "=<") || ieq(s, ">=") ||
           ieq(s, ">") || ieq(s, "=>") || ieq(s, "=") || ieq(s, "==");
}
static char rel_char(const char *s) {
    if (teq(s, "<=") || teq(s, "<") || teq(s, "=<")) return 'L';
    if (teq(s, ">=") || teq(s, ">") || teq(s, "=>")) return 'G';
    return 'E';
}

/* Same out-of-memory rule as toks_push: allocate or exit. */
static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (!r) { fprintf(stderr, "out of memory\n"); exit(1); }
    memcpy(r, s, n);
    return r;
}

/* Suffix test, and it is asymmetric: the FILE name is folded to lower case but
 * `ext` is compared verbatim, so an extension must be passed in lower case --
 * which every call site here does (".cbf", ".lp").  Passing ".MPS" would match
 * nothing, and the fallback for an unrecognised suffix is MPS, so the cost of
 * getting it wrong is a format guess, not an error. */
static int has_ext(const char *fn, const char *ext) {
    size_t ln = strlen(fn), le = strlen(ext);
    if (ln < le) return 0;
    for (size_t i = 0; i < le; i++)
        if (tolower((unsigned char)fn[ln - le + i]) != ext[i]) return 0;
    return 1;
}

/* growable arrays */
typedef struct { double *a; int n, cap; } DArr;
typedef struct { int *a; int n, cap; } IArr;
static void dpush(DArr *v, double x) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        double *b = (double *)realloc(v->a, (size_t)v->cap * sizeof(double));
        if (!b) { fprintf(stderr, "out of memory\n"); exit(1); }
        v->a = b;
    }
    v->a[v->n++] = x;
}
static void ipush(IArr *v, int x) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        int *b = (int *)realloc(v->a, (size_t)v->cap * sizeof(int));
        if (!b) { fprintf(stderr, "out of memory\n"); exit(1); }
        v->a = b;
    }
    v->a[v->n++] = x;
}

/* name -> index lookup */
typedef struct { char **nm; int n, cap; } Names;
/* Name -> index, exact and CASE SENSITIVE (ROW0 and row0 are two names in
 * either format), by linear scan: reading a model with N distinct rows or
 * columns costs O(N^2) comparisons, which is fine for the files here and is the
 * first thing to replace if a large MPS ever has to be parsed.  create=1 appends
 * and returns the new index; create=0 answers -1, which is how the reader says
 * "this token is not one of my names" without distinguishing it from an empty
 * table.  The strings are duplicated on insert, so names_free owns them. */
static int names_get(Names *N, const char *s, int create) {
    for (int i = 0; i < N->n; i++)
        if (strcmp(N->nm[i], s) == 0) return i;
    if (!create) return -1;
    if (N->n == N->cap) {
        N->cap = N->cap ? N->cap * 2 : 16;
        char **nn = (char **)realloc(N->nm, (size_t)N->cap * sizeof(char *));
        if (!nn) { fprintf(stderr, "out of memory\n"); exit(1); }
        N->nm = nn;
    }
    N->nm[N->n] = xstrdup(s);
    return N->n++;
}
static void names_free(Names *N) {
    for (int i = 0; i < N->n; i++) free(N->nm[i]);
    free(N->nm);
}

/* A number is a token that strtod consumes ENTIRELY.  The whole-token rule is
 * what keeps a free-format reader unambiguous, where a name and a value sit in
 * the same whitespace-separated stream with no column to tell them apart: it
 * rejects both leading junk (end == s) and a value with a tail stuck on it
 * ("1.5x"), which would otherwise read as 1.5 and quietly move a coefficient.
 * A token that overflows is not questioned -- strtod returns an Inf and so do
 * we, and the model that gets built is the file's problem, not the parser's. */
static int parse_num(const char *s, double *v) {
    char *end = NULL;
    double x = strtod(s, &end);
    if (!end || *end != 0 || end == s) return 0;
    *v = x;
    return 1;
}

/* ================= MPS writer ================= */
/* The writer names rows and columns as the TASK calls them when the name is
 * non-empty and free of whitespace (MPS/LP names are whitespace-delimited), and
 * falls back to the positional key otherwise. The reader populates the name
 * tables (T112), so a read -> write -> read round-trip now preserves the labels;
 * before this the writer coined x%d/c%d/obj and the labels were lost. */
static const char *w_name(const char *name, char *buf, size_t bufn,
                          const char *fmt, int idx) {
    if (name && name[0]) {
        int ok = 1;
        for (const char *p = name; *p; p++)
            if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') { ok = 0; break; }
        if (ok) return name;
    }
    snprintf(buf, bufn, fmt, idx);
    return buf;
}
static const char *mps_rowname(PRIMALtask_t t, int i, char *buf, size_t n) {
    const char *nm = "";
    PRIMAL_getconnameidx(t, i, &nm);
    return w_name(nm, buf, n, "c%d", i);
}
static const char *mps_colname(PRIMALtask_t t, int j, char *buf, size_t n) {
    const char *nm = "";
    PRIMAL_getvarnameidx(t, j, &nm);
    return w_name(nm, buf, n, "x%d", j);
}
static const char *mps_objname(PRIMALtask_t t, char *buf, size_t n) {
    const char *nm = "";
    PRIMAL_getobjname(t, &nm);
    return w_name(nm, buf, n, "obj", 0);
}

static PRIMALrescodee mps_write(PRIMALtask_t t, FILE *f) {
    int numcon = 0, numvar = 0;
    PRIMALobjsensee sense;
    double cfix = 0.0;
    PRIMALrescodee rc;
    if ((rc = PRIMAL_getnumcon(t, &numcon))) return rc;
    if ((rc = PRIMAL_getnumvar(t, &numvar))) return rc;
    if ((rc = PRIMAL_getobjsense(t, &sense))) return rc;
    PRIMAL_getcfix(t, &cfix);

    fprintf(f, "NAME primalclone\n");
    if (sense == PRIMAL_OPTIMIZE_MAXIMIZE)
        fprintf(f, "OBJSENSE\n    MAX\n");
    {   /* the objective's own name (reference OBJNAME), when it is a token an MPS
         * can carry; the `N` row name is separate (it is the reader's row 0). */
        const char *nm = "";
        PRIMAL_getobjname(t, &nm);
        if (nm && nm[0]) {
            int ok = 1;
            for (const char *p = nm; *p; p++)
                if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') { ok = 0; break; }
            if (ok) fprintf(f, "OBJNAME\n    %s\n", nm);
        }
    }
    /* Our reader counts the MPS `N` row as a task row (row 0), so a model read
     * from a file has row 0 = the free, all-zero objective row. The writer must
     * emit exactly that shape or the round-trip grows by one row: it writes the
     * `N` row for row 0 and the constraints for rows 1..numcon-1. A hand-built
     * task whose row 0 is a real constraint keeps the old path (an `N` row plus
     * every row as a constraint), documented: for it the re-read has one row
     * more. */
    int start = 0;
    if (numcon > 0) {
        PRIMALboundkeye bk0; double bl0, bu0;
        PRIMAL_getconbound(t, 0, &bk0, &bl0, &bu0);
        int free0 = (!isfinite(bl0) && !isfinite(bu0));
        int anyA = 0;
        for (int j = 0; j < numvar && !anyA; j++) {
            double a; if (PRIMAL_getaij(t, 0, j, &a) == PRIMAL_RES_OK && a != 0.0) anyA = 1;
        }
        if (free0 && !anyA) start = 1;
    }
    char ob[32];
    /* When row 0 is the file's free `N` row, the `N` line must carry THAT row's
     * name (the reader stores it in conname[0]); otherwise the objective is
     * implicit and `on` is the objective's own name. */
    const char *on = (start == 1) ? mps_rowname(t, 0, ob, sizeof ob)
                                  : mps_objname(t, ob, sizeof ob);
    fprintf(f, "ROWS\n N %s\n", on);
    for (int i = start; i < numcon; i++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        int flo = isfinite(bl), fup = isfinite(bu);
        char ty;
        if (bk == PRIMAL_BK_FX ||
            (flo && fup && fabs(bl - bu) <= 1e-12 * (1.0 + fabs(bl))))
            ty = 'E';
        else if (!flo) ty = 'L';
        else if (!fup) ty = 'G';
        else ty = 'L';   /* ranged -> L row + RANGES */
        char rb[32];
        fprintf(f, " %c %s\n", ty, mps_rowname(t, i, rb, sizeof rb));
    }
    fprintf(f, "COLUMNS\n");
    for (int j = 0; j < numvar; j++) {
        PRIMALvariabletypee vt;
        PRIMAL_getvartype(t, j, &vt);
        if (vt != PRIMAL_VAR_TYPE_CONT)
            fprintf(f, "    MARKER                 'MARKER'                 'INTORG'\n");
        double cj;
        PRIMAL_getcj(t, j, &cj);
        char cb[32];
        const char *cn = mps_colname(t, j, cb, sizeof cb);
        if (cj != 0.0)
            fprintf(f, "    %s        %s        %.17g\n", cn, on, cj);
        for (int i = start; i < numcon; i++) {
            double aij;
            PRIMAL_getaij(t, i, j, &aij);
            if (aij != 0.0) {
                char rb2[32];
                fprintf(f, "    %s        %s         %.17g\n", cn,
                        mps_rowname(t, i, rb2, sizeof rb2), aij);
            }
        }
        if (vt != PRIMAL_VAR_TYPE_CONT)
            fprintf(f, "    MARKER                 'MARKER'                 'INTEND'\n");
    }
    fprintf(f, "RHS\n");
    if (cfix != 0.0)
        fprintf(f, "    rhs       %s        %.17g\n", on, -cfix);
    for (int i = start; i < numcon; i++) {
        PRIMALboundkeye bk; double bl, bu, b;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        switch (bk) {
            case PRIMAL_BK_LO: b = bl; break;
            case PRIMAL_BK_UP: b = bu; break;
            case PRIMAL_BK_FX: b = bl; break;
            default:        b = isfinite(bu) ? bu : bl; break;
        }
        if (isfinite(b)) {
            char rb[32];
            fprintf(f, "    rhs       %s         %.17g\n",
                    mps_rowname(t, i, rb, sizeof rb), b);
        }
    }
    int any_range = 0;
    for (int i = start; i < numcon && !any_range; i++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        if (bk == PRIMAL_BK_RA) any_range = 1;
    }
    if (any_range) {
        fprintf(f, "RANGES\n");
        for (int i = start; i < numcon; i++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getconbound(t, i, &bk, &bl, &bu);
            if (bk == PRIMAL_BK_RA && isfinite(bl) && isfinite(bu)) {
                char rb[32];
                fprintf(f, "    rng       %s         %.17g\n",
                        mps_rowname(t, i, rb, sizeof rb), fabs(bu - bl));
            }
        }
    }
    fprintf(f, "BOUNDS\n");
    for (int j = 0; j < numvar; j++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
        char cb[32];
        const char *cn = mps_colname(t, j, cb, sizeof cb);
        if (bk == PRIMAL_BK_FX) {
            fprintf(f, " FX bnd       %s         %.17g\n", cn, bl);
            continue;
        }
        if (bk == PRIMAL_BK_FR) {
            fprintf(f, " FR bnd       %s\n", cn);
            continue;
        }
        if (isfinite(bl) && bl != 0.0)
            fprintf(f, " LO bnd       %s         %.17g\n", cn, bl);
        if (isfinite(bu))
            fprintf(f, " UP bnd       %s         %.17g\n", cn, bu);
        if (bk == PRIMAL_BK_UP && !isfinite(bl) && isfinite(bu) && bu < 0.0)
            fprintf(f, " MI bnd       %s\n", cn);  /* make lower -inf explicit */
    }
    /* probe for a quadratic objective (upper triangle) */
    int has_q = 0;
    for (int i = 0; i < numvar && !has_q; i++)
        for (int j = i; j < numvar && !has_q; j++) {
            double qij;
            if (PRIMAL_getqobjij(t, i, j, &qij) == PRIMAL_RES_OK && qij != 0.0)
                has_q = 1;
        }
    if (has_q) {
        /* QUADOBJ (formato CPLEX) invece di QSECTION: e' quello che HiGHS
         * riconosce (QSECTION gli fa segnalare un errore di sintassi), e va
         * SENZA nome di riga: con "QUADOBJ obj" HiGHS scarta l'intero obiettivo
         * (anche lineare); con il solo "QUADOBJ" lo legge. La semantica e' la
         * stessa: le entrate valgono per 0.5 x'Qx. */
        fprintf(f, "QUADOBJ\n");
        for (int i = 0; i < numvar; i++)
            for (int j = i; j < numvar; j++) {
                double qij;
                PRIMAL_getqobjij(t, i, j, &qij);
                if (qij != 0.0) {
                    char cbi[32], cbj[32];
                    fprintf(f, "    %s        %s         %.17g\n",
                            mps_colname(t, i, cbi, sizeof cbi),
                            mps_colname(t, j, cbj, sizeof cbj), qij);
                }
            }
    }
    /* CSECTION (MOSEK extension): the cones, so a conic model round-trips. */
    {
        int ncon = 0;
        PRIMAL_getnumcone(t, &ncon);
        for (int k = 0; k < ncon; k++) {
            PRIMALconetypee ct; double par = 0.0; int nmem = 0;
            PRIMAL_getconeinfo(t, k, &ct, &par, &nmem);
            const char *tn = ct == PRIMAL_CT_QUAD ? "QUAD" : ct == PRIMAL_CT_RQUAD ? "RQUAD" :
                             ct == PRIMAL_CT_PEXP ? "PEXP" : ct == PRIMAL_CT_DEXP ? "DEXP" :
                             ct == PRIMAL_CT_PPOW ? "PPOW" : ct == PRIMAL_CT_RPOW ? "DPOW" : NULL;
            if (!tn) continue;
            const char *nm = NULL; PRIMAL_getconenameidx(t, k, &nm);
            char kn[32];
            int safe = nm && *nm;
            for (const char *q = nm; safe && *q; q++) if (*q == ' ' || *q == '\t') safe = 0;
            if (!safe) { snprintf(kn, sizeof kn, "k%d", k); nm = kn; }
            int *mem = (int *)malloc((size_t)(nmem > 0 ? nmem : 1) * sizeof(int));
            if (!mem) continue;
            PRIMAL_getcone(t, k, &ct, &nmem, mem);
            fprintf(f, "CSECTION %s %.17g %s", nm, par, tn);
            for (int e = 0; e < nmem; e++) {
                char cb[32];
                fprintf(f, " %s", mps_colname(t, mem[e], cb, sizeof cb));
            }
            fprintf(f, "\n");
            free(mem);
        }
    }
    fprintf(f, "ENDATA\n");
    return PRIMAL_RES_OK;
}

/* ================= LP writer ================= */
/* gather entries of one constraint row via getaij (dense scan) */
static void lp_row_terms(FILE *f, PRIMALtask_t t, int numvar, int con) {
    int first = 1;
    for (int j = 0; j < numvar; j++) {
        double aij;
        if (PRIMAL_getaij(t, con, j, &aij) != PRIMAL_RES_OK || aij == 0.0) continue;
        char cb[32];
        fprintf(f, " %s%.17g %s", aij < 0 ? "-" : "+", fabs(aij),
                mps_colname(t, j, cb, sizeof cb));
        first = 0;
    }
    if (first) { char cb[32]; fprintf(f, " 0 %s", mps_colname(t, 0, cb, sizeof cb)); }
}

static PRIMALrescodee lp_write(PRIMALtask_t t, FILE *f) {
    int numcon = 0, numvar = 0;
    PRIMALobjsensee sense;
    double cfix = 0.0;
    PRIMALrescodee rc;
    if ((rc = PRIMAL_getnumcon(t, &numcon))) return rc;
    if ((rc = PRIMAL_getnumvar(t, &numvar))) return rc;
    if ((rc = PRIMAL_getobjsense(t, &sense))) return rc;
    PRIMAL_getcfix(t, &cfix);
    /* The LP format does support a quadratic objective/constraint (ref. 16.1),
     * but this writer emits only the linear part; refuse instead of writing a
     * file that has silently lost the quadratic terms (OPF and MPS emit it). */
    {
        int nqo = 0;
        if (PRIMAL_getnumqobjnz(t, &nqo) == PRIMAL_RES_OK && nqo > 0) return PRIMAL_RES_ERR_ARG;
        for (int i = 0; i < numcon; i++) {
            int nq = 0;
            PRIMAL_getnumqconknz(t, i, &nq);
            if (nq > 0) return PRIMAL_RES_ERR_ARG;
        }
    }

    fprintf(f, "\\ PRIMAL-clone LP export\n");
    fprintf(f, "%s\n", sense == PRIMAL_OPTIMIZE_MAXIMIZE ? "Maximize" : "Minimize");
    fprintf(f, " obj:");
    {
        int first = 1;
        for (int j = 0; j < numvar; j++) {
            double cj;
            PRIMAL_getcj(t, j, &cj);
            if (cj == 0.0) continue;
            char cb[32];
            fprintf(f, " %s%.17g %s", cj < 0 ? "-" : "+", fabs(cj),
                    mps_colname(t, j, cb, sizeof cb));
            first = 0;
        }
        if (cfix != 0.0) {
            fprintf(f, " %s%.17g", cfix < 0 ? "-" : "+", fabs(cfix));
            first = 0;
        }
        if (first) fprintf(f, " 0");
    }
    fprintf(f, "\nSubject To\n");
    for (int i = 0; i < numcon; i++) {
        PRIMALboundkeye bk; double bl, bu;
        PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        switch (bk) {
            case PRIMAL_BK_FX:
                { char rb[32]; fprintf(f, " %s:", mps_rowname(t, i, rb, sizeof rb)); }
                lp_row_terms(f, t, numvar, i);
                fprintf(f, " = %.17g\n", bl);
                break;
            case PRIMAL_BK_LO:
                { char rb[32]; fprintf(f, " %s:", mps_rowname(t, i, rb, sizeof rb)); }
                lp_row_terms(f, t, numvar, i);
                fprintf(f, " >= %.17g\n", bl);
                break;
            case PRIMAL_BK_UP:
                { char rb[32]; fprintf(f, " %s:", mps_rowname(t, i, rb, sizeof rb)); }
                lp_row_terms(f, t, numvar, i);
                fprintf(f, " <= %.17g\n", bu);
                break;
            default: /* RA: two constraints */
                { char rb[32]; fprintf(f, " %s_a:", mps_rowname(t, i, rb, sizeof rb)); }
                lp_row_terms(f, t, numvar, i);
                fprintf(f, " >= %.17g\n", bl);
                { char rb[32]; fprintf(f, " %s_b:", mps_rowname(t, i, rb, sizeof rb)); }
                lp_row_terms(f, t, numvar, i);
                fprintf(f, " <= %.17g\n", bu);
                break;
        }
    }
    {
        int any_bnd = 0, any_int = 0, any_bin = 0;
        for (int j = 0; j < numvar; j++) {
            PRIMALboundkeye bk; double bl, bu;
            PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
            if (!(bk == PRIMAL_BK_LO && bl == 0.0 && !isfinite(bu))) any_bnd = 1;
            PRIMALvariabletypee vt;
            PRIMAL_getvartype(t, j, &vt);
            if (vt == PRIMAL_VAR_TYPE_INT) any_int = 1;
            if (vt == PRIMAL_VAR_TYPE_INT_BIN) any_bin = 1;
        }
        if (any_bnd) {
            fprintf(f, "Bounds\n");
            for (int j = 0; j < numvar; j++) {
                PRIMALboundkeye bk; double bl, bu;
                PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
                char cb[32];
                const char *cn = mps_colname(t, j, cb, sizeof cb);
                if (bk == PRIMAL_BK_FX) { fprintf(f, " %s = %.17g\n", cn, bl); continue; }
                if (bk == PRIMAL_BK_FR) { fprintf(f, " %s free\n", cn); continue; }
                if (isfinite(bl) && bl != 0.0) fprintf(f, " %s >= %.17g\n", cn, bl);
                if (isfinite(bu)) fprintf(f, " %s <= %.17g\n", cn, bu);
            }
        }
        if (any_int) {
            fprintf(f, "General\n");
            for (int j = 0; j < numvar; j++) {
                PRIMALvariabletypee vt;
                PRIMAL_getvartype(t, j, &vt);
                if (vt == PRIMAL_VAR_TYPE_INT) { char cb[32]; fprintf(f, " %s\n", mps_colname(t, j, cb, sizeof cb)); }
            }
        }
        if (any_bin) {
            fprintf(f, "Binaries\n");
            for (int j = 0; j < numvar; j++) {
                PRIMALvariabletypee vt;
                PRIMAL_getvartype(t, j, &vt);
                if (vt == PRIMAL_VAR_TYPE_INT_BIN) { char cb[32]; fprintf(f, " %s\n", mps_colname(t, j, cb, sizeof cb)); }
            }
        }
    }
    fprintf(f, "End\n");
    return PRIMAL_RES_OK;
}

/* ================= MPS reader ================= */
static PRIMALrescodee mps_read(PRIMALtask_t t, const Toks *T) {
    Names rows = {0, 0, 0}, cols = {0, 0, 0};
    char *rty = NULL;               /* row type chars 'N','L','G','E' */
    int objrow = -1;
    int nvar = 0;
    const char *objnm = NULL;   /* owned by the tokenizer; putobjname duplicates */
    PRIMALrescodee rc = PRIMAL_RES_OK;

    IArr *csub = NULL;
    DArr *cval = NULL;
    char *rhs_set = NULL;           /* rows that got an explicit RHS entry */
    double *cj = NULL, *bl = NULL, *bu = NULL, *rlo = NULL, *rup = NULL;
    PRIMALvariabletypee *vt = NULL;
    int *qi = NULL, *qj = NULL;
    double *qv = NULL;
    int q_cap = 64, q_n = 0;
    int *cqi = NULL, *cqj = NULL, *cqk = NULL;   /* QCMATRIX: (con, i, j, val) */
    double *cqv = NULL;
    int cq_cap = 64, cq_n = 0;
    /* CSECTION (MOSEK extension): cones */
    int cone_acc_n = 0, cone_acc_cap = 0;
    int *ca_type = NULL, *ca_nmem = NULL; double *ca_par = NULL; int **ca_mem = NULL;

    /* ---- pass 1: ROWS section (row names/types) ---- */
    {
        int i = 0;
        while (i < T->n && !tok_is(T, i, "ROWS")) i++;
        if (i == T->n) return PRIMAL_RES_ERR_FILE;
        i++;
        int cap = 16;
        rows.nm = (char **)malloc((size_t)cap * sizeof(char *));
        rty = (char *)malloc((size_t)cap);
        if (!rows.nm || !rty) return PRIMAL_RES_ERR_ALLOC;
        rows.cap = cap;
        while (i < T->n && !is_mps_section(T->tok[i])) {
            if (i + 1 >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            const char *ty = T->tok[i++];
            const char *nm = T->tok[i++];
            /* Two rows with one name are not a label problem but a model one:
             * every section below looks rows up by name, names_get answers the
             * FIRST occurrence, and the second row receives no entries at all.
             * Refused here, where nothing has reached the task yet. */
            for (int k = 0; k < rows.n; k++)
                if (teq(rows.nm[k], nm)) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            if (rows.n == rows.cap) {
                rows.cap *= 2;
                char **nn = (char **)realloc(rows.nm, (size_t)rows.cap * sizeof(char *));
                char *nt = (char *)realloc(rty, (size_t)rows.cap);
                if (!nn || !nt) { rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
                rows.nm = nn; rty = nt;
            }
            if (strlen(ty) != 1 || !strchr("NLGE", ty[0])) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            rty[rows.n] = ty[0];
            rows.nm[rows.n] = xstrdup(nm);
            if (ty[0] == 'N' && objrow < 0) objrow = rows.n;
            rows.n++;
        }
    }

    /* ---- pass 2: COLUMNS (var names + sizes) ---- */
    {
        int i = 0;
        while (i < T->n && !tok_is(T, i, "COLUMNS")) i++;
        if (i == T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
        i++;
        while (i < T->n && !is_mps_section(T->tok[i])) {
            if (teq(T->tok[i], "MARKER")) {
                i++;
                if (i < T->n && teq(T->tok[i], "'MARKER'")) i++;
                if (i < T->n && (teq(T->tok[i], "'INTORG'") || teq(T->tok[i], "'INTEND'"))) i++;
                else { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                continue;
            }
            names_get(&cols, T->tok[i], 1);
            int cline = T->line[i];
            i++;
            /* entries: (known-row, numeric-value) pairs ON THE SAME LINE. The
             * tokenizer keeps the line index so the loop cannot run past the
             * end of the line: with all-numeric names (netlib `blend`) the next
             * line's column name is itself a valid row name, so it would be
             * eaten as a row and a value like `-3.` would become a column. */
            while (i + 1 < T->n && T->line[i] == cline && !is_mps_section(T->tok[i]) &&
                   !teq(T->tok[i], "MARKER") &&
                   names_get(&rows, T->tok[i], 0) >= 0 &&
                   parse_num(T->tok[i + 1], &(double){0.0}))
                i += 2;
        }
        nvar = cols.n;
    }
    if (nvar == 0) { rc = PRIMAL_RES_ERR_FILE; goto fail; }

    /* ---- scratch ---- */
    csub = (IArr *)calloc((size_t)nvar, sizeof(IArr));
    cval = (DArr *)calloc((size_t)nvar, sizeof(DArr));
    cj = (double *)calloc((size_t)nvar, sizeof(double));
    bl = (double *)malloc((size_t)nvar * sizeof(double));
    bu = (double *)malloc((size_t)nvar * sizeof(double));
    vt = (PRIMALvariabletypee *)calloc((size_t)nvar, sizeof(PRIMALvariabletypee));
    rlo = (double *)malloc((size_t)(rows.n > 0 ? rows.n : 1) * sizeof(double));
    rup = (double *)malloc((size_t)(rows.n > 0 ? rows.n : 1) * sizeof(double));
    qi = (int *)malloc((size_t)q_cap * sizeof(int));
    qj = (int *)malloc((size_t)q_cap * sizeof(int));
    qv = (double *)malloc((size_t)q_cap * sizeof(double));
    if (!csub || !cval || !cj || !bl || !bu || !vt || !rlo || !rup || !qi || !qj || !qv) {
        rc = PRIMAL_RES_ERR_ALLOC; goto fail;
    }
    rhs_set = (char *)calloc((size_t)(rows.n > 0 ? rows.n : 1), 1);
    if (!rhs_set) { rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
    for (int j = 0; j < nvar; j++) { bl[j] = 0.0; bu[j] = INF; }
    for (int r = 0; r < rows.n; r++) { rlo[r] = -INF; rup[r] = INF; }

    /* ---- pass 3: data sections ---- */
    PRIMALobjsensee sense = PRIMAL_OPTIMIZE_MINIMIZE;
    double cfix = 0.0;
    int i = 0;
    while (i < T->n) {
        const char *sec = T->tok[i];
        if (teq(sec, "NAME")) {
            /* the problem name: this task has no table for it (declared) */
            i++; continue;
        }
        if (teq(sec, "OBJNAME")) {
            i++;
            /* Advance past the name only if this token is one: a file that
             * writes OBJNAME with no name must not lose the section after it. */
            if (i < T->n && !is_mps_section(T->tok[i])) { objnm = T->tok[i]; i++; }
            continue;
        }
        if (teq(sec, "OBJSENSE")) {
            i++;
            if (i < T->n && (teq(T->tok[i], "MAX") || teq(T->tok[i], "MAXIMIZE")))
                sense = PRIMAL_OPTIMIZE_MAXIMIZE;
            else if (i < T->n && (teq(T->tok[i], "MIN") || teq(T->tok[i], "MINIMIZE")))
                sense = PRIMAL_OPTIMIZE_MINIMIZE;
            i++;
            continue;
        }
        if (teq(sec, "ROWS")) {
            i++;
            while (i < T->n && !is_mps_section(T->tok[i])) i += 2;
            continue;
        }
        if (teq(sec, "COLUMNS")) {
            i++;
            int inint = 0;
            while (i < T->n && !is_mps_section(T->tok[i])) {
                if (teq(T->tok[i], "MARKER")) {
                    i++;
                    if (i < T->n && teq(T->tok[i], "'MARKER'")) i++;
                    if (i < T->n && teq(T->tok[i], "'INTORG'")) { inint = 1; i++; }
                    else if (i < T->n && teq(T->tok[i], "'INTEND'")) { inint = 0; i++; }
                    else { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                    continue;
                }
                int cline = T->line[i];
                int c = names_get(&cols, T->tok[i++], 0);
                if (c < 0) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                if (inint) vt[c] = PRIMAL_VAR_TYPE_INT;
                while (i + 1 < T->n && T->line[i] == cline && !is_mps_section(T->tok[i]) &&
                       !teq(T->tok[i], "MARKER") &&
                       names_get(&rows, T->tok[i], 0) >= 0 &&
                       parse_num(T->tok[i + 1], &(double){0.0})) {
                    int r = names_get(&rows, T->tok[i], 0);
                    double v;
                    int oknum = parse_num(T->tok[i + 1], &v);
                    i += 2;
                    if (r < 0 || !oknum) continue;
                    if (r == objrow) { cj[c] += v; continue; }
                    if (rty[r] == 'N') continue;   /* free row: ignore */
                    ipush(&csub[c], r);
                    dpush(&cval[c], v);
                }
            }
            continue;
        }
        if (teq(sec, "RHS")) {
            i++;
            while (i < T->n && !is_mps_section(T->tok[i])) {
                /* entry = (row, value); a leading bound-set name is skipped */
                if (names_get(&rows, T->tok[i], 0) < 0 ||
                    i + 1 >= T->n || !parse_num(T->tok[i + 1], &(double){0.0})) {
                    i++;   /* set name or junk */
                    continue;
                }
                int r = names_get(&rows, T->tok[i], 0);
                double v;
                parse_num(T->tok[i + 1], &v);
                i += 2;
                if (r == objrow) { cfix = -v; continue; }
                if (rty[r] == 'N') continue;
                rhs_set[r] = 1;
                if (rty[r] == 'L') rup[r] = v;
                else if (rty[r] == 'G') rlo[r] = v;
                else { rlo[r] = v; rup[r] = v; }
            }
            continue;
        }
        if (teq(sec, "RANGES")) {
            i++;
            while (i < T->n && !is_mps_section(T->tok[i])) {
                if (names_get(&rows, T->tok[i], 0) < 0 ||
                    i + 1 >= T->n || !parse_num(T->tok[i + 1], &(double){0.0})) {
                    i++;
                    continue;
                }
                int r = names_get(&rows, T->tok[i], 0);
                double v;
                parse_num(T->tok[i + 1], &v);
                i += 2;
                if (rty[r] == 'L') rlo[r] = rup[r] - fabs(v);
                else if (rty[r] == 'G') rup[r] = rlo[r] + fabs(v);
                else {
                    if (v > 0) rup[r] = rlo[r] + v;
                    else if (v < 0) rlo[r] = rlo[r] + v;
                }
            }
            continue;
        }
        if (teq(sec, "BOUNDS")) {
            i++;
            while (i < T->n && !is_mps_section(T->tok[i])) {
                const char *ty = T->tok[i++];
                if (i >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                i++;                                   /* bound-set name */
                if (i >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                int c = names_get(&cols, T->tok[i++], 0);
                if (c < 0) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                double v = 0.0;
                int has_v = 0;
                if (i < T->n && !is_mps_section(T->tok[i]) &&
                    !teq(T->tok[i], "MARKER") &&
                    (isdigit((unsigned char)T->tok[i][0]) ||
                     T->tok[i][0] == '-' || T->tok[i][0] == '+' ||
                     T->tok[i][0] == '.')) {
                    v = strtod(T->tok[i++], NULL); has_v = 1;
                }
                if (teq(ty, "LO")) { if (!has_v) { rc = PRIMAL_RES_ERR_FILE; goto fail; } bl[c] = v; }
                else if (teq(ty, "UP")) {
                    if (!has_v) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                    bu[c] = v;
                    if (v < 0.0 && bl[c] == 0.0) bl[c] = -INF;  /* classic MPS rule */
                }
                else if (teq(ty, "FX")) {
                    if (!has_v) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                    bl[c] = bu[c] = v;
                }
                else if (teq(ty, "FR")) { bl[c] = -INF; bu[c] = INF; }
                else if (teq(ty, "MI")) { bl[c] = -INF; }
                else if (teq(ty, "PL")) { bu[c] = INF; }
                else if (teq(ty, "BV")) { bl[c] = 0.0; bu[c] = 1.0; vt[c] = PRIMAL_VAR_TYPE_INT_BIN; }
                else { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            }
            continue;
        }
        if (teq(sec, "QSECTION") || teq(sec, "QUADOBJ") || teq(sec, "QMATRIX")) {
            i++;
            /* QSECTION/QMATRIX portano il nome della riga obiettivo; QUADOBJ
             * (CPLEX) NO: saltare il primo token li' mangerebbe la prima
             * entrata della matrice. */
            if (!teq(sec, "QUADOBJ") && i < T->n && !is_mps_section(T->tok[i])) i++;
            while (i + 2 < T->n && !is_mps_section(T->tok[i])) {
                int a = names_get(&cols, T->tok[i], 0);
                int b = names_get(&cols, T->tok[i + 1], 0);
                double v;
                int oknum = parse_num(T->tok[i + 2], &v);
                i += 3;
                if (a < 0 || b < 0 || !oknum) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                if (q_n == q_cap) {
                    q_cap *= 2;
                    int *ti = (int *)realloc(qi, (size_t)q_cap * sizeof(int));
                    int *tj = (int *)realloc(qj, (size_t)q_cap * sizeof(int));
                    double *tv = (double *)realloc(qv, (size_t)q_cap * sizeof(double));
                    if (!ti || !tj || !tv) { rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
                    qi = ti; qj = tj; qv = tv;
                }
                qi[q_n] = a; qj[q_n] = b; qv[q_n] = v; q_n++;
            }
            continue;
        }
        if (teq(sec, "QCMATRIX")) {
            /* header carries the constraint name, then records (vname1 vname2 val) */
            i++;
            if (i >= T->n || is_mps_section(T->tok[i])) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            int r = names_get(&rows, T->tok[i], 0);
            i++;
            if (r < 0) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            if (!cqk) {
                cqk = (int *)malloc((size_t)cq_cap * sizeof(int));
                cqi = (int *)malloc((size_t)cq_cap * sizeof(int));
                cqj = (int *)malloc((size_t)cq_cap * sizeof(int));
                cqv = (double *)malloc((size_t)cq_cap * sizeof(double));
                if (!cqk || !cqi || !cqj || !cqv) { rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
            }
            while (i + 2 < T->n && !is_mps_section(T->tok[i])) {
                int a = names_get(&cols, T->tok[i], 0);
                int bb = names_get(&cols, T->tok[i + 1], 0);
                double v;
                int oknum = parse_num(T->tok[i + 2], &v);
                i += 3;
                if (a < 0 || bb < 0 || !oknum) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
                if (cq_n == cq_cap) {
                    cq_cap *= 2;
                    int *t1 = (int *)realloc(cqk, (size_t)cq_cap * sizeof(int));
                    int *t2 = (int *)realloc(cqi, (size_t)cq_cap * sizeof(int));
                    int *t3 = (int *)realloc(cqj, (size_t)cq_cap * sizeof(int));
                    double *t4 = (double *)realloc(cqv, (size_t)cq_cap * sizeof(double));
                    if (!t1 || !t2 || !t3 || !t4) { rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
                    cqk = t1; cqi = t2; cqj = t3; cqv = t4;
                }
                cqk[cq_n] = r; cqi[cq_n] = a; cqj[cq_n] = bb; cqv[cq_n] = v; cq_n++;
            }
            continue;
        }
        if (teq(sec, "CSECTION")) {
            /* header CSECTION kname [value] ktype, then the member variables */
            i++;
            if (i >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            i++;                                    /* cone name (not stored) */
            double cpar = 0.0;
            { double vv; if (i < T->n && parse_num(T->tok[i], &vv)) { cpar = vv; i++; } }
            if (i >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto fail; }
            const char *kt = T->tok[i++];
            int ct;
            if (teq(kt, "QUAD")) ct = PRIMAL_CT_QUAD;
            else if (teq(kt, "RQUAD")) ct = PRIMAL_CT_RQUAD;
            else if (teq(kt, "PEXP")) ct = PRIMAL_CT_PEXP;
            else if (teq(kt, "DEXP")) ct = PRIMAL_CT_DEXP;
            else if (teq(kt, "PPOW")) ct = PRIMAL_CT_PPOW;
            else if (teq(kt, "DPOW")) ct = PRIMAL_CT_RPOW;
            else { rc = PRIMAL_RES_ERR_FILE; goto fail; }   /* ZERO non rappresentato */
            int *mem = NULL, nmem = 0, memcap = 0;
            while (i < T->n && !is_mps_section(T->tok[i])) {
                int a = names_get(&cols, T->tok[i], 0);
                i++;
                if (a < 0) { free(mem); rc = PRIMAL_RES_ERR_FILE; goto fail; }
                if (nmem == memcap) {
                    memcap = memcap ? 2 * memcap : 8;
                    int *t2 = (int *)realloc(mem, (size_t)memcap * sizeof(int));
                    if (!t2) { free(mem); rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
                    mem = t2;
                }
                mem[nmem++] = a;
            }
            if (cone_acc_n == cone_acc_cap) {
                cone_acc_cap = cone_acc_cap ? 2 * cone_acc_cap : 8;
                int *t1 = (int *)realloc(ca_type, (size_t)cone_acc_cap * sizeof(int));
                int *t2 = (int *)realloc(ca_nmem, (size_t)cone_acc_cap * sizeof(int));
                double *t3 = (double *)realloc(ca_par, (size_t)cone_acc_cap * sizeof(double));
                int **t4 = (int **)realloc(ca_mem, (size_t)cone_acc_cap * sizeof(int *));
                if (!t1 || !t2 || !t3 || !t4) { free(mem); free(t1); free(t2); free(t3); free(t4); rc = PRIMAL_RES_ERR_ALLOC; goto fail; }
                ca_type = t1; ca_nmem = t2; ca_par = t3; ca_mem = t4;
            }
            ca_type[cone_acc_n] = ct; ca_par[cone_acc_n] = cpar; ca_nmem[cone_acc_n] = nmem; ca_mem[cone_acc_n] = mem;
            cone_acc_n++;
            continue;
        }
        if (teq(sec, "ENDATA")) break;
        i++;
    }

    /* ---- finalize into task ---- */
    rc = PRIMAL_appendvars(t, nvar);
    if (rc) goto fail;
    if (rows.n > 0) {
        rc = PRIMAL_appendcons(t, rows.n);
        if (rc) goto fail;
    }
    for (int j = 0; j < nvar && !rc; j++) {
        if (cj[j] != 0.0) rc = PRIMAL_putcj(t, j, cj[j]);
        if (!rc && csub[j].n > 0)
            rc = PRIMAL_putacol(t, j, csub[j].n, csub[j].a, cval[j].a);
        if (rc) break;
        PRIMALboundkeye b;
        if (bl[j] == -INF && bu[j] == INF) b = PRIMAL_BK_FR;
        else if (bl[j] == -INF) b = PRIMAL_BK_UP;
        else if (bu[j] == INF) b = PRIMAL_BK_LO;
        else if (fabs(bl[j] - bu[j]) <= 1e-12 * (1.0 + fabs(bl[j]))) b = PRIMAL_BK_FX;
        else b = PRIMAL_BK_RA;
        rc = PRIMAL_putvarbound(t, j, b, bl[j], bu[j]);
        if (rc) break;
        rc = PRIMAL_putvartype(t, j, vt[j]);
        if (rc) break;
    }
    /* MPS default: a row with no RHS entry has RHS 0. The RHS section lists
     * only the nonzero ones, so an E/L/G row it omits must still get its bound
     * (an E row with no RHS is "= 0", not free). */
    for (int r = 0; r < rows.n; r++) {
        if (r == objrow || rty[r] == 'N' || rhs_set[r]) continue;
        if (rty[r] == 'L') rup[r] = 0.0;
        else if (rty[r] == 'G') rlo[r] = 0.0;
        else { rlo[r] = 0.0; rup[r] = 0.0; }
    }
    for (int r = 0; r < rows.n && !rc; r++) {
        if (r == objrow || rty[r] == 'N') continue;
        PRIMALboundkeye b;
        if (rlo[r] == -INF && rup[r] == INF) b = PRIMAL_BK_FR;
        else if (rlo[r] == -INF) b = PRIMAL_BK_UP;
        else if (rup[r] == INF) b = PRIMAL_BK_LO;
        else if (fabs(rlo[r] - rup[r]) <= 1e-12 * (1.0 + fabs(rlo[r]))) b = PRIMAL_BK_FX;
        else b = PRIMAL_BK_RA;
        rc = PRIMAL_putconbound(t, r, b, rlo[r], rup[r]);
    }
    /* The names are part of what was read: ROWS entry r and COLUMNS entry j
     * are constraints r and variables j for every other put here, so the file's
     * label lands on the index it labels.  Duplicates cannot reach name_put
     * because pass 1 refused the file. */
    for (int j = 0; j < nvar && !rc; j++) rc = PRIMAL_putvarname(t, j, cols.nm[j]);
    for (int r = 0; r < rows.n && !rc; r++) rc = PRIMAL_putconname(t, r, rows.nm[r]);
    if (!rc && objnm) rc = PRIMAL_putobjname(t, objnm);
    if (!rc && cq_n > 0) {
        for (int r = 0; r < rows.n && !rc; r++) {
            int cnt = 0;
            for (int e = 0; e < cq_n; e++) if (cqk[e] == r) cnt++;
            if (cnt == 0) continue;
            int *si = (int *)malloc((size_t)cnt * sizeof(int));
            int *sj = (int *)malloc((size_t)cnt * sizeof(int));
            double *sv = (double *)malloc((size_t)cnt * sizeof(double));
            if (!si || !sj || !sv) { free(si); free(sj); free(sv); rc = PRIMAL_RES_ERR_ALLOC; break; }
            int z = 0;
            for (int e = 0; e < cq_n; e++) if (cqk[e] == r) { si[z] = cqi[e]; sj[z] = cqj[e]; sv[z] = cqv[e]; z++; }
            PRIMALboundkeye bk; double bl_, bu_;
            PRIMAL_getconbound(t, r, &bk, &bl_, &bu_);
            /* Only an UP (<=) row is representable here: the row's quadratic form
             * is +1/2 x'Qx.  A G/E row is refused rather than silently dropping
             * the quadratic part (declared deviation). */
            if (bk != PRIMAL_BK_UP) { free(si); free(sj); free(sv); rc = PRIMAL_RES_ERR_FILE; break; }
            rc = PRIMAL_putqconk(t, r, cnt, si, sj, sv);
            free(si); free(sj); free(sv);
        }
    }
    for (int c = 0; c < cone_acc_n && !rc; c++)
        rc = PRIMAL_appendcone(t, (PRIMALconetypee)ca_type[c], ca_par[c], ca_nmem[c], ca_mem[c]);
    if (!rc && q_n > 0) rc = PRIMAL_putqobj(t, q_n, qi, qj, qv);
    if (!rc) rc = PRIMAL_putcfix(t, cfix);
    if (!rc) rc = PRIMAL_putobjsense(t, sense);

fail:
    free(csub); free(cval); free(cj); free(rhs_set);
    free(bl); free(bu); free(vt); free(rlo); free(rup);
    for (int c = 0; c < cone_acc_n; c++) free(ca_mem[c]);
    free(ca_type); free(ca_nmem); free(ca_par); free(ca_mem);
    free(qi); free(qj); free(qv);
    free(cqi); free(cqj); free(cqk); free(cqv);
    names_free(&rows); names_free(&cols);
    free(rty);
    return rc;
}

/* ================= LP reader ================= */
typedef struct {
    IArr sub; DArr val;
    char rel;         /* 'L', 'G', 'E' */
    double rhs;
    const char *nm;   /* the label before ':', owned by the tokenizer */
} LpCon;

static PRIMALrescodee lp_read(PRIMALtask_t t, const Toks *T) {
    Names cols = {0, 0, 0};
    LpCon *cons = NULL;
    int ncon = 0, con_cap = 0;
    double *cj = NULL;
    int cjcap = 0;
    double *blo = NULL, *bup = NULL;
    int *lofin = NULL, *upfin = NULL;
    int bcap = 0;
    IArr gen = {0, 0, 0}, bins = {0, 0, 0};
    PRIMALobjsensee sense = PRIMAL_OPTIMIZE_MINIMIZE;
    double cfix = 0.0;
    const char *objnm = NULL;   /* the label before the objective's ':' */
    PRIMALrescodee rc = PRIMAL_RES_OK;

#define ENSURE_CJ(C) do { \
        if ((C) >= cjcap) { \
            int nc = cjcap ? cjcap * 2 : 8; \
            if (nc <= (C)) nc = (C) + 1; \
            double *ncj = (double *)realloc(cj, (size_t)nc * sizeof(double)); \
            if (!ncj) { rc = PRIMAL_RES_ERR_ALLOC; goto done; } \
            for (int q = cjcap; q < nc; q++) ncj[q] = 0.0; \
            cj = ncj; cjcap = nc; \
        } } while (0)
#define ENSURE_BND(C) do { \
        if ((C) >= bcap) { \
            int nb = bcap ? bcap * 2 : 8; \
            if (nb <= (C)) nb = (C) + 1; \
            double *tl = (double *)realloc(blo, (size_t)nb * sizeof(double)); \
            double *tu = (double *)realloc(bup, (size_t)nb * sizeof(double)); \
            int *fl = (int *)realloc(lofin, (size_t)nb * sizeof(int)); \
            int *fu = (int *)realloc(upfin, (size_t)nb * sizeof(int)); \
            if (!tl || !tu || !fl || !fu) { rc = PRIMAL_RES_ERR_ALLOC; goto done; } \
            blo = tl; bup = tu; lofin = fl; upfin = fu; \
            for (int q = bcap; q < nb; q++) { \
                blo[q] = 0.0; bup[q] = INF; lofin[q] = 0; upfin[q] = 0; \
            } \
            bcap = nb; \
        } } while (0)

    int i = 0;
    if (T->n > 0 && (ieq(T->tok[0], "minimize") || ieq(T->tok[0], "minimum") ||
                     ieq(T->tok[0], "min"))) { sense = PRIMAL_OPTIMIZE_MINIMIZE; i = 1; }
    else if (T->n > 0 && (ieq(T->tok[0], "maximize") || ieq(T->tok[0], "maximum") ||
                          ieq(T->tok[0], "max"))) { sense = PRIMAL_OPTIMIZE_MAXIMIZE; i = 1; }

    /* objective */
    if (i < T->n && tok_is(T, i + 1, ":")) { objnm = T->tok[i]; i += 2; }
    {
        double sign = 1.0;
        while (i < T->n && !is_lp_section(T->tok[i])) {
            const char *tk = T->tok[i];
            if (tk[0] == '[' || tk[0] == ']') { rc = PRIMAL_RES_ERR_FILE; goto done; }
            if (ieq(tk, "+")) { sign = 1.0; i++; continue; }
            if (ieq(tk, "-")) { sign = -1.0; i++; continue; }
            double v;
            if (parse_num(tk, &v)) {
                if (i + 1 < T->n && !is_lp_section(T->tok[i + 1]) &&
                    !tok_is(T, i + 1, ":")) {
                    int c = names_get(&cols, T->tok[i + 1], 1);
                    ENSURE_CJ(c);
                    cj[c] += sign * v;
                    i += 2;
                } else {
                    cfix += sign * v;
                    i++;
                }
                sign = 1.0;
                continue;
            }
            int c = names_get(&cols, tk, 1);
            ENSURE_CJ(c);
            cj[c] += sign;
            i++;
            sign = 1.0;
        }
    }

    /* sections */
    while (i < T->n) {
        const char *sec = T->tok[i];
        if (ieq(sec, "subject") || ieq(sec, "such") || ieq(sec, "st") || ieq(sec, "s.t.")) {
            if (i + 1 < T->n && (ieq(T->tok[i + 1], "to") || ieq(T->tok[i + 1], "that"))) i += 2;
            else i++;
            while (i < T->n && !is_lp_section(T->tok[i])) {
                if (ncon == con_cap) {
                    con_cap = con_cap ? con_cap * 2 : 8;
                    LpCon *nc = (LpCon *)realloc(cons, (size_t)con_cap * sizeof(LpCon));
                    if (!nc) { rc = PRIMAL_RES_ERR_ALLOC; goto done; }
                    cons = nc;
                }
                LpCon *C = &cons[ncon];
                memset(C, 0, sizeof *C);
                C->rel = 'L'; C->rhs = 0.0;
                if (tok_is(T, i + 1, ":")) {
                    C->nm = T->tok[i];
                    for (int k = 0; k < ncon; k++)
                        if (cons[k].nm && teq(cons[k].nm, C->nm)) {
                            rc = PRIMAL_RES_ERR_FILE; goto done;
                        }
                    i += 2;
                }
                double lsign = 1.0, lconst = 0.0;
                while (i < T->n && !is_rel(T->tok[i]) && !is_lp_section(T->tok[i]) &&
                       !tok_is(T, i, ":") && !tok_is(T, i + 1, ":")) {
                    if (T->tok[i][0] == '[' || T->tok[i][0] == ']') { rc = PRIMAL_RES_ERR_FILE; goto done; }
                    if (ieq(T->tok[i], "+")) { lsign = 1.0; i++; continue; }
                    if (ieq(T->tok[i], "-")) { lsign = -1.0; i++; continue; }
                    double v;
                    if (parse_num(T->tok[i], &v)) {
                        if (i + 2 < T->n && !is_rel(T->tok[i + 1]) &&
                            !is_lp_section(T->tok[i + 1]) && !tok_is(T, i + 1, ":") &&
                            !tok_is(T, i + 2, ":")) {
                            int c = names_get(&cols, T->tok[i + 1], 1);
                            ipush(&C->sub, c); dpush(&C->val, lsign * v);
                            i += 2;
                        } else { lconst += lsign * v; i++; }
                    } else {
                        int c = names_get(&cols, T->tok[i], 1);
                        ipush(&C->sub, c); dpush(&C->val, lsign);
                        i++;
                    }
                    lsign = 1.0;
                }
                if (i >= T->n || !is_rel(T->tok[i])) { rc = PRIMAL_RES_ERR_FILE; goto done; }
                char r2 = rel_char(T->tok[i++]);
                double rsign = 1.0, rconst = 0.0;
                while (i < T->n && !is_lp_section(T->tok[i]) && !is_rel(T->tok[i]) &&
                       !tok_is(T, i, ":") && !tok_is(T, i + 1, ":")) {
                    if (T->tok[i][0] == '[' || T->tok[i][0] == ']') { rc = PRIMAL_RES_ERR_FILE; goto done; }
                    if (ieq(T->tok[i], "+")) { rsign = 1.0; i++; continue; }
                    if (ieq(T->tok[i], "-")) { rsign = -1.0; i++; continue; }
                    double v;
                    if (parse_num(T->tok[i], &v)) {
                        if (i + 2 < T->n && !is_lp_section(T->tok[i + 1]) &&
                            !is_rel(T->tok[i + 1]) && !tok_is(T, i + 1, ":") &&
                            !tok_is(T, i + 2, ":")) {
                            /* coef var on the right: move left negated */
                            int c = names_get(&cols, T->tok[i + 1], 1);
                            ipush(&C->sub, c); dpush(&C->val, -rsign * v);
                            i += 2;
                        } else { rconst += rsign * v; i++; }
                    } else {
                        int c = names_get(&cols, T->tok[i], 1);
                        ipush(&C->sub, c); dpush(&C->val, -rsign);
                        i++;
                    }
                    rsign = 1.0;
                }
                C->rel = r2;
                C->rhs = rconst - lconst;
                ncon++;
            }
            continue;
        }
        if (ieq(sec, "bounds") || ieq(sec, "bound")) {
            i++;
            while (i < T->n && !is_lp_section(T->tok[i])) {
                double sgn = 1.0;
                int k = i;
                if (ieq(T->tok[k], "-")) { sgn = -1.0; k++; }
                else if (ieq(T->tok[k], "+")) k++;
                double v;
                if (parse_num(T->tok[k], &v)) {
                    v *= sgn;
                    i = k + 1;
                    if (i >= T->n || !is_rel(T->tok[i])) { rc = PRIMAL_RES_ERR_FILE; goto done; }
                    i++;   /* rel; direction comes from the numeric side */
                    if (i >= T->n) { rc = PRIMAL_RES_ERR_FILE; goto done; }
                    int c = names_get(&cols, T->tok[i++], 1);
                    ENSURE_BND(c);
                    lofin[c] = 1; blo[c] = v;
                    if (i < T->n && is_rel(T->tok[i])) {
                        i++;
                        double s2 = 1.0;
                        if (ieq(T->tok[i], "-")) { s2 = -1.0; i++; }
                        else if (ieq(T->tok[i], "+")) i++;
                        double v2;
                        if (i >= T->n || !parse_num(T->tok[i], &v2)) { rc = PRIMAL_RES_ERR_FILE; goto done; }
                        upfin[c] = 1; bup[c] = v2 * s2;
                        i++;
                    }
                } else {
                    int c = names_get(&cols, T->tok[i++], 1);
                    ENSURE_BND(c);
                    if (i < T->n && ieq(T->tok[i], "free")) {
                        i++;
                        lofin[c] = 1; blo[c] = -INF;
                        upfin[c] = 1; bup[c] = INF;
                    } else if (i < T->n && is_rel(T->tok[i])) {
                        char rc2 = rel_char(T->tok[i]); i++;
                        double s2 = 1.0;
                        if (ieq(T->tok[i], "-")) { s2 = -1.0; i++; }
                        else if (ieq(T->tok[i], "+")) i++;
                        double v2;
                        if (i >= T->n || !parse_num(T->tok[i], &v2)) { rc = PRIMAL_RES_ERR_FILE; goto done; }
                        v2 *= s2;
                        if (rc2 == 'L') { upfin[c] = 1; bup[c] = v2; }
                        else if (rc2 == 'G') { lofin[c] = 1; blo[c] = v2; }
                        else { lofin[c] = 1; blo[c] = v2; upfin[c] = 1; bup[c] = v2; }
                        i++;
                    } else { rc = PRIMAL_RES_ERR_FILE; goto done; }
                }
            }
            continue;
        }
        if (ieq(sec, "general") || ieq(sec, "generals") || ieq(sec, "gen") ||
            ieq(sec, "integer") || ieq(sec, "integers")) {
            i++;
            while (i < T->n && !is_lp_section(T->tok[i]))
                ipush(&gen, names_get(&cols, T->tok[i++], 1));
            continue;
        }
        if (ieq(sec, "binaries") || ieq(sec, "binary") || ieq(sec, "bin")) {
            i++;
            while (i < T->n && !is_lp_section(T->tok[i]))
                ipush(&bins, names_get(&cols, T->tok[i++], 1));
            continue;
        }
        if (ieq(sec, "end")) break;
        i++;
    }

    int nvar = cols.n;
    if (nvar == 0) { rc = PRIMAL_RES_ERR_FILE; goto done; }

    double *bl = (double *)malloc((size_t)nvar * sizeof(double));
    double *bu = (double *)malloc((size_t)nvar * sizeof(double));
    PRIMALboundkeye *bk = (PRIMALboundkeye *)malloc((size_t)nvar * sizeof(PRIMALboundkeye));
    PRIMALvariabletypee *vt = (PRIMALvariabletypee *)calloc((size_t)nvar, sizeof(PRIMALvariabletypee));
    if (!bl || !bu || !bk || !vt) {
        free(bl); free(bu); free(bk); free(vt);
        rc = PRIMAL_RES_ERR_ALLOC; goto done;
    }
    for (int j = 0; j < nvar; j++) {
        bl[j] = (j < bcap && lofin[j]) ? blo[j] : 0.0;
        bu[j] = (j < bcap && upfin[j]) ? bup[j] : INF;
    }
    for (int q = 0; q < gen.n; q++) vt[gen.a[q]] = PRIMAL_VAR_TYPE_INT;
    for (int q = 0; q < bins.n; q++) {
        vt[bins.a[q]] = PRIMAL_VAR_TYPE_INT_BIN;
        bl[bins.a[q]] = 0.0; bu[bins.a[q]] = 1.0;
    }
    for (int j = 0; j < nvar; j++)
        if (bu[j] < 0.0 && !(j < bcap && lofin[j])) bl[j] = -INF;

    rc = PRIMAL_appendvars(t, nvar);
    if (rc) { free(bl); free(bu); free(bk); free(vt); goto done; }
    if (ncon > 0) {
        rc = PRIMAL_appendcons(t, ncon);
        if (rc) { free(bl); free(bu); free(bk); free(vt); goto done; }
    }
    for (int j = 0; j < nvar; j++) {
        if (j < cjcap && cj[j] != 0.0) {
            rc = PRIMAL_putcj(t, j, cj[j]);
            if (rc) break;
        }
        PRIMALboundkeye b;
        if (bl[j] == -INF && bu[j] == INF) b = PRIMAL_BK_FR;
        else if (bl[j] == -INF) b = PRIMAL_BK_UP;
        else if (bu[j] == INF) b = PRIMAL_BK_LO;
        else if (fabs(bl[j] - bu[j]) <= 1e-12 * (1.0 + fabs(bl[j]))) b = PRIMAL_BK_FX;
        else b = PRIMAL_BK_RA;
        rc = PRIMAL_putvarbound(t, j, b, bl[j], bu[j]);
        if (rc) break;
        rc = PRIMAL_putvartype(t, j, vt[j]);
        if (rc) break;
    }
    free(bl); free(bu); free(bk); free(vt);
    for (int k = 0; k < ncon && !rc; k++) {
        LpCon *C = &cons[k];
        if (C->sub.n > 0) {
            rc = PRIMAL_putarow(t, k, C->sub.n, C->sub.a, C->val.a);
            if (rc) break;
        }
        PRIMALboundkeye b;
        if (C->rel == 'L') b = PRIMAL_BK_UP;
        else if (C->rel == 'G') b = PRIMAL_BK_LO;
        else b = PRIMAL_BK_FX;
        double lo = (C->rel == 'G' || C->rel == 'E') ? C->rhs : -INF;
        double up = (C->rel == 'L' || C->rel == 'E') ? C->rhs : INF;
        rc = PRIMAL_putconbound(t, k, b, lo, up);
    }
    /* The labels are part of what was read, at the index they label.  A
     * constraint written without one stays without one: there is no name to
     * put, not an empty one to put. */
    for (int j = 0; j < nvar && !rc; j++) rc = PRIMAL_putvarname(t, j, cols.nm[j]);
    for (int k = 0; k < ncon && !rc; k++)
        if (cons[k].nm) rc = PRIMAL_putconname(t, k, cons[k].nm);
    if (!rc && objnm) rc = PRIMAL_putobjname(t, objnm);
    if (!rc) rc = PRIMAL_putcfix(t, cfix);
    if (!rc) rc = PRIMAL_putobjsense(t, sense);

done:
    free(cj);
    free(blo); free(bup); free(lofin); free(upfin);
    free(gen.a); free(bins.a);
    if (cons)
        for (int k = 0; k < ncon; k++) {
            free(cons[k].sub.a); free(cons[k].val.a);
        }
    free(cons);
    names_free(&cols);
    return rc;
}

/* ======================= OPF format (MOSEK ref. 16.3) =======================
 * Tag-structured, row-oriented; an alternative to LP/MPS.  Supported here:
 *   [comment] (ignored) [hints]/[vendor]/[solutions] (ignored)
 *   [variables] (ordering), [objective min|max [name]] (linear + quadratic),
 *   [constraints]/[con] (linear), [bounds]/[b], [bounds]/[cone], [integer].
 * Names may be quoted.  Expressions accept + - * ( ) ^2 and implicit products.
 * The data callbacks READ_OPF / READ_OPF_SECTION / WRITE_OPF are emitted.
 * Declared deviations: quadratic CONSTRAINTS, [solutions], [vendor] parameters
 * and the `zero` cone are not read; OPF `dpow` maps to our RPOW; the writer
 * prints names positionally when they are empty and does not emit solutions. */

#define OPF_NM 256

typedef struct { char nm[OPF_NM]; int idx; } OpfName;

typedef struct {
    const char *p;
    PRIMALtask_t t;
    OpfName *var; int nvar, vcap;
    OpfName *con; int ncon, ccap;
    double *lo, *up;
    int rc;
} Opf;

static void opf_ws(Opf *o) {
    for (;;) {
        while (*o->p && isspace((unsigned char)*o->p)) o->p++;
        if (*o->p == '#') { while (*o->p && *o->p != '\n') o->p++; }
        else break;
    }
}
static int opf_name(Opf *o, char *out, int cap) {
    opf_ws(o);
    int n = 0;
    if (*o->p == '\'' || *o->p == '"') {
        char q = *o->p++;
        while (*o->p && *o->p != q) { char c = *o->p++; if (c == '\\' && *o->p) c = *o->p++; if (n < cap-1) out[n++] = c; }
        if (*o->p == q) o->p++;
        out[n] = 0; return n > 0;
    }
    if (!isalpha((unsigned char)*o->p)) return 0;
    while (*o->p && (isalnum((unsigned char)*o->p) || *o->p == '_' || *o->p == '{' || *o->p == '}')) {
        if (n < cap-1) out[n++] = *o->p; o->p++;
    }
    out[n] = 0; return 1;
}
static int opf_num(Opf *o, double *v) {
    opf_ws(o);
    const char *s = o->p; char *end = NULL;
    double d = strtod(s, &end);
    if (end == s) return 0;
    o->p = end; *v = d; return 1;
}
/* '[tag' / '[/tag'; returns 1 open, 0 close, -1 none. */
static int opf_tag(Opf *o, char *tag, int cap) {
    opf_ws(o);
    if (*o->p != '[') return -1;
    o->p++;
    int close = 0;
    if (*o->p == '/') { close = 1; o->p++; }
    int n = 0;
    while (*o->p && *o->p != ']' && !isspace((unsigned char)*o->p)) { if (n < cap-1) tag[n++] = *o->p; o->p++; }
    tag[n] = 0;
    if (close) { while (*o->p && *o->p != ']') o->p++; if (*o->p == ']') o->p++; }
    return close ? 0 : 1;
}
static void opf_to_bracket(Opf *o) {
    opf_ws(o);
    while (*o->p && *o->p != ']') {
        if (*o->p == '\'' || *o->p == '"') { char q = *o->p++; while (*o->p && *o->p != q) { if (*o->p == '\\' && o->p[1]) o->p++; o->p++; } if (*o->p == q) o->p++; }
        else o->p++;
    }
    if (*o->p == ']') o->p++;
}
static void opf_skip(Opf *o, const char *tag) {
    char close[OPF_NM]; snprintf(close, sizeof close, "[/%s]", tag);
    const char *q = strstr(o->p, close);
    if (q) o->p = q + strlen(close);
}

static int opf_find(OpfName *a, int n, const char *nm) { for (int i = 0; i < n; i++) if (strcmp(a[i].nm, nm) == 0) return a[i].idx; return -1; }
static int opf_var(Opf *o, const char *nm) {
    int e = opf_find(o->var, o->nvar, nm);
    if (e >= 0) return e;
    if (PRIMAL_appendvars(o->t, 1) != PRIMAL_RES_OK) { o->rc = PRIMAL_RES_ERR_ALLOC; return -1; }
    int idx = o->nvar;
    if (o->nvar == o->vcap) {
        o->vcap = o->vcap ? o->vcap * 2 : 8;
        OpfName *nv = (OpfName *)realloc(o->var, (size_t)o->vcap * sizeof(OpfName));
        double *nl = (double *)realloc(o->lo, (size_t)o->vcap * sizeof(double));
        double *nu = (double *)realloc(o->up, (size_t)o->vcap * sizeof(double));
        if (!nv || !nl || !nu) { free(nv); free(nl); free(nu); o->rc = PRIMAL_RES_ERR_ALLOC; return -1; }
        o->var = nv; o->lo = nl; o->up = nu;
    }
    strncpy(o->var[idx].nm, nm, OPF_NM-1); o->var[idx].nm[OPF_NM-1] = 0; o->var[idx].idx = idx;
    o->lo[idx] = -INFINITY; o->up[idx] = INFINITY;
    o->nvar++;
    PRIMAL_putvarname(o->t, idx, nm);
    return idx;
}

/* expression: polynomial of degree <= 2 over the variables */
typedef struct {
    Opf *o;
    double c;
    double *lin; int lcap;
    int nq, qcap; int *qi, *qj; double *qv;
    int err;
} OpfE;

static void oe_free(OpfE *e) { free(e->lin); free(e->qi); free(e->qj); free(e->qv); e->lin = NULL; e->qi = e->qj = NULL; e->qv = NULL; e->lcap = 0; e->nq = e->qcap = 0; }
static void oe_lin(OpfE *e) {
    if (e->lcap < e->o->nvar) {
        int nc = e->o->nvar + 8;
        double *nl = (double *)realloc(e->lin, (size_t)nc * sizeof(double));
        if (!nl) { e->err = 1; return; }
        for (int i = e->lcap; i < nc; i++) nl[i] = 0.0;
        e->lin = nl; e->lcap = nc;
    }
}
static void oe_addlin(OpfE *e, int idx, double v) { oe_lin(e); if (!e->err) e->lin[idx] += v; }
static void oe_addq(OpfE *e, int i, int j, double v) {
    if (i < j) { int t = i; i = j; j = t; }
    if (e->nq == e->qcap) {
        e->qcap = e->qcap ? e->qcap * 2 : 8;
        int *qi = (int *)realloc(e->qi, (size_t)e->qcap * sizeof(int));
        int *qj = (int *)realloc(e->qj, (size_t)e->qcap * sizeof(int));
        double *qv = (double *)realloc(e->qv, (size_t)e->qcap * sizeof(double));
        if (!qi || !qj || !qv) { free(qi); free(qj); free(qv); e->err = 1; return; }
        e->qi = qi; e->qj = qj; e->qv = qv;
    }
    e->qi[e->nq] = i; e->qj[e->nq] = j; e->qv[e->nq] = v; e->nq++;
}
static int oe_deg(OpfE *e) { oe_lin(e); if (e->nq > 0) return 2; for (int i = 0; i < e->o->nvar; i++) if (e->lin[i] != 0.0) return 1; return 0; }
static void oe_scale(OpfE *e, double s) { e->c *= s; oe_lin(e); for (int i = 0; i < e->o->nvar; i++) e->lin[i] *= s; for (int k = 0; k < e->nq; k++) e->qv[k] *= s; }
static void oe_addto(OpfE *e, const OpfE *g) { oe_lin((OpfE *)g); e->c += g->c; oe_lin(e); for (int i = 0; i < e->o->nvar; i++) e->lin[i] += g->lin[i]; for (int k = 0; k < g->nq; k++) oe_addq(e, g->qi[k], g->qj[k], g->qv[k]); }
static void oe_mul(OpfE *e, const OpfE *a, const OpfE *b) {   /* e = a*b */
    int da = oe_deg((OpfE *)a), db = oe_deg((OpfE *)b);
    memset(e, 0, sizeof *e); e->o = a->o;
    if ((da == 2 && db != 0) || (db == 2 && da != 0)) { e->err = 1; return; }
    if (da == 2 || db == 2) {
        const OpfE *q = (da == 2) ? a : b; double s = (da == 2) ? b->c : a->c;
        e->c = q->c * s;
        oe_lin(e);
        for (int i = 0; i < e->o->nvar; i++) e->lin[i] = q->lin[i] * s;
        for (int k = 0; k < q->nq; k++) oe_addq(e, q->qi[k], q->qj[k], q->qv[k] * s);
        return;
    }
    e->c = a->c * b->c;
    oe_lin(e);
    for (int i = 0; i < e->o->nvar; i++) e->lin[i] = a->c * b->lin[i] + b->c * a->lin[i];
    for (int i = 0; i < e->o->nvar; i++) if (a->lin[i] != 0.0)
        for (int j = 0; j < e->o->nvar; j++) if (b->lin[j] != 0.0) oe_addq(e, i, j, a->lin[i] * b->lin[j]);
}

static void opf_factor(Opf *o, OpfE *e);
static void opf_prod(Opf *o, OpfE *e);
static void opf_expr(Opf *o, OpfE *e);

static void opf_factor(Opf *o, OpfE *e) {
    memset(e, 0, sizeof *e); e->o = o;
    opf_ws(o);
    char c = *o->p;
    if (c == '(') {
        o->p++; opf_expr(o, e); opf_ws(o);
        if (*o->p == ')') o->p++; else e->err = 1;
        return;
    }
    if (c == '\'' || c == '"') {
        char nm[OPF_NM]; if (!opf_name(o, nm, sizeof nm)) { e->err = 1; return; }
        int idx = opf_var(o, nm); if (idx < 0) { e->err = 1; return; }
        oe_addlin(e, idx, 1.0); return;
    }
    if (isalpha((unsigned char)c)) {
        char nm[OPF_NM]; if (!opf_name(o, nm, sizeof nm)) { e->err = 1; return; }
        opf_ws(o); int ex = 1;
        if (*o->p == '^') { o->p++; opf_ws(o); double d; if (!opf_num(o, &d)) { e->err = 1; return; } ex = (int)(d + 0.5); }
        int idx = opf_var(o, nm); if (idx < 0) { e->err = 1; return; }
        if (ex == 1) oe_addlin(e, idx, 1.0);
        else if (ex == 2) oe_addq(e, idx, idx, 1.0);
        else e->err = 1;
        return;
    }
    if (isdigit((unsigned char)c) || c == '.') { double d; if (!opf_num(o, &d)) { e->err = 1; return; } e->c = d; return; }
    e->err = 1;
}
static void opf_prod(Opf *o, OpfE *e) {
    opf_factor(o, e);
    for (;;) {
        opf_ws(o); char c = *o->p;
        int expl = (c == '*');
        if (expl) { o->p++; c = '\0'; opf_ws(o); c = *o->p; }
        if (expl || c == '(' || c == '\'' || c == '"' || isalpha((unsigned char)c) || isdigit((unsigned char)c) || c == '.') {
            OpfE g; opf_factor(o, &g);
            OpfE r; oe_mul(&r, e, &g);
            oe_free(e); oe_free(&g); *e = r;
            if (e->err) return;
        } else break;
    }
}
static void opf_expr(Opf *o, OpfE *e) {
    opf_ws(o);
    int neg = 0;
    if (*o->p == '+') o->p++;
    else if (*o->p == '-') { o->p++; neg = 1; }
    opf_prod(o, e);
    if (neg) oe_scale(e, -1.0);
    for (;;) {
        opf_ws(o); char c = *o->p;
        if (c == '+' || c == '-') { o->p++; OpfE g; opf_prod(o, &g); if (c == '-') oe_scale(&g, -1.0); oe_addto(e, &g); oe_free(&g); if (e->err) return; }
        else break;
    }
}

/* relation: 0 '<=', 1 '>=', 2 '=', 3 '<', 4 '>' */
static int opf_rel(Opf *o, int *kind) {
    opf_ws(o); char c = *o->p;
    if (c == '<') { if (o->p[1] == '=') { o->p += 2; *kind = 0; } else { o->p++; *kind = 3; } return 1; }
    if (c == '>') { if (o->p[1] == '=') { o->p += 2; *kind = 1; } else { o->p++; *kind = 4; } return 1; }
    if (c == '=') { o->p += (o->p[1] == '=') ? 2 : 1; *kind = 2; return 1; }
    return 0;
}
/* comma-separated variable names; '*' = all current variables */
static int opf_varlist(Opf *o, int *vars, int *np, int cap) {
    opf_ws(o);
    if (*o->p == '*') { o->p++; for (int j = 0; j < o->nvar; j++) if (*np < cap) vars[(*np)++] = j; return 0; }
    for (;;) {
        opf_ws(o); char nm[OPF_NM];
        if (!opf_name(o, nm, sizeof nm)) break;
        int idx = opf_var(o, nm); if (idx < 0) return -1;
        if (*np < cap) vars[(*np)++] = idx;
        opf_ws(o);
        if (*o->p == ',') { o->p++; continue; }
        break;
    }
    return *np > 0 ? 0 : -1;
}

static void opf_con(Opf *o, const char *nm) {
    OpfE term[3]; int nt = 0, rel[2], nr = 0;
    opf_expr(o, &term[nt++]);
    int k;
    while (nt < 3 && opf_rel(o, &k)) { rel[nr++] = k; opf_expr(o, &term[nt++]); }
    if (term[0].err) { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
    int ei = -1;
    for (int i = 0; i < nt; i++) if (oe_deg(&term[i]) > 0) { if (ei >= 0) { o->rc = PRIMAL_RES_ERR_FILE; goto done; } ei = i; }
    if (ei < 0) { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
    {
        double lo = -INFINITY, up = INFINITY; int fx = 0;
        if (nt == 1) { /* empty body */ }
        else if (ei == 0) {           /* expr rel c */
            double c = term[1].c;
            if (rel[0] == 0) up = c; else if (rel[0] == 1) lo = c; else if (rel[0] == 2) { lo = up = c; fx = 1; } else { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
        } else if (ei == 1 && nt == 2) {   /* c rel expr */
            double c = term[0].c;
            if (rel[0] == 0) lo = c; else if (rel[0] == 1) up = c; else if (rel[0] == 2) { lo = up = c; fx = 1; } else { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
        } else if (ei == 1 && nt == 3) {   /* a rel0 expr rel1 b */
            double a = term[0].c, b = term[2].c;
            if (rel[0] == 0) lo = a; else if (rel[0] == 1) up = a; else { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
            if (rel[1] == 0) up = b; else if (rel[1] == 1) lo = b; else { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
        } else { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
        int ki = o->ncon++;
        if (PRIMAL_appendcons(o->t, 1) != PRIMAL_RES_OK) { o->rc = PRIMAL_RES_ERR_ALLOC; goto done; }
        if (nm && *nm) PRIMAL_putconname(o->t, ki, nm);
        {
            PRIMALboundkeye bk = fx ? PRIMAL_BK_FX :
                (isinf(lo) && isinf(up)) ? PRIMAL_BK_FR :
                isinf(lo) ? PRIMAL_BK_UP : isinf(up) ? PRIMAL_BK_LO : PRIMAL_BK_RA;
            /* Quadratic constraints: a convex row q(x) <= b is stored with the UP
             * convention (qv = a off-diagonal, 2a diagonal, so quad_row_value's
             * +1/2 x'Qx equals q); a row q(x) >= b is written as -q(x) <= -b
             * (UP).  An equality or ranged quadratic is not representable here
             * (declared) and is refused. */
            int nq = term[ei].nq;
            int neg = 0;
            double bnd = up;
            if (nq > 0) {
                if (bk == PRIMAL_BK_FX || bk == PRIMAL_BK_RA || bk == PRIMAL_BK_FR) { o->rc = PRIMAL_RES_ERR_FILE; goto done; }
                neg = (bk == PRIMAL_BK_LO);
                bnd = neg ? -lo : up;
            }
            {
                int nz = 0;
                int *sub = (int *)malloc((size_t)(o->nvar > 0 ? o->nvar : 1) * sizeof(int));
                double *val = (double *)malloc((size_t)(o->nvar > 0 ? o->nvar : 1) * sizeof(double));
                if (!sub || !val) { free(sub); free(val); o->rc = PRIMAL_RES_ERR_ALLOC; goto done; }
                for (int j = 0; j < o->nvar; j++) if (term[ei].lin[j] != 0.0) { sub[nz] = j; val[nz] = neg ? -term[ei].lin[j] : term[ei].lin[j]; nz++; }
                if (nz > 0) PRIMAL_putarow(o->t, ki, nz, sub, val);
                free(sub); free(val);
            }
            if (nq > 0) {
                int *qi = (int *)malloc((size_t)nq * sizeof(int));
                int *qj = (int *)malloc((size_t)nq * sizeof(int));
                double *qv = (double *)malloc((size_t)nq * sizeof(double));
                if (!qi || !qj || !qv) { free(qi); free(qj); free(qv); o->rc = PRIMAL_RES_ERR_ALLOC; goto done; }
                for (int q = 0; q < nq; q++) {
                    int ii = term[ei].qi[q], jj = term[ei].qj[q];
                    double aa = neg ? -term[ei].qv[q] : term[ei].qv[q];
                    qi[q] = ii; qj[q] = jj;
                    qv[q] = (ii == jj) ? 2.0 * aa : aa;
                }
                PRIMALrescodee qrc = PRIMAL_putqconk(o->t, ki, nq, qi, qj, qv);
                free(qi); free(qj); free(qv);
                if (qrc != PRIMAL_RES_OK) { o->rc = qrc; goto done; }
                PRIMAL_putconbound(o->t, ki, PRIMAL_BK_UP, 0.0, bnd);
            } else {
                PRIMAL_putconbound(o->t, ki, bk, lo, up);
            }
        }
    }
done:
    for (int i = 0; i < nt; i++) oe_free(&term[i]);
}

static void opf_bound(Opf *o) {
    int vars[1024]; int nv = 0;
    opf_ws(o);
    double first;
    if (opf_num(o, &first)) {
        int r1; if (!opf_rel(o, &r1)) { o->rc = PRIMAL_RES_ERR_FILE; return; }
        if (opf_varlist(o, vars, &nv, 1024) < 0) { o->rc = PRIMAL_RES_ERR_FILE; return; }
        opf_ws(o); int r2 = -1; double second = 0;
        if (*o->p != '[' && opf_rel(o, &r2) && !opf_num(o, &second)) { o->rc = PRIMAL_RES_ERR_FILE; return; }
        for (int i = 0; i < nv; i++) {
            int j = vars[i];
            if (r1 == 0) o->lo[j] = first; else if (r1 == 1) o->up[j] = first; else { o->lo[j] = first; o->up[j] = first; }
            if (r2 >= 0) { if (r2 == 0) o->up[j] = second; else if (r2 == 1) o->lo[j] = second; else { o->lo[j] = second; o->up[j] = second; } }
        }
    } else {
        if (opf_varlist(o, vars, &nv, 1024) < 0) { o->rc = PRIMAL_RES_ERR_FILE; return; }
        opf_ws(o); int r = -1;
        if (opf_rel(o, &r)) {
            double v; if (!opf_num(o, &v)) { o->rc = PRIMAL_RES_ERR_FILE; return; }
            for (int i = 0; i < nv; i++) { int j = vars[i];
                if (r == 0) o->up[j] = v; else if (r == 1) o->lo[j] = v; else { o->lo[j] = v; o->up[j] = v; } }
        } else {
            char w[OPF_NM];
            if (!opf_name(o, w, sizeof w) || strcmp(w, "free") != 0) { o->rc = PRIMAL_RES_ERR_FILE; return; }
            for (int i = 0; i < nv; i++) { int j = vars[i]; o->lo[j] = -INFINITY; o->up[j] = INFINITY; }
        }
    }
}

static void opf_cone(Opf *o) {
    char ctype[OPF_NM]; if (!opf_name(o, ctype, sizeof ctype)) { o->rc = PRIMAL_RES_ERR_FILE; return; }
    double alpha = 0.5; char nm[OPF_NM]; nm[0] = 0;
    opf_ws(o);
    if (*o->p != ']') {
        char t[OPF_NM];
        if (opf_num(o, &alpha)) { opf_ws(o); if (*o->p != ']') opf_name(o, nm, sizeof nm); }
        else if (opf_name(o, t, sizeof t)) { alpha = strtod(t, NULL); opf_ws(o); if (*o->p != ']') opf_name(o, nm, sizeof nm); }
    }
    (void)nm;
    opf_to_bracket(o);
    int mem[4096]; int nmem = 0;
    for (;;) {
        opf_ws(o); if (*o->p == '[') break;
        char v[OPF_NM]; if (!opf_name(o, v, sizeof v)) break;
        int idx = opf_var(o, v); if (idx < 0) { o->rc = PRIMAL_RES_ERR_FILE; break; }
        if (nmem < 4096) mem[nmem++] = idx;
        opf_ws(o); if (*o->p == ',') o->p++;
    }
    opf_skip(o, "cone");
    PRIMALconetypee ct; double par = 0.0; int known = 1;
    if (strcmp(ctype, "quad") == 0) ct = PRIMAL_CT_QUAD;
    else if (strcmp(ctype, "rquad") == 0) ct = PRIMAL_CT_RQUAD;
    else if (strcmp(ctype, "pexp") == 0) ct = PRIMAL_CT_PEXP;
    else if (strcmp(ctype, "dexp") == 0) ct = PRIMAL_CT_DEXP;
    else if (strcmp(ctype, "ppow") == 0) { ct = PRIMAL_CT_PPOW; par = alpha; }
    else if (strcmp(ctype, "dpow") == 0) { ct = PRIMAL_CT_RPOW; par = alpha; }
    else known = 0;
    if (!known) { o->rc = PRIMAL_RES_ERR_FILE; return; }
    if (PRIMAL_appendcone(o->t, ct, par, nmem, mem) != PRIMAL_RES_OK) o->rc = PRIMAL_RES_ERR_FILE;
}

static void opf_variables(Opf *o) {
    opf_to_bracket(o);
    for (;;) { opf_ws(o); if (*o->p == '[') break; char nm[OPF_NM]; if (!opf_name(o, nm, sizeof nm)) break; opf_var(o, nm); }
    opf_skip(o, "variables");
}
static void opf_objective(Opf *o) {
    char w[OPF_NM], nm[OPF_NM]; nm[0] = 0;
    if (!opf_name(o, w, sizeof w)) { o->rc = PRIMAL_RES_ERR_FILE; return; }
    opf_ws(o); if (*o->p != ']') opf_name(o, nm, sizeof nm);
    opf_to_bracket(o);
    PRIMAL_putobjsense(o->t, (w[0] == 'm' && w[1] == 'a') ? PRIMAL_OPTIMIZE_MAXIMIZE : PRIMAL_OPTIMIZE_MINIMIZE);
    if (nm[0]) PRIMAL_putobjname(o->t, nm);
    OpfE e; opf_expr(o, &e);
    if (e.err) { o->rc = PRIMAL_RES_ERR_FILE; }
    else {
        PRIMAL_putcfix(o->t, e.c);
        for (int j = 0; j < o->nvar; j++) if (e.lin && e.lin[j] != 0.0) PRIMAL_putcj(o->t, j, e.lin[j]);
        for (int q = 0; q < e.nq; q++) { int i = e.qi[q], j = e.qj[q]; PRIMAL_putqobjij(o->t, i, j, (i == j) ? 2.0 * e.qv[q] : e.qv[q]); }
    }
    oe_free(&e);
    opf_skip(o, "objective");
}
static void opf_constraints(Opf *o) {
    opf_to_bracket(o);
    char tag[OPF_NM];
    for (;;) {
        opf_ws(o); if (*o->p != '[') break;
        int k = opf_tag(o, tag, sizeof tag); if (k != 1) break;
        if (strcmp(tag, "con") != 0) { opf_skip(o, tag); continue; }
        char nm[OPF_NM]; nm[0] = 0;
        opf_ws(o); if (*o->p != ']') opf_name(o, nm, sizeof nm);
        opf_to_bracket(o);
        opf_con(o, nm);
        opf_skip(o, "con");
    }
}
static void opf_bounds(Opf *o) {
    opf_to_bracket(o);
    char tag[OPF_NM];
    for (;;) {
        opf_ws(o); if (*o->p != '[') break;
        int k = opf_tag(o, tag, sizeof tag); if (k != 1) break;
        if (strcmp(tag, "b") == 0) { opf_to_bracket(o); opf_bound(o); opf_skip(o, "b"); }
        else if (strcmp(tag, "cone") == 0) opf_cone(o);
        else opf_skip(o, tag);
    }
}
static void opf_integer(Opf *o) {
    opf_to_bracket(o);
    for (;;) { opf_ws(o); if (*o->p == '[') break; char nm[OPF_NM]; if (!opf_name(o, nm, sizeof nm)) break; int idx = opf_var(o, nm); if (idx >= 0) PRIMAL_putvartype(o->t, idx, PRIMAL_VAR_TYPE_INT); }
    opf_skip(o, "integer");
}

static int opf_parse(Opf *o) {
    char tag[OPF_NM];
    for (;;) {
        int k = opf_tag(o, tag, sizeof tag);
        if (k < 0) break;
        if (k == 0) continue;
        primal_cb_notify(o->t, PRIMAL_CALLBACK_READ_OPF_SECTION);
        if (strcmp(tag, "comment") == 0) opf_skip(o, "comment");
        else if (strcmp(tag, "hints") == 0) opf_skip(o, "hints");
        else if (strcmp(tag, "vendor") == 0) opf_skip(o, "vendor");
        else if (strcmp(tag, "solutions") == 0) opf_skip(o, "solutions");
        else if (strcmp(tag, "variables") == 0) opf_variables(o);
        else if (strcmp(tag, "objective") == 0) opf_objective(o);
        else if (strcmp(tag, "constraints") == 0) opf_constraints(o);
        else if (strcmp(tag, "bounds") == 0) opf_bounds(o);
        else if (strcmp(tag, "integer") == 0) opf_integer(o);
        else opf_skip(o, tag);
        if (o->rc != PRIMAL_RES_OK) break;
    }
    return o->rc;
}

PRIMALrescodee opf_read(PRIMALtask_t t, const char *data) {
    if (!t || !data) return PRIMAL_RES_ERR_NULL;
    {
        int nv = 0, nc = 0;
        PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
        if (nv != 0 || nc != 0) return PRIMAL_RES_ERR_ARG;
    }
    Opf o; memset(&o, 0, sizeof o); o.p = data; o.t = t;
    primal_cb_notify(t, PRIMAL_CALLBACK_READ_OPF);
    opf_parse(&o);
    if (o.rc == PRIMAL_RES_OK) {
        for (int j = 0; j < o.nvar; j++) {
            double lo = o.lo[j], up = o.up[j];
            PRIMALboundkeye bk = (isinf(lo) && isinf(up) && lo < 0 && up > 0) ? PRIMAL_BK_FR :
                (isinf(lo) && lo < 0) ? PRIMAL_BK_UP :
                (isinf(up) && up > 0) ? PRIMAL_BK_LO : PRIMAL_BK_RA;
            if (lo == up) bk = PRIMAL_BK_FX;
            PRIMAL_putvarbound(t, j, bk, lo, up);
        }
    }
    free(o.var); free(o.con); free(o.lo); free(o.up);
    return o.rc;
}

/* -------- writer -------- */
static void opf_pname(FILE *f, const char *nm) {
    if (!nm || !*nm) return;
    int simple = isalpha((unsigned char)nm[0]) ? 1 : 0;
    for (const char *p = nm; *p && simple; p++)
        if (!(isalnum((unsigned char)*p) || *p == '_' || *p == '{' || *p == '}')) simple = 0;
    if (simple) fputs(nm, f);
    else fprintf(f, "'%s'", nm);
}
static void opf_vname(FILE *f, PRIMALtask_t t, int j) {
    const char *nm = NULL;
    PRIMAL_getvarnameidx(t, j, &nm);
    if (nm && *nm) opf_pname(f, nm); else fprintf(f, "x%d", j);
}
static void opf_cterm(FILE *f, int *first, double coef, PRIMALtask_t t, int j, int power) {
    if (coef == 0.0) return;
    fputs(*first ? (coef > 0 ? "" : "- ") : (coef > 0 ? " + " : " - "), f);
    double a = fabs(coef);
    if (a != 1.0) fprintf(f, "%.12g ", a);
    opf_vname(f, t, j);
    if (power == 2) fputs(" ^ 2", f);
    *first = 0;
}
PRIMALrescodee opf_write(PRIMALtask_t t, FILE *f) {
    if (!t || !f) return PRIMAL_RES_ERR_NULL;
    primal_cb_notify(t, PRIMAL_CALLBACK_WRITE_OPF);
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
    PRIMALobjsensee sense = PRIMAL_OPTIMIZE_MINIMIZE; PRIMAL_getobjsense(t, &sense);
    double cfix = 0.0; PRIMAL_getcfix(t, &cfix);
    const char *obj = NULL; PRIMAL_getobjname(t, &obj);
    fprintf(f, "[comment]\nWritten by PrimalSolver.\n[/comment]\n");
    fprintf(f, "[variables]\n");
    for (int j = 0; j < nv; j++) { if (j) fputc(' ', f); opf_vname(f, t, j); }
    fprintf(f, "\n[/variables]\n");
    fprintf(f, "[objective %s", sense == PRIMAL_OPTIMIZE_MAXIMIZE ? "maximize" : "minimize");
    if (obj && *obj) fprintf(f, " '%s'", obj);
    fprintf(f, "]\n");
    {
        int first = 1;
        for (int j = 0; j < nv; j++) { double cj = 0; PRIMAL_getcj(t, j, &cj); opf_cterm(f, &first, cj, t, j, 1); }
        for (int i = 0; i < nv; i++) for (int j = 0; j <= i; j++) {
            double q = 0; PRIMAL_getqobjij(t, i, j, &q); if (q == 0.0) continue;
            if (i == j) opf_cterm(f, &first, q / 2.0, t, i, 2);
            else { fputs(first ? (q > 0 ? "" : "- ") : (q > 0 ? " + " : " - "), f);
                   double a = fabs(q); if (a != 1.0) fprintf(f, "%.12g ", a);
                   opf_vname(f, t, i); fputc(' ', f); opf_vname(f, t, j); first = 0; }
        }
        if (first || cfix != 0.0) fprintf(f, "%s%.12g", first ? "" : " + ", cfix);
    }
    fprintf(f, "\n[/objective]\n");
    fprintf(f, "[constraints]\n");
    for (int i = 0; i < nc; i++) {
        const char *nm = NULL; PRIMAL_getconnameidx(t, i, &nm);
        PRIMALboundkeye bk; double bl = 0, bu = 0; PRIMAL_getconbound(t, i, &bk, &bl, &bu);
        fprintf(f, "[con"); if (nm && *nm) fprintf(f, " '%s'", nm); fprintf(f, "] ");
        if (bk == PRIMAL_BK_RA) fprintf(f, "%.12g <= ", bl);
        int nz = 0;
        int *sub = (int *)malloc((size_t)(nv > 0 ? nv : 1) * sizeof(int));
        double *val = (double *)malloc((size_t)(nv > 0 ? nv : 1) * sizeof(double));
        if (sub && val) PRIMAL_getarow(t, i, sub, val, nv, &nz);
        int first = 1;
        for (int e = 0; e < nz; e++) opf_cterm(f, &first, val[e], t, sub[e], 1);
        /* quadratic terms: 1/2 x'Qx on an UP row, -1/2 x'Qx otherwise (the
         * reader's convention), so qcon(i,j) prints as qv on the diagonal and
         * qv/2 (signed) off it. */
        {
            int nq = 0;
            PRIMAL_getnumqconknz(t, i, &nq);
            if (nq > 0) {
                int *qi = (int *)malloc((size_t)nq * sizeof(int));
                int *qj = (int *)malloc((size_t)nq * sizeof(int));
                double *qv = (double *)malloc((size_t)nq * sizeof(double));
                int nr = 0;
                if (qi && qj && qv) PRIMAL_getqconk(t, i, qi, qj, qv, nq, &nr);
                int ups = (bk == PRIMAL_BK_UP);
                for (int e = 0; e < nr; e++) {
                    int ii = qi[e], jj = qj[e];
                    double a = ups ? ((ii == jj) ? 0.5 * qv[e] : qv[e])
                                   : ((ii == jj) ? -0.5 * qv[e] : -qv[e]);
                    if (a == 0.0) continue;
                    fputs(first ? (a > 0 ? "" : "- ") : (a > 0 ? " + " : " - "), f);
                    double m = fabs(a);
                    if (m != 1.0) fprintf(f, "%.12g ", m);
                    opf_vname(f, t, ii);
                    if (ii != jj) { fputc(' ', f); opf_vname(f, t, jj); }
                    first = 0;
                }
                free(qi); free(qj); free(qv);
            }
        }
        if (first) fputs("0", f);
        free(sub); free(val);
        if (bk == PRIMAL_BK_UP) fprintf(f, " <= %.12g", bu);
        else if (bk == PRIMAL_BK_LO) fprintf(f, " >= %.12g", bl);
        else if (bk == PRIMAL_BK_FX) fprintf(f, " = %.12g", bl);
        else if (bk == PRIMAL_BK_RA) fprintf(f, " <= %.12g", bu);
        fprintf(f, " [/con]\n");
    }
    fprintf(f, "[/constraints]\n");
    fprintf(f, "[bounds]\n");
    for (int j = 0; j < nv; j++) {
        PRIMALboundkeye bk; double bl = 0, bu = 0; PRIMAL_getvarbound(t, j, &bk, &bl, &bu);
        fprintf(f, "[b] ");
        if (bk == PRIMAL_BK_FR) { opf_vname(f, t, j); fputs(" free", f); }
        else if (bk == PRIMAL_BK_LO) { fprintf(f, "%.12g <= ", bl); opf_vname(f, t, j); }
        else if (bk == PRIMAL_BK_UP) { opf_vname(f, t, j); fprintf(f, " <= %.12g", bu); }
        else if (bk == PRIMAL_BK_FX) { opf_vname(f, t, j); fprintf(f, " = %.12g", bl); }
        else { fprintf(f, "%.12g <= ", bl); opf_vname(f, t, j); fprintf(f, " <= %.12g", bu); }
        fprintf(f, " [/b]\n");
    }
    {
        int ncon = 0; PRIMAL_getnumcone(t, &ncon);
        for (int k = 0; k < ncon; k++) {
            PRIMALconetypee ct; double par = 0; int nmem = 0;
            PRIMAL_getconeinfo(t, k, &ct, &par, &nmem);
            int *mem = (int *)malloc((size_t)(nmem > 0 ? nmem : 1) * sizeof(int));
            if (mem) PRIMAL_getcone(t, k, &ct, &nmem, mem);
            const char *tn = ct == PRIMAL_CT_QUAD ? "quad" : ct == PRIMAL_CT_RQUAD ? "rquad" :
                             ct == PRIMAL_CT_PEXP ? "pexp" : ct == PRIMAL_CT_DEXP ? "dexp" :
                             ct == PRIMAL_CT_PPOW ? "ppow" : ct == PRIMAL_CT_RPOW ? "dpow" : "quad";
            fprintf(f, "[cone %s", tn);
            if (ct == PRIMAL_CT_PPOW || ct == PRIMAL_CT_RPOW) fprintf(f, " %.17g", par);
            fputs("] ", f);
            for (int e = 0; e < nmem; e++) { if (e) fputs(", ", f); opf_vname(f, t, mem[e]); }
            fprintf(f, " [/cone]\n");
            free(mem);
        }
    }
    fprintf(f, "[/bounds]\n");
    {
        int any = 0;
        for (int j = 0; j < nv; j++) { PRIMALvariabletypee vt; PRIMAL_getvartype(t, j, &vt); if (vt == PRIMAL_VAR_TYPE_INT) any = 1; }
        if (any) {
            fprintf(f, "[integer]\n");
            for (int j = 0; j < nv; j++) { PRIMALvariabletypee vt; PRIMAL_getvartype(t, j, &vt); if (vt == PRIMAL_VAR_TYPE_INT) { opf_vname(f, t, j); fputc(' ', f); } }
            fprintf(f, "\n[/integer]\n");
        }
    }
    return PRIMAL_RES_OK;
}

/* ================= dispatch ================= */
/**
 * Writes a task to a file in the format determined by the extension.
 *
 * @param t        [in] Task handle.
 * @param filename [in] Output file path. Extension determines format:
 *                   - .cbf  -> CBF format
 *                   - .lp   -> LP format
 *                   - other -> MPS format (default)
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or filename is NULL,
 *         PRIMAL_RES_ERR_FILE if file cannot be opened.
 *
 * @note Uses `has_ext` to detect format by extension (case-insensitive).
 *       The file is opened in "w" mode (truncated).
 *
 * @example
 * PRIMALrescodee rc = primalio_write(task, "model.mps");
 * // Writes model.mps in MPS format
 */
PRIMALrescodee primalio_write(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    PRIMALrescodee rc = has_ext(filename, ".cbf") ? cbf_write(t, f) :
                     has_ext(filename, ".lp") ? lp_write(t, f) :
                     has_ext(filename, ".opf") ? opf_write(t, f) : mps_write(t, f);
    fclose(f);
    return rc;
}

/**
 * Reads a task from a file, auto-detecting format by extension.
 *
 * @param t        [in] Task handle (must be empty: numvar=0, numcon=0).
 * @param filename [in] Input file path. Extension determines format:
 *                    - .cbf -> CBF format
 *                    - .lp  -> LP format
 *                    - other -> MPS format
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or filename is NULL,
 *         PRIMAL_RES_ERR_FILE if file cannot be opened or is invalid,
 *         PRIMAL_RES_ERR_ARG if task is not empty.
 *
 * @note Auto-detects format by extension. Reads entire file into memory
 *       before parsing. MPS/LP readers are free-format with full section support.
 *       CBF reader handles conic/SDP sections including CHANGE section.
 *
 * @example
 * PRIMALtask_t task;
 * PRIMAL_maketask(env, 0, 0, &task);
 * PRIMALrescodee rc = primalio_read(task, "model.mps");
 * if (rc == PRIMAL_RES_OK) { // task populated }
 */
PRIMALrescodee primalio_read(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    {
        int nv = 0, nc = 0;
        PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
        if (nv != 0 || nc != 0) return PRIMAL_RES_ERR_ARG;
    }
    if (has_ext(filename, ".cbf")) {
        FILE *f = fopen(filename, "r");
        if (!f) return PRIMAL_RES_ERR_FILE;
        PRIMALrescodee rc = cbf_read(t, f);
        fclose(f);
        return rc;
    }
    if (has_ext(filename, ".opf")) {
        FILE *f = fopen(filename, "r");
        if (!f) return PRIMAL_RES_ERR_FILE;
        size_t cap = 65536, len = 0;
        char *buf = (char *)malloc(cap);
        if (!buf) { fclose(f); return PRIMAL_RES_ERR_ALLOC; }
        for (;;) {
            if (len + 4096 + 1 > cap) { cap *= 2; char *nb = (char *)realloc(buf, cap); if (!nb) { free(buf); fclose(f); return PRIMAL_RES_ERR_ALLOC; } buf = nb; }
            size_t got = fread(buf + len, 1, 4096, f);
            len += got;
            if (got < 4096) break;
        }
        fclose(f);
        buf[len] = 0;
        PRIMALrescodee rc = opf_read(t, buf);
        free(buf);
        return rc;
    }
    Toks T = {0};
    if (!toks_load(filename, &T)) return PRIMAL_RES_ERR_FILE;
    PRIMALrescodee rc = has_ext(filename, ".lp") ? lp_read(t, &T) : mps_read(t, &T);
    toks_free(&T);
    return rc;
}

PRIMALrescodee primalio_read_format(PRIMALtask_t t, const char *filename, int format) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    {
        int nv = 0, nc = 0;
        PRIMAL_getnumvar(t, &nv); PRIMAL_getnumcon(t, &nc);
        if (nv != 0 || nc != 0) return PRIMAL_RES_ERR_ARG;
    }
    if (format == PRIMAL_DATA_FORMAT_EXTENSION) return primalio_read(t, filename);
    if (format == PRIMAL_DATA_FORMAT_OP) {
        FILE *f = fopen(filename, "r");
        if (!f) return PRIMAL_RES_ERR_FILE;
        size_t cap = 65536, len = 0;
        char *buf = (char *)malloc(cap);
        if (!buf) { fclose(f); return PRIMAL_RES_ERR_ALLOC; }
        for (;;) {
            if (len + 4096 + 1 > cap) { cap *= 2; char *nb = (char *)realloc(buf, cap);
                if (!nb) { free(buf); fclose(f); return PRIMAL_RES_ERR_ALLOC; } buf = nb; }
            size_t got = fread(buf + len, 1, 4096, f);
            len += got;
            if (got < 4096) break;
        }
        fclose(f);
        buf[len] = 0;
        PRIMALrescodee rc = opf_read(t, buf);
        free(buf);
        return rc;
    }
    if (format == PRIMAL_DATA_FORMAT_CB) {
        FILE *f = fopen(filename, "r");
        if (!f) return PRIMAL_RES_ERR_FILE;
        PRIMALrescodee rc = cbf_read(t, f);
        fclose(f);
        return rc;
    }
    if (format == PRIMAL_DATA_FORMAT_MPS || format == PRIMAL_DATA_FORMAT_LP ||
        format == PRIMAL_DATA_FORMAT_FREE_MPS) {
        Toks T = {0};
        if (!toks_load(filename, &T)) return PRIMAL_RES_ERR_FILE;
        PRIMALrescodee rc = (format == PRIMAL_DATA_FORMAT_LP) ? lp_read(t, &T) : mps_read(t, &T);
        toks_free(&T);
        return rc;
    }
    /* TASK (5), PTF (6) and JSON_TASK (8) have no reader here (declared). */
    return PRIMAL_RES_ERR_ARG;
}
