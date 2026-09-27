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
/* primal_solio.c - solution I/O (text/binary/JSON) and basis setters.
 * Shares primal_priv.h.
 */
#include "primal_priv.h"
#include <ctype.h>
#include <errno.h>

/* A complete image is parsed into independent storage. No task state is touched
 * until every required field, dimension, enum and finite value is validated. */
enum { SF_SOLSTA, SF_PROSTA, SF_POBJ, SF_DOBJ, SF_DIMS, SF_SKC, SF_SKX,
       SF_X, SF_Y, SF_SLC, SF_SUC, SF_SLX, SF_SUX, SF_BARX, SF_BARSJ, SF_COUNT };
static const char *const sol_tags[SF_COUNT] = {
    "solsta", "prosta", "pobj", "dobj", "dims", "skc", "skx", "xx", "y",
    "slc", "suc", "slx", "sux", "barx", "barsj"
};
typedef struct {
    double *v[SF_COUNT], *snx, *xc, *pray, *dray;
    size_t n[SF_COUNT];
    PRIMALstakeye *skc, *skx;
} SolImage;

static void sol_image_free(SolImage *s) {
    for (int k = 0; k < SF_COUNT; k++) free(s->v[k]);
    free(s->snx); free(s->xc); free(s->pray); free(s->dray);
    free(s->skc); free(s->skx);
}

static PRIMALrescodee sol_image_init(PRIMALtask_t t, SolImage *s) {
    memset(s, 0, sizeof *s);
    size_t nb = 0, nv = (size_t)t->numvar, nc = (size_t)t->numcon;
    for (int j = 0; j < t->numbarvar; j++) {
        size_t d = (size_t)t->barDim[j];
        if (!d || d > SIZE_MAX / d || d*d > SIZE_MAX / sizeof(double) - nb)
            return PRIMAL_RES_ERR_ALLOC;
        nb += d*d;
        if (!t->barx[j] || !t->barsj[j]) return PRIMAL_RES_ERR_ARG;
    }
    const size_t counts[SF_COUNT] = {1,1,1,1,3,nc,nv,nv,nc,nc,nc,nv,nv,nb,nb};
    for (int k = 0; k < SF_COUNT; k++) {
        s->n[k] = counts[k];
        s->v[k] = (double *)calloc(counts[k] ? counts[k] : 1, sizeof(double));
        if (!s->v[k]) return PRIMAL_RES_ERR_ALLOC;
    }
    s->snx = (double *)calloc(nv ? nv : 1, sizeof(double));
    s->xc = (double *)calloc(nc ? nc : 1, sizeof(double));
    s->pray = (double *)calloc(nv ? nv : 1, sizeof(double));
    s->dray = (double *)calloc(nc ? nc : 1, sizeof(double));
    s->skc = (PRIMALstakeye *)calloc(nc ? nc : 1, sizeof(PRIMALstakeye));
    s->skx = (PRIMALstakeye *)calloc(nv ? nv : 1, sizeof(PRIMALstakeye));
    if (!s->snx || !s->xc || !s->pray || !s->dray || !s->skc || !s->skx)
        return PRIMAL_RES_ERR_ALLOC;
    return PRIMAL_RES_OK;
}

static PRIMALrescodee sol_image_validate(PRIMALtask_t t, const SolImage *s) {
    for (int k = 0; k < SF_COUNT; k++)
        for (size_t i = 0; i < s->n[k]; i++)
            if (!isfinite(s->v[k][i])) return PRIMAL_RES_ERR_FILE;
    if (s->v[SF_DIMS][0] != t->numvar || s->v[SF_DIMS][1] != t->numcon ||
        s->v[SF_DIMS][2] != t->numbarvar) return PRIMAL_RES_ERR_ARG;
    double ss = s->v[SF_SOLSTA][0], ps = s->v[SF_PROSTA][0];
    if (!(ss == 0 || ss == 1 || ss == 2 || ss == 5 || ss == 6 || ss == 9) ||
        ps < 0 || ps > 8 || ps != floor(ps)) return PRIMAL_RES_ERR_FILE;
    for (int k = SF_SKC; k <= SF_SKX; k++)
        for (size_t i = 0; i < s->n[k]; i++) {
            double v = s->v[k][i];
            if (v < PRIMAL_SK_UNDEF || v > PRIMAL_SK_UPR || v != floor(v))
                return PRIMAL_RES_ERR_FILE;
        }
    return PRIMAL_RES_OK;
}

static void sol_image_commit(PRIMALtask_t t, SolImage *s) {
#define SOL_TAKE(member, src) do { free(t->member); t->member = (src); (src) = NULL; } while (0)
    SOL_TAKE(x, s->v[SF_X]); SOL_TAKE(y, s->v[SF_Y]);
    SOL_TAKE(slc, s->v[SF_SLC]); SOL_TAKE(suc, s->v[SF_SUC]);
    SOL_TAKE(slx, s->v[SF_SLX]); SOL_TAKE(sux, s->v[SF_SUX]);
    SOL_TAKE(snx, s->snx); SOL_TAKE(xc, s->xc);
    SOL_TAKE(pray, s->pray); SOL_TAKE(dray, s->dray);
    for (int i = 0; i < t->numcon; i++) s->skc[i] = (PRIMALstakeye)s->v[SF_SKC][i];
    for (int j = 0; j < t->numvar; j++) s->skx[j] = (PRIMALstakeye)s->v[SF_SKX][j];
    SOL_TAKE(skc, s->skc); SOL_TAKE(skx, s->skx);
    t->skccap = t->numcon > 0 ? t->numcon : 1;
    t->skxcap = t->numvar > 0 ? t->numvar : 1;
#undef SOL_TAKE
    size_t offset = 0;
    for (int j = 0; j < t->numbarvar; j++) {
        size_t n = (size_t)t->barDim[j] * (size_t)t->barDim[j];
        memcpy(t->barx[j], s->v[SF_BARX] + offset, n * sizeof(double));
        memcpy(t->barsj[j], s->v[SF_BARSJ] + offset, n * sizeof(double));
        offset += n;
    }
    free(t->soc_dual); t->soc_dual = NULL; t->nsoc_dual = 0;
    t->has_xc = t->has_pray = t->has_dray = 0;
    t->mip_result = t->mip_bound_defined = 0;
    /* A file has no authenticated model identity or optimality proof. Keep its
     * vectors for explicit inspection; only optimization can publish a verdict. */
    t->has_sol = 1; t->result_stale = 1;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN; t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    t->pobj = s->v[SF_POBJ][0]; t->dobj = s->v[SF_DOBJ][0];
}

/* Tokens have bounded storage; an overlong token is an error, never a prefix. */
static int sol_token(FILE *f, char *buf, size_t cap) {
    int c;
    do { c = fgetc(f); } while (c != EOF && isspace((unsigned char)c));
    if (c == EOF) return 0;
    size_t n = 0;
    do {
        if (c == 0 || n + 1 >= cap) return -1;
        buf[n++] = (char)c; c = fgetc(f);
    } while (c != EOF && !isspace((unsigned char)c));
    buf[n] = 0;
    return 1;
}
static int sol_number(FILE *f, double *out) {
    char token[128], *end;
    if (sol_token(f, token, sizeof token) != 1) return 0;
    errno = 0;
    double v = strtod(token, &end);
    if (end == token || *end || errno || !isfinite(v)) return 0;
    *out = v; return 1;
}



static PRIMALrescodee sol_capture(PRIMALtask_t t, SolImage *s) {
    PRIMALrescodee rc = sol_image_init(t, s);
    if (rc != PRIMAL_RES_OK) return rc;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    s->v[SF_SOLSTA][0] = t->solsta; s->v[SF_PROSTA][0] = t->prosta;
    s->v[SF_POBJ][0] = t->pobj; s->v[SF_DOBJ][0] = t->dobj;
    s->v[SF_DIMS][0] = t->numvar; s->v[SF_DIMS][1] = t->numcon;
    s->v[SF_DIMS][2] = t->numbarvar;
    const double *vectors[] = {t->x, t->y, t->slc, t->suc, t->slx, t->sux};
    for (int k = SF_X; k <= SF_SUX; k++) {
        if (s->n[k] && !vectors[k-SF_X]) return PRIMAL_RES_ERR_ARG;
        if (s->n[k]) memcpy(s->v[k], vectors[k-SF_X], s->n[k]*sizeof(double));
    }
    if (t->skc) for (int i = 0; i < t->numcon; i++) s->v[SF_SKC][i] = t->skc[i];
    if (t->skx) for (int j = 0; j < t->numvar; j++) s->v[SF_SKX][j] = t->skx[j];
    size_t off = 0;
    for (int j = 0; j < t->numbarvar; j++) {
        size_t n = (size_t)t->barDim[j] * (size_t)t->barDim[j];
        memcpy(s->v[SF_BARX]+off, t->barx[j], n*sizeof(double));
        memcpy(s->v[SF_BARSJ]+off, t->barsj[j], n*sizeof(double)); off += n;
    }
    return sol_image_validate(t, s);
}

PRIMALrescodee PRIMAL_writesolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(whichsol)) return PRIMAL_RES_ERR_ARG;
    SolImage s; PRIMALrescodee rc = sol_capture(t, &s);
    if (rc != PRIMAL_RES_OK) { sol_image_free(&s); return rc; }
    FILE *f = fopen(filename, "w");
    if (!f) { sol_image_free(&s); return PRIMAL_RES_ERR_FILE; }
    fprintf(f, "PRIMAL-SOLUTION 1\n");
    for (int k = 0; k < SF_COUNT; k++) {
        fprintf(f, "%s %zu", sol_tags[k], s.n[k]);
        for (size_t i = 0; i < s.n[k]; i++) fprintf(f, " %.17g", s.v[k][i]);
        fputc('\n', f);
    }
    int failed = ferror(f); if (fclose(f)) failed = 1;
    sol_image_free(&s); return failed ? PRIMAL_RES_ERR_FILE : PRIMAL_RES_OK;
}

PRIMALrescodee PRIMAL_readsolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!sol_key_ok(whichsol)) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    SolImage s; PRIMALrescodee rc = sol_image_init(t, &s);
    if (rc != PRIMAL_RES_OK) goto done;
    rc = PRIMAL_RES_ERR_FILE;
    char token[128];
    if (sol_token(f, token, sizeof token) != 1 || strcmp(token, "PRIMAL-SOLUTION")) goto done;
    if (sol_token(f, token, sizeof token) != 1 || strcmp(token, "1")) goto done;
    unsigned seen = 0;
    int result;
    while ((result = sol_token(f, token, sizeof token)) == 1) {
        int k;
        for (k = 0; k < SF_COUNT && strcmp(token, sol_tags[k]); k++) {}
        if (k == SF_COUNT || (seen & (1u << k))) goto done;
        double count;
        if (!sol_number(f, &count)) goto done;
        if ((k == SF_SKC || k == SF_SKX) && count == 0) s.n[k] = 0;
        if (count != (double)s.n[k]) goto done;
        for (size_t i = 0; i < s.n[k]; i++) if (!sol_number(f, &s.v[k][i])) goto done;
        seen |= 1u << k;
    }
    if (result < 0 || ferror(f) || seen != (1u << SF_COUNT)-1) goto done;
    rc = sol_image_validate(t, &s);
    if (rc == PRIMAL_RES_OK) sol_image_commit(t, &s);
done:
    sol_image_free(&s); fclose(f); return rc;
}

/* Both file and callback writers emit the same complete native binary format. */
static void sol_emit(PRIMALhwritefunc emit, void *handle, const void *data, size_t bytes) {
    const char *p = (const char *)data;
    while (bytes) {
        int n = bytes > INT_MAX ? INT_MAX : (int)bytes;
        emit(handle, p, n); p += n; bytes -= (size_t)n;
    }
}
static void sol_binary_emit(PRIMALtask_t t, const SolImage *s, PRIMALhwritefunc emit, void *handle) {
    static const char magic[16] = "PRIMAL-BSOL 1";
    int dims[3] = {t->numvar, t->numcon, t->numbarvar};
    int ss = (int)s->v[SF_SOLSTA][0], ps = (int)s->v[SF_PROSTA][0];
    sol_emit(emit, handle, magic, sizeof magic);
    sol_emit(emit, handle, dims, sizeof dims);
    sol_emit(emit, handle, &ss, sizeof ss); sol_emit(emit, handle, &ps, sizeof ps);
    sol_emit(emit, handle, s->v[SF_POBJ], sizeof(double));
    sol_emit(emit, handle, s->v[SF_DOBJ], sizeof(double));
    for (int k = SF_X; k <= SF_SUX; k++) sol_emit(emit, handle, s->v[k], s->n[k]*sizeof(double));
    size_t off = 0;
    for (int j = 0; j < t->numbarvar; j++) {
        size_t n = (size_t)t->barDim[j] * (size_t)t->barDim[j];
        sol_emit(emit, handle, s->v[SF_BARX]+off, n*sizeof(double));
        sol_emit(emit, handle, s->v[SF_BARSJ]+off, n*sizeof(double)); off += n;
    }
}
static void sol_file_emit(void *handle, const char *data, int len) {
    (void)fwrite(data, 1, (size_t)len, (FILE *)handle);
}
PRIMALrescodee PRIMAL_writebsolution(PRIMALtask_t t, const char *filename, int compress) {
    if (compress != 0) return PRIMAL_RES_ERR_ARG;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    SolImage s; PRIMALrescodee rc = sol_capture(t, &s);
    if (rc != PRIMAL_RES_OK) { sol_image_free(&s); return rc; }
    FILE *f = fopen(filename, "wb");
    if (!f) { sol_image_free(&s); return PRIMAL_RES_ERR_FILE; }
    sol_binary_emit(t, &s, sol_file_emit, f);
    int failed = ferror(f); if (fclose(f)) failed = 1;
    sol_image_free(&s); return failed ? PRIMAL_RES_ERR_FILE : PRIMAL_RES_OK;
}
PRIMALrescodee PRIMAL_writebsolutionhandle(PRIMALtask_t t, PRIMALhwritefunc func, void *handle, int compress) {
    if (compress != 0) return PRIMAL_RES_ERR_ARG;
    if (!t || !func) return PRIMAL_RES_ERR_NULL;
    SolImage s; PRIMALrescodee rc = sol_capture(t, &s);
    if (rc == PRIMAL_RES_OK) sol_binary_emit(t, &s, func, handle);
    sol_image_free(&s); return rc;
}
PRIMALrescodee PRIMAL_readbsolution(PRIMALtask_t t, const char *filename, int compress) {
    if (compress != 0) return PRIMAL_RES_ERR_ARG;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "rb");
    if (!f) return PRIMAL_RES_ERR_FILE;
    SolImage s; PRIMALrescodee rc = sol_image_init(t, &s);
    if (rc != PRIMAL_RES_OK) goto done;
    rc = PRIMAL_RES_ERR_FILE;
    char magic[16]; static const char expected[16] = "PRIMAL-BSOL 1";
    int dims[3], ss, ps;
#define SOL_READ(dst, count) do { size_t n_ = (count); if (fread((dst), sizeof *(dst), n_, f) != n_) goto done; } while (0)
    SOL_READ(magic, 16);
    if (memcmp(magic, expected, 16)) goto done;
    SOL_READ(dims, 3);
    if (dims[0] != t->numvar || dims[1] != t->numcon || dims[2] != t->numbarvar) {
        rc = PRIMAL_RES_ERR_ARG; goto done;
    }
    for (int i = 0; i < 3; i++) s.v[SF_DIMS][i] = dims[i];
    SOL_READ(&ss, 1); SOL_READ(&ps, 1);
    s.v[SF_SOLSTA][0] = ss; s.v[SF_PROSTA][0] = ps;
    SOL_READ(s.v[SF_POBJ], 1); SOL_READ(s.v[SF_DOBJ], 1);
    for (int k = SF_X; k <= SF_SUX; k++) SOL_READ(s.v[k], s.n[k]);
    size_t off = 0;
    for (int j = 0; j < t->numbarvar; j++) {
        size_t n = (size_t)t->barDim[j] * (size_t)t->barDim[j];
        SOL_READ(s.v[SF_BARX]+off, n); SOL_READ(s.v[SF_BARSJ]+off, n); off += n;
    }
    if (fgetc(f) != EOF || ferror(f)) goto done;
    rc = sol_image_validate(t, &s);
    if (rc == PRIMAL_RES_OK) sol_image_commit(t, &s);
done:
#undef SOL_READ
    sol_image_free(&s); fclose(f); return rc;
}

PRIMALrescodee PRIMAL_writejsonsol(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    SolImage s; PRIMALrescodee rc = sol_capture(t, &s);
    if (rc != PRIMAL_RES_OK) { sol_image_free(&s); return rc; }
    FILE *f = fopen(filename, "w");
    if (!f) { sol_image_free(&s); return PRIMAL_RES_ERR_FILE; }
    fputc('{', f);
    for (int k = 0; k < SF_COUNT; k++) {
        fprintf(f, "%s\"%s\":", k ? "," : "", sol_tags[k]);
        if (k <= SF_DOBJ) fprintf(f, "%.17g", s.v[k][0]);
        else {
            fputc('[', f);
            for (size_t i = 0; i < s.n[k]; i++) fprintf(f, "%s%.17g", i ? "," : "", s.v[k][i]);
            fputc(']', f);
        }
    }
    fputs("}\n", f);
    int failed = ferror(f); if (fclose(f)) failed = 1;
    sol_image_free(&s); return failed ? PRIMAL_RES_ERR_FILE : PRIMAL_RES_OK;
}
static void sol_json_space(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}
static int sol_json_number(const char **p, double *value) {
    sol_json_space(p);
    const char *start = *p, *end = start;
    if (*end == '-') end++;
    if (*end == '0') end++;
    else { if (*end < '1' || *end > '9') return 0; while (*end >= '0' && *end <= '9') end++; }
    if (*end == '.') {
        end++; if (*end < '0' || *end > '9') return 0;
        while (*end >= '0' && *end <= '9') end++;
    }
    if (*end == 'e' || *end == 'E') {
        end++; if (*end == '+' || *end == '-') end++;
        if (*end < '0' || *end > '9') return 0;
        while (*end >= '0' && *end <= '9') end++;
    }
    errno = 0; char *parsed;
    double v = strtod(start, &parsed);
    if (parsed != end || errno || !isfinite(v)) return 0;
    *value = v; *p = end; return 1;
}
PRIMALrescodee PRIMAL_readjsonstring(PRIMALtask_t t, const char *data) {
    if (!t || !data) return PRIMAL_RES_ERR_NULL;
    SolImage s; PRIMALrescodee rc = sol_image_init(t, &s);
    if (rc != PRIMAL_RES_OK) goto done;
    rc = PRIMAL_RES_ERR_FILE;
    const char *p = data; unsigned seen = 0;
    sol_json_space(&p); if (*p != '{') goto done; p++;
    for (;;) {
        sol_json_space(&p); if (*p != '"') goto done; p++;
        const char *start = p;
        while (*p && *p != '"') p++;
        if (!*p) goto done;
        int k;
        for (k = 0; k < SF_COUNT; k++)
            if ((size_t)(p-start) == strlen(sol_tags[k]) && !memcmp(start, sol_tags[k], (size_t)(p-start))) break;
        if (k == SF_COUNT || (seen & (1u << k))) goto done;
        p++; sol_json_space(&p); if (*p != ':') goto done; p++;
        if (k <= SF_DOBJ) { if (!sol_json_number(&p, s.v[k])) goto done; }
        else {
            sol_json_space(&p); if (*p != '[') goto done; p++; sol_json_space(&p);
            size_t n = 0;
            if (*p != ']') for (;;) {
                if (n >= s.n[k] || !sol_json_number(&p, &s.v[k][n++])) goto done;
                sol_json_space(&p);
                if (*p != ',') break;
                p++;
            }
            if (*p != ']' || (n != s.n[k] && !((k == SF_SKC || k == SF_SKX) && n == 0))) goto done;
            s.n[k] = n; p++;
        }
        seen |= 1u << k;
        sol_json_space(&p);
        if (*p == '}') { p++; break; }
        if (*p != ',') goto done;
        p++;
    }
    sol_json_space(&p);
    if (*p || seen != (1u << SF_COUNT)-1) goto done;
    rc = sol_image_validate(t, &s);
    if (rc == PRIMAL_RES_OK) sol_image_commit(t, &s);
done:
    sol_image_free(&s); return rc;
}

/* Write the interior-point solution to a text file. */
PRIMALrescodee PRIMAL_writesolutionfile(PRIMALtask_t t, const char *filename) {
    return PRIMAL_writesolution(t, PRIMAL_SOL_ITR, filename);
}
/* Read a text solution file into the task. */
PRIMALrescodee PRIMAL_readsolutionfile(PRIMALtask_t t, const char *filename) {
    return PRIMAL_readsolution(t, PRIMAL_SOL_ITR, filename);
}
/* Read a JSON solution file into the task solution buffers. */
PRIMALrescodee PRIMAL_readjsonsol(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { fclose(f); return PRIMAL_RES_ERR_ALLOC; }
    size_t got;
    while ((got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); fclose(f); return PRIMAL_RES_ERR_ALLOC; }
            buf = nb;
        }
    }
    if (ferror(f) || memchr(buf, 0, len)) { free(buf); fclose(f); return PRIMAL_RES_ERR_FILE; }
    buf[len] = '\0';
    fclose(f);
    PRIMALrescodee rc = PRIMAL_readjsonstring(t, buf);
    free(buf);
    return rc;
}

/* Write the basis status keys to a text basis file. */
PRIMALrescodee PRIMAL_writebasis(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!t->skc || !t->skx) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    for (int i = 0; i < t->numcon; i++) {
        if (t->skc[i] == PRIMAL_SK_BAS) fprintf(f, " XBASIC       c%d\n", i);
        else if (t->skc[i] == PRIMAL_SK_UPR) fprintf(f, " XUPPER       c%d\n", i);
        else if (t->skc[i] == PRIMAL_SK_LOW) fprintf(f, " XLOWER       c%d\n", i);
    }
    for (int j = 0; j < t->numvar; j++) {
        if (t->skx[j] == PRIMAL_SK_BAS) fprintf(f, " XBASIC       x%d\n", j);
        else if (t->skx[j] == PRIMAL_SK_UPR) fprintf(f, " XUPPER       x%d\n", j);
        else if (t->skx[j] == PRIMAL_SK_LOW) fprintf(f, " XLOWER       x%d\n", j);
    }
    fclose(f);
    return PRIMAL_RES_OK;
}

/* Read basis status keys from a text basis file into the task. */
PRIMALrescodee PRIMAL_readbasis(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    PRIMALstakeye *kc = (PRIMALstakeye *)lazy_grow(t->skc, &t->skccap,
                             t->numcon > 0 ? t->numcon : 1, sizeof(PRIMALstakeye));
    if (!kc) { fclose(f); return PRIMAL_RES_ERR_ALLOC; }
    t->skc = kc;
    PRIMALstakeye *kx = (PRIMALstakeye *)lazy_grow(t->skx, &t->skxcap,
                             t->numvar > 0 ? t->numvar : 1, sizeof(PRIMALstakeye));
    if (!kx) { fclose(f); return PRIMAL_RES_ERR_ALLOC; }
    t->skx = kx;
    for (int i = 0; i < t->numcon; i++) t->skc[i] = PRIMAL_SK_BAS;
    for (int j = 0; j < t->numvar; j++) t->skx[j] = PRIMAL_SK_BAS;
    char line[512];
    int bad = 0;
    while (fgets(line, sizeof line, f)) {
        char ty[32], nm[64];
        if (sscanf(line, "%31s %63s", ty, nm) != 2) continue;
        PRIMALstakeye st;
        if (strcmp(ty, "XBASIC") == 0) st = PRIMAL_SK_BAS;
        else if (strcmp(ty, "XLOWER") == 0) st = PRIMAL_SK_LOW;
        else if (strcmp(ty, "XUPPER") == 0) st = PRIMAL_SK_UPR;
        else continue;
        int idx = -1;
        char kind = 0;
        if (nm[0] == 'x' && sscanf(nm, "x%d", &idx) == 1) kind = 'x';
        else if (nm[0] == 'c' && sscanf(nm, "c%d", &idx) == 1) kind = 'c';
        if (kind == 'x' && idx >= 0 && idx < t->numvar) t->skx[idx] = st;
        else if (kind == 'c' && idx >= 0 && idx < t->numcon) t->skc[idx] = st;
        else bad = 1;
    }
    fclose(f);
    return bad ? PRIMAL_RES_ERR_FILE : PRIMAL_RES_OK;
}

/* Store caller-supplied constraint status keys in the task. */
PRIMALrescodee PRIMAL_putskc(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc) {
    (void)which;
    if (!t || !skc) return PRIMAL_RES_ERR_NULL;
    PRIMALstakeye *p = (PRIMALstakeye *)lazy_grow(t->skc, &t->skccap,
                            t->numcon > 0 ? t->numcon : 1, sizeof(PRIMALstakeye));
    if (!p) return PRIMAL_RES_ERR_ALLOC;
    t->skc = p;
    for (int i = 0; i < t->numcon; i++) t->skc[i] = skc[i];
    return PRIMAL_RES_OK;
}

/* Store caller-supplied variable status keys in the task. */
PRIMALrescodee PRIMAL_putskx(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skx) {
    (void)which;
    if (!t || !skx) return PRIMAL_RES_ERR_NULL;
    PRIMALstakeye *p = (PRIMALstakeye *)lazy_grow(t->skx, &t->skxcap,
                            t->numvar > 0 ? t->numvar : 1, sizeof(PRIMALstakeye));
    if (!p) return PRIMAL_RES_ERR_ALLOC;
    t->skx = p;
    for (int j = 0; j < t->numvar; j++) t->skx[j] = skx[j];
    return PRIMAL_RES_OK;
}

/* Copy the stored constraint status keys into the caller buffer. */
PRIMALrescodee PRIMAL_getskc(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skc) {
    (void)which;
    if (!t || !skc) return PRIMAL_RES_ERR_NULL;
    if (!t->skc) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < t->numcon; i++) skc[i] = t->skc[i];
    return PRIMAL_RES_OK;
}

/* Copy the stored variable status keys into the caller buffer. */
PRIMALrescodee PRIMAL_getskx(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skx) {
    (void)which;
    if (!t || !skx) return PRIMAL_RES_ERR_NULL;
    if (!t->skx) return PRIMAL_RES_ERR_ARG;
    for (int j = 0; j < t->numvar; j++) skx[j] = t->skx[j];
    return PRIMAL_RES_OK;
}

/* =====================================================================
 * Sensitivity (LP post-optimal analysis of the current solution).
 * The solution stays optimal while it remains dual feasible (reduced
 * costs) and primal feasible (bounds). Using the slc/suc/slx/sux split:
 *  - cost c_j (nonbasic var at lower/upper): move c_j until its
 *    reduced cost z_j changes sign (z_j = -(c_j + A_j'y))
 *  - cost c_j (basic var): moving c_j changes every dual; without the
 *    basis it has no closed form -> ERR_ARG (declared deviation)
 *  - RHS of active row i (y_i != 0): move the active bound until
 *    a basic var hits its bound; needs x_B -> without an exposed basis
 *    the row world is used: the row stays active while a primal feasible
 *    solution exists, approximated with the bracket (no exposed basic
 *    var) -> return the interval where the DUALS stay optimal by checking
 *    structural complementarity: for the clone return the range where the
 *    row CAN stay active given the variable bounds (geometric analysis for
 *    single-variable rows, not in general): documented deviation, ERR_ARG
 *    for rows with more than one free nonbasic variable... In practice:
 *    implement the full cost range (nonbasic case + recomputation for the
 *    basic case via re-solve bisection) and the RHS range via re-solve
 *    bisection.
 * ===================================================================== */
