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
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Write a tagged double vector to the text solution file. */
static void sol_wr_vec(FILE *f, const char *tag, const double *v, int n) {
    fprintf(f, "%s %d", tag, n);
    for (int i = 0; i < n; i++) fprintf(f, " %.17g", v[i]);
    fprintf(f, "\n");
}
/* Write a tagged integer vector to the text solution file. */
static void sol_wr_ivec(FILE *f, const char *tag, const int *v, int n) {
    fprintf(f, "%s %d", tag, n);
    for (int i = 0; i < n; i++) fprintf(f, " %d", v[i]);
    fprintf(f, "\n");
}
/* Write the published solution to a text file with the given name. */
PRIMALrescodee PRIMAL_writesolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename) {
    (void)whichsol;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    fprintf(f, "PRIMAL-SOLUTION 1\n");
    double sc[1] = {(double)t->solsta};   sol_wr_vec(f, "solsta", sc, 1);
    double pr[1] = {(double)t->prosta};   sol_wr_vec(f, "prosta", pr, 1);
    double ob[1] = {t->pobj};             sol_wr_vec(f, "pobj", ob, 1);
    double db[1] = {t->dobj};             sol_wr_vec(f, "dobj", db, 1);
    double nv[3] = {t->numvar, t->numcon, t->numbarvar}; sol_wr_vec(f, "dims", nv, 3);
    if (t->skc) { int *s = (int *)malloc((size_t)t->numcon * sizeof(int));
        for (int i = 0; i < t->numcon; i++) s[i] = (int)t->skc[i];
        sol_wr_ivec(f, "skc", s, t->numcon); free(s); } else fprintf(f, "skc 0\n");
    if (t->skx) { int *s = (int *)malloc((size_t)t->numvar * sizeof(int));
        for (int j = 0; j < t->numvar; j++) s[j] = (int)t->skx[j];
        sol_wr_ivec(f, "skx", s, t->numvar); free(s); } else fprintf(f, "skx 0\n");
    sol_wr_vec(f, "xx", t->x, t->numvar);
    sol_wr_vec(f, "y", t->y, t->numcon);
    sol_wr_vec(f, "slc", t->slc, t->numcon);
    sol_wr_vec(f, "suc", t->suc, t->numcon);
    sol_wr_vec(f, "slx", t->slx, t->numvar);
    sol_wr_vec(f, "sux", t->sux, t->numvar);
    int tot = 0;
    for (int j = 0; j < t->numbarvar; j++) tot += t->barDim[j] * t->barDim[j];
    double *bx = (double *)malloc((size_t)(tot > 0 ? tot : 1) * sizeof(double));
    if (bx) {
        int w = 0;
        for (int j = 0; j < t->numbarvar; j++) {
            int d = t->barDim[j];
            for (int k = 0; k < d * d; k++) bx[w++] = t->barx[j] ? t->barx[j][k] : 0.0;
        }
        sol_wr_vec(f, "barx", bx, tot);
        w = 0;
        for (int j = 0; j < t->numbarvar; j++) {
            int d = t->barDim[j];
            for (int k = 0; k < d * d; k++) bx[w++] = t->barsj[j] ? t->barsj[j][k] : 0.0;
        }
        sol_wr_vec(f, "barsj", bx, tot);
        free(bx);
    }
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Read a text solution file back into the task solution buffers. */
PRIMALrescodee PRIMAL_readsolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename) {
    (void)whichsol;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "r");
    if (!f) return PRIMAL_RES_ERR_FILE;
    char line[64];
    if (!fgets(line, sizeof line, f) || strncmp(line, "PRIMAL-SOLUTION", 15) != 0) {
        fclose(f); return PRIMAL_RES_ERR_FILE;
    }
    PRIMALrescodee rc = opt_prepare(t);
    if (rc != PRIMAL_RES_OK) { fclose(f); return rc; }
    t->has_sol = 1;
    double *buf = NULL; int bufn = 0;
    while (fgets(line, sizeof line, f)) {
        char tag[32];
        if (sscanf(line, "%31s", tag) != 1) continue;
        const char *p = line + strlen(tag);
        char *end;
        long n = strtol(p, &end, 10);
        if (end == p || n < 0) continue;
        p = end;
        if (n > bufn) {
            double *nb = (double *)realloc(buf, (size_t)(n > 0 ? n : 1) * sizeof(double));
            if (!nb) { free(buf); fclose(f); return PRIMAL_RES_ERR_ALLOC; }
            buf = nb; bufn = (int)n;
        }
        for (long i = 0; i < n; i++) {
            buf[i] = strtod(p, &end);
            if (end == p) break;
            p = end;
        }
        if (strcmp(tag, "solsta") == 0) t->solsta = (PRIMALsolstae)(int)buf[0];
        else if (strcmp(tag, "prosta") == 0) t->prosta = (PRIMALprostae)(int)buf[0];
        else if (strcmp(tag, "pobj") == 0) t->pobj = buf[0];
        else if (strcmp(tag, "dobj") == 0) t->dobj = buf[0];
        else if (strcmp(tag, "skc") == 0 && t->skc)
            for (long i = 0; i < n && i < t->numcon; i++) t->skc[i] = (PRIMALstakeye)(int)buf[i];
        else if (strcmp(tag, "skx") == 0 && t->skx)
            for (long i = 0; i < n && i < t->numvar; i++) t->skx[i] = (PRIMALstakeye)(int)buf[i];
        else if (strcmp(tag, "xx") == 0)
            for (long i = 0; i < n && i < t->numvar; i++) t->x[i] = buf[i];
        else if (strcmp(tag, "y") == 0)
            for (long i = 0; i < n && i < t->numcon; i++) t->y[i] = buf[i];
        else if (strcmp(tag, "slc") == 0)
            for (long i = 0; i < n && i < t->numcon; i++) t->slc[i] = buf[i];
        else if (strcmp(tag, "suc") == 0)
            for (long i = 0; i < n && i < t->numcon; i++) t->suc[i] = buf[i];
        else if (strcmp(tag, "slx") == 0)
            for (long i = 0; i < n && i < t->numvar; i++) t->slx[i] = buf[i];
        else if (strcmp(tag, "sux") == 0)
            for (long i = 0; i < n && i < t->numvar; i++) t->sux[i] = buf[i];
        else if (strcmp(tag, "barx") == 0 || strcmp(tag, "barsj") == 0) {
            int w = 0;
            for (int j = 0; j < t->numbarvar; j++) {
                int d = t->barDim[j];
                double *dst = (strcmp(tag, "barx") == 0) ? t->barx[j] : t->barsj[j];
                for (int k = 0; k < d * d; k++) if (w < n && dst) dst[k] = buf[w++];
            }
        }
    }
    free(buf);
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Write the interior-point solution to a text file. */
PRIMALrescodee PRIMAL_writesolutionfile(PRIMALtask_t t, const char *filename) {
    return PRIMAL_writesolution(t, PRIMAL_SOL_ITR, filename);
}
/* Read a text solution file into the task. */
PRIMALrescodee PRIMAL_readsolutionfile(PRIMALtask_t t, const char *filename) {
    return PRIMAL_readsolution(t, PRIMAL_SOL_ITR, filename);
}
/* Proprietary binary dump: the same description, written as records (tag, n, values). */
/* Write the published solution to a binary file with a magic header. */
PRIMALrescodee PRIMAL_writebsolution(PRIMALtask_t t, const char *filename, int compress) {
    (void)compress;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, "wb");
    if (!f) return PRIMAL_RES_ERR_FILE;
    const char magic[16] = "PRIMAL-BSOL 1";
    fwrite(magic, 1, sizeof magic, f);
    int dims[3] = {t->numvar, t->numcon, t->numbarvar};
    fwrite(dims, sizeof(int), 3, f);
    fwrite(&t->solsta, sizeof(int), 1, f);
    fwrite(&t->prosta, sizeof(int), 1, f);
    fwrite(&t->pobj, sizeof(double), 1, f);
    fwrite(&t->dobj, sizeof(double), 1, f);
    fwrite(t->x, sizeof(double), (size_t)t->numvar, f);
    fwrite(t->y, sizeof(double), (size_t)t->numcon, f);
    fwrite(t->slc, sizeof(double), (size_t)t->numcon, f);
    fwrite(t->suc, sizeof(double), (size_t)t->numcon, f);
    fwrite(t->slx, sizeof(double), (size_t)t->numvar, f);
    fwrite(t->sux, sizeof(double), (size_t)t->numvar, f);
    for (int j = 0; j < t->numbarvar; j++) {
        int d = t->barDim[j];
        if (t->barx[j]) fwrite(t->barx[j], sizeof(double), (size_t)d * d, f);
        if (t->barsj[j]) fwrite(t->barsj[j], sizeof(double), (size_t)d * d, f);
    }
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Read a binary solution file back into the task solution buffers. */
PRIMALrescodee PRIMAL_readbsolution(PRIMALtask_t t, const char *filename, int compress) {
    (void)compress;
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    FILE *f = fopen(filename, "rb");
    if (!f) return PRIMAL_RES_ERR_FILE;
    char magic[16];
    if (fread(magic, 1, sizeof magic, f) != sizeof magic ||
        strncmp(magic, "PRIMAL-BSOL", 11) != 0) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    int dims[3] = {0, 0, 0};
    if (fread(dims, sizeof(int), 3, f) != 3) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (dims[0] != t->numvar || dims[1] != t->numcon || dims[2] != t->numbarvar) {
        fclose(f); return PRIMAL_RES_ERR_ARG;   /* different dimensions: not applicable */
    }
    PRIMALrescodee rc = opt_prepare(t);
    if (rc != PRIMAL_RES_OK) { fclose(f); return rc; }
    t->has_sol = 1;
    int iv = 0;
    if (fread(&iv, sizeof(int), 1, f) != 1) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    t->solsta = (PRIMALsolstae)iv;
    if (fread(&iv, sizeof(int), 1, f) != 1) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    t->prosta = (PRIMALprostae)iv;
    if (fread(&t->pobj, sizeof(double), 1, f) != 1) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (fread(&t->dobj, sizeof(double), 1, f) != 1) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numvar > 0 && fread(t->x, sizeof(double), (size_t)t->numvar, f) != (size_t)t->numvar) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numcon > 0 && fread(t->y, sizeof(double), (size_t)t->numcon, f) != (size_t)t->numcon) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numcon > 0 && fread(t->slc, sizeof(double), (size_t)t->numcon, f) != (size_t)t->numcon) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numcon > 0 && fread(t->suc, sizeof(double), (size_t)t->numcon, f) != (size_t)t->numcon) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numvar > 0 && fread(t->slx, sizeof(double), (size_t)t->numvar, f) != (size_t)t->numvar) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    if (t->numvar > 0 && fread(t->sux, sizeof(double), (size_t)t->numvar, f) != (size_t)t->numvar) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    for (int j = 0; j < t->numbarvar; j++) {
        int d = t->barDim[j];
        if (t->barx[j] && fread(t->barx[j], sizeof(double), (size_t)d * d, f) != (size_t)d * d) { fclose(f); return PRIMAL_RES_ERR_FILE; }
        if (t->barsj[j] && fread(t->barsj[j], sizeof(double), (size_t)d * d, f) != (size_t)d * d) { fclose(f); return PRIMAL_RES_ERR_FILE; }
    }
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Reference JSON (JSOL): a flat object holding the solution. Here too the
 * reference format was not read: this is a proprietary JSON. */
/* Write one named double array as a JSON member. */
static void json_wr_arr(FILE *f, const char *key, const double *v, int n) {
    fprintf(f, ",\"%s\":[", key);
    for (int i = 0; i < n; i++) fprintf(f, "%s%.17g", i ? "," : "", v[i]);
    fprintf(f, "]");
}
/* Write the published solution as a flat JSON object. */
PRIMALrescodee PRIMAL_writejsonsol(PRIMALtask_t t, const char *filename) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    fprintf(f, "{\"pobj\":%.17g,\"dobj\":%.17g,\"solsta\":%d,\"prosta\":%d",
            t->pobj, t->dobj, (int)t->solsta, (int)t->prosta);
    json_wr_arr(f, "xx", t->x, t->numvar);
    json_wr_arr(f, "y", t->y, t->numcon);
    json_wr_arr(f, "slc", t->slc, t->numcon);
    json_wr_arr(f, "suc", t->suc, t->numcon);
    json_wr_arr(f, "slx", t->slx, t->numvar);
    json_wr_arr(f, "sux", t->sux, t->numvar);
    fprintf(f, "}\n");
    fclose(f);
    return PRIMAL_RES_OK;
}
/* Parse one numeric JSON member by key into val. */
static int json_num(const char *data, const char *key, double *val) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(data, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    char *end;
    double v = strtod(p + 1, &end);
    if (end == p + 1) return 0;
    *val = v;
    return 1;
}
/* Parse one JSON array member by key into out, reporting its length. */
static int json_arr(const char *data, const char *key, double *out, int maxn, int *n) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(data, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != '[') p++;
    if (*p != '[') return 0;
    p++;
    int w = 0;
    while (*p && *p != ']') {
        char *end;
        double v = strtod(p, &end);
        if (end == p) { p++; continue; }
        if (w < maxn) out[w] = v;
        w++;
        p = end;
    }
    *n = w;
    return 1;
}
/* Parse a JSON solution string into the task solution buffers. */
PRIMALrescodee PRIMAL_readjsonstring(PRIMALtask_t t, const char *data) {
    if (!t || !data) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = opt_prepare(t);
    if (rc != PRIMAL_RES_OK) return rc;
    t->has_sol = 1;
    double v;
    int n;
    if (json_num(data, "solsta", &v)) t->solsta = (PRIMALsolstae)(int)v;
    if (json_num(data, "prosta", &v)) t->prosta = (PRIMALprostae)(int)v;
    if (json_num(data, "pobj", &v)) t->pobj = v;
    if (json_num(data, "dobj", &v)) t->dobj = v;
    if (json_arr(data, "xx", t->x, t->numvar, &n)) {}
    if (json_arr(data, "y", t->y, t->numcon, &n)) {}
    if (json_arr(data, "slc", t->slc, t->numcon, &n)) {}
    if (json_arr(data, "suc", t->suc, t->numcon, &n)) {}
    if (json_arr(data, "slx", t->slx, t->numvar, &n)) {}
    if (json_arr(data, "sux", t->sux, t->numvar, &n)) {}
    return PRIMAL_RES_OK;
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
    buf[len] = '\0';
    fclose(f);
    PRIMALrescodee rc = PRIMAL_readjsonstring(t, buf);
    free(buf);
    return rc;
}

/* Write the binary solution image through a caller-supplied write callback. */
PRIMALrescodee PRIMAL_writebsolutionhandle(PRIMALtask_t t, PRIMALhwritefunc func,
                                           void *handle, int compress) {
    (void)compress;
    if (!t || !func) return PRIMAL_RES_ERR_NULL;
    if (!t->has_sol) return PRIMAL_RES_ERR_ARG;
    int dims[3] = {t->numvar, t->numcon, t->numbarvar};
    func(handle, "PRIMAL-BSOL 1", 16);
    func(handle, (const char *)dims, (int)sizeof dims);
    int iv = (int)t->solsta; func(handle, (const char *)&iv, (int)sizeof iv);
    iv = (int)t->prosta; func(handle, (const char *)&iv, (int)sizeof iv);
    func(handle, (const char *)&t->pobj, (int)sizeof(double));
    func(handle, (const char *)&t->dobj, (int)sizeof(double));
    func(handle, (const char *)t->x, (int)((size_t)t->numvar * sizeof(double)));
    func(handle, (const char *)t->y, (int)((size_t)t->numcon * sizeof(double)));
    return PRIMAL_RES_OK;
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
