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
/* scip_cbf.c - a CBF model built through SCIP's C API, the way an application
 * would write it, and solved by SCIP (SCIP-SDP when the model has PSD parts).
 *
 * Neither SCIP's nor SCIP-SDP's CBF reader takes every cone (SCIP-SDP's reads
 * only F/L+/L-/L=, SCIP's has no EXP or PSD), so this builds the model itself:
 *   - VAR groups: bounds (L+, L-, L=, F) on the variables, or a cone on them;
 *   - CON rows u = A x + <F, X> + b: L+/L-/L= rows are linear constraints; a
 *     Q cone takes a one-variable row a x + b as it is (SCIP's SOC constraint
 *     has a coefficient and an offset per term); otherwise a member gets a
 *     variable of its own, tied to its row by a linear equality;
 *   - Q: SCIP's SOC constraint; QR as the SOC  ||(sqrt2 w, u0-u1)|| <= u0+u1;
 *   - EXP (u0 >= u1 exp(u2/u1)): u1 exp(u2/u1) <= u0 when u1 is a constant;
 *     when u1 is a variable the epsilon-perspective (u1+e) exp(u2/(u1+e)) - e
 *     <= u0, u1 >= 0 (e = 1e-7): the plain perspective is undefined at u1 = 0,
 *     where the closed cone still holds points (u2 <= 0), and SCIP would then
 *     accept u2 > 0 -- the syn/rsyn families switch u1 with a binary;
 *   - @k:POW (u0^a u1^(1-a) >= |u2..|): powers of the two (nonnegative) bases;
 *   - PSDVAR: one variable per lower-triangle entry and an SDP constraint on
 *     them; PSDCON: the SDP constraint  sum_j H_j x_j + D >= 0  directly
 *     (SCIP-SDP's form is  sum_j A_j y_j - A_0 >= 0, so A_0 = -D);
 *   - INT: integer variables.
 * The plugins are SCIP-SDP's for a model with PSD parts, SCIP's otherwise,
 * each with its default settings.
 *
 * Usage:  scip_cbf FILE.cbf [TIMELIMIT]
 * Prints one CSV line: status,seconds,objective -- status ok, infeasible,
 * timeout or n/a; seconds is SCIP's solving time (model building excluded).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/cons_nonlinear.h"
#include "scipsdp/scipsdpdefplugins.h"
#include "scipsdp/cons_sdp.h"

#define EPS_PERSPECTIVE 1e-7

typedef struct { char name[32]; int size; } Group;
typedef struct { int a, b, c, d; double v; } Ent;   /* up to four indices + value */
typedef struct { Ent *e; int n, cap; } Ents;

static void push(Ents *s, int a, int b, int c, int d, double v) {
    if (s->n == s->cap) {
        s->cap = s->cap ? 2 * s->cap : 64;
        s->e = (Ent *)realloc(s->e, (size_t)s->cap * sizeof(Ent));
        if (!s->e) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    s->e[s->n].a = a; s->e[s->n].b = b; s->e[s->n].c = c; s->e[s->n].d = d; s->e[s->n].v = v;
    s->n++;
}

typedef struct {
    int max, nv, ncon, nvg, ncg, nint, npsdvar, npsdcon, npow;
    Group *vg, *cg;
    int *ints, *psdvar, *psdcon;
    double (*pow)[2];
    double objb;
    Ents obja, objf, a, b, f, h, d;
} Cbf;

/* the next significant line (no comments, no blanks), trimmed; NULL at EOF */
static char *line(FILE *fp, char *buf, int len) {
    while (fgets(buf, len, fp)) {
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        char *e = p + strlen(p);
        while (e > p && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
        if (*p == 0 || *p == '#') continue;
        return p;
    }
    return NULL;
}

static int read_groups(FILE *fp, char *buf, int *n, Group **g, int *ng) {
    char *l = line(fp, buf, 4096);
    if (!l || sscanf(l, "%d %d", n, ng) != 2) return 0;
    *g = (Group *)calloc((size_t)(*ng > 0 ? *ng : 1), sizeof(Group));
    for (int k = 0; k < *ng; k++) {
        if (!(l = line(fp, buf, 4096)) || sscanf(l, "%31s %d", (*g)[k].name, &(*g)[k].size) != 2) return 0;
    }
    return 1;
}

static int read_cbf(const char *path, Cbf *m) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    memset(m, 0, sizeof *m);
    char buf[4096], *l;
    int ok = 1;
    while (ok && (l = line(fp, buf, sizeof buf))) {
        char kw[32];
        snprintf(kw, sizeof kw, "%s", l);
        int cnt = 0;
        if (!strcmp(kw, "VER")) { line(fp, buf, sizeof buf); }
        else if (!strcmp(kw, "OBJSENSE")) { l = line(fp, buf, sizeof buf); m->max = l && !strcmp(l, "MAX"); }
        else if (!strcmp(kw, "VAR")) ok = read_groups(fp, buf, &m->nv, &m->vg, &m->nvg);
        else if (!strcmp(kw, "CON")) ok = read_groups(fp, buf, &m->ncon, &m->cg, &m->ncg);
        else if (!strcmp(kw, "POWCONES")) {
            int tot;
            l = line(fp, buf, sizeof buf);
            ok = l && sscanf(l, "%d %d", &m->npow, &tot) == 2;
            m->pow = calloc((size_t)(m->npow > 0 ? m->npow : 1), sizeof *m->pow);
            for (int k = 0; ok && k < m->npow; k++) {
                int len = atoi(line(fp, buf, sizeof buf));
                for (int q = 0; q < len; q++) {
                    double v = atof(line(fp, buf, sizeof buf));
                    if (q < 2) m->pow[k][q] = v;
                }
                if (len != 2) ok = 0;            /* only two-alpha power cones */
            }
        } else if (!strcmp(kw, "INT") || !strcmp(kw, "PSDVAR") || !strcmp(kw, "PSDCON")) {
            int n = atoi(line(fp, buf, sizeof buf));
            int *v = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
            for (int k = 0; k < n; k++) v[k] = atoi(line(fp, buf, sizeof buf));
            if (kw[0] == 'I') { m->ints = v; m->nint = n; }
            else if (kw[3] == 'V') { m->psdvar = v; m->npsdvar = n; }
            else { m->psdcon = v; m->npsdcon = n; }
        } else if (!strcmp(kw, "OBJBCOORD")) { m->objb = atof(line(fp, buf, sizeof buf)); }
        else if (!strcmp(kw, "OBJACOORD") || !strcmp(kw, "ACOORD") || !strcmp(kw, "BCOORD") ||
                 !strcmp(kw, "OBJFCOORD") || !strcmp(kw, "FCOORD") || !strcmp(kw, "HCOORD") ||
                 !strcmp(kw, "DCOORD")) {
            /* number of indices before the value, and where the entries go */
            int ni = !strcmp(kw, "OBJACOORD") || !strcmp(kw, "BCOORD") ? 1
                   : !strcmp(kw, "ACOORD") || !strcmp(kw, "OBJFCOORD") ? (kw[0] == 'A' ? 2 : 3)
                   : !strcmp(kw, "DCOORD") ? 3 : 4;
            Ents *s = !strcmp(kw, "OBJACOORD") ? &m->obja : !strcmp(kw, "ACOORD") ? &m->a
                    : !strcmp(kw, "BCOORD") ? &m->b : !strcmp(kw, "OBJFCOORD") ? &m->objf
                    : !strcmp(kw, "FCOORD") ? &m->f : !strcmp(kw, "HCOORD") ? &m->h : &m->d;
            cnt = atoi(line(fp, buf, sizeof buf));
            for (int k = 0; ok && k < cnt; k++) {
                int ix[4] = {0, 0, 0, 0};
                double v = 0.0;
                l = line(fp, buf, sizeof buf);
                char *p = l;
                for (int q = 0; q < ni; q++) ix[q] = (int)strtol(p, &p, 10);
                v = strtod(p, NULL);
                push(s, ix[0], ix[1], ix[2], ix[3], v);
            }
        } else ok = 0;                           /* CHANGE and unknown sections */
    }
    fclose(fp);
    return ok;
}

/* ----------------------------------------------------------------- model */
typedef struct {
    SCIP *scip;
    SCIP_VAR **x;                /* scalar variables */
    SCIP_VAR ***X;               /* PSDVAR entries: X[k][p*(p+1)/2+q], p >= q */
    int ncons;
} Model;

static SCIP_RETCODE addcons(Model *M, SCIP_CONS *c) {
    SCIP_CALL( SCIPaddCons(M->scip, c) );
    SCIP_CALL( SCIPreleaseCons(M->scip, &c) );
    M->ncons++;
    return SCIP_OKAY;
}

static SCIP_RETCODE newvar(Model *M, SCIP_VAR **v, double lb, double ub, double obj) {
    char nm[32];
    snprintf(nm, sizeof nm, "w%d", SCIPgetNVars(M->scip));
    SCIP_CALL( SCIPcreateVarBasic(M->scip, v, nm, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS) );
    SCIP_CALL( SCIPaddVar(M->scip, *v) );
    return SCIP_OKAY;
}

static SCIP_RETCODE exprvar(Model *M, SCIP_EXPR **e, SCIP_VAR *v) {
    return SCIPcreateExprVar(M->scip, e, v, NULL, NULL);
}

/* the cone `name` (size n) on the member variables v; isconst[k] marks
 * a member fixed to cval[k] (isconst[k]) */
static SCIP_RETCODE cone(Model *M, const Cbf *m, const char *name, int n, SCIP_VAR **v,
                         const int *isconst, const double *cval) {
    SCIP *scip = M->scip;
    SCIP_CONS *c;
    double inf = SCIPinfinity(scip);
    char nm[32];
    snprintf(nm, sizeof nm, "k%d", M->ncons);
    if (!strcmp(name, "Q")) {
        if (n == 1) {
            SCIP_CALL( SCIPchgVarLb(scip, v[0], 0.0) );
            return SCIP_OKAY;
        }
        SCIP_CALL( SCIPchgVarLb(scip, v[0], 0.0) );
        SCIP_CALL( SCIPcreateConsBasicSOCNonlinear(scip, &c, nm, n - 1, v + 1, NULL, NULL, 0.0,
                                                   v[0], 1.0, 0.0) );
        return addcons(M, c);
    }
    if (!strcmp(name, "QR")) {                   /* 2 u0 u1 >= ||w||^2, u0, u1 >= 0 */
        SCIP_VAR *s, *dlt, **lhs = (SCIP_VAR **)malloc((size_t)(n - 1) * sizeof(SCIP_VAR *));
        double *coef = (double *)malloc((size_t)(n - 1) * sizeof(double));
        SCIP_CALL( SCIPchgVarLb(scip, v[0], 0.0) );
        SCIP_CALL( SCIPchgVarLb(scip, v[1], 0.0) );
        SCIP_CALL( newvar(M, &s, 0.0, inf, 0.0) );
        SCIP_CALL( newvar(M, &dlt, -inf, inf, 0.0) );
        SCIP_VAR *ls[3] = {s, v[0], v[1]};
        double cs[3] = {1.0, -1.0, -1.0}, cd[3] = {1.0, -1.0, 1.0};
        SCIP_CALL( SCIPcreateConsBasicLinear(scip, &c, "qrs", 3, ls, cs, 0.0, 0.0) );
        SCIP_CALL( addcons(M, c) );
        ls[0] = dlt;
        SCIP_CALL( SCIPcreateConsBasicLinear(scip, &c, "qrd", 3, ls, cd, 0.0, 0.0) );
        SCIP_CALL( addcons(M, c) );
        for (int k = 2; k < n; k++) { lhs[k - 2] = v[k]; coef[k - 2] = sqrt(2.0); }
        lhs[n - 2] = dlt; coef[n - 2] = 1.0;
        SCIP_CALL( SCIPcreateConsBasicSOCNonlinear(scip, &c, nm, n - 1, lhs, coef, NULL, 0.0, s, 1.0, 0.0) );
        SCIP_CALL( SCIPreleaseVar(scip, &s) );
        SCIP_CALL( SCIPreleaseVar(scip, &dlt) );
        free(lhs); free(coef);
        return addcons(M, c);
    }
    if (!strcmp(name, "EXP")) {                  /* u0 >= u1 exp(u2/u1) */
        SCIP_EXPR *e0, *e2, *ex, *prod, *sum;
        SCIP_CALL( exprvar(M, &e0, v[0]) );
        SCIP_CALL( exprvar(M, &e2, v[2]) );
        if (isconst[1] && cval[1] > 0.0) {       /* c exp(u2/c) - u0 <= 0 */
            SCIP_EXPR *sc, *ch[2];
            double cf = 1.0 / cval[1];
            SCIP_CALL( SCIPcreateExprSum(scip, &sc, 1, &e2, &cf, 0.0, NULL, NULL) );
            SCIP_CALL( SCIPcreateExprExp(scip, &ex, sc, NULL, NULL) );
            ch[0] = ex; ch[1] = e0;
            double w[2] = {cval[1], -1.0};
            SCIP_CALL( SCIPcreateExprSum(scip, &sum, 2, ch, w, 0.0, NULL, NULL) );
            SCIP_CALL( SCIPreleaseExpr(scip, &sc) );
        } else if (isconst[1]) {                 /* closure at u1 = 0: u0 >= 0, u2 <= 0 */
            SCIP_CALL( SCIPchgVarLb(scip, v[0], 0.0) );
            SCIP_CALL( SCIPchgVarUb(scip, v[2], 0.0) );
            SCIP_CALL( SCIPreleaseExpr(scip, &e0) );
            SCIP_CALL( SCIPreleaseExpr(scip, &e2) );
            return SCIP_OKAY;
        } else if (isconst[0] && cval[0] > 0.0 && !getenv("SCIP_CBF_PERSPECTIVE")) {
            /* c >= u1 exp(u2/u1) with c constant  <=>  u2 <= u1 ln(c/u1):
             * u2 - u1 ln c - entropy(u1) <= 0, entropy(u1) = -u1 ln u1 -- exact
             * (closure included: at u1 = 0 it reads u2 <= 0) and convex to
             * SCIP, which knows the entropy expression */
            SCIP_EXPR *e1, *ent, *ch[3];
            double w[3] = {1.0, -log(cval[0]), -1.0};
            SCIP_CALL( SCIPchgVarLb(scip, v[1], 0.0) );
            SCIP_CALL( exprvar(M, &e1, v[1]) );
            SCIP_CALL( SCIPcreateExprEntropy(scip, &ent, e1, NULL, NULL) );
            ch[0] = e2; ch[1] = e1; ch[2] = ent;
            SCIP_CALL( SCIPcreateExprSum(scip, &sum, 3, ch, w, 0.0, NULL, NULL) );
            SCIP_CALL( SCIPcreateExprVar(scip, &ex, v[1], NULL, NULL) );   /* released below */
            SCIP_CALL( SCIPreleaseExpr(scip, &ent) );
            SCIP_CALL( SCIPreleaseExpr(scip, &e1) );
        } else {                                 /* (u1+e) exp(u2/(u1+e)) - e - u0 <= 0 */
            SCIP_EXPR *e1, *pe, *inv, *ratio, *ch[2];
            double one = 1.0;
            SCIP_CALL( SCIPchgVarLb(scip, v[1], 0.0) );
            SCIP_CALL( exprvar(M, &e1, v[1]) );
            SCIP_CALL( SCIPcreateExprSum(scip, &pe, 1, &e1, &one, EPS_PERSPECTIVE, NULL, NULL) );
            SCIP_CALL( SCIPcreateExprPow(scip, &inv, pe, -1.0, NULL, NULL) );
            ch[0] = e2; ch[1] = inv;
            SCIP_CALL( SCIPcreateExprProduct(scip, &ratio, 2, ch, 1.0, NULL, NULL) );
            SCIP_CALL( SCIPcreateExprExp(scip, &ex, ratio, NULL, NULL) );
            ch[0] = pe; ch[1] = ex;
            SCIP_CALL( SCIPcreateExprProduct(scip, &prod, 2, ch, 1.0, NULL, NULL) );
            ch[0] = prod; ch[1] = e0;
            double w[2] = {1.0, -1.0};
            SCIP_CALL( SCIPcreateExprSum(scip, &sum, 2, ch, w, -EPS_PERSPECTIVE, NULL, NULL) );
            SCIP_CALL( SCIPreleaseExpr(scip, &prod) );
            SCIP_CALL( SCIPreleaseExpr(scip, &ratio) );
            SCIP_CALL( SCIPreleaseExpr(scip, &inv) );
            SCIP_CALL( SCIPreleaseExpr(scip, &pe) );
            SCIP_CALL( SCIPreleaseExpr(scip, &e1) );
        }
        SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &c, nm, sum, -inf, 0.0) );
        SCIP_CALL( SCIPreleaseExpr(scip, &sum) );
        SCIP_CALL( SCIPreleaseExpr(scip, &ex) );
        SCIP_CALL( SCIPreleaseExpr(scip, &e2) );
        SCIP_CALL( SCIPreleaseExpr(scip, &e0) );
        return addcons(M, c);
    }
    if (name[0] == '@') {                        /* @k:POW, u0^a u1^(1-a) >= ||u2..|| */
        int k = atoi(name + 1);
        double a = m->pow[k][0] / (m->pow[k][0] + m->pow[k][1]);
        SCIP_EXPR *b0, *b1, *p0, *p1, *ch[2], *g;
        SCIP_CALL( SCIPchgVarLb(scip, v[0], 0.0) );
        SCIP_CALL( SCIPchgVarLb(scip, v[1], 0.0) );
        SCIP_CALL( exprvar(M, &b0, v[0]) );
        SCIP_CALL( exprvar(M, &b1, v[1]) );
        SCIP_CALL( SCIPcreateExprPow(scip, &p0, b0, a, NULL, NULL) );
        SCIP_CALL( SCIPcreateExprPow(scip, &p1, b1, 1.0 - a, NULL, NULL) );
        ch[0] = p0; ch[1] = p1;
        SCIP_CALL( SCIPcreateExprProduct(scip, &g, 2, ch, 1.0, NULL, NULL) );
        if (n == 3) {                            /* g - u2 >= 0 and g + u2 >= 0 */
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                SCIP_EXPR *e2, *s, *cc[2];
                double w[2] = {1.0, (double)sgn};
                SCIP_CALL( exprvar(M, &e2, v[2]) );
                cc[0] = g; cc[1] = e2;
                SCIP_CALL( SCIPcreateExprSum(scip, &s, 2, cc, w, 0.0, NULL, NULL) );
                snprintf(nm, sizeof nm, "k%d", M->ncons);
                SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &c, nm, s, 0.0, inf) );
                SCIP_CALL( SCIPreleaseExpr(scip, &s) );
                SCIP_CALL( SCIPreleaseExpr(scip, &e2) );
                SCIP_CALL( addcons(M, c) );
            }
        } else {                                 /* sqrt(sum u_k^2) - g <= 0 */
            SCIP_EXPR **sq = (SCIP_EXPR **)malloc((size_t)(n - 2) * sizeof(SCIP_EXPR *));
            SCIP_EXPR *ss, *rt, *s, *cc[2];
            for (int q = 2; q < n; q++) {
                SCIP_EXPR *eq;
                SCIP_CALL( exprvar(M, &eq, v[q]) );
                SCIP_CALL( SCIPcreateExprPow(scip, &sq[q - 2], eq, 2.0, NULL, NULL) );
                SCIP_CALL( SCIPreleaseExpr(scip, &eq) );
            }
            SCIP_CALL( SCIPcreateExprSum(scip, &ss, n - 2, sq, NULL, 0.0, NULL, NULL) );
            SCIP_CALL( SCIPcreateExprPow(scip, &rt, ss, 0.5, NULL, NULL) );
            cc[0] = rt; cc[1] = g;
            double w[2] = {1.0, -1.0};
            SCIP_CALL( SCIPcreateExprSum(scip, &s, 2, cc, w, 0.0, NULL, NULL) );
            SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, &c, nm, s, -inf, 0.0) );
            SCIP_CALL( SCIPreleaseExpr(scip, &s) );
            SCIP_CALL( SCIPreleaseExpr(scip, &rt) );
            SCIP_CALL( SCIPreleaseExpr(scip, &ss) );
            for (int q = 0; q < n - 2; q++) SCIP_CALL( SCIPreleaseExpr(scip, &sq[q]) );
            free(sq);
            SCIP_CALL( addcons(M, c) );
        }
        SCIP_CALL( SCIPreleaseExpr(scip, &g) );
        SCIP_CALL( SCIPreleaseExpr(scip, &p1) );
        SCIP_CALL( SCIPreleaseExpr(scip, &p0) );
        SCIP_CALL( SCIPreleaseExpr(scip, &b1) );
        SCIP_CALL( SCIPreleaseExpr(scip, &b0) );
        return SCIP_OKAY;
    }
    return SCIP_INVALIDDATA;                     /* EXP*, POW*, ... */
}

static int tri(int p, int q) { return p * (p + 1) / 2 + q; }   /* p >= q */

/* one SDP constraint  sum_i A_i y_i - A_0 >= 0  from per-variable lower-
 * triangle entry lists; entries (var, p, q, v) with var -1 for A_0 */
static SCIP_RETCODE sdpcons(Model *M, int dim, int nvars, SCIP_VAR **vars, const Ents *ent) {
    int *cnt = (int *)calloc((size_t)nvars + 1, sizeof(int));
    for (int k = 0; k < ent->n; k++) cnt[ent->e[k].a + 1]++;
    int **col = (int **)malloc((size_t)(nvars > 0 ? nvars : 1) * sizeof(int *));
    int **row = (int **)malloc((size_t)(nvars > 0 ? nvars : 1) * sizeof(int *));
    double **val = (double **)malloc((size_t)(nvars > 0 ? nvars : 1) * sizeof(double *));
    int *nnz = (int *)calloc((size_t)(nvars > 0 ? nvars : 1), sizeof(int));
    for (int i = 0; i < nvars; i++) {
        col[i] = (int *)malloc((size_t)(cnt[i + 1] + 1) * sizeof(int));
        row[i] = (int *)malloc((size_t)(cnt[i + 1] + 1) * sizeof(int));
        val[i] = (double *)malloc((size_t)(cnt[i + 1] + 1) * sizeof(double));
    }
    int *ccol = (int *)malloc((size_t)(cnt[0] + 1) * sizeof(int));
    int *crow = (int *)malloc((size_t)(cnt[0] + 1) * sizeof(int));
    double *cval = (double *)malloc((size_t)(cnt[0] + 1) * sizeof(double));
    int nc = 0, tot = 0;
    for (int k = 0; k < ent->n; k++) {
        const Ent *e = &ent->e[k];
        int p = e->b > e->c ? e->b : e->c, q = e->b > e->c ? e->c : e->b;
        if (e->a < 0) { crow[nc] = p; ccol[nc] = q; cval[nc] = e->d ? -e->v : e->v; nc++; }
        else { int i = e->a; row[i][nnz[i]] = p; col[i][nnz[i]] = q; val[i][nnz[i]] = e->v; nnz[i]++; tot++; }
    }
    /* drop variables with no entry */
    int w = 0;
    for (int i = 0; i < nvars; i++) {
        if (nnz[i] == 0) { free(col[i]); free(row[i]); free(val[i]); continue; }
        col[w] = col[i]; row[w] = row[i]; val[w] = val[i]; nnz[w] = nnz[i]; vars[w] = vars[i]; w++;
    }
    SCIP_CONS *c;
    char nm[32];
    snprintf(nm, sizeof nm, "sdp%d", M->ncons);
    SCIP_CALL( SCIPcreateConsSdp(M->scip, &c, nm, w, tot, dim, nnz, col, row, val, vars,
                                 nc, ccol, crow, cval, TRUE) );
    SCIP_CALL( addcons(M, c) );
    for (int i = 0; i < w; i++) { free(col[i]); free(row[i]); free(val[i]); }
    free(col); free(row); free(val); free(nnz); free(cnt); free(ccol); free(crow); free(cval);
    return SCIP_OKAY;
}

static SCIP_RETCODE build(Model *M, const Cbf *m) {
    SCIP *scip = M->scip;
    double inf = SCIPinfinity(scip);
    char nm[64];
    char *isint = (char *)calloc((size_t)(m->nv > 0 ? m->nv : 1), 1);
    for (int k = 0; k < m->nint; k++) isint[m->ints[k]] = 1;
    double *cobj = (double *)calloc((size_t)(m->nv > 0 ? m->nv : 1), sizeof(double));
    for (int k = 0; k < m->obja.n; k++) cobj[m->obja.e[k].a] += m->obja.e[k].v;

    /* scalar variables, bounded by their VAR group */
    M->x = (SCIP_VAR **)calloc((size_t)(m->nv > 0 ? m->nv : 1), sizeof(SCIP_VAR *));
    for (int g = 0, j0 = 0; g < m->nvg; j0 += m->vg[g].size, g++) {
        const char *t = m->vg[g].name;
        double lb = !strcmp(t, "L+") || !strcmp(t, "L=") ? 0.0 : -inf;
        double ub = !strcmp(t, "L-") || !strcmp(t, "L=") ? 0.0 : inf;
        for (int j = j0; j < j0 + m->vg[g].size; j++) {
            snprintf(nm, sizeof nm, "x%d", j);
            SCIP_CALL( SCIPcreateVarBasic(scip, &M->x[j], nm, lb, ub, cobj[j],
                                          isint[j] ? SCIP_VARTYPE_INTEGER : SCIP_VARTYPE_CONTINUOUS) );
            SCIP_CALL( SCIPaddVar(scip, M->x[j]) );
        }
    }
    /* PSDVAR entries, with their objective <F, X> (off-diagonals count twice) */
    M->X = (SCIP_VAR ***)calloc((size_t)(m->npsdvar > 0 ? m->npsdvar : 1), sizeof(SCIP_VAR **));
    for (int k = 0; k < m->npsdvar; k++) {
        int n = m->psdvar[k], L = n * (n + 1) / 2;
        double *o = (double *)calloc((size_t)L, sizeof(double));
        for (int t = 0; t < m->objf.n; t++) {
            const Ent *e = &m->objf.e[t];
            if (e->a != k) continue;
            int p = e->b > e->c ? e->b : e->c, q = e->b > e->c ? e->c : e->b;
            o[tri(p, q)] += p == q ? e->v : 2.0 * e->v;
        }
        M->X[k] = (SCIP_VAR **)calloc((size_t)L, sizeof(SCIP_VAR *));
        for (int p = 0; p < n; p++)
            for (int q = 0; q <= p; q++) {
                snprintf(nm, sizeof nm, "X%d_%d_%d", k, p, q);
                SCIP_CALL( SCIPcreateVarBasic(scip, &M->X[k][tri(p, q)], nm, p == q ? 0.0 : -inf,
                                              inf, o[tri(p, q)], SCIP_VARTYPE_CONTINUOUS) );
                SCIP_CALL( SCIPaddVar(scip, M->X[k][tri(p, q)]) );
            }
        /* X_k >= 0: one SDP constraint with E_pq on each entry variable */
        Ents en = {0};
        for (int p = 0; p < n; p++)
            for (int q = 0; q <= p; q++) push(&en, tri(p, q), p, q, 0, 1.0);
        SCIP_CALL( sdpcons(M, n, L, M->X[k], &en) );
        free(en.e); free(o);
    }
    /* VAR cones */
    for (int g = 0, j0 = 0; g < m->nvg; j0 += m->vg[g].size, g++) {
        const char *t = m->vg[g].name;
        if (!strcmp(t, "F") || !strcmp(t, "L+") || !strcmp(t, "L-") || !strcmp(t, "L=")) continue;
        int *zero = (int *)calloc((size_t)m->vg[g].size, sizeof(int));
        double *zv = (double *)calloc((size_t)m->vg[g].size, sizeof(double));
        SCIP_CALL( cone(M, m, t, m->vg[g].size, M->x + j0, zero, zv) );
        free(zero); free(zv);
    }
    /* CON rows: coefficient lists per row (scalar part, then PSDVAR entries) */
    int nr = m->ncon;
    int *rc = (int *)calloc((size_t)nr + 1, sizeof(int));
    double *rb = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    for (int k = 0; k < m->a.n; k++) rc[m->a.e[k].a]++;
    for (int k = 0; k < m->f.n; k++) rc[m->f.e[k].a]++;
    for (int k = 0; k < m->b.n; k++) rb[m->b.e[k].a] += m->b.e[k].v;
    SCIP_VAR ***rv = (SCIP_VAR ***)calloc((size_t)(nr > 0 ? nr : 1), sizeof(SCIP_VAR **));
    double **rw = (double **)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double *));
    int *rn = (int *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(int));
    for (int r = 0; r < nr; r++) {
        rv[r] = (SCIP_VAR **)malloc((size_t)(rc[r] + 2) * sizeof(SCIP_VAR *));
        rw[r] = (double *)malloc((size_t)(rc[r] + 2) * sizeof(double));
    }
    for (int k = 0; k < m->a.n; k++) {
        int r = m->a.e[k].a;
        rv[r][rn[r]] = M->x[m->a.e[k].b]; rw[r][rn[r]++] = m->a.e[k].v;
    }
    for (int k = 0; k < m->f.n; k++) {          /* (row, psdvar, p, q, v) */
        const Ent *e = &m->f.e[k];
        int r = e->a, p = e->c > e->d ? e->c : e->d, q = e->c > e->d ? e->d : e->c;
        rv[r][rn[r]] = M->X[e->b][tri(p, q)]; rw[r][rn[r]++] = p == q ? e->v : 2.0 * e->v;
    }
    for (int g = 0, r0 = 0; g < m->ncg; r0 += m->cg[g].size, g++) {
        const char *t = m->cg[g].name;
        int sz = m->cg[g].size;
        if (!strcmp(t, "F")) continue;
        if (!strcmp(t, "L+") || !strcmp(t, "L-") || !strcmp(t, "L=")) {
            for (int r = r0; r < r0 + sz; r++) {
                SCIP_CONS *c;
                double lhs = t[1] == '-' ? -inf : -rb[r], rhs = t[1] == '+' ? inf : -rb[r];
                snprintf(nm, sizeof nm, "r%d", r);
                SCIP_CALL( SCIPcreateConsBasicLinear(scip, &c, nm, rn[r], rv[r], rw[r], lhs, rhs) );
                SCIP_CALL( addcons(M, c) );
            }
            continue;
        }
        /* Q on rows: a member that is one variable, a x + b, enters SCIP's SOC
         * constraint directly (coefficient a, offset b/a) and a constant one
         * adds b^2 under the root; only a member over several variables gets a
         * variable of its own. */
        if (!strcmp(t, "Q") && sz > 1) {
            SCIP_VAR **lv = (SCIP_VAR **)malloc((size_t)sz * sizeof(SCIP_VAR *));
            SCIP_VAR **made = (SCIP_VAR **)malloc((size_t)sz * sizeof(SCIP_VAR *));
            double *la = (double *)malloc((size_t)sz * sizeof(double));
            double *lo = (double *)malloc((size_t)sz * sizeof(double));
            double gam = 0.0, ra = 1.0, ro = 0.0;
            SCIP_VAR *rhs = NULL;
            int nl = 0, nmade = 0;
            for (int q = 0; q < sz; q++) {
                int r = r0 + q;
                SCIP_VAR *xv = NULL;
                double a = 1.0, off = 0.0;
                if (rn[r] == 1 && rw[r][0] != 0.0) {
                    xv = rv[r][0]; a = rw[r][0]; off = rb[r] / a;
                } else if (rn[r] == 0 && q > 0) {
                    gam += rb[r] * rb[r];
                    continue;
                } else {                     /* a variable for the row */
                    SCIP_CONS *c;
                    SCIP_CALL( newvar(M, &xv, rn[r] == 0 ? rb[r] : -inf, rn[r] == 0 ? rb[r] : inf, 0.0) );
                    made[nmade++] = xv;
                    rv[r][rn[r]] = xv; rw[r][rn[r]] = -1.0;
                    snprintf(nm, sizeof nm, "r%d", r);
                    SCIP_CALL( SCIPcreateConsBasicLinear(scip, &c, nm, rn[r] + 1, rv[r], rw[r], -rb[r], -rb[r]) );
                    SCIP_CALL( addcons(M, c) );
                }
                if (q == 0) { rhs = xv; ra = a; ro = off; }
                else { lv[nl] = xv; la[nl] = a; lo[nl] = off; nl++; }
            }
            SCIP_CONS *c;
            snprintf(nm, sizeof nm, "k%d", M->ncons);
            SCIP_CALL( SCIPcreateConsBasicSOCNonlinear(scip, &c, nm, nl, lv, la, lo, gam, rhs, ra, ro) );
            SCIP_CALL( addcons(M, c) );
            for (int q = 0; q < nmade; q++) SCIP_CALL( SCIPreleaseVar(scip, &made[q]) );
            free(lv); free(made); free(la); free(lo);
            continue;
        }
        /* any other cone on rows: one variable per member, u_r = A x + <F, X> + b */
        SCIP_VAR **u = (SCIP_VAR **)malloc((size_t)sz * sizeof(SCIP_VAR *));
        int *isc = (int *)calloc((size_t)sz, sizeof(int));
        double *cv = (double *)calloc((size_t)sz, sizeof(double));
        for (int q = 0; q < sz; q++) {
            int r = r0 + q;
            SCIP_CONS *c;
            isc[q] = rn[r] == 0;
            cv[q] = rb[r];
            SCIP_CALL( newvar(M, &u[q], isc[q] ? rb[r] : -inf, isc[q] ? rb[r] : inf, 0.0) );
            rv[r][rn[r]] = u[q]; rw[r][rn[r]] = -1.0;
            snprintf(nm, sizeof nm, "r%d", r);
            SCIP_CALL( SCIPcreateConsBasicLinear(scip, &c, nm, rn[r] + 1, rv[r], rw[r], -rb[r], -rb[r]) );
            SCIP_CALL( addcons(M, c) );
        }
        SCIP_CALL( cone(M, m, t, sz, u, isc, cv) );
        for (int q = 0; q < sz; q++) SCIP_CALL( SCIPreleaseVar(scip, &u[q]) );
        free(u); free(isc); free(cv);
    }
    /* PSDCON c:  sum_j H_j x_j + D >= 0, i.e. A_j = H_j, A_0 = -D */
    for (int c = 0; c < m->npsdcon; c++) {
        Ents en = {0};
        for (int k = 0; k < m->h.n; k++)
            if (m->h.e[k].a == c) push(&en, m->h.e[k].b, m->h.e[k].c, m->h.e[k].d, 0, m->h.e[k].v);
        for (int k = 0; k < m->d.n; k++)
            if (m->d.e[k].a == c) push(&en, -1, m->d.e[k].b, m->d.e[k].c, 1, m->d.e[k].v);
        SCIP_VAR **vs = (SCIP_VAR **)malloc((size_t)(m->nv > 0 ? m->nv : 1) * sizeof(SCIP_VAR *));
        memcpy(vs, M->x, (size_t)m->nv * sizeof(SCIP_VAR *));
        SCIP_CALL( sdpcons(M, m->psdcon[c], m->nv, vs, &en) );
        free(vs); free(en.e);
    }
    if (m->objb != 0.0) SCIP_CALL( SCIPaddOrigObjoffset(scip, m->objb) );
    SCIP_CALL( SCIPsetObjsense(scip, m->max ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE) );
    for (int r = 0; r < nr; r++) { free(rv[r]); free(rw[r]); }
    free(rv); free(rw); free(rn); free(rc); free(rb); free(isint); free(cobj);
    return SCIP_OKAY;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FILE.cbf [TIMELIMIT]\n", argv[0]); return 2; }
    Cbf m;
    if (!read_cbf(argv[1], &m)) { printf("unsupported,,\n"); return 1; }
    Model M = {0};
    SCIP_CALL( SCIPcreate(&M.scip) );
    int psd = m.npsdvar > 0 || m.npsdcon > 0, nonlin = 0;
    for (int g = 0; g < m.nvg; g++) nonlin |= strcmp(m.vg[g].name, "F") && m.vg[g].name[0] != 'L';
    for (int g = 0; g < m.ncg; g++) nonlin |= strcmp(m.cg[g].name, "F") && m.cg[g].name[0] != 'L';
    if (psd) SCIP_CALL( SCIPSDPincludeDefaultPlugins(M.scip) );
    else SCIP_CALL( SCIPincludeDefaultPlugins(M.scip) );
    if (psd) {
        /* SCIP-SDP's default feasibility tolerance (1e-5) admits points that
         * are not feasible in these models -- near-feasible optima of
         * infeasible SDPs, 0.7% too good on CBLIB port -- 1e-7 does not */
        SCIP_CALL( SCIPsetRealParam(M.scip, "numerics/feastol", 1e-7) );
        SCIP_CALL( SCIPsetRealParam(M.scip, "relaxing/SDP/sdpsolverfeastol", 1e-7) );
        /* with nonlinear (cone) constraints as well, the SDP relaxations would
         * hold the linear rows and the SDP blocks only -- the nonlinear ones
         * reach them as cuts that, on port, never lift the root bound -- so
         * SCIP-SDP's LP approach: SDP blocks by eigenvector cuts, everything
         * in one outer approximation */
        if (nonlin) SCIP_CALL( SCIPsetIntParam(M.scip, "misc/solvesdps", 0) );
    }
    if (!getenv("SCIP_CBF_VERBOSE")) SCIPsetMessagehdlrQuiet(M.scip, TRUE);
    if (argc > 2) SCIP_CALL( SCIPsetRealParam(M.scip, "limits/time", atof(argv[2])) );
    /* SCIP_CBF_SET=file: extra SCIP parameters (diagnostics) */
    if (getenv("SCIP_CBF_SET")) SCIP_CALL( SCIPreadParams(M.scip, getenv("SCIP_CBF_SET")) );
    SCIP_CALL( SCIPcreateProbBasic(M.scip, argv[1]) );
    SCIP_RETCODE rc = build(&M, &m);
    if (rc != SCIP_OKAY) { printf("unsupported,,\n"); return 1; }
    SCIP_CALL( SCIPsolve(M.scip) );
    SCIP_STATUS st = SCIPgetStatus(M.scip);
    double t = SCIPgetSolvingTime(M.scip);
    if (getenv("SCIP_CBF_SOL") && SCIPgetNSols(M.scip) > 0) {   /* x for checking */
        FILE *sf = fopen(getenv("SCIP_CBF_SOL"), "w");
        SCIP_SOL *sol = SCIPgetBestSol(M.scip);
        for (int j = 0; sf && j < m.nv; j++) fprintf(sf, "%.17g\n", SCIPgetSolVal(M.scip, sol, M.x[j]));
        if (sf) fclose(sf);
    }
    if (st == SCIP_STATUS_OPTIMAL && SCIPgetNSols(M.scip) > 0)
        printf("ok,%.6f,%.12g\n", t, SCIPgetPrimalbound(M.scip));
    else if (st == SCIP_STATUS_INFEASIBLE)
        printf("infeasible,%.6f,\n", t);
    else if (st == SCIP_STATUS_TIMELIMIT)
        printf("timeout,,\n");
    else
        printf("n/a,,\n");
    for (int j = 0; j < m.nv; j++) SCIP_CALL( SCIPreleaseVar(M.scip, &M.x[j]) );
    for (int k = 0; k < m.npsdvar; k++) {
        for (int i = 0; i < m.psdvar[k] * (m.psdvar[k] + 1) / 2; i++)
            SCIP_CALL( SCIPreleaseVar(M.scip, &M.X[k][i]) );
        free(M.X[k]);
    }
    free(M.X); free(M.x);
    SCIP_CALL( SCIPfree(&M.scip) );
    return 0;
}
