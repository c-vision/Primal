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

/* make_figures.c - writes the figures of algorithms/ that are computed:
 *
 *   simplex-pivots.svg       analytic   feasible polygon, level lines, the pivots of a
 *                                       textbook Dantzig tableau run (computed here)
 *   lp-central-path.svg      measured   iterates and residuals of the dense LP interior
 *                                       point engine (ipm_solve_std) on the same LP
 *   lp-central-path-trace.csv           the numbers behind lp-central-path.svg
 *   qp-contours.svg          analytic   contours of a convex QP and its active constraint
 *   presolve-reduction.svg   computed   the rows and columns removed by lp_presolve on a 3x3 LP and the
 *                                       multipliers lifted back by presolve_postsolve_dir
 *   cholesky-ordering.svg    measured   the pattern of an arrowhead matrix and of its Cholesky factor with
 *                                       the natural order and with AMD (spchol_factor, spchol_factor_ord)
 *   ordering-fill.csv                   the nonzeros of the factors for the arrowhead and for 2-D grids
 *   elliptope-slices.svg     analytic   three slices of the 3x3 unit-diagonal PSD set and the line that touches the best one
 *   exp-cone-section.svg     analytic   sections of t >= u exp(v/u) and the optimum of min t s.t. u + v = 2
 *   power-cone-section.svg   analytic   sections u = 1 of the power cones and the optimum of min t with u = 1, v = 2
 *   supporting-hyperplane.svg analytic  the optimal vertex of the same LP, its normal cone and the
 *                                       supporting lines that touch the polygon there
 *
 * Every figure is checked before it is written: the pivot run and the interior point
 * must reach the optimum the public API reports, and the QP optimum must satisfy the
 * KKT conditions.  The program exits non-zero if a check fails.
 *
 * Build and run from this directory (needs out/libprimal.a, built by `make`):
 *   cc -std=c99 -O2 -I../.. -o /tmp/make_figures make_figures.c ../../out/libprimal.a -lm -lpthread
 *   /tmp/make_figures <git revision shown in the captions>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "primal.h"
#include "ipm.h"
#include "presolve.h"
#include "linalg.h"
#include "expcone.h"

#define INK "#17212b"
#define BLUE "#155097"
#define FILL "#dbeafe"
#define ORANGE "#b45309"
#define GREY "#6b7280"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static const char *g_rev = "unknown";

/* ------------------------------------------------------------------------- */
/* Small SVG toolkit: a data window mapped to a pixel rectangle.              */

typedef struct { double x0, x1, y0, y1, px, py, pw, ph; } Plot;

static double PX(const Plot *p, double x) { return p->px + (x - p->x0) / (p->x1 - p->x0) * p->pw; }
static double PY(const Plot *p, double y) { return p->py + p->ph - (y - p->y0) / (p->y1 - p->y0) * p->ph; }

/* write s with the XML special characters escaped */
static void xml_text(FILE *f, const char *s) {
    for (; *s; s++) {
        if (*s == '<') fputs("&lt;", f);
        else if (*s == '>') fputs("&gt;", f);
        else if (*s == '&') fputs("&amp;", f);
        else fputc(*s, f);
    }
}

static void svg_begin(FILE *f, int w, int h, const char *title, const char *desc) {
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" "
               "viewBox=\"0 0 %d %d\" role=\"img\" aria-labelledby=\"title desc\">\n", w, h, w, h);
    fputs("  <title id=\"title\">", f); xml_text(f, title); fputs("</title>\n  <desc id=\"desc\">", f);
    xml_text(f, desc); fputs("</desc>\n", f);
    fprintf(f, "  <defs>\n"
               "    <marker id=\"arr\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" "
               "markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0 0L10 5L0 10z\" fill=\"%s\"/></marker>\n"
               "    <marker id=\"arrb\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" "
               "markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0 0L10 5L0 10z\" fill=\"%s\"/></marker>\n"
               "  </defs>\n", INK, ORANGE);
    fprintf(f, "  <rect width=\"%d\" height=\"%d\" fill=\"#fff\"/>\n", w, h);
}

static void svg_end(FILE *f) { fputs("</svg>\n", f); }

static void text(FILE *f, double x, double y, const char *anchor, int size, const char *fill, const char *s) {
    fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"%s\" font-family=\"sans-serif\" font-size=\"%d\" fill=\"%s\" stroke=\"#fff\" stroke-width=\"3.5\" "
               "paint-order=\"stroke\" stroke-linejoin=\"round\">", x, y, anchor, size, fill);
    xml_text(f, s);
    fputs("</text>\n", f);
}

static void line(FILE *f, double x1, double y1, double x2, double y2, const char *stroke,
                 double w, const char *dash, const char *marker) {
    fprintf(f, "  <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"%.1f\"",
            x1, y1, x2, y2, stroke, w);
    if (dash) fprintf(f, " stroke-dasharray=\"%s\"", dash);
    if (marker) fprintf(f, " marker-end=\"url(#%s)\"", marker);
    fputs("/>\n", f);
}

/* axes with integer ticks; xl/yl are the axis names */
static void axes(FILE *f, const Plot *p, const char *xl, const char *yl, int xstep, int ystep) {
    line(f, p->px, p->py + p->ph, p->px + p->pw + 14, p->py + p->ph, INK, 1.5, NULL, "arr");
    line(f, p->px, p->py + p->ph, p->px, p->py - 14, INK, 1.5, NULL, "arr");
    for (int v = (int)ceil(p->x0); v <= (int)floor(p->x1); v += xstep) {
        char s[16]; snprintf(s, sizeof s, "%d", v);
        line(f, PX(p, v), p->py + p->ph, PX(p, v), p->py + p->ph + 5, INK, 1, NULL, NULL);
        text(f, PX(p, v), p->py + p->ph + 20, "middle", 14, INK, s);
    }
    for (int v = (int)ceil(p->y0); v <= (int)floor(p->y1); v += ystep) {
        char s[16]; snprintf(s, sizeof s, "%d", v);
        line(f, p->px - 5, PY(p, v), p->px, PY(p, v), INK, 1, NULL, NULL);
        text(f, p->px - 9, PY(p, v) + 5, "end", 14, INK, s);
    }
    text(f, p->px + p->pw + 8, p->py + p->ph + 22, "start", 16, INK, xl);
    text(f, p->px - 6, p->py - 18, "middle", 16, INK, yl);
}

/* ------------------------------------------------------------------------- */
/* The LP shared by the simplex and interior-point chapters:                   */
/*   maximize 3x + 2y   s.t.  x + y <= 4,  x <= 2,  x, y >= 0                */

#define LP_M 2
#define LP_N 4        /* x, y and two slacks */
static const double LP_A[LP_M * LP_N] = { 1, 1, 1, 0,
                                          1, 0, 0, 1 };
static const double LP_B[LP_M] = { 4, 2 };
static const double LP_C[LP_N] = { -3, -2, 0, 0 };   /* minimization form */

/* Reference answer from the public API; the figures are checked against it. */
static double g_ref_y[LP_M];   /* row multipliers reported by PRIMAL for the reference LP */

static void lp_reference(double *x, double *y, double *obj) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 2); PRIMAL_appendcons(t, 2);
    for (int j = 0; j < 2; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, 0, 3.0); PRIMAL_putcj(t, 1, 2.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 4.0);
    PRIMAL_putarow(t, 1, 1, (int[]){0}, (double[]){1});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_UP, -INFINITY, 2.0);
    if (PRIMAL_optimize(t) != PRIMAL_RES_OK) { fputs("lp_reference: optimize failed\n", stderr); exit(1); }
    double xx[2];
    PRIMAL_getxx(t, PRIMAL_SOL_BAS, xx);
    PRIMAL_gety(t, PRIMAL_SOL_BAS, g_ref_y);
    PRIMAL_getprimalobj(t, PRIMAL_SOL_BAS, obj);
    *x = xx[0]; *y = xx[1];
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
}

static void fail(const char *msg) { fprintf(stderr, "make_figures: check failed: %s\n", msg); exit(1); }

/* Vertices of the feasible polygon from all pairs of boundary lines. */
static int polygon(double v[][2], int maxv) {
    /* lines a*x + b*y = c, with the feasible side a*x + b*y <= c */
    static const double L[4][3] = { { -1, 0, 0 }, { 0, -1, 0 }, { 1, 1, 4 }, { 1, 0, 2 } };
    int n = 0;
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) {
            double det = L[i][0] * L[j][1] - L[i][1] * L[j][0];
            if (fabs(det) < 1e-12) continue;
            double x = (L[i][2] * L[j][1] - L[i][1] * L[j][2]) / det;
            double y = (L[i][0] * L[j][2] - L[i][2] * L[j][0]) / det;
            int ok = 1;
            for (int k = 0; k < 4; k++) if (L[k][0] * x + L[k][1] * y > L[k][2] + 1e-9) ok = 0;
            if (ok && n < maxv) { v[n][0] = x; v[n][1] = y; n++; }
        }
    /* order counter-clockwise around the centroid */
    double cx = 0, cy = 0;
    for (int i = 0; i < n; i++) { cx += v[i][0]; cy += v[i][1]; }
    cx /= n; cy /= n;
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && atan2(v[j][1] - cy, v[j][0] - cx) < atan2(v[j - 1][1] - cy, v[j - 1][0] - cx); j--) {
            double t0 = v[j][0], t1 = v[j][1];
            v[j][0] = v[j - 1][0]; v[j][1] = v[j - 1][1]; v[j - 1][0] = t0; v[j - 1][1] = t1;
        }
    return n;
}

/* Dantzig tableau on the standard form, starting from the slack basis.
 * Records the (x, y) vertex after every pivot, the entering and leaving column. */
typedef struct { double x, y; int enter, leave; } Pivot;

static int tableau_run(Pivot *out, int maxp) {
    double T[LP_M + 1][LP_N + 1];
    int basis[LP_M] = { 2, 3 };
    for (int i = 0; i < LP_M; i++) {
        for (int j = 0; j < LP_N; j++) T[i][j] = LP_A[i * LP_N + j];
        T[i][LP_N] = LP_B[i];
    }
    for (int j = 0; j < LP_N; j++) T[LP_M][j] = LP_C[j];     /* reduced costs, basis = slacks */
    T[LP_M][LP_N] = 0.0;
    int np = 0;
    out[np].x = 0; out[np].y = 0; out[np].enter = -1; out[np].leave = -1; np++;
    for (int it = 0; it < 10; it++) {
        int e = -1; double best = -1e-12;
        for (int j = 0; j < LP_N; j++) if (T[LP_M][j] < best) { best = T[LP_M][j]; e = j; }   /* Dantzig */
        if (e < 0) break;
        int r = -1; double ratio = 1e300;
        for (int i = 0; i < LP_M; i++)
            if (T[i][e] > 1e-12 && T[i][LP_N] / T[i][e] < ratio) { ratio = T[i][LP_N] / T[i][e]; r = i; }
        if (r < 0) fail("tableau: unbounded");
        double pv = T[r][e];
        for (int j = 0; j <= LP_N; j++) T[r][j] /= pv;
        for (int i = 0; i <= LP_M; i++) {
            if (i == r) continue;
            double f = T[i][e];
            for (int j = 0; j <= LP_N; j++) T[i][j] -= f * T[r][j];
        }
        int left = basis[r]; basis[r] = e;
        double xv[LP_N] = { 0, 0, 0, 0 };
        for (int i = 0; i < LP_M; i++) xv[basis[i]] = T[i][LP_N];
        if (np < maxp) { out[np].x = xv[0]; out[np].y = xv[1]; out[np].enter = e; out[np].leave = left; np++; }
    }
    return np;
}

static const char *colname(int j) { static const char *nm[4] = { "x", "y", "slack 1", "slack 2" }; return nm[j]; }

/* ------------------------------------------------------------------------- */
/* Figure 1: simplex pivots                                                   */

static void fig_simplex(const char *dir) {
    double ref[2], refobj;
    lp_reference(&ref[0], &ref[1], &refobj);
    double V[8][2]; int nv = polygon(V, 8);
    Pivot P[8]; int np = tableau_run(P, 8);
    if (nv != 4) fail("polygon: expected 4 vertices");
    if (np != 3) fail("tableau: expected 2 pivots");
    if (fabs(P[np - 1].x - ref[0]) > 1e-9 || fabs(P[np - 1].y - ref[1]) > 1e-9) fail("tableau optimum differs from PRIMAL");
    if (fabs(3 * ref[0] + 2 * ref[1] - 10.0) > 1e-9 || fabs(refobj - 10.0) > 1e-7) fail("objective is not 10");
    /* certificate by substitution: multipliers (2, 1) on the two rows give 3x + 2y <= 8 + 2 */
    if (fabs(2 * 4 + 1 * 2 - refobj) > 1e-7) fail("dual bound 2*4 + 1*2 differs from the optimum");

    char name[512]; snprintf(name, sizeof name, "%s/simplex-pivots.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open simplex-pivots.svg");
    Plot p = { 0, 5, 0, 5, 70, 60, 470, 330 };
    char desc[900];
    snprintf(desc, sizeof desc, "Feasible polygon with vertices (0,0), (2,0), (2,2) and (0,4) for maximize 3x+2y with "
             "x+y<=4 and x<=2. Two simplex pivots go from (0,0) to (2,0) and then to (2,2), where the objective is 10. "
             "Dashed lines are the level sets 3x+2y=6 and 3x+2y=10. Analytic figure, computed at revision %s.", g_rev);
    svg_begin(f, 700, 500, "Feasible polygon and two simplex pivots", desc);
    text(f, 350, 30, "middle", 20, INK, "Two simplex pivots on max 3x + 2y");
    /* polygon */
    fputs("  <path d=\"", f);
    for (int i = 0; i < nv; i++) fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, V[i][0]), PY(&p, V[i][1]));
    fprintf(f, "Z\" fill=\"%s\" stroke=\"none\"/>\n", FILL);
    /* constraint lines */
    line(f, PX(&p, 0), PY(&p, 4), PX(&p, 4), PY(&p, 0), BLUE, 3, NULL, NULL);
    line(f, PX(&p, 2), PY(&p, 0), PX(&p, 2), PY(&p, 5), BLUE, 3, NULL, NULL);
    /* level sets of the objective */
    line(f, PX(&p, 0), PY(&p, 3), PX(&p, 2), PY(&p, 0), GREY, 1.5, "7 5", NULL);
    line(f, PX(&p, 0), PY(&p, 5), PX(&p, 10.0 / 3.0), PY(&p, 0), GREY, 1.5, "7 5", NULL);
    axes(f, &p, "x", "y", 1, 1);
    text(f, PX(&p, 0.12), PY(&p, 3.0) + 20, "start", 14, GREY, "3x + 2y = 6");
    text(f, PX(&p, 0.35), PY(&p, 4.85), "start", 14, GREY, "3x + 2y = 10");
    text(f, PX(&p, 3.05), PY(&p, 1.30), "start", 14, BLUE, "x + y = 4");
    text(f, PX(&p, 2.08), PY(&p, 4.60), "start", 14, BLUE, "x = 2");
    /* pivots */
    for (int i = 1; i < np; i++) {
        line(f, PX(&p, P[i - 1].x) + (i == 1 ? 6 : 0), PY(&p, P[i - 1].y) - (i == 2 ? 6 : 0),
             PX(&p, P[i].x) - (i == 1 ? 8 : 0), PY(&p, P[i].y) + (i == 2 ? 8 : 0), ORANGE, 3.5, NULL, "arrb");
    }
    /* vertices: open circle = start, square = optimum, filled circle = other visited vertex */
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"6\" fill=\"#fff\" stroke=\"%s\" stroke-width=\"2.5\"/>\n",
            PX(&p, P[0].x), PY(&p, P[0].y), INK);
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"%s\"/>\n", PX(&p, P[1].x), PY(&p, P[1].y), INK);
    fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"12\" height=\"12\" fill=\"%s\"/>\n",
            PX(&p, P[2].x) - 6, PY(&p, P[2].y) - 6, INK);
    /* the unvisited vertex (0,4) gets a small cross, found from the polygon */
    for (int i = 0; i < nv; i++) {
        int visited = 0;
        for (int k = 0; k < np; k++) if (fabs(V[i][0] - P[k].x) < 1e-9 && fabs(V[i][1] - P[k].y) < 1e-9) visited = 1;
        if (!visited) {
            double cx = PX(&p, V[i][0]), cy = PY(&p, V[i][1]);
            line(f, cx - 5, cy - 5, cx + 5, cy + 5, INK, 2, NULL, NULL);
            line(f, cx - 5, cy + 5, cx + 5, cy - 5, INK, 2, NULL, NULL);
            text(f, cx + 10, cy - 8, "start", 14, INK, "(0, 4)  z = 8, not visited");
        }
    }
    text(f, PX(&p, 0) + 12, PY(&p, 0) - 10, "start", 14, INK, "(0, 0)  z = 0");
    text(f, PX(&p, 2) + 10, PY(&p, 0) - 10, "start", 14, INK, "(2, 0)  z = 6");
    text(f, PX(&p, 2) + 12, PY(&p, 2) - 12, "start", 14, INK, "(2, 2)  z = 10, optimal");
    {
        char s1[128], s2[128];
        snprintf(s1, sizeof s1, "Pivot 1 (arrow 1): %s enters the basis, %s leaves", colname(P[1].enter), colname(P[1].leave));
        snprintf(s2, sizeof s2, "Pivot 2 (arrow 2): %s enters the basis, %s leaves", colname(P[2].enter), colname(P[2].leave));
        text(f, PX(&p, 1.0), PY(&p, 0) - 14, "middle", 16, ORANGE, "1");
        text(f, PX(&p, 2.0) + 12, PY(&p, 1.0) + 5, "start", 16, ORANGE, "2");
        text(f, 70, 440, "start", 14, ORANGE, s1);
        text(f, 70, 460, "start", 14, ORANGE, s2);
    }
    /* direction of improvement */
    line(f, PX(&p, 0.6), PY(&p, 1.25), PX(&p, 1.2), PY(&p, 1.65), INK, 2, NULL, "arr");
    text(f, PX(&p, 0.1), PY(&p, 1.85), "start", 14, INK, "increasing 3x + 2y");
    text(f, 70, 482, "start", 13, GREY, "Open circle: start. Filled circle: visited vertex. Square: optimum. Cross: unvisited vertex.");
    svg_end(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- */
/* Figure 2: iterates and residuals of the dense LP interior point            */

#define MAXTRACE 40
typedef struct { double x[LP_N], y[LP_M], z[LP_N], rp, rd, mu, gap; int status; } Iter;

/* The same three quantities ipm_solve_std tests, computed from an iterate. */
static void measures(const double *x, const double *y, const double *z, Iter *it) {
    double bnorm = 1.0, cnorm = 1.0, xz = 0.0, pobj = 0.0;
    for (int i = 0; i < LP_M; i++) bnorm = fmax(bnorm, fabs(LP_B[i]));
    for (int j = 0; j < LP_N; j++) cnorm = fmax(cnorm, fabs(LP_C[j]));
    double fp = 0, fd = 0;
    for (int i = 0; i < LP_M; i++) {
        double s = -LP_B[i];
        for (int j = 0; j < LP_N; j++) s += LP_A[i * LP_N + j] * x[j];
        fp = fmax(fp, fabs(s));
    }
    for (int j = 0; j < LP_N; j++) {
        double s = LP_C[j] - z[j];
        for (int i = 0; i < LP_M; i++) s -= LP_A[i * LP_N + j] * y[i];
        fd = fmax(fd, fabs(s));
        xz += x[j] * z[j];
        pobj += LP_C[j] * x[j];
    }
    it->rp = fp / bnorm; it->rd = fd / cnorm; it->mu = xz / LP_N; it->gap = fabs(xz) / (1.0 + fabs(pobj));
}

static int ipm_trace(Iter *tr, int max) {
    int n = 0;
    for (int k = 0; k < max; k++) {
        Iter *it = &tr[n];
        /* run k iterations from the default start; the engine is deterministic */
        it->status = ipm_solve_std(LP_A, NULL, LP_M, LP_N, LP_B, LP_C, 1e-8, 1e-8, 1e-8, k, it->x, it->y, it->z, NULL, NULL);
        measures(it->x, it->y, it->z, it);
        /* the call that reports OPTIMAL stops at the top of the loop, so it returns the
         * iterate of the previous call: do not record it twice */
        if (it->status == 0 && n > 0) { tr[n - 1].status = 0; break; }
        n++;
        if (it->status == 0) break;
    }
    return n;
}

static double lg(double v) { return log10(v < 1e-16 ? 1e-16 : v); }

static void fig_ipm(const char *dir) {
    double ref[2], refobj;
    lp_reference(&ref[0], &ref[1], &refobj);
    Iter tr[MAXTRACE]; int n = ipm_trace(tr, MAXTRACE);
    if (tr[n - 1].status != 0) fail("interior point did not converge on the figure LP");
    if (fabs(tr[n - 1].x[0] - ref[0]) > 1e-6 || fabs(tr[n - 1].x[1] - ref[1]) > 1e-6) fail("interior point optimum differs from PRIMAL");
    if (fabs(tr[0].x[0] - 1.0) > 0 || fabs(tr[0].z[0] - 1.0) > 0) fail("unexpected starting point");
    if (!(tr[n - 1].mu < tr[0].mu * 1e-6)) fail("complementarity did not fall");

    /* the data behind the figure */
    char cname[512]; snprintf(cname, sizeof cname, "%s/lp-central-path-trace.csv", dir);
    FILE *c = fopen(cname, "w"); if (!c) fail("cannot open the csv");
    fprintf(c, "# make_figures.c at revision %s; dense LP engine ipm_solve_std, Mehrotra predictor-corrector\n", g_rev);
    fputs("# LP: min -3x-2y s.t. x+y+s1=4, x+s2=2, x,y,s1,s2>=0; start x=z=1, y=0; tolerances 1e-8\n", c);
    fputs("# rp=max|Ax-b|/max(1,|b|), rd=max|c-A'y-z|/max(1,|c|), mu=x'z/n, gap=|x'z|/(1+|c'x|)\n", c);
    fputs("iter,x,y,s1,s2,rp,rd,mu,gap\n", c);
    for (int k = 0; k < n; k++)
        fprintf(c, "%d,%.10g,%.10g,%.10g,%.10g,%.6e,%.6e,%.6e,%.6e\n", k, tr[k].x[0], tr[k].x[1], tr[k].x[2], tr[k].x[3],
                tr[k].rp, tr[k].rd, tr[k].mu, tr[k].gap);
    fclose(c);

    char name[512]; snprintf(name, sizeof name, "%s/lp-central-path.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open lp-central-path.svg");
    char desc[1100];
    snprintf(desc, sizeof desc, "Left: the iterates of the interior point method in the (x,y) plane for maximize 3x+2y "
             "with x+y<=4 and x<=2; they start at (1,1) and approach the vertex (2,2) without touching the boundary. "
             "Right: base-10 logarithm of the primal residual, the dual residual and the complementarity mu against the "
             "iteration number, falling from about 1 to below 1e-8 in %d iterations. Measured with ipm_solve_std at revision %s.",
             n - 1, g_rev);
    svg_begin(f, 900, 470, "Interior point iterates and residuals on a small LP", desc);
    text(f, 450, 28, "middle", 20, INK, "Interior point on max 3x + 2y: iterates and residuals (measured)");

    /* left panel: polygon and iterates */
    Plot p = { 0, 4.4, 0, 4.4, 60, 70, 330, 300 };
    double V[8][2]; int nv = polygon(V, 8);
    fputs("  <path d=\"", f);
    for (int i = 0; i < nv; i++) fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, V[i][0]), PY(&p, V[i][1]));
    fprintf(f, "Z\" fill=\"%s\" stroke=\"%s\" stroke-width=\"2.5\"/>\n", FILL, BLUE);
    axes(f, &p, "x", "y", 1, 1);
    fputs("  <polyline fill=\"none\" stroke=\"" ORANGE "\" stroke-width=\"1.8\" points=\"", f);
    for (int k = 0; k < n; k++) fprintf(f, "%.1f,%.1f ", PX(&p, tr[k].x[0]), PY(&p, tr[k].x[1]));
    fputs("\"/>\n", f);
    for (int k = 0; k < n; k++) {
        double cx = PX(&p, tr[k].x[0]), cy = PY(&p, tr[k].x[1]);
        if (k == 0) fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"6\" fill=\"#fff\" stroke=\"%s\" stroke-width=\"2.5\"/>\n", cx, cy, INK);
        else if (k == n - 1) fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"11\" height=\"11\" fill=\"%s\"/>\n", cx - 5.5, cy - 5.5, INK);
        else fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"3.5\" fill=\"%s\"/>\n", cx, cy, ORANGE);
    }
    text(f, PX(&p, tr[0].x[0]) + 10, PY(&p, tr[0].x[1]) + 20, "start", 14, INK, "start (1, 1)");
    text(f, PX(&p, 2) + 12, PY(&p, 2) - 12, "start", 14, INK, "(2, 2)");
    text(f, 60 + 165, 442, "middle", 13, GREY, "Open circle: start. Square: final iterate.");

    /* right panel: log10 residuals */
    Plot q = { 0, (double)(n - 1), -16, 1, 500, 70, 360, 300 };
    for (int e = -16; e <= 0; e += 4) {
        char s[16]; snprintf(s, sizeof s, "1e%d", e);
        line(f, q.px, PY(&q, e), q.px + q.pw, PY(&q, e), "#e5e7eb", 1, NULL, NULL);
        text(f, q.px - 8, PY(&q, e) + 5, "end", 13, INK, s);
    }
    line(f, q.px, q.py + q.ph, q.px + q.pw + 12, q.py + q.ph, INK, 1.5, NULL, "arr");
    line(f, q.px, q.py + q.ph, q.px, q.py - 12, INK, 1.5, NULL, "arr");
    for (int k = 0; k < n; k++) {
        char s[16]; snprintf(s, sizeof s, "%d", k);
        double xk = PX(&q, k);
        line(f, xk, q.py + q.ph, xk, q.py + q.ph + 5, INK, 1, NULL, NULL);
        text(f, xk, q.py + q.ph + 20, "middle", 13, INK, s);
    }
    text(f, q.px + q.pw / 2, q.py + q.ph + 42, "middle", 15, INK, "iteration");
    {
        static const char *names[3] = { "primal residual", "dual residual", "complementarity mu" };
        static const char *dash[3] = { NULL, "7 4", "2 4" };
        static const char *stroke[3] = { BLUE, ORANGE, INK };
        for (int s = 0; s < 3; s++) {
            fprintf(f, "  <polyline fill=\"none\" stroke=\"%s\" stroke-width=\"2.2\"", stroke[s]);
            if (dash[s]) fprintf(f, " stroke-dasharray=\"%s\"", dash[s]);
            fputs(" points=\"", f);
            for (int k = 0; k < n; k++) {
                double v = s == 0 ? tr[k].rp : s == 1 ? tr[k].rd : tr[k].mu;
                fprintf(f, "%.1f,%.1f ", PX(&q, k), PY(&q, lg(v)));
            }
            fputs("\"/>\n", f);
            for (int k = 0; k < n; k++) {
                double v = s == 0 ? tr[k].rp : s == 1 ? tr[k].rd : tr[k].mu;
                if (s == 0) fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"3\" fill=\"%s\"/>\n", PX(&q, k), PY(&q, lg(v)), stroke[s]);
                else if (s == 1) fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"6\" height=\"6\" fill=\"%s\"/>\n", PX(&q, k) - 3, PY(&q, lg(v)) - 3, stroke[s]);
                else fprintf(f, "  <path d=\"M%.1f %.1fl3.5 6h-7z\" fill=\"%s\"/>\n", PX(&q, k), PY(&q, lg(v)) - 3.5, stroke[s]);
            }
            line(f, q.px + 14, q.py + q.ph - 62 + 20 * s, q.px + 54, q.py + q.ph - 62 + 20 * s, stroke[s], 2.2, dash[s], NULL);
            text(f, q.px + 62, q.py + q.ph - 57 + 20 * s, "start", 13, INK, names[s]);
        }
    }
    text(f, q.px + q.pw / 2, 442, "middle", 13, GREY, "Values at or below 1e-16 are drawn on the bottom line.");
    svg_end(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- */
/* Figure 3: a convex QP and its active constraint                            */
/*   minimize (x1^2 + x2^2)/2   s.t.  x1 + x2 >= 2,  x1 <= 3                 */

static void fig_qp(const char *dir) {
    /* reference from the public API: the optimum and the multiplier of the active row */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 2); PRIMAL_appendcons(t, 1);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_UP, -INFINITY, 3.0);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putqobj(t, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putarow(t, 0, 2, (int[]){0, 1}, (double[]){1, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_LO, 2.0, INFINITY);
    if (PRIMAL_optimize(t) != PRIMAL_RES_OK) fail("qp: optimize failed");
    double x[2], obj, y[1];
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_gety(t, PRIMAL_SOL_ITR, y);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    /* KKT by substitution: gradient (x1, x2) = lambda (1, 1), lambda >= 0, row active, x1 < 3 */
    double lam = fabs(y[0]);
    if (fabs(x[0] - 1.0) > 1e-6 || fabs(x[1] - 1.0) > 1e-6) fail("qp optimum is not (1, 1)");
    if (fabs(obj - 1.0) > 1e-7) fail("qp objective is not 1");
    if (fabs(lam - 1.0) > 1e-5) fail("qp multiplier is not 1");
    if (fabs(x[0] + x[1] - 2.0) > 1e-6) fail("qp row is not active");

    char name[512]; snprintf(name, sizeof name, "%s/qp-contours.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open qp-contours.svg");
    Plot p = { -1, 4, -1, 4, 70, 60, 440, 340 };
    char desc[900];
    snprintf(desc, sizeof desc, "Circular contours of (x1^2+x2^2)/2 centred at the origin, the feasible region above the line "
             "x1+x2=2 and to the left of x1=3, and the constrained minimum (1,1) with objective 1, where the gradient (1,1) "
             "equals the multiplier 1 times the row normal (1,1). Analytic figure, optimum checked against PRIMAL at revision %s.", g_rev);
    svg_begin(f, 700, 500, "Contours of a convex QP and its active constraint", desc);
    fprintf(f, "  <clipPath id=\"win\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n", p.px, p.py, p.pw, p.ph);
    text(f, 350, 30, "middle", 20, INK, "A convex QP: contours and the active constraint");
    /* feasible region: x1 + x2 >= 2 and x1 <= 3, inside the window */
    fputs("  <g clip-path=\"url(#win)\">\n", f);
    double xa = -1, ya = 2 - xa;                                  /* the line x1 + x2 = 2 at the left edge: (-1, 3) */
    fprintf(f, "  <path d=\"M%.1f %.1fL%.1f %.1fL%.1f %.1fL%.1f %.1fZ\" fill=\"%s\"/>\n",
            PX(&p, xa), PY(&p, ya), PX(&p, 3), PY(&p, -1), PX(&p, 3), PY(&p, 4), PX(&p, xa), PY(&p, 4), FILL);
    /* contours f = 0.5, 1, 2, 4 -> radius sqrt(2 f) */
    static const double lev[4] = { 0.5, 1.0, 2.0, 4.0 };
    for (int i = 0; i < 4; i++) {
        double r = sqrt(2.0 * lev[i]);
        fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"none\" stroke=\"%s\" stroke-width=\"%.1f\"%s/>\n",
                PX(&p, 0), PY(&p, 0), r * p.pw / (p.x1 - p.x0), i == 1 ? ORANGE : GREY, i == 1 ? 2.5 : 1.5,
                i == 1 ? "" : " stroke-dasharray=\"7 5\"");
    }
    fputs("  </g>\n", f);
    axes(f, &p, "x1", "x2", 1, 1);
    /* constraints */
    line(f, PX(&p, -1), PY(&p, 3), PX(&p, 3), PY(&p, -1), BLUE, 3.5, NULL, NULL);
    line(f, PX(&p, 3), PY(&p, -1), PX(&p, 3), PY(&p, 4), BLUE, 2, "10 6", NULL);
    text(f, PX(&p, -0.95), PY(&p, 3.25), "start", 14, BLUE, "x1 + x2 = 2  (active)");
    text(f, PX(&p, 3.08), PY(&p, 3.85), "start", 14, BLUE, "x1 = 3  (inactive)");
    /* contour labels */
    /* gradient and row normal at the optimum */
    line(f, PX(&p, 1), PY(&p, 1), PX(&p, 0.35), PY(&p, 0.35), INK, 2.5, NULL, "arr");
    line(f, PX(&p, 1), PY(&p, 1), PX(&p, 1.8), PY(&p, 1.8), ORANGE, 2.5, NULL, "arrb");
    text(f, PX(&p, 1.8) + 8, PY(&p, 1.8) + 4, "start", 14, ORANGE, "row normal (1, 1)");
    text(f, PX(&p, -0.9), PY(&p, 0.72), "start", 14, INK, "-gradient (-1, -1)");
    fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"12\" height=\"12\" fill=\"%s\"/>\n", PX(&p, 1) - 6, PY(&p, 1) - 6, INK);
    text(f, PX(&p, 1) - 12, PY(&p, 1) - 14, "end", 14, INK, "optimum (1, 1), f = 1");
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"#fff\" stroke=\"%s\" stroke-width=\"2\"/>\n", PX(&p, 0), PY(&p, 0), INK);
    text(f, PX(&p, -0.9), PY(&p, -0.62), "start", 14, INK, "O: unconstrained minimum (0, 0), infeasible");
    text(f, 70, 452, "start", 13, GREY, "Circles: level sets of f = (x1^2 + x2^2) / 2 at 0.5, 1 (solid), 2 and 4.");
    text(f, 70, 472, "start", 13, GREY, "At the optimum the gradient (1, 1) equals the multiplier 1 times the row normal (1, 1).");
    svg_end(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- */
/* Figure 4: a supporting hyperplane at the optimal vertex                    */
/*   the gradient (3, 2) lies in the cone of the active-constraint normals    */

static void fig_hyperplane(const char *dir) {
    double ref[2], refobj;
    lp_reference(&ref[0], &ref[1], &refobj);
    /* active rows at the optimum (2, 2): x + y = 4 with normal (1, 1) and x = 2 with normal (1, 0) */
    const double n1[2] = { 1, 1 }, n2[2] = { 1, 0 }, g[2] = { 3, 2 };
    double det = n1[0] * n2[1] - n1[1] * n2[0];
    double l1 = (g[0] * n2[1] - g[1] * n2[0]) / det, l2 = (n1[0] * g[1] - n1[1] * g[0]) / det;
    if (fabs(l1 - 2.0) > 1e-12 || fabs(l2 - 1.0) > 1e-12) fail("gradient is not 2 n1 + 1 n2");
    if (l1 < 0 || l2 < 0) fail("negative multiplier: the vertex would not be optimal");
    if (fabs(ref[0] - 2.0) > 1e-9 || fabs(ref[1] - 2.0) > 1e-9) fail("optimum is not (2, 2)");
    /* PRIMAL reports the same multipliers (up to its documented sign convention) */
    if (fabs(fabs(g_ref_y[0]) - l1) > 1e-7 || fabs(fabs(g_ref_y[1]) - l2) > 1e-7) fail("PRIMAL multipliers differ from (2, 1)");
    /* every point of the polygon satisfies g.x <= g.(2,2) = 10: the line 3x + 2y = 10 supports it */
    double V[8][2]; int nv = polygon(V, 8);
    for (int i = 0; i < nv; i++) if (g[0] * V[i][0] + g[1] * V[i][1] > 10.0 + 1e-12) fail("a vertex lies beyond the supporting line");

    char name[512]; snprintf(name, sizeof name, "%s/supporting-hyperplane.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open supporting-hyperplane.svg");
    Plot p = { -0.5, 5, -0.5, 5, 70, 60, 470, 330 };
    char desc[900];
    snprintf(desc, sizeof desc, "The polygon with vertices (0,0), (2,0), (2,2) and (0,4). At the optimal vertex (2,2) the two active "
             "constraints have normals (1,1) and (1,0); the objective gradient (3,2) equals 2 times the first plus 1 times the second, "
             "so it lies in the cone they span, shaded. Every line through (2,2) whose normal lies in that cone supports the polygon; "
             "the line 3x+2y=10 is one of them. Analytic figure, multipliers checked against PRIMAL at revision %s.", g_rev);
    svg_begin(f, 700, 500, "A supporting hyperplane at the optimal vertex", desc);
    text(f, 350, 30, "middle", 20, INK, "Supporting lines at the optimal vertex");
    fprintf(f, "  <clipPath id=\"win\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n  <g clip-path=\"url(#win)\">\n", p.px, p.py, p.pw, p.ph);
    /* the normal cone at (2, 2): the wedge between the rays along n1 and n2 */
    fprintf(f, "  <path d=\"M%.1f %.1fL%.1f %.1fL%.1f %.1fZ\" fill=\"#fde68a\" stroke=\"none\"/>\n", PX(&p, 2), PY(&p, 2),
            PX(&p, 2 + 3 * n1[0]), PY(&p, 2 + 3 * n1[1]), PX(&p, 2 + 3 * n2[0]), PY(&p, 2 + 3 * n2[1]));
    fputs("  <path d=\"", f);
    for (int i = 0; i < nv; i++) fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, V[i][0]), PY(&p, V[i][1]));
    fprintf(f, "Z\" fill=\"%s\" stroke=\"none\"/>\n", FILL);
    /* supporting lines through (2, 2): normals n1, n2 and g */
    line(f, PX(&p, -0.5), PY(&p, 4.5), PX(&p, 5), PY(&p, -1), BLUE, 3, NULL, NULL);
    line(f, PX(&p, 2), PY(&p, -0.5), PX(&p, 2), PY(&p, 5), BLUE, 3, NULL, NULL);
    line(f, PX(&p, 0), PY(&p, 5), PX(&p, 11.0 / 3.0), PY(&p, -0.5), ORANGE, 2.5, "9 5", NULL);
    fputs("  </g>\n", f);
    axes(f, &p, "x", "y", 1, 1);
    /* arrows from the vertex: n1, n2 (blue) and g (orange), g = 2 n1 + n2 */
    line(f, PX(&p, 2), PY(&p, 2), PX(&p, 2 + 0.9 * n1[0]), PY(&p, 2 + 0.9 * n1[1]), BLUE, 3, NULL, "arr");
    line(f, PX(&p, 2), PY(&p, 2), PX(&p, 2 + 0.9 * n2[0]), PY(&p, 2 + 0.9 * n2[1]), BLUE, 3, NULL, "arr");
    line(f, PX(&p, 2), PY(&p, 2), PX(&p, 2 + 0.55 * g[0]), PY(&p, 2 + 0.55 * g[1]), ORANGE, 3.5, NULL, "arrb");
    fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"12\" height=\"12\" fill=\"%s\"/>\n", PX(&p, 2) - 6, PY(&p, 2) - 6, INK);
    text(f, PX(&p, 2.15), PY(&p, 3.55), "start", 14, BLUE, "n1 = (1, 1)");
    text(f, PX(&p, 2 + 0.9) + 8, PY(&p, 2) + 22, "start", 14, BLUE, "n2 = (1, 0)");
    text(f, PX(&p, 2 + 0.55 * g[0]) + 10, PY(&p, 2 + 0.55 * g[1]) + 4, "start", 14, ORANGE, "g = (3, 2)");
    text(f, PX(&p, 0.1), PY(&p, 3.55), "start", 14, BLUE, "x + y = 4");
    text(f, PX(&p, 2.08), PY(&p, 4.75), "start", 14, BLUE, "x = 2");
    text(f, PX(&p, 0.1), PY(&p, 4.85), "start", 14, ORANGE, "3x + 2y = 10");
    text(f, PX(&p, 0.1), PY(&p, 0.35), "start", 14, INK, "feasible polygon");
    text(f, PX(&p, 2) - 10, PY(&p, 2) + 24, "end", 14, INK, "(2, 2)");
    text(f, 70, 452, "start", 14, INK, "g = 2 n1 + 1 n2: the gradient lies in the cone of the active normals (shaded),");
    text(f, 70, 472, "start", 14, INK, "so every line through (2, 2) with a normal in that cone supports the polygon.");
    svg_end(f);
    fclose(f);
}


/* ------------------------------------------------------------------------- */
/* Figure 5: presolve reductions and the homogeneous lift                     */
/*   x = 2 ; x + y = 5 ; y + z = 1 ; x, y, z >= 0   (infeasible)             */

static void cell(FILE *f, double x, double y, double w, double h, const char *fill, const char *stroke) {
    fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1.5\"/>\n", x, y, w, h, fill, stroke);
}

static void fig_presolve(const char *dir) {
    /* A in CSC: columns x = (r0, r1), y = (r1, r2), z = (r2) */
    int Ap[4] = { 0, 2, 4, 5 }, Ar[5] = { 0, 1, 1, 2, 2 };
    double Av[5] = { 1, 1, 1, 1, 1 }, b[3] = { 2, 5, 1 }, c[3] = { 0, 0, 0 };
    Presolve *p = NULL;
    if (lp_presolve(Ap, Ar, Av, b, c, 3, 3, 1e-9, 1, &p) != 0 || !p) fail("lp_presolve failed");
    const int *rp, *ri; const double *rv, *br, *cr; int mr, nr;
    presolve_reduced(p, &rp, &ri, &rv, &br, &cr, &mr, &nr);
    if (mr != 1 || nr != 1 || fabs(br[0] - 2.0) > 1e-12) fail("presolve did not reduce to the single row -z = 2");
    /* the surviving row z = -2 is stored negated (-z = 2, so that b >= 0); its Farkas vector is y = 1:
     * b'y = 2 > 0 and A'y = -1 <= 0, and the postsolve gives the sign back */
    double yred[1] = { 1.0 }, rfull[3], yfull[3];
    presolve_postsolve_dir(p, NULL, yred, rfull, yfull);
    presolve_free(p);
    double bty = b[0] * yfull[0] + b[1] * yfull[1] + b[2] * yfull[2];
    double aty[3] = { yfull[0] + yfull[1], yfull[1] + yfull[2], yfull[2] };
    if (fabs(yfull[0] + 1) > 1e-12 || fabs(yfull[1] - 1) > 1e-12 || fabs(yfull[2] + 1) > 1e-12) fail("lifted vector is not (-1, 1, -1)");
    if (fabs(bty - 2.0) > 1e-12) fail("b'y is not invariant under the reduction");
    for (int j = 0; j < 3; j++) if (aty[j] > 1e-12) fail("A'y has a positive entry");

    char name[512]; snprintf(name, sizeof name, "%s/presolve-reduction.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open presolve-reduction.svg");
    char desc[1000];
    snprintf(desc, sizeof desc, "Three panels showing the 3 by 3 constraint matrix of x=2, x+y=5, y+z=1 with x,y,z nonnegative. "
             "Singleton row 1 fixes x=2 and is removed with column x; singleton row 2 then fixes y=3 and is removed with column y; "
             "the surviving row 3 reads z=-2, which is infeasible with a Farkas multiplier of -1. Replaying the removals in reverse "
             "with the homogeneous lift gives the multipliers (-1, 1, -1) on the original rows, with b'y=2 and A'y<=0. "
             "Computed with lp_presolve and presolve_postsolve_dir at revision %s.", g_rev);
    svg_begin(f, 900, 430, "Presolve removes singleton rows and lifts a Farkas vector back", desc);
    text(f, 450, 28, "middle", 20, INK, "Presolve on x = 2, x + y = 5, y + z = 1");
    static const char *rname[3] = { "r1", "r2", "r3" }, *cname[3] = { "x", "y", "z" };
    static const char *rhs[3][3] = { { "2", "3", "-2" }, { "5", "3", "" }, { "1", "", "" } };
    static const char *title[3] = { "original", "after r1 fixes x = 2", "after r2 fixes y = 3" };
    /* which entries are alive in each panel */
    int alive_r[3][3] = { { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 1 } };
    int alive_c[3][3] = { { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 1 } };
    const double cw = 46, chh = 46;
    for (int pn = 0; pn < 3; pn++) {
        double ox = 50 + pn * 290, oy = 90;
        text(f, ox + 1.5 * cw + 20, oy - 28, "middle", 15, INK, title[pn]);
        for (int j = 0; j < 3; j++) text(f, ox + 40 + j * cw + cw / 2, oy - 4, "middle", 15, alive_c[pn][j] ? INK : GREY, cname[j]);
        text(f, ox + 40 + 3 * cw + 24, oy - 4, "middle", 15, INK, "rhs");
        for (int i = 0; i < 3; i++) {
            text(f, ox + 30, oy + i * chh + chh / 2 + 5, "end", 15, alive_r[pn][i] ? INK : GREY, rname[i]);
            for (int j = 0; j < 3; j++) {
                int nz = 0; for (int q = Ap[j]; q < Ap[j + 1]; q++) if (Ar[q] == i) nz = 1;
                int live = alive_r[pn][i] && alive_c[pn][j];
                cell(f, ox + 40 + j * cw, oy + i * chh, cw, chh, live ? (nz ? FILL : "#fff") : "#f3f4f6", live ? BLUE : "#d1d5db");
                if (nz) text(f, ox + 40 + j * cw + cw / 2, oy + i * chh + chh / 2 + 5, "middle", 16, live ? INK : GREY, "1");
            }
            /* right-hand side after the substitutions made so far */
            double bv = b[i]; if (pn >= 1 && i >= 1) bv -= 2.0 * (i == 1); if (pn >= 2 && i == 2) bv -= 3.0;
            char s[16]; snprintf(s, sizeof s, "%g", bv);
            text(f, ox + 40 + 3 * cw + 24, oy + i * chh + chh / 2 + 5, "middle", 15, alive_r[pn][i] ? INK : GREY, s);
            if (!alive_r[pn][i]) line(f, ox + 38, oy + i * chh + chh / 2, ox + 40 + 3 * cw + 44, oy + i * chh + chh / 2, GREY, 1.5, NULL, NULL);
        }
        (void)rhs;
        if (pn < 2) line(f, ox + 40 + 3 * cw + 55, oy + 1.5 * chh, ox + 280, oy + 1.5 * chh, INK, 2, NULL, "arr");
    }
    text(f, 50, 270, "start", 15, INK, "Panel 3: the surviving row r3 reads z = -2 with z >= 0, so the reduced problem is infeasible.");
    text(f, 50, 296, "start", 15, INK, "Farkas vector of the reduced row: y3 = -1, since b'y = (-2)(-1) = 2 > 0 and A'y = -1 <= 0.");
    text(f, 50, 330, "start", 15, ORANGE, "Reverse replay, homogeneous: y2 = -(A_{3y} y3) / A_{2y} = 1,   y1 = -(A_{2x} y2) / A_{1x} = -1.");
    text(f, 50, 356, "start", 15, ORANGE, "Lifted: y = (-1, 1, -1);  b'y = -2 + 5 - 1 = 2 (unchanged);  A'y = (0, 0, -1) <= 0.");
    text(f, 50, 396, "start", 13, GREY, "Shaded cell: nonzero of an active row and column. Grey: removed. The right-hand side column shows the substitution b_k - A_kj x_j.");
    svg_end(f);
    fclose(f);
}


/* ------------------------------------------------------------------------- */
/* Figure 6: Cholesky fill with the natural order and with AMD                */
/*   an arrowhead matrix (the star graph), hub in the first position          */

static int to_lower_csc(int n, const double *D, int *Kp, int *Ki, double *Kx) {
    int nz = 0; Kp[0] = 0;
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) if (D[(size_t)i * n + j] != 0.0) { Ki[nz] = i; Kx[nz] = D[(size_t)i * n + j]; nz++; }
        Kp[j + 1] = nz;
    }
    return nz;
}

/* draw an n x n pattern: mark[i*n+j] = 0 none, 1 entry of K, 2 fill-in */
static void spy(FILE *f, double ox, double oy, int n, const char *mark, double cs, const char *label) {
    text(f, ox + n * cs / 2, oy - 10, "middle", 14, INK, label);
    fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"#fff\" stroke=\"%s\" stroke-width=\"1.5\"/>\n", ox, oy, n * cs, n * cs, INK);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double x = ox + j * cs, y = oy + i * cs;
            if (mark[i * n + j] == 1) fprintf(f, "  <rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\"/>\n", x + 2, y + 2, cs - 4, cs - 4, BLUE);
            else if (mark[i * n + j] == 2) fprintf(f, "  <path d=\"M%.1f %.1fL%.1f %.1fL%.1f %.1fL%.1f %.1fZ\" fill=\"%s\"/>\n", x + cs / 2, y + 1.5, x + cs - 1.5, y + cs / 2, x + cs / 2, y + cs - 1.5, x + 1.5, y + cs / 2, ORANGE);
        }
}

static void fig_ordering(const char *dir) {
    const int n = 10;
    double *D = (double *)calloc((size_t)n * n, sizeof(double));
    int *Kp = (int *)malloc((size_t)(n + 1) * sizeof(int)), *Ki = (int *)malloc((size_t)n * n * sizeof(int));
    double *Kx = (double *)malloc((size_t)n * n * sizeof(double));
    if (!D || !Kp || !Ki || !Kx) fail("ordering: out of memory");
    for (int i = 0; i < n; i++) D[i * n + i] = (i == 0) ? 12.0 : 4.0;     /* diagonally dominant, hence SPD */
    for (int i = 1; i < n; i++) D[i * n] = D[i] = 1.0;                      /* hub = index 0 */
    int nzK = to_lower_csc(n, D, Kp, Ki, Kx);
    SpChol *L0 = spchol_factor(n, Kp, Ki, Kx), *L1 = spchol_factor_ord(n, Kp, Ki, Kx);
    if (!L0 || !L1 || !L1->perm) fail("ordering: factorization failed");
    int nnz0 = L0->Lp[n], nnz1 = L1->Lp[n];
    if (nnz0 != n * (n + 1) / 2) fail("natural order did not fill the triangle");
    if (nnz1 != nzK) fail("AMD order was expected to create no fill on the arrowhead");
    /* both factors must give the same solution */
    double rhs0[10], rhs1[10], x0[10], x1[10];
    for (int i = 0; i < n; i++) rhs0[i] = rhs1[i] = 1.0 + i;
    if (spchol_solve(L0, rhs0) || spchol_solve_ord(L1, rhs1)) fail("solve failed");
    for (int i = 0; i < n; i++) { x0[i] = rhs0[i]; x1[i] = rhs1[i]; if (fabs(x0[i] - x1[i]) > 1e-12 * (1 + fabs(x0[i]))) fail("the two orders disagree"); }
    for (int i = 0; i < n; i++) {                                           /* residual of the ordered solve */
        double r = -(1.0 + i);
        for (int j = 0; j < n; j++) r += D[i * n + j] * x1[j];
        if (fabs(r) > 1e-12) fail("residual of the ordered solve is large");
    }

    char *mA = (char *)calloc((size_t)n * n, 1), *mB = (char *)calloc((size_t)n * n, 1), *mC = (char *)calloc((size_t)n * n, 1), *mD = (char *)calloc((size_t)n * n, 1);
    if (!mA || !mB || !mC || !mD) fail("ordering: out of memory");
    for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) if (D[i * n + j] != 0.0) mA[i * n + j] = 1;       /* K, natural order */
    for (int j = 0; j < n; j++) for (int p = L0->Lp[j]; p < L0->Lp[j + 1]; p++) mB[L0->Li[p] * n + j] = mA[L0->Li[p] * n + j] ? 1 : 2;
    for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) if (D[L1->perm[i] * n + L1->perm[j]] != 0.0) mC[i * n + j] = 1;   /* P K P' */
    for (int j = 0; j < n; j++) for (int p = L1->Lp[j]; p < L1->Lp[j + 1]; p++) mD[L1->Li[p] * n + j] = mC[L1->Li[p] * n + j] ? 1 : 2;

    /* the fill of 2-D grids, measured, goes to a csv next to the figure */
    char cname[512]; snprintf(cname, sizeof cname, "%s/ordering-fill.csv", dir);
    FILE *c = fopen(cname, "w"); if (!c) fail("cannot open ordering-fill.csv");
    fprintf(c, "# make_figures.c at revision %s; spchol_factor (natural order) and spchol_factor_ord (AMD)\n", g_rev);
    fputs("# arrowhead: 10 x 10, hub first.  grid: g x g five-point Laplacian plus 0.1 I, natural row-by-row order\n", c);
    fputs("case,n,nnz_K_lower,nnz_L_natural,nnz_L_amd\n", c);
    fprintf(c, "arrowhead,%d,%d,%d,%d\n", n, nzK, nnz0, nnz1);
    for (int g = 10; g <= 40; g += 10) {
        int N = g * g, nz = 0;
        int *Gp = (int *)malloc((size_t)(N + 1) * sizeof(int)), *Gi = (int *)malloc((size_t)3 * N * sizeof(int));
        double *Gx = (double *)malloc((size_t)3 * N * sizeof(double));
        if (!Gp || !Gi || !Gx) fail("ordering: out of memory");
        Gp[0] = 0;
        for (int j = 0; j < N; j++) {
            int r = j / g, cc = j % g;
            Gi[nz] = j; Gx[nz++] = 4.1;
            if (cc + 1 < g) { Gi[nz] = j + 1; Gx[nz++] = -1.0; }
            if (r + 1 < g) { Gi[nz] = j + g; Gx[nz++] = -1.0; }
            Gp[j + 1] = nz;
        }
        SpChol *A0 = spchol_factor(N, Gp, Gi, Gx), *A1 = spchol_factor_ord(N, Gp, Gi, Gx);
        if (!A0 || !A1) fail("ordering: grid factorization failed");
        if (A1->Lp[N] >= A0->Lp[N]) fail("AMD did not reduce the fill of a grid");
        fprintf(c, "grid%dx%d,%d,%d,%d,%d\n", g, g, N, nz, A0->Lp[N], A1->Lp[N]);
        spchol_free(A0); spchol_free(A1); free(Gp); free(Gi); free(Gx);
    }
    fclose(c);

    char name[512]; snprintf(name, sizeof name, "%s/cholesky-ordering.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open cholesky-ordering.svg");
    char desc[1000];
    snprintf(desc, sizeof desc, "Sparsity patterns of a 10 by 10 arrowhead matrix and of its Cholesky factor. In the natural order, with the hub first, "
             "the factor fills the whole lower triangle: %d nonzeros against %d in the matrix. After the AMD ordering, which places the hub "
             "next to last, the factor has exactly the pattern of the matrix, %d nonzeros, and no fill. Measured with spchol_factor and "
             "spchol_factor_ord at revision %s.", nnz0, nzK, nnz1, g_rev);
    svg_begin(f, 720, 600, "Cholesky fill with the natural order and with AMD", desc);
    text(f, 360, 28, "middle", 20, INK, "Fill-in of the Cholesky factor of an arrowhead matrix");
    char label[128];
    const double cs = 20, ox1 = 60, ox2 = 400, oy1 = 80, oy2 = 340;
    snprintf(label, sizeof label, "K, natural order (%d nonzeros, lower)", nzK);          spy(f, ox1, oy1, n, mA, cs, label);
    snprintf(label, sizeof label, "L, natural order (%d nonzeros)", nnz0);                spy(f, ox2, oy1, n, mB, cs, label);
    snprintf(label, sizeof label, "P K P', AMD order (%d nonzeros, lower)", nzK);         spy(f, ox1, oy2, n, mC, cs, label);
    snprintf(label, sizeof label, "L, AMD order (%d nonzeros)", nnz1);                    spy(f, ox2, oy2, n, mD, cs, label);
    {   int hub_pos = -1; for (int k = 0; k < n; k++) if (L1->perm[k] == 0) hub_pos = k;
        char leg[200]; snprintf(leg, sizeof leg, "Square: nonzero of K. Diamond: fill-in created by the factorization. The hub is row and column 0; AMD moves it to position %d.", hub_pos);
        text(f, 70, 582, "start", 13, GREY, leg); }
    svg_end(f);
    fclose(f);
    spchol_free(L0); spchol_free(L1); free(D); free(Kp); free(Ki); free(Kx); free(mA); free(mB); free(mC); free(mD);
}

/* ------------------------------------------------------------------------- */
/* Figure 8: slices of the 3x3 elliptope                                       */
/*   X = [1 a b; a 1 c; b c 1] >= 0  <=>  a^2 + b^2 - 2c ab <= 1 - c^2        */

/* Minimum of a + b + c over the slice X23 = c, solved by PRIMAL (obj = X12 + X13 + X23). */
static double elliptope_slice_min(double c) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    int dim = 3; PRIMAL_appendbarvars(t, 1, &dim); PRIMAL_appendcons(t, 4);
    int mC;
    PRIMAL_appendsparsesymmat(t, 3, 3, (int[]){1, 2, 2}, (int[]){0, 0, 1}, (double[]){0.5, 0.5, 0.5}, &mC);
    PRIMAL_putbarcj(t, 0, 1, (int[]){mC}, (double[]){1.0});
    for (int i = 0; i < 3; i++) {
        int mE;
        PRIMAL_appendsparsesymmat(t, 3, 1, (int[]){i}, (int[]){i}, (double[]){1.0}, &mE);
        PRIMAL_putbaraij(t, i, 0, 1, (int[]){mE}, (double[]){1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    int mO;
    PRIMAL_appendsparsesymmat(t, 3, 1, (int[]){2}, (int[]){1}, (double[]){0.5}, &mO);   /* <E,X> = X23 */
    PRIMAL_putbaraij(t, 3, 0, 1, (int[]){mO}, (double[]){1.0});
    PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, c, c);
    if (PRIMAL_optimize(t) != PRIMAL_RES_OK) fail("elliptope: optimize failed");
    double obj; PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return obj;
}

static void fig_elliptope(const char *dir) {
    static const double cs[3] = { 0.5, 0.0, -0.5 };
    double best[3];
    for (int i = 0; i < 3; i++) {
        best[i] = cs[i] - sqrt(2.0) * sqrt(1.0 + cs[i]);          /* closed form of the slice minimum */
        double got = elliptope_slice_min(cs[i]);
        if (fabs(got - best[i]) > 1e-6) fail("elliptope slice minimum differs from the closed form");
    }
    if (fabs(best[2] + 1.5) > 1e-12) fail("elliptope: optimum is not -3/2");

    char name[512]; snprintf(name, sizeof name, "%s/elliptope-slices.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open elliptope-slices.svg");
    Plot p = { -1.3, 1.3, -1.3, 1.3, 90, 60, 400, 400 };
    char desc[900];
    snprintf(desc, sizeof desc, "Three slices of the set of 3x3 correlation matrices with unit diagonal, in the plane of the entries X12 "
             "and X13, for X23 = 0.5, 0 and -0.5. The line X12 + X13 = -1 touches the slice X23 = -0.5 at (-0.5, -0.5), the optimum "
             "-3/2 of the triangle max-cut SDP. Analytic figure; the minimum on each slice is checked against PRIMAL at revision %s.", g_rev);
    svg_begin(f, 780, 540, "Slices of the 3x3 elliptope", desc);
    text(f, 390, 30, "middle", 20, INK, "Slices of the set X >= 0 with unit diagonal (3x3)");
    axes(f, &p, "X12", "X13", 1, 1);
    static const char *col[3] = { GREY, BLUE, ORANGE };
    for (int k = 0; k < 3; k++) {
        double c = cs[k];
        fprintf(f, "  <path d=\"");
        for (int i = 0; i <= 240; i++) {
            double th = 2.0 * M_PI * i / 240.0;
            double pp = sqrt(1.0 + c) * cos(th), qq = sqrt(1.0 - c) * sin(th);
            double a = (pp + qq) / sqrt(2.0), b = (pp - qq) / sqrt(2.0);
            fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, a), PY(&p, b));
        }
        fprintf(f, "Z\" fill=\"%s\" fill-opacity=\"%s\" stroke=\"%s\" stroke-width=\"%.1f\"/>\n",
                k == 2 ? FILL : "none", k == 2 ? "0.8" : "1", col[k], k == 2 ? 3.0 : 2.0);
    }
    /* the level line a + b = -1, tangent to the c = -0.5 slice */
    line(f, PX(&p, -1.3), PY(&p, 0.3), PX(&p, 0.3), PY(&p, -1.3), INK, 2, "8 5", NULL);
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"6\" fill=\"%s\"/>\n", PX(&p, -0.5), PY(&p, -0.5), INK);
    text(f, PX(&p, -0.5) + 12, PY(&p, -0.5) + 22, "start", 14, INK, "optimum (-1/2, -1/2)");
    text(f, PX(&p, 0.12), PY(&p, -1.2), "end", 14, INK, "X12 + X13 = -1");
    text(f, PX(&p, 0.95), PY(&p, 0.85), "start", 14, GREY, "X23 = 0.5");
    text(f, PX(&p, 1.06), PY(&p, -0.05), "start", 14, BLUE, "X23 = 0");
    text(f, PX(&p, 0.1), PY(&p, 0.08), "middle", 14, ORANGE, "X23 = -0.5");
    char s[160];
    text(f, 545, 140, "start", 15, INK, "Best objective per slice");
    snprintf(s, sizeof s, "X23 = 0.5:  %.4f", best[0]); text(f, 545, 168, "start", 14, GREY, s);
    snprintf(s, sizeof s, "X23 = 0:  %.4f", best[1]); text(f, 545, 192, "start", 14, BLUE, s);
    snprintf(s, sizeof s, "X23 = -0.5:  %.4f", best[2]); text(f, 545, 216, "start", 14, ORANGE, s);
    text(f, 545, 250, "start", 13, GREY, "Each slice is the ellipse");
    text(f, 545, 270, "start", 13, GREY, "a^2 + b^2 - 2 c a b <= 1 - c^2");
    text(f, 545, 290, "start", 13, GREY, "with a = X12, b = X13, c = X23.");
    text(f, 90, 508, "start", 13, GREY, "The line X12 + X13 = -1 touches only the slice X23 = -0.5: that slice holds the optimum.");
    svg_end(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- */
/* Figure 9: sections of the exponential cone                                  */
/*   PEXP = { t >= u exp(v/u), u > 0 }: the section at height u is t >= u exp(v/u) */

static void fig_expcone(const char *dir) {
    /* the barrier accepts exactly the interior: tested on both sides of the boundary curve */
    for (int k = 0; k < 40; k++) {
        double u = 0.5 + 0.1 * k, v = -3.0 + 0.12 * k, tb = u * exp(v / u), z[3], f;
        z[0] = tb * (1.0 + 1e-6); z[1] = u; z[2] = v;
        if (expcone_barrier(EXPCONE_PEXP, 0, z, &f) != 0) fail("expcone: a point above the boundary is rejected");
        z[0] = tb * (1.0 - 1e-6);
        if (expcone_barrier(EXPCONE_PEXP, 0, z, &f) == 0) fail("expcone: a point below the boundary is accepted");
    }
    /* the optimum of min t s.t. u + v = 2, (t,u,v) in PEXP is (2,2,0): objective by PRIMAL */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 3);
    for (int j = 0; j < 3; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, 0, 1.0);
    PRIMAL_appendcons(t, 1);
    PRIMAL_putarow(t, 0, 2, (int[]){1, 2}, (double[]){1, 1});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 2.0, 2.0);
    PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){0, 1, 2});
    if (PRIMAL_optimize(t) != PRIMAL_RES_OK) fail("expcone: optimize failed");
    double x[3], obj; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x); PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    if (fabs(x[0] - 2.0) > 1e-6 || fabs(x[1] - 2.0) > 1e-6 || fabs(x[2]) > 1e-6 || fabs(obj - 2.0) > 1e-8)
        fail("expcone: the optimum is not (2, 2, 0) with value 2");

    char name[512]; snprintf(name, sizeof name, "%s/exp-cone-section.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open exp-cone-section.svg");
    Plot p = { -3, 2.5, 0, 6, 80, 60, 480, 340 };
    char desc[900];
    snprintf(desc, sizeof desc, "Two sections of the exponential cone t >= u exp(v/u) in the (v, t) plane: the curve t = exp(v) for u = 1 "
             "with the region above it shaded, and the curve t = 2 exp(v/2) for u = 2. The point (v, t) = (0, 2) of the section u = 2 is the "
             "optimum (t,u,v) = (2,2,0) of minimize t subject to u + v = 2. Analytic figure; membership tested with expcone_barrier and the optimum "
             "checked against PRIMAL at revision %s.", g_rev);
    svg_begin(f, 760, 480, "Sections of the exponential cone", desc);
    text(f, 380, 30, "middle", 20, INK, "Sections of the exponential cone t >= u exp(v / u)");
    fprintf(f, "  <clipPath id=\"win\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n", p.px, p.py, p.pw, p.ph);
    fputs("  <g clip-path=\"url(#win)\">\n  <path d=\"", f);
    for (int i = 0; i <= 200; i++) {
        double v = p.x0 + (p.x1 - p.x0) * i / 200.0;
        fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, v), PY(&p, exp(v)));
    }
    fprintf(f, "L%.1f %.1fL%.1f %.1fZ\" fill=\"%s\"/>\n", PX(&p, p.x1), PY(&p, p.y1), PX(&p, p.x0), PY(&p, p.y1), FILL);
    for (int k = 0; k < 2; k++) {
        double u = k == 0 ? 1.0 : 2.0;
        fprintf(f, "  <polyline fill=\"none\" stroke=\"%s\" stroke-width=\"3\" points=\"", k == 0 ? BLUE : ORANGE);
        for (int i = 0; i <= 200; i++) {
            double v = p.x0 + (p.x1 - p.x0) * i / 200.0;
            fprintf(f, "%.1f,%.1f ", PX(&p, v), PY(&p, u * exp(v / u)));
        }
        fputs("\"/>\n", f);
    }
    fputs("  </g>\n", f);
    axes(f, &p, "v", "t", 1, 1);
    text(f, PX(&p, -2.9), PY(&p, 5.4), "start", 14, GREY, "interior: u ln(t/u) - v > 0");
    text(f, PX(&p, 1.0), PY(&p, 2.0) + 18, "start", 14, BLUE, "u = 1:  t = e^v");
    text(f, PX(&p, 1.42), PY(&p, 3.5), "start", 14, ORANGE, "u = 2:  t = 2 e^(v/2)");
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"7\" fill=\"%s\" stroke=\"#fff\" stroke-width=\"2\"/>\n", PX(&p, 0), PY(&p, 2), INK);
    text(f, PX(&p, 0) - 12, PY(&p, 2) - 14, "end", 14, INK, "optimum (t, u, v) = (2, 2, 0)");
    text(f, 80, 440, "start", 13, GREY, "Each section is the epigraph of a convex curve; the section at height u is the one at u = 1 stretched by u.");
    text(f, 80, 458, "start", 13, GREY, "The optimum of min t s.t. u + v = 2 is on the boundary of the cone, where the barrier blows up.");
    svg_end(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- */
/* Figure 10: sections of the power cones                                      */
/*   PPOW: t^a u^(1-a) >= |v|; the section u = 1 is t >= |v|^(1/a)            */
/*   RPOW: sqrt2 t^a u^(1-a) >= |v|; for a = 1/2 it is t >= v^2 / 2           */

static void fig_powercone(const char *dir) {
    struct { int kind; double a; const char *name; const char *col; } C[4] = {
        { EXPCONE_PPOW, 0.3, "PPOW a = 0.3:  t = |v|^(1/0.3)", ORANGE },
        { EXPCONE_PPOW, 0.5, "PPOW a = 0.5:  t = v^2", BLUE },
        { EXPCONE_PPOW, 0.7, "PPOW a = 0.7:  t = |v|^(1/0.7)", GREY },
        { EXPCONE_RPOW, 0.5, "RPOW a = 0.5:  t = v^2 / 2", INK },
    };
    /* the barrier accepts exactly the points above each curve (section u = 1) */
    for (int c = 0; c < 4; c++)
        for (int k = 1; k < 30; k++) {
            double v = 0.1 * k, tb = C[c].kind == EXPCONE_RPOW ? v * v / 2.0 : pow(v, 1.0 / C[c].a), z[3], f;
            z[0] = tb * (1.0 + 1e-6); z[1] = 1.0; z[2] = v;
            if (expcone_barrier(C[c].kind, C[c].a, z, &f) != 0) fail("powercone: a point above the boundary is rejected");
            z[0] = tb * (1.0 - 1e-6);
            if (expcone_barrier(C[c].kind, C[c].a, z, &f) == 0) fail("powercone: a point below the boundary is accepted");
        }
    /* RPOW with a = 1/2 is the rotated second-order cone 2 t u >= v^2 */
    for (int k = 0; k < 50; k++) {
        double z[3] = { 0.3 + 0.07 * k, 0.4 + 0.03 * k, 1.0 + 0.05 * k }, f;
        int inside = 2.0 * z[0] * z[1] - z[2] * z[2] > 0.0;
        if ((expcone_barrier(EXPCONE_RPOW, 0.5, z, &f) == 0) != inside) fail("powercone: RPOW(1/2) differs from the rotated quadratic cone");
    }
    /* the example: min t s.t. u = 1, v = 2, (t,u,v) in PPOW(0.3): t = 2^(1/0.3) */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 3);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_FX, 2.0, 2.0);
    PRIMAL_putcj(t, 0, 1.0);
    PRIMAL_appendcone(t, PRIMAL_CT_PPOW, 0.3, 3, (int[]){0, 1, 2});
    if (PRIMAL_optimize(t) != PRIMAL_RES_OK) fail("powercone: optimize failed");
    double x[3]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    double topt = pow(2.0, 1.0 / 0.3);
    if (fabs(x[0] - topt) > 1e-6) fail("powercone: the optimum is not 2^(1/0.3)");

    char name[512]; snprintf(name, sizeof name, "%s/power-cone-section.svg", dir);
    FILE *f = fopen(name, "w"); if (!f) fail("cannot open power-cone-section.svg");
    Plot p = { -3, 3, 0, 11, 80, 60, 440, 340 };
    char desc[900];
    snprintf(desc, sizeof desc, "Sections at u = 1 of power cones: the curves t = |v|^(1/a) for a = 0.3, 0.5 and 0.7 and the rotated cone "
             "t = v^2/2 for a = 1/2; each cone is the region above its curve. The point (v, t) = (2, 10.079) on the a = 0.3 curve is the optimum "
             "of minimize t with u = 1, v = 2. Analytic figure; membership tested with expcone_barrier and the optimum checked against PRIMAL "
             "at revision %s.", g_rev);
    svg_begin(f, 760, 480, "Sections of the power cones", desc);
    text(f, 380, 30, "middle", 20, INK, "Sections u = 1 of the power cones");
    fprintf(f, "  <clipPath id=\"win\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n", p.px, p.py, p.pw, p.ph);
    fputs("  <g clip-path=\"url(#win)\">\n  <path d=\"", f);
    for (int i = 0; i <= 240; i++) {
        double v = p.x0 + (p.x1 - p.x0) * i / 240.0;
        fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, v), PY(&p, pow(fabs(v), 1.0 / 0.3)));
    }
    fprintf(f, "L%.1f %.1fL%.1f %.1fZ\" fill=\"%s\"/>\n", PX(&p, p.x1), PY(&p, p.y1), PX(&p, p.x0), PY(&p, p.y1), FILL);
    for (int c = 0; c < 4; c++) {
        fprintf(f, "  <polyline fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" points=\"", C[c].col, c == 0 ? "3.5" : "2.2");
        for (int i = 0; i <= 240; i++) {
            double v = p.x0 + (p.x1 - p.x0) * i / 240.0;
            double tv = C[c].kind == EXPCONE_RPOW ? v * v / 2.0 : pow(fabs(v), 1.0 / C[c].a);
            fprintf(f, "%.1f,%.1f ", PX(&p, v), PY(&p, tv));
        }
        fputs("\"/>\n", f);
    }
    fputs("  </g>\n", f);
    axes(f, &p, "v", "t", 1, 2);
    for (int c = 0; c < 4; c++) {
        line(f, 545, 100 + 24 * c, 575, 100 + 24 * c, C[c].col, c == 0 ? 3.5 : 2.2, NULL, NULL);
        text(f, 583, 105 + 24 * c, "start", 13, INK, C[c].name);
    }
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"7\" fill=\"%s\" stroke=\"#fff\" stroke-width=\"2\"/>\n", PX(&p, 2), PY(&p, topt), INK);
    text(f, PX(&p, 2) + 12, PY(&p, topt) + 4, "start", 14, INK, "(v, t) = (2, 10.08)");
    text(f, 80, 440, "start", 13, GREY, "Shaded: the cone a = 0.3. Each cone is the region above its curve; all curves are even in v.");
    text(f, 80, 458, "start", 13, GREY, "The optimum of min t with u = 1, v = 2 is t = 2^(1/0.3) = 10.0794, on the boundary.");
    svg_end(f);
    fclose(f);
}

int main(int argc, char **argv) {
    const char *dir = ".";
    if (argc > 1) g_rev = argv[1];
    if (argc > 2) dir = argv[2];
    fig_simplex(dir);
    fig_ipm(dir);
    fig_qp(dir);
    fig_hyperplane(dir);
    fig_presolve(dir);
    fig_ordering(dir);
    fig_elliptope(dir);
    fig_expcone(dir);
    fig_powercone(dir);
    printf("make_figures: wrote simplex-pivots.svg, lp-central-path.svg, lp-central-path-trace.csv, qp-contours.svg, supporting-hyperplane.svg, presolve-reduction.svg, cholesky-ordering.svg, ordering-fill.csv, elliptope-slices.svg, exp-cone-section.svg, power-cone-section.svg in %s (revision %s)\n", dir, g_rev);
    return 0;
}
