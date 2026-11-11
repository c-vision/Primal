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

/* make_animations.c - writes the SVG frames of the animated figures of algorithms/.
 *
 *   simplex-walk   the tableau pivots of the LP of simplex.md, vertex by vertex, with the level
 *                  line of the objective sweeping up and the ratio test written out
 *   ipm-approach   the iterates of ipm_solve_std on the same LP, with the residuals drawn as they fall
 *
 * Every frame is computed from the same data as the still figures (make_figures.c is included
 * for its LP, its tableau and its interior-point trace) and the final positions are checked
 * against the optimum the public API reports.  For each animation <name> the program writes
 * <dir>/<name>-NN.svg and a list <dir>/<name>.frames with one "CS file.svg" line per frame
 * (CS = display time in hundredths of a second); make_gifs.sh turns the lists into GIFs.
 *
 *   cc -std=c99 -O2 -I../.. -o make_animations make_animations.c <libprimal.a> -lm -lpthread
 *   make_animations <revision> <dir>
 */
#define main make_figures_main
#include "make_figures.c"
#undef main

typedef struct { FILE *list; const char *dir; const char *name; int n; } Seq;

static FILE *seq_open(Seq *s, const char *dir, const char *name) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.frames", dir, name);
    s->list = fopen(path, "w"); if (!s->list) fail("cannot open the frame list");
    s->dir = dir; s->name = name; s->n = 0;
    return s->list;
}

/* start frame number s->n, hold it for cs hundredths of a second */
static FILE *frame_open(Seq *s, int cs, int w, int h, const char *title, const char *desc) {
    char path[512], rel[256];
    snprintf(rel, sizeof rel, "%s-%02d.svg", s->name, s->n++);
    snprintf(path, sizeof path, "%s/%s", s->dir, rel);
    FILE *f = fopen(path, "w"); if (!f) fail("cannot open a frame");
    fprintf(s->list, "%d %s\n", cs, rel);
    svg_begin(f, w, h, title, desc);
    return f;
}

/* ------------------------------------------------------------------------- */
/* Simplex: the tableau run of make_figures.c with the numbers of every pivot   */

typedef struct {
    double x, y;                 /* vertex after the pivot */
    int enter, leave;
    double rc;                   /* reduced cost of the entering column */
    double ratio[LP_M];          /* b_i / a_ie for a_ie > 0, else -1 */
    double rcnow[LP_N];          /* reduced costs after the pivot */
    double z;                    /* objective 3x + 2y after the pivot */
} Step;

static int tableau_steps(Step *out, int maxp) {
    double T[LP_M + 1][LP_N + 1];
    int basis[LP_M] = { 2, 3 };
    for (int i = 0; i < LP_M; i++) {
        for (int j = 0; j < LP_N; j++) T[i][j] = LP_A[i * LP_N + j];
        T[i][LP_N] = LP_B[i];
    }
    for (int j = 0; j < LP_N; j++) T[LP_M][j] = LP_C[j];
    T[LP_M][LP_N] = 0.0;
    int np = 0;
    memset(&out[0], 0, sizeof out[0]); out[0].enter = out[0].leave = -1;
    for (int j = 0; j < LP_N; j++) out[0].rcnow[j] = T[LP_M][j];
    np = 1;
    for (int it = 0; it < 10; it++) {
        int e = -1; double best = -1e-12;
        for (int j = 0; j < LP_N; j++) if (T[LP_M][j] < best) { best = T[LP_M][j]; e = j; }
        if (e < 0) break;
        Step *s = &out[np];
        memset(s, 0, sizeof *s);
        s->enter = e; s->rc = best;
        int r = -1; double ratio = 1e300;
        for (int i = 0; i < LP_M; i++) {
            s->ratio[i] = T[i][e] > 1e-12 ? T[i][LP_N] / T[i][e] : -1.0;
            if (T[i][e] > 1e-12 && T[i][LP_N] / T[i][e] < ratio) { ratio = T[i][LP_N] / T[i][e]; r = i; }
        }
        if (r < 0) fail("tableau: unbounded");
        double pv = T[r][e];
        for (int j = 0; j <= LP_N; j++) T[r][j] /= pv;
        for (int i = 0; i <= LP_M; i++) {
            if (i == r) continue;
            double fct = T[i][e];
            for (int j = 0; j <= LP_N; j++) T[i][j] -= fct * T[r][j];
        }
        s->leave = basis[r]; basis[r] = e;
        double xv[LP_N] = { 0, 0, 0, 0 };
        for (int i = 0; i < LP_M; i++) xv[basis[i]] = T[i][LP_N];
        s->x = xv[0]; s->y = xv[1]; s->z = 3 * xv[0] + 2 * xv[1];
        for (int j = 0; j < LP_N; j++) s->rcnow[j] = T[LP_M][j];
        if (++np >= maxp) break;
    }
    return np;
}

static void simplex_frame(Seq *sq, int cs, const Plot *p, const Step *S, int k, double t,
                          const char *l1, const char *l2, const char *l3) {
    double V[8][2]; int nv = polygon(V, 8);
    /* position: between vertex k-1 and k at fraction t (t = 1: at k) */
    double px, py;
    if (k == 0) { px = S[0].x; py = S[0].y; }
    else { px = S[k - 1].x + t * (S[k].x - S[k - 1].x); py = S[k - 1].y + t * (S[k].y - S[k - 1].y); }
    double z = 3 * px + 2 * py;
    char desc[400];
    snprintf(desc, sizeof desc, "Simplex on max 3x+2y with x+y<=4, x<=2: the current point (%.2f, %.2f) with objective %.2f, "
             "on the way from (0,0) through (2,0) to (2,2). Computed at revision %s.", px, py, z, g_rev);
    FILE *f = frame_open(sq, cs, 700, 500, "Simplex walking along the edges of the polygon", desc);
    fprintf(f, "  <clipPath id=\"win\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n", p->px, p->py, p->pw, p->ph);
    text(f, 350, 30, "middle", 20, INK, "Simplex: from vertex to vertex, always uphill");
    fputs("  <path d=\"", f);
    for (int i = 0; i < nv; i++) fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(p, V[i][0]), PY(p, V[i][1]));
    fprintf(f, "Z\" fill=\"%s\" stroke=\"none\"/>\n", FILL);
    line(f, PX(p, 0), PY(p, 4), PX(p, 4), PY(p, 0), BLUE, 3, NULL, NULL);
    line(f, PX(p, 2), PY(p, 0), PX(p, 2), PY(p, 5), BLUE, 3, NULL, NULL);
    axes(f, p, "x", "y", 1, 1);
    text(f, PX(p, 3.05), PY(p, 1.30), "start", 14, BLUE, "x + y = 4");
    text(f, PX(p, 2.08), PY(p, 4.60), "start", 14, BLUE, "x = 2");
    /* level line 3x + 2y = z through the current point, direction (2, -3) */
    fputs("  <g clip-path=\"url(#win)\">\n", f);
    line(f, PX(p, px - 4.0), PY(p, py + 6.0), PX(p, px + 4.0), PY(p, py - 6.0), GREY, 2, "7 5", NULL);
    fputs("  </g>\n", f);
    char s[64]; snprintf(s, sizeof s, "dashed: 3x + 2y = %.1f", z);
    text(f, PX(p, 2.9), PY(p, 4.85), "start", 14, GREY, s);
    /* the path so far */
    for (int i = 1; i <= k; i++) {
        double x1 = S[i - 1].x, y1 = S[i - 1].y, x2 = S[i].x, y2 = S[i].y;
        if (i == k) { x2 = px; y2 = py; }
        line(f, PX(p, x1), PY(p, y1), PX(p, x2), PY(p, y2), ORANGE, 4, NULL, NULL);
    }
    for (int i = 0; i < k || (k == 0 && i == 0); i++)
        fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"%s\"/>\n", PX(p, S[i].x), PY(p, S[i].y), INK);
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"8\" fill=\"%s\" stroke=\"#fff\" stroke-width=\"2\"/>\n", PX(p, px), PY(p, py), ORANGE);
    snprintf(s, sizeof s, "z = %.1f", z);
    text(f, PX(p, px) + 14, PY(p, py) - 12, "start", 15, INK, s);
    text(f, 70, 440, "start", 15, INK, l1);
    text(f, 70, 462, "start", 14, ORANGE, l2);
    text(f, 70, 484, "start", 14, GREY, l3);
    svg_end(f);
    fclose(f);
}

static void anim_simplex(const char *dir) {
    double ref[2], refobj;
    lp_reference(&ref[0], &ref[1], &refobj);
    Step S[8]; int ns = tableau_steps(S, 8);
    if (ns != 3) fail("simplex animation: expected 2 pivots");
    if (fabs(S[ns - 1].x - ref[0]) > 1e-9 || fabs(S[ns - 1].y - ref[1]) > 1e-9) fail("simplex animation: optimum differs from PRIMAL");
    if (fabs(S[ns - 1].z - refobj) > 1e-7) fail("simplex animation: objective differs from PRIMAL");
    if (fabs(S[ns - 1].rcnow[2] - g_ref_y[0] * -1.0) > 1e-9 && fabs(S[ns - 1].rcnow[2] - fabs(g_ref_y[0])) > 1e-9)
        fail("simplex animation: final reduced cost of slack 1 is not the row multiplier");
    Seq sq; seq_open(&sq, dir, "simplex-walk");
    Plot p = { 0, 5, 0, 5, 70, 60, 470, 330 };
    char a[200], b[200], c[200];
    snprintf(a, sizeof a, "Start at the vertex (0, 0): the slack variables are basic, z = 0.");
    snprintf(b, sizeof b, "Reduced costs of (x, y): (%g, %g). Both are negative, so z can still grow.", S[0].rcnow[0], S[0].rcnow[1]);
    snprintf(c, sizeof c, "Dantzig's rule picks the most negative one.");
    simplex_frame(&sq, 150, &p, S, 0, 1.0, a, b, c);
    for (int k = 1; k < ns; k++) {
        char r1[32], r2[32];
        if (S[k].ratio[0] >= 0) snprintf(r1, sizeof r1, "%g", S[k].ratio[0]); else snprintf(r1, sizeof r1, "-");
        if (S[k].ratio[1] >= 0) snprintf(r2, sizeof r2, "%g", S[k].ratio[1]); else snprintf(r2, sizeof r2, "-");
        snprintf(a, sizeof a, "Pivot %d: %s enters the basis (reduced cost %g).", k, colname(S[k].enter), S[k].rc);
        snprintf(b, sizeof b, "Ratio test: row 1 gives %s, row 2 gives %s; the smallest wins, %s leaves.", r1, r2, colname(S[k].leave));
        snprintf(c, sizeof c, "The point slides along an edge until the leaving variable reaches 0.");
        simplex_frame(&sq, 90, &p, S, k, 0.0, a, b, c);
        for (int t = 1; t <= 6; t++) simplex_frame(&sq, 8, &p, S, k, t / 6.0, a, b, c);
        if (k < ns - 1) {
            snprintf(a, sizeof a, "Vertex (%g, %g) reached: z = %g.", S[k].x, S[k].y, S[k].z);
            snprintf(b, sizeof b, "Reduced costs now (%g, %g, %g, %g): %s is still negative.",
                     S[k].rcnow[0], S[k].rcnow[1], S[k].rcnow[2], S[k].rcnow[3], colname(S[k + 1].enter));
            snprintf(c, sizeof c, "Next pivot.");
            simplex_frame(&sq, 110, &p, S, k, 1.0, a, b, c);
        }
    }
    snprintf(a, sizeof a, "Vertex (%g, %g): all reduced costs are >= 0, so this is optimal, z = %g.", S[ns - 1].x, S[ns - 1].y, S[ns - 1].z);
    snprintf(b, sizeof b, "Reduced costs of the slacks (%g, %g) are the dual multipliers of the two rows.", S[ns - 1].rcnow[2], S[ns - 1].rcnow[3]);
    snprintf(c, sizeof c, "Certificate: %g * 4 + %g * 2 = %g. The unvisited vertex (0, 4) has z = 8.",
             S[ns - 1].rcnow[2], S[ns - 1].rcnow[3], S[ns - 1].rcnow[2] * 4 + S[ns - 1].rcnow[3] * 2);
    simplex_frame(&sq, 400, &p, S, ns - 1, 1.0, a, b, c);
    fclose(sq.list);
}

/* ------------------------------------------------------------------------- */
/* Interior point: iterate by iterate, with the residuals drawn as they fall     */

static void ipm_frame(Seq *sq, int cs, const Iter *tr, int n, int k, double t, const char *l1, const char *l2) {
    /* current position: between iterate k-1 and k at fraction t */
    double px = tr[k].x[0], py = tr[k].x[1];
    if (k > 0) { px = tr[k - 1].x[0] + t * (tr[k].x[0] - tr[k - 1].x[0]); py = tr[k - 1].x[1] + t * (tr[k].x[1] - tr[k - 1].x[1]); }
    int kc = (k > 0 && t < 1.0) ? k - 1 : k;        /* residuals shown up to the iterate already reached */
    char desc[400];
    snprintf(desc, sizeof desc, "Interior point on max 3x+2y with x+y<=4, x<=2: iteration %d of %d, current point (%.3f, %.3f). "
             "Measured with ipm_solve_std at revision %s.", k, n - 1, px, py, g_rev);
    FILE *f = frame_open(sq, cs, 900, 470, "Interior point iterates and residuals", desc);
    text(f, 450, 28, "middle", 20, INK, "Interior point: through the inside, towards the vertex");
    Plot p = { 0, 4.4, 0, 4.4, 60, 70, 330, 300 };
    double V[8][2]; int nv = polygon(V, 8);
    fputs("  <path d=\"", f);
    for (int i = 0; i < nv; i++) fprintf(f, "%c%.1f %.1f", i ? 'L' : 'M', PX(&p, V[i][0]), PY(&p, V[i][1]));
    fprintf(f, "Z\" fill=\"%s\" stroke=\"%s\" stroke-width=\"2.5\"/>\n", FILL, BLUE);
    axes(f, &p, "x", "y", 1, 1);
    for (int i = 1; i <= k; i++) {
        double x2 = tr[i].x[0], y2 = tr[i].x[1];
        if (i == k) { x2 = px; y2 = py; }
        line(f, PX(&p, tr[i - 1].x[0]), PY(&p, tr[i - 1].x[1]), PX(&p, x2), PY(&p, y2), ORANGE, 2, NULL, NULL);
    }
    for (int i = 0; i <= kc; i++)
        fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%s\" fill=\"%s\"/>\n", PX(&p, tr[i].x[0]), PY(&p, tr[i].x[1]),
                i == 0 ? "4.5" : "3.5", i == 0 ? INK : ORANGE);
    fprintf(f, "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"7\" fill=\"%s\" stroke=\"#fff\" stroke-width=\"2\"/>\n", PX(&p, px), PY(&p, py), ORANGE);
    text(f, PX(&p, 2) + 12, PY(&p, 2) - 12, "start", 14, INK, "optimum (2, 2)");
    text(f, PX(&p, 1) + 10, PY(&p, 1) + 24, "start", 14, INK, "start (1, 1)");

    Plot q = { 0, (double)(n - 1), -16, 1, 500, 70, 360, 300 };
    for (int e = -16; e <= 0; e += 4) {
        char s[16]; snprintf(s, sizeof s, "1e%d", e);
        line(f, q.px, PY(&q, e), q.px + q.pw, PY(&q, e), "#e5e7eb", 1, NULL, NULL);
        text(f, q.px - 8, PY(&q, e) + 5, "end", 13, INK, s);
    }
    line(f, q.px, q.py + q.ph, q.px + q.pw + 12, q.py + q.ph, INK, 1.5, NULL, "arr");
    line(f, q.px, q.py + q.ph, q.px, q.py - 12, INK, 1.5, NULL, "arr");
    for (int i = 0; i < n; i++) {
        char s[16]; snprintf(s, sizeof s, "%d", i);
        line(f, PX(&q, i), q.py + q.ph, PX(&q, i), q.py + q.ph + 5, INK, 1, NULL, NULL);
        text(f, PX(&q, i), q.py + q.ph + 20, "middle", 13, INK, s);
    }
    text(f, q.px + q.pw / 2, q.py + q.ph + 42, "middle", 15, INK, "iteration");
    static const char *names[3] = { "primal residual", "dual residual", "complementarity mu" };
    static const char *dash[3] = { NULL, "7 4", "2 4" };
    static const char *stroke[3] = { BLUE, ORANGE, INK };
    for (int s = 0; s < 3; s++) {
        fprintf(f, "  <polyline fill=\"none\" stroke=\"%s\" stroke-width=\"2.2\"", stroke[s]);
        if (dash[s]) fprintf(f, " stroke-dasharray=\"%s\"", dash[s]);
        fputs(" points=\"", f);
        for (int i = 0; i <= kc; i++) {
            double v = s == 0 ? tr[i].rp : s == 1 ? tr[i].rd : tr[i].mu;
            fprintf(f, "%.1f,%.1f ", PX(&q, i), PY(&q, lg(v)));
        }
        fputs("\"/>\n", f);
        line(f, q.px + 14, q.py + q.ph - 62 + 20 * s, q.px + 54, q.py + q.ph - 62 + 20 * s, stroke[s], 2.2, dash[s], NULL);
        text(f, q.px + 62, q.py + q.ph - 57 + 20 * s, "start", 13, INK, names[s]);
    }
    text(f, 60, 424, "start", 15, INK, l1);
    text(f, 60, 446, "start", 14, GREY, l2);
    svg_end(f);
    fclose(f);
}

static void anim_ipm(const char *dir) {
    double ref[2], refobj;
    lp_reference(&ref[0], &ref[1], &refobj);
    Iter tr[MAXTRACE]; int n = ipm_trace(tr, MAXTRACE);
    if (tr[n - 1].status != 0) fail("ipm animation: interior point did not converge");
    if (fabs(tr[n - 1].x[0] - ref[0]) > 1e-6 || fabs(tr[n - 1].x[1] - ref[1]) > 1e-6) fail("ipm animation: optimum differs from PRIMAL");
    Seq sq; seq_open(&sq, dir, "ipm-approach");
    char a[200], b[200];
    for (int k = 0; k < n; k++) {
        if (k == 0) {
            snprintf(a, sizeof a, "Iteration 0: start strictly inside, at (1, 1).");
            snprintf(b, sizeof b, "At the start: mu = %.2g, primal residual %.2g, dual residual %.2g.", tr[0].mu, tr[0].rp, tr[0].rd);
            ipm_frame(&sq, 150, tr, n, 0, 1.0, a, b);
            continue;
        }
        snprintf(a, sizeof a, "Iteration %d: one Newton step towards a smaller mu, kept inside by the step length.", k);
        snprintf(b, sizeof b, "After the step: mu = %.2g, primal residual %.2g, dual residual %.2g.", tr[k].mu, tr[k].rp, tr[k].rd);
        for (int t = 1; t <= 4; t++) ipm_frame(&sq, 8, tr, n, k, t / 4.0, a, b);
        ipm_frame(&sq, k == n - 1 ? 400 : 45, tr, n, k, 1.0, a, b);
    }
    fclose(sq.list);
}

int main(int argc, char **argv) {
    const char *dir = ".";
    if (argc > 1) g_rev = argv[1];
    if (argc > 2) dir = argv[2];
    anim_simplex(dir);
    anim_ipm(dir);
    printf("make_animations: wrote the frames of simplex-walk and ipm-approach in %s (revision %s)\n", dir, g_rev);
    return 0;
}
