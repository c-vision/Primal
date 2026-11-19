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
 *   simplex_walk   the tableau pivots of the LP of simplex.md, vertex by vertex, with the level
 *                  line of the objective sweeping up and the ratio test written out
 *   ipm_approach   the iterates of ipm_solve_std on the same LP, with the residuals drawn as they fall
 *
 * The solver traces reuse the still-figure code. Additional sequences show an analytic
 * central path, sections of the exponential cone and a checked branch-and-bound example.
 * For each animation <name> the program writes
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

static void plot_clip(FILE *f, const Plot *p) {
    fprintf(f, "<clipPath id=\"plot\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n",
            p->px, p->py, p->pw, p->ph);
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
    Seq sq; seq_open(&sq, dir, "simplex_walk");
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
    text(f, 450, 28, "middle", 20, INK, "Interior point: positive slacks and falling residuals");
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
    Seq sq; seq_open(&sq, dir, "ipm_approach");
    char a[200], b[200], previous[200];
    for (int k = 0; k < n; k++) {
        if (k == 0) {
            snprintf(a, sizeof a, "Iteration 0: positive coordinates and slacks, with an equality residual.");
            snprintf(b, sizeof b, "At the start: mu = %.2g, primal residual %.2g, dual residual %.2g.", tr[0].mu, tr[0].rp, tr[0].rd);
            ipm_frame(&sq, 150, tr, n, 0, 1.0, a, b);
            continue;
        }
        snprintf(a, sizeof a, "Iteration %d: the step length keeps the standard-form coordinates and dual slacks positive.", k);
        snprintf(b, sizeof b, "After the step: mu = %.2g, primal residual %.2g, dual residual %.2g.", tr[k].mu, tr[k].rp, tr[k].rd);
        snprintf(previous, sizeof previous, "Plot through iteration %d. Moving marker illustrates the step to iteration %d.", k - 1, k);
        for (int t = 1; t < 4; t++) ipm_frame(&sq, 8, tr, n, k, t / 4.0, a, previous);
        ipm_frame(&sq, 8, tr, n, k, 1.0, a, b);
        ipm_frame(&sq, k == n - 1 ? 400 : 45, tr, n, k, 1.0, a, b);
    }
    fclose(sq.list);
}

/* Exact minimizers of 1+r-mu*log(r*(1-r)) along x1+x2=1. */
static double segment_barrier(double mu, double r) {
    return 1 + r - mu * log(r * (1 - r));
}

static void anim_central_path(const char *dir) {
    const double scales[] = {2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.005};
    Seq seq;
    seq_open(&seq, dir, "central_path");
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++) {
        double mu = scales[i];
        double r = 2 * mu / (1 + 2 * mu + sqrt(1 + 4 * mu * mu));
        if (!(r > 0 && r < 1) || fabs(1 - mu/r + mu/(1-r)) > 1e-12)
            fail("central path: invalid barrier minimizer");
        char details[130];
        snprintf(details, sizeof details, "mu = %.3g, minimizing x2 = %.4f, x1 = %.4f", mu, r, 1-r);
        FILE *f = frame_open(&seq, i == 7 ? 240 : 95, 720, 432,
                             "Exact central path on a line segment", details);
        Plot p = {0, 1, 1, 5.4, 76, 65, 560, 230};
        text(f, 360, 32, "middle", 21, INK, "A barrier keeps the minimizer away from the endpoints");
        axes(f, &p, "x2", "value", 1, 1);
        line(f, PX(&p, 0), PY(&p, 1), PX(&p, 1), PY(&p, 2), GREY, 1.7, "5 4", NULL);
        plot_clip(f, &p);
        fputs("  <polyline clip-path=\"url(#plot)\" fill=\"none\" stroke=\"" BLUE "\" stroke-width=\"2.6\" points=\"", f);
        for (int j = 1; j < 240; j++) {
            double s = j / 240.0;
            fprintf(f, "%.1f,%.1f ", PX(&p, s), PY(&p, segment_barrier(mu, s)));
        }
        fputs("\"/>\n", f);
        double x = PX(&p, r);
        line(f, x, PY(&p, segment_barrier(mu, r)), x, 349, ORANGE, 1.6, "5 4", NULL);
        fprintf(f, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"5.5\" fill=\"%s\"/>\n",
                x, PY(&p, segment_barrier(mu, r)), ORANGE);
        line(f, p.px, 349, p.px+p.pw, 349, INK, 2, NULL, NULL);
        fprintf(f, "<circle cx=\"%.1f\" cy=\"349\" r=\"6\" fill=\"%s\"/>\n", x, ORANGE);
        text(f, 360, 382, "middle", 16, INK, details);
        text(f, 360, 410, "middle", 14, GREY, "Blue: barrier objective. Dashed diagonal: original objective.");
        svg_end(f);
        fclose(f);
    }
    fclose(seq.list);
}

static void anim_exponential_sections(const char *dir) {
    const double scales[] = {2, 1.6, 1.2, 1, 0.75, 0.5, 0.35, 0.2};
    Seq seq;
    seq_open(&seq, dir, "exponential_sections");
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++) {
        double u = scales[i];
        Plot p = {-2, 2, 0, 6, 70, 65, 580, 245};
        char label[90];
        snprintf(label, sizeof label, "Fixed scale u = %.2g. The boundary crosses v = 0 at t = u.", u);
        FILE *f = frame_open(&seq, i == 7 ? 240 : 95, 720, 412, "Cross-sections of the exponential cone", label);
        text(f, 360, 32, "middle", 22, INK, "Exponential cone sections at fixed u");
        plot_clip(f, &p);
        fprintf(f, "  <path clip-path=\"url(#plot)\" d=\"M %.1f %.1f ", PX(&p,p.x0), PY(&p,p.y1));
        for (int k = 0; k <= 200; k++) {
            double v = p.x0 + (p.x1-p.x0)*k/200;
            double t = u * exp(v/u);
            double point[3] = {t * 1.001, u, v}, value;
            if (expcone_barrier(EXPCONE_PEXP, 0, point, &value) != 0)
                fail("exponential animation: point above the boundary rejected");
            point[0] = t * 0.999;
            if (expcone_barrier(EXPCONE_PEXP, 0, point, &value) == 0)
                fail("exponential animation: point below the boundary accepted");
            fprintf(f, "L %.1f %.1f ", PX(&p,v), PY(&p,t));
        }
        fprintf(f, "L %.1f %.1f Z\" fill=\"%s\"/>\n", PX(&p,p.x1), PY(&p,p.y1), FILL);
        fputs("<polyline clip-path=\"url(#plot)\" fill=\"none\" stroke=\"" BLUE "\" stroke-width=\"2\" points=\"", f);
        for (int k = 0; k <= 200; k++) {
            double v = p.x0 + (p.x1-p.x0)*k/200;
            fprintf(f, "%.1f,%.1f ", PX(&p,v), PY(&p,u * exp(v/u)));
        }
        fputs("\"/>\n", f);
        line(f, PX(&p,0), PY(&p,0), PX(&p,0), PY(&p,p.y1), GREY, 1.2, "5 4", NULL);
        axes(f, &p, "v", "t", 1, 1);
        fprintf(f, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"5.5\" fill=\"%s\"/>\n", PX(&p,0), PY(&p,u), ORANGE);
        text(f, 360, 363, "middle", 16, INK, label);
        text(f, 360, 390, "middle", 15, GREY, "Shading marks feasible values t at or above u exp(v/u).");
        svg_end(f);
        fclose(f);
    }
    fclose(seq.list);
}

static void anim_branch_and_bound(const char *dir) {
    static const int xpos[] = {360, 176, 530, 438, 622};
    static const int ypos[] = {85, 177, 177, 269, 269};
    static const int parent[] = {-1, 0, 0, 2, 2};
    static const char *box[] = {"root", "x2 >= 3", "x2 <= 2", "x1 >= 2", "x1 <= 1"};
    static const char *bounds[] = {"bound -3.75", "infeasible", "bound -3.5", "bound -3", "bound -3"};
    static const char *caption[] = {
        "The root relaxation gives (1.25, 2.5). Branch on the fractional x2.",
        "The branch x2 >= 3 needs x1 >= 1.75, so it violates the first row.",
        "Branch on x1. A diving heuristic finds an incumbent with value -3.",
        "The branch x1 >= 2 attains -3 at the integer point (2, 1).",
        "The last bound is -3. This branch cannot improve the incumbent."
    };
    int best = 0;
    for (int x = 0; x <= 2; x++)
        for (int y = 0; y <= 3; y++)
            if (2*x+y <= 5 && -4*x+4*y <= 5 && -x-y < best) best = -x-y;
    if (best != -3) fail("branch-and-bound animation: wrong integer optimum");
    Seq seq;
    seq_open(&seq, dir, "branch_and_bound");
    for (int current = 0; current < 5; current++) {
        FILE *f = frame_open(&seq, current == 4 ? 300 : 175, 720, 420,
                             "Five nodes in a branch-and-bound search", caption[current]);
        text(f, 360, 33, "middle", 22, INK, "Branch and bound on a two-variable integer program");
        for (int i = 1; i <= current; i++) {
            int p = parent[i];
            line(f, xpos[p], ypos[p]+24, xpos[i], ypos[i]-24, GREY, 1.5, NULL, "arr");
        }
        for (int i = 0; i <= current; i++) {
            fprintf(f, "<rect x=\"%d\" y=\"%d\" width=\"158\" height=\"48\" rx=\"5\" fill=\"%s\" stroke=\"%s\" stroke-width=\"%d\"/>\n",
                    xpos[i]-79, ypos[i]-24, FILL, i == current ? ORANGE : BLUE, i == current ? 3 : 1);
            char name[48];
            snprintf(name, sizeof name, "%d. %s", i+1, box[i]);
            text(f, xpos[i], ypos[i]-3, "middle", 15, INK, name);
            text(f, xpos[i], ypos[i]+15, "middle", 13, GREY, bounds[i]);
        }
        text(f, 360, 345, "middle", 16, INK, caption[current]);
        text(f, 360, 381, "middle", 15, GREY,
             current < 2 ? "No incumbent found yet." : "Incumbent: (2, 1), objective -3.");
        svg_end(f);
        fclose(f);
    }
    fclose(seq.list);
}

static double tangent_master(double x, const double *anchors, int count) {
    double t = 0;
    for (int i = 0; i < count; i++) {
        double a = anchors[i];
        t = fmax(t, 2 * a * x - a * a);
    }
    return t;
}

static void anim_tangent_cuts(const char *dir) {
    const double anchors[] = {0, 1, 0.5};
    const double trial[] = {1, 0.5, 0.5};
    const double endpoints[] = {0, 0.25, 0.5, 0.75, 1};
    const double lower[] = {-1, -0.5, -0.25};
    const double upper[] = {0, -0.25, -0.25};
    Plot p = {0, 1, -0.25, 1.05, 82, 60, 550, 250};
    Seq seq;
    seq_open(&seq, dir, "tangent_cuts");
    for (int step = 0; step < 3; step++) {
        double x = trial[step];
        double t = tangent_master(x, anchors, step + 1);
        double best = t - x;
        /* Each linear piece takes its minimum at an endpoint. */
        for (size_t i = 0; i < sizeof endpoints / sizeof *endpoints; i++)
            best = fmin(best, tangent_master(endpoints[i], anchors, step + 1) - endpoints[i]);
        if (t - x != lower[step] || best != lower[step] || x * x - x != upper[step])
            fail("tangent cuts: invalid bounds");
        char label[100];
        snprintf(label, sizeof label, "Round %d: lower bound %g, upper bound %g.", step + 1, lower[step], upper[step]);
        FILE *f = frame_open(&seq, step == 2 ? 290 : 190, 720, 420,
                             "Tangent cuts for the parabola t = x squared", label);
        text(f, 360, 30, "middle", 21, INK, "Tangent cuts tighten the lower bound");
        axes(f, &p, "x", "t", 1, 1);
        line(f, PX(&p, 0.5), p.py + p.ph, PX(&p, 0.5), p.py + p.ph + 5, INK, 1, NULL, NULL);
        text(f, PX(&p, 0.5), p.py + p.ph + 20, "middle", 14, INK, "0.5");
        plot_clip(f, &p);
        fputs("<g clip-path=\"url(#plot)\">\n", f);
        for (int i = 0; i <= step; i++) {
            double a = anchors[i];
            line(f, PX(&p, 0), PY(&p, -a*a), PX(&p, 1), PY(&p, 2*a-a*a), GREY, 1.6, "5 5", NULL);
        }
        fputs("<polyline fill=\"none\" stroke=\"" BLUE "\" stroke-width=\"2.5\" points=\"", f);
        for (int i = 0; i <= 100; i++) {
            double u = i / 100.0;
            fprintf(f, "%.1f,%.1f ", PX(&p, u), PY(&p, u*u));
        }
        fputs("\"/>\n</g>\n", f);
        line(f, PX(&p, x), PY(&p, t), PX(&p, x), PY(&p, x*x), ORANGE, 2, NULL, NULL);
        fprintf(f, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"6\" fill=\"%s\"/>\n", PX(&p, x), PY(&p, t), ORANGE);
        text(f, 360, 363, "middle", 16, INK, label);
        text(f, 360, 390, "middle", 14, GREY, "Blue: parabola. Dashed: cuts. Orange: master point and feasibility shortfall.");
        svg_end(f);
        fclose(f);
    }
    fclose(seq.list);
}

int main(int argc, char **argv) {
    const char *dir = ".";
    if (argc > 1) g_rev = argv[1];
    if (argc > 2) dir = argv[2];
    anim_simplex(dir);
    anim_ipm(dir);
    anim_central_path(dir);
    anim_exponential_sections(dir);
    anim_branch_and_bound(dir);
    anim_tangent_cuts(dir);
    printf("make_animations: wrote frames for six animations in %s (revision %s)\n", dir, g_rev);
    return 0;
}
