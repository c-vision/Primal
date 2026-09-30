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

/* primal.c - the command-line front end: read a model file, solve it and
 * print the solution.  The model comes from a file in any format the readers
 * accept (MPS, CPLEX LP, OPF, CBF), detected by PRIMAL_readdataautoformat;
 * the algorithm is chosen by the dispatcher from what the model contains, so
 * LP, QP, SOCP, SDP, exp/power and MIP/MIQP all go through the same command.
 *
 * This is the program the issue asks for.  The whole implementation lives in
 * primal_main(argc, argv, out) so the test suite can drive it without exec;
 * main() is only a call to it and is compiled out with -DPRIMAL_NO_MAIN when
 * the file is linked into run_tests.
 *
 * Supported options: see usage() below.  Example inputs: lp_examples/.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "primal.h"

#define MAX_PARAMS 64

/* Parsed command line. */
typedef struct {
    const char *model;      /* positional: the model file (NULL if none) */
    const char *solsave;    /* --solution-file FILE */
    const char *modelsave;  /* --write FILE */
    const char *params[MAX_PARAMS];  /* --param NAME=VALUE, applied in order */
    int nparams;
    int max_iter;           /* --max-iter N, -1 = not set */
    double tol_pfeas, tol_dfeas, tol_gap;
    int full;               /* --full: also y, slacks */
    int brief;              /* --brief: only status and objective */
    int quiet;              /* -q: suppress the header */
    int sensitivity;        /* --sensitivity: also the LP cost/RHS ranges */
} cli_options;

/* Print the usage text. */
static void usage(FILE *out, const char *prog) {
    fprintf(out,
        "Usage: %s <model-file> [options]\n"
        "\n"
        "Read a model, solve it and print the solution.\n"
        "The model file may be MPS (.mps), CPLEX LP (.lp), OPF (.opf) or CBF\n"
        "(.cbf); the format is detected from the file, and the algorithm is\n"
        "chosen from the model itself (LP/QP/SOCP/SDP/exp-power/MIP).\n"
        "\n"
        "Options:\n"
        "  -o, --solution-file FILE  also write the solution to FILE (.json -> JSON)\n"
        "      --write FILE          write the model back (format conversion)\n"
        "      --max-iter N          iteration cap (IPAR_INTPNT_MAX_ITERATIONS)\n"
        "      --tol-pfeas V         primal feasibility tolerance\n"
        "      --tol-dfeas V         dual feasibility tolerance\n"
        "      --tol-gap V           relative gap tolerance\n"
        "      --param NAME=VALUE    set a named int or double parameter\n"
        "      --full                also print the duals and the slacks\n"
        "      --sensitivity         also print the LP cost/RHS ranges\n"
        "      --brief               only the status and the objective\n"
        "  -q, --quiet               suppress the header\n"
        "  -h, --help                print this help and exit\n"
        "  -V, --version             print the version and exit\n"
        "\n"
        "The input format is the interface: write the model in one of the\n"
        "formats above (lp_examples/README.md has one of each).\n", prog);
}

/* True when s ends with the suffix suf. */
static int ends_with(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + (ls - lf), suf) == 0;
}

/* Print one dense vector of n entries as "name[i] = value". */
/* One line per entry.  When the model carries a name for the entry (from a file
 * that names rows/columns) it is printed instead of the index; the name table is
 * the task's, read back through the public getters. */
static void print_vec(FILE *out, PRIMALtask_t t, int is_var,
                      const char *name, const double *v, int n) {
    for (int i = 0; i < n; i++) {
        const char *nm = NULL;
        if (is_var) PRIMAL_getvarnameidx(t, i, &nm);
        else        PRIMAL_getconnameidx(t, i, &nm);
        if (nm && nm[0]) fprintf(out, "%s[%s] = %.10g\n", name, nm, v[i]);
        else             fprintf(out, "%s[%d] = %.10g\n", name, i, v[i]);
    }
}

/* Print the status, objective, feasibility and the solution of a solved task.
 * brief stops after the objective; full adds y and the slack vectors. */
static void print_solution(PRIMALtask_t t, FILE *out, int full, int brief) {
    char ss[PRIMAL_MAX_STR_LEN], ps[PRIMAL_MAX_STR_LEN];
    PRIMALsolstae solsta = PRIMAL_SOL_STA_UNKNOWN;
    PRIMALprostae prosta = PRIMAL_PRO_STA_UNKNOWN;
    PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &solsta);
    PRIMAL_getprosta(t, PRIMAL_SOL_ITR, &prosta);
    PRIMAL_solstatostr(t, solsta, ss);
    PRIMAL_prostatostr(t, prosta, ps);
    fprintf(out, "status  : %s   prosta=%s  solsta=%s\n", ss, ps, ss);

    double pobj = 0.0, dobj = 0.0, pinf = 0.0, dinf = 0.0;
    int hp = PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &pobj) == PRIMAL_RES_OK;
    int hd = PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj) == PRIMAL_RES_OK;
    int hpi = PRIMAL_getprimalinfeas(t, PRIMAL_SOL_ITR, &pinf) == PRIMAL_RES_OK;
    int hdi = PRIMAL_getdualinfeas(t, PRIMAL_SOL_ITR, &dinf) == PRIMAL_RES_OK;

    fprintf(out, "obj     :");
    if (hp) fprintf(out, " primal = %.10g", pobj); else fprintf(out, " primal = n/a");
    if (hd) fprintf(out, "   dual = %.10g", dobj); else fprintf(out, "   dual = n/a");
    fprintf(out, "\n");
    if (!brief) {
        /* Primal infeasibility answers the user's question -- is the returned
         * point feasible? -- and is always shown.  The dual infeasibility is a
         * diagnostic and is shown only with --full. */
        fprintf(out, "viol    :");
        if (hpi) fprintf(out, " primal = %.3e", pinf); else fprintf(out, " primal = n/a");
        if (full) { if (hdi) fprintf(out, "   dual = %.3e", dinf); else fprintf(out, "   dual = n/a"); }
        fprintf(out, "\n");
    }
    if (brief) return;

    int nv = 0, nc = 0, nb = 0;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getnumcon(t, &nc);
    PRIMAL_getnumbarvar(t, &nb);

    if (nv > 0) {
        double *x = (double *)malloc((size_t)nv * sizeof(double));
        if (x && PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == PRIMAL_RES_OK) print_vec(out, t, 1, "x", x, nv);
        else fprintf(out, "x       : (no primal point published)\n");
        free(x);
    }
    for (int j = 0; j < nb; j++) {
        int d = 0;
        PRIMAL_getdimbarvarj(t, j, &d);
        double *X = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
        if (X && PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, j, X) == PRIMAL_RES_OK) {
            fprintf(out, "barx[%d] =\n", j);
            for (int i = 0; i < d; i++) {
                fprintf(out, "  ");
                for (int k = 0; k < d; k++) fprintf(out, " %.8g", X[(size_t)i * d + k]);
                fprintf(out, "\n");
            }
        }
        free(X);
    }
    if (full && nc > 0) {
        double *y = (double *)malloc((size_t)nc * sizeof(double));
        if (y && PRIMAL_gety(t, PRIMAL_SOL_ITR, y) == PRIMAL_RES_OK) print_vec(out, t, 0, "y", y, nc);
        free(y);
    }
    if (full) {
        double *v = nv > 0 ? (double *)malloc((size_t)nv * sizeof(double)) : NULL;
        if (v && PRIMAL_getslx(t, PRIMAL_SOL_ITR, v) == PRIMAL_RES_OK) print_vec(out, t, 1, "slx", v, nv);
        if (v && PRIMAL_getsux(t, PRIMAL_SOL_ITR, v) == PRIMAL_RES_OK) print_vec(out, t, 1, "sux", v, nv);
        free(v);
        v = nc > 0 ? (double *)malloc((size_t)nc * sizeof(double)) : NULL;
        if (v && PRIMAL_getslc(t, PRIMAL_SOL_ITR, v) == PRIMAL_RES_OK) print_vec(out, t, 0, "slc", v, nc);
        if (v && PRIMAL_getsuc(t, PRIMAL_SOL_ITR, v) == PRIMAL_RES_OK) print_vec(out, t, 0, "suc", v, nc);
        free(v);
    }
}

/* Render one range endpoint (may be +-inf) into b. */
static const char *rng_str(char *b, size_t n, double v) {
    if (v == INFINITY) { snprintf(b, n, "inf"); return b; }
    if (v == -INFINITY) { snprintf(b, n, "-inf"); return b; }
    snprintf(b, n, "%.10g", v);
    return b;
}

/* Print the LP post-optimal ranges: the interval of each objective coefficient
 * and of each active RHS over which the published optimum stays optimal, from
 * the public sensitivity API (PRIMAL_costsensitivity / PRIMAL_rhssensitivity).
 * That API refuses anything but a solved LP, so on other classes this says so
 * instead of printing a range that does not exist. */
static void print_ranges(PRIMALtask_t t, FILE *out) {
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getnumcon(t, &nc);
    double lo = 0.0, up = 0.0;
    if (nv > 0 && PRIMAL_costsensitivity(t, 0, &lo, &up) != PRIMAL_RES_OK) {
        fprintf(out, "ranges  : not available (cost/RHS ranges are LP-only)\n");
        return;
    }
    for (int j = 0; j < nv; j++) {
        if (PRIMAL_costsensitivity(t, j, &lo, &up) != PRIMAL_RES_OK) continue;
        const char *nm = NULL;
        PRIMAL_getvarnameidx(t, j, &nm);
        char b1[32], b2[32];
        fprintf(out, "cost[%s] in [%s, %s]\n", nm && nm[0] ? nm : "?",
                rng_str(b1, sizeof b1, lo), rng_str(b2, sizeof b2, up));
    }
    for (int i = 0; i < nc; i++) {
        if (PRIMAL_rhssensitivity(t, i, &lo, &up) != PRIMAL_RES_OK) break;
        const char *nm = NULL;
        PRIMAL_getconnameidx(t, i, &nm);
        char b1[32], b2[32];
        fprintf(out, "rhs [%s] in [%s, %s]\n", nm && nm[0] ? nm : "?",
                rng_str(b1, sizeof b1, lo), rng_str(b2, sizeof b2, up));
    }
}

/* Apply the --param NAME=VALUE list to a task.  Returns 0 on success, -1 on an
 * unknown parameter name or a malformed NAME=VALUE. */
static int apply_params(PRIMALtask_t t, const cli_options *o, FILE *err) {
    for (int i = 0; i < o->nparams; i++) {
        const char *eq = strchr(o->params[i], '=');
        if (!eq) { fprintf(err, "primal: --param needs NAME=VALUE, got '%s'\n", o->params[i]); return -1; }
        char name[256];
        size_t nl = (size_t)(eq - o->params[i]);
        if (nl >= sizeof name) nl = sizeof name - 1;
        memcpy(name, o->params[i], nl);
        name[nl] = 0;
        const char *val = eq + 1;
        int id = 0;
        if (PRIMAL_isintparname(t, name, &id) == PRIMAL_RES_OK)
            PRIMAL_putnaintparam(t, name, (int)strtol(val, NULL, 10));
        else if (PRIMAL_isdouparname(t, name, &id) == PRIMAL_RES_OK)
            PRIMAL_putnadouparam(t, name, strtod(val, NULL));
        else { fprintf(err, "primal: unknown parameter '%s'\n", name); return -1; }
    }
    return 0;
}

/* Entry point.  out receives the normal output (stdout for the program, a
 * temporary file in the tests); errors always go to stderr.  Returns 0 when
 * the model is solved to optimality, 1 when it is not, 2 on a usage or I/O
 * error. */
int primal_main(int argc, char **argv, FILE *out) {
    cli_options o;
    memset(&o, 0, sizeof o);
    o.max_iter = -1;
    const char *prog = (argc > 0 && argv[0]) ? argv[0] : "primal";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(out, prog); return 0; }
        else if (!strcmp(a, "-V") || !strcmp(a, "--version")) {
            int maj = 0, min = 0, rev = 0;
            PRIMAL_getversion(&maj, &min, &rev);
            fprintf(out, "PrimalSolver %d.%d.%d\n", maj, min, rev);
            return 0;
        }
        else if (!strcmp(a, "-o") || !strcmp(a, "--solution-file")) {
            if (++i >= argc) { fprintf(stderr, "primal: %s needs a FILE\n", a); return 2; }
            o.solsave = argv[i];
        }
        else if (!strcmp(a, "--write")) {
            if (++i >= argc) { fprintf(stderr, "primal: --write needs a FILE\n"); return 2; }
            o.modelsave = argv[i];
        }
        else if (!strcmp(a, "--max-iter")) {
            if (++i >= argc) { fprintf(stderr, "primal: --max-iter needs N\n"); return 2; }
            o.max_iter = (int)strtol(argv[i], NULL, 10);
        }
        else if (!strcmp(a, "--tol-pfeas")) {
            if (++i >= argc) { fprintf(stderr, "primal: --tol-pfeas needs V\n"); return 2; }
            o.tol_pfeas = strtod(argv[i], NULL);
        }
        else if (!strcmp(a, "--tol-dfeas")) {
            if (++i >= argc) { fprintf(stderr, "primal: --tol-dfeas needs V\n"); return 2; }
            o.tol_dfeas = strtod(argv[i], NULL);
        }
        else if (!strcmp(a, "--tol-gap")) {
            if (++i >= argc) { fprintf(stderr, "primal: --tol-gap needs V\n"); return 2; }
            o.tol_gap = strtod(argv[i], NULL);
        }
        else if (!strcmp(a, "--param")) {
            if (++i >= argc) { fprintf(stderr, "primal: --param needs NAME=VALUE\n"); return 2; }
            if (o.nparams >= MAX_PARAMS) { fprintf(stderr, "primal: too many --param\n"); return 2; }
            o.params[o.nparams++] = argv[i];
        }
        else if (!strcmp(a, "--full")) o.full = 1;
        else if (!strcmp(a, "--sensitivity") || !strcmp(a, "--ranges")) o.sensitivity = 1;
        else if (!strcmp(a, "--brief")) o.brief = 1;
        else if (!strcmp(a, "-q") || !strcmp(a, "--quiet")) o.quiet = 1;
        else if (a[0] == '-' && a[1] != 0) { fprintf(stderr, "primal: unknown option '%s'\n", a); usage(stderr, prog); return 2; }
        else {
            if (o.model) { fprintf(stderr, "primal: only one model file is accepted\n"); return 2; }
            o.model = a;
        }
    }
    if (!o.model) { usage(stderr, prog); return 2; }

    PRIMALenv_t env = NULL;
    PRIMALtask_t t = NULL;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) { fprintf(stderr, "primal: cannot create the environment\n"); return 2; }
    if (PRIMAL_maketask(env, 0, 0, &t) != PRIMAL_RES_OK) { fprintf(stderr, "primal: cannot create the task\n"); PRIMAL_deleteenv(&env); return 2; }

    /* options that need the task */
    if (o.max_iter >= 0) PRIMAL_putintparam(t, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, o.max_iter);
    PRIMAL_putdouparam(t, PRIMAL_DPAR_INTPNT_TOL_PFEAS, o.tol_pfeas > 0 ? o.tol_pfeas : 1e-8);
    PRIMAL_putdouparam(t, PRIMAL_DPAR_INTPNT_TOL_DFEAS, o.tol_dfeas > 0 ? o.tol_dfeas : 1e-8);
    PRIMAL_putdouparam(t, PRIMAL_DPAR_INTPNT_TOL_REL_GAP, o.tol_gap > 0 ? o.tol_gap : 1e-8);
    if (apply_params(t, &o, stderr) != 0) { PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env); return 2; }

    PRIMALrescodee rc = PRIMAL_readdataautoformat(t, o.model);
    if (rc != PRIMAL_RES_OK) {
        fprintf(stderr, "primal: cannot read '%s' (rc=%d)\n", o.model, (int)rc);
        PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
        return 2;
    }

    if (o.modelsave) {
        PRIMALrescodee w = PRIMAL_writedata(t, o.modelsave);
        if (w != PRIMAL_RES_OK) fprintf(stderr, "primal: cannot write '%s' (rc=%d)\n", o.modelsave, (int)w);
    }

    int nv = 0, nc = 0, nb = 0;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getnumcon(t, &nc);
    PRIMAL_getnumbarvar(t, &nb);
    if (!o.quiet) {
        int maj = 0, min = 0, rev = 0;
        PRIMAL_getversion(&maj, &min, &rev);
        fprintf(out, "PrimalSolver %d.%d.%d\n", maj, min, rev);
        fprintf(out, "file    : %s   (numvar=%d  numcon=%d", o.model, nv, nc);
        if (nb > 0) fprintf(out, "  numbarvar=%d", nb);
        fprintf(out, ")\n");
    }

    rc = PRIMAL_optimize(t);

    if (o.solsave) {
        PRIMALrescodee w = ends_with(o.solsave, ".json")
            ? PRIMAL_writejsonsol(t, o.solsave)
            : PRIMAL_writesolution(t, PRIMAL_SOL_ITR, o.solsave);
        if (w != PRIMAL_RES_OK) fprintf(stderr, "primal: cannot write the solution to '%s' (rc=%d)\n", o.solsave, (int)w);
    }

    print_solution(t, out, o.full, o.brief);
    if (o.sensitivity) print_ranges(t, out);

    PRIMALsolstae solsta = PRIMAL_SOL_STA_UNKNOWN;
    PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &solsta);
    int optimal = (rc == PRIMAL_RES_OK) &&
                  (solsta == PRIMAL_SOL_STA_OPTIMAL || solsta == PRIMAL_SOL_STA_INTEGER_OPTIMAL);

    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return optimal ? 0 : 1;
}

#ifndef PRIMAL_NO_MAIN
int main(int argc, char **argv) { return primal_main(argc, argv, stdout); }
#endif
