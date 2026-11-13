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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* pexp_matrix.c -- the measurement matrix for the SOC / exponential-cone
 * engines.  It exists to LOCALISE cost, not to make the solver faster: the
 * instance family is parameterised so that the number of blocks, the density of
 * the links and the degeneracy can be moved one at a time.
 *
 *   pexp_matrix <family> <n> <d> <rho> <rep>
 *
 *   family  soc | exp | mix
 *   n       distinct blocks (observations for exp/mix, cone blocks for soc)
 *   d       dimension of x
 *   rho     fraction of the d features that each block links to (0 < rho <= 1)
 *   rep     copies of every block; rep>1 makes the model exactly degenerate
 *           (identical rows and identical cones -> the dual has a whole face
 *           of optima)
 *
 * The three families share the same ingredients so the numbers are comparable:
 *
 *   soc   min  c'x + 0.1 * sum_k w_k      s.t.  (w_k, 1, x) in RQUAD
 *         (n cone blocks over one x, no rows)
 *
 *   exp   min  sum_i v_i                  s.t.  u_i = a_i'x + b_i
 *                                               (v_i, 1, u_i) in PEXP
 *         (n cones, n rows; a_i has round(rho*d) nonzeros)
 *
 *   mix   the README regression model: 1 RQUAD for the quadratic penalty plus
 *         n PEXP cones, n rows
 *
 * Output, one line:
 *   matrix,<family>,<n>,<d>,<rho>,<rep>,<nblocks>,<nrows>,<seconds>,<pobj>,<pinf>,<dinf>
 *
 * The KKT order, the structural nonzeros, the chosen route and the iteration
 * count are not reachable through the public API; the runner
 * (bench/pexp_matrix.py) reads them from the GMB_DBG trace.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "primal.h"

static uint32_t rng_state;
static double draw(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (double)((rng_state >> 16) & 0x7fffu) / 32767.0;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <soc|exp|mix> <n> <d> <rho> <rep>\n", argv[0]);
        return 2;
    }
    const char *fam = argv[1];
    int n = atoi(argv[2]);
    int d = atoi(argv[3]);
    double rho = atof(argv[4]);
    int rep = atoi(argv[5]);
    if (n < 1 || d < 1 || rep < 1 || !(rho > 0.0 && rho <= 1.0)) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }
    int is_soc = strcmp(fam, "soc") == 0;
    int is_exp = strcmp(fam, "exp") == 0;
    int is_mix = strcmp(fam, "mix") == 0;
    if (!is_soc && !is_exp && !is_mix) { fprintf(stderr, "unknown family\n"); return 2; }
    rng_state = 11235813u;   /* the sequence bench/pexp_matrix.py repeats */

    int nb = n * rep;              /* cone blocks after replication */
    int nrow = (is_soc ? 0 : nb);  /* exp/mix: one linking row per block */
    int m = (int)(rho * d + 0.5);  /* nonzeros per row */
    if (m < 1) m = 1;
    if (m > d) m = d;

    /* Variables: x (d, free), then per family
     *   soc : w_0..w_{nb-1} (free), one
     *   exp : v_0..v_{nb-1} (>=0), u_0..u_{nb-1} (free), one
     *   mix : v, u and one w */
    /* soc: each block k owns its own x slice (x_k in R^d), so the blocks are
     * independent and the optimum is unique -- n identical blocks over one x
     * would be exactly degenerate and would measure that, not the engine. */
    const int dx = is_soc ? nb * d : d;
    const int X0 = 0;
    const int V0 = dx;
    const int U0 = (is_soc ? dx : dx + nb);
    const int ONE = (is_soc ? dx + nb : dx + 2 * nb);
    const int W = ONE + 1;      /* mix only */
    const int NV = (is_soc ? dx + nb + 1 : is_exp ? dx + 2 * nb + 1 : dx + 2 * nb + 2);

    /* Instance data, drawn once: the replicas reuse it verbatim (that is the
     * degeneracy: identical rows and cones). */
    double *A = (double *)malloc((size_t)n * (size_t)d * sizeof(double));
    double *bias = (double *)malloc((size_t)n * sizeof(double));
    int *idx = (int *)malloc((size_t)n * (size_t)m * sizeof(int));
    double *coef = (double *)malloc((size_t)n * (size_t)m * sizeof(double));
    double *cx = (double *)malloc((size_t)(nb * d) * sizeof(double));
    if (!A || !bias || !idx || !coef || !cx) return 2;
    for (size_t q = 0; q < (size_t)n * (size_t)d; q++) A[q] = (draw() * 2.0 - 1.0) * 0.5;
    for (int i = 0; i < n; i++) bias[i] = draw() * 2.0 - 1.0;
    for (int j = 0; j < nb * d; j++) cx[j] = draw() * 2.0 - 1.0;
    for (int i = 0; i < n; i++) {
        /* scattered link set of size m: the stride is coprime with d */
        int step = 1 + (int)(draw() * (double)(d > 1 ? d - 2 : 0) + 0.5);
        if (step >= d) step = 1;
        while (d > 1 && step > 1) {   /* keep gcd(step,d)=1 so the set is wide */
            int a = step, bb = d;
            while (bb) { int t2 = a % bb; a = bb; bb = t2; }
            if (a == 1) break;
            step++;
        }
        int j0 = (int)(draw() * (double)d);
        for (int k = 0; k < m; k++) {
            int j = (j0 + k * step) % d;
            idx[(size_t)i * m + k] = X0 + j;
            coef[(size_t)i * m + k] = A[(size_t)i * d + j];
        }
    }

    PRIMALenv_t env;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) return 2;
    PRIMALtask_t task;
    if (PRIMAL_maketask(env, 0, 0, &task) != PRIMAL_RES_OK) return 2;
    if (PRIMAL_appendvars(task, NV) != PRIMAL_RES_OK) return 2;
    if (nrow > 0 && PRIMAL_appendcons(task, nrow) != PRIMAL_RES_OK) return 2;
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    for (int j = 0; j < NV; j++)
        PRIMAL_putvarbound(task, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(task, ONE, PRIMAL_BK_FX, 1.0, 1.0);

    if (is_soc) {
        for (int j = 0; j < nb * d; j++) PRIMAL_putcj(task, X0 + j, cx[j]);
        for (int k = 0; k < nb; k++) {
            PRIMAL_putcj(task, V0 + k, 0.1);
            /* w_k >= 0 is implied by the rotated cone; saying it keeps the
             * standard form from splitting w into a free pair, whose lineality
             * direction is exactly what stalls an interior-point method. */
            PRIMAL_putvarbound(task, V0 + k, PRIMAL_BK_LO, 0.0, INFINITY);
        }
    } else if (is_exp) {
        for (int i = 0; i < nb; i++) {
            PRIMAL_putvarbound(task, V0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
            PRIMAL_putcj(task, V0 + i, 1.0);
        }
    } else {
        for (int i = 0; i < nb; i++) {
            PRIMAL_putvarbound(task, V0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
            PRIMAL_putcj(task, V0 + i, 1.0);
        }
        PRIMAL_putcj(task, W, 0.1);
    }

    if (nrow > 0) {
        int *sub = (int *)malloc((size_t)(m + 1) * sizeof(int));
        double *val = (double *)malloc((size_t)(m + 1) * sizeof(double));
        if (!sub || !val) return 2;
        for (int i = 0; i < n; i++) {
            int base = i % n;   /* replica r of block i is row i + n*r */
            for (int r = 0; r < rep; r++) {
                int row = i + n * r;
                for (int k = 0; k < m; k++) {
                    sub[k] = idx[(size_t)base * m + k];
                    val[k] = coef[(size_t)base * m + k];
                }
                sub[m] = U0 + row;
                val[m] = -1.0;
                PRIMAL_putarow(task, row, m + 1, sub, val);
                PRIMAL_putconbound(task, row, PRIMAL_BK_FX, -bias[base], -bias[base]);
            }
        }
        free(sub); free(val);
    }

    int *mem = (int *)malloc((size_t)(d + 2) * sizeof(int));
    if (!mem) return 2;
    if (is_soc) {
        for (int k = 0; k < nb; k++) {
            mem[0] = V0 + k; mem[1] = ONE;
            for (int j = 0; j < d; j++) mem[2 + j] = X0 + k * d + j;
            PRIMAL_appendcone(task, PRIMAL_CT_RQUAD, 0.0, d + 2, mem);
        }
    } else {
        for (int i = 0; i < nb; i++) {
            mem[0] = V0 + i; mem[1] = ONE; mem[2] = U0 + i;
            PRIMAL_appendcone(task, PRIMAL_CT_PEXP, 0.0, 3, mem);
        }
        if (is_mix) {
            mem[0] = W; mem[1] = ONE;
            for (int j = 0; j < d; j++) mem[2 + j] = X0 + j;
            PRIMAL_appendcone(task, PRIMAL_CT_RQUAD, 0.0, d + 2, mem);
        }
    }

    double t0 = now_sec();
    PRIMALrescodee rc = PRIMAL_optimize(task);
    double t1 = now_sec();

    double pobj = 0.0, dobj = 0.0, pinf = 0.0, dinf = 0.0;
    PRIMAL_getprimalobj(task, PRIMAL_SOL_ITR, &pobj);
    PRIMAL_getdualobj(task, PRIMAL_SOL_ITR, &dobj);
    PRIMAL_getprimalinfeas(task, PRIMAL_SOL_ITR, &pinf);
    PRIMAL_getdualinfeas(task, PRIMAL_SOL_ITR, &dinf);
    /* PEXP_MATRIX_DUMP=1 reports the model as the solver saw it and the worst
     * cone slack of the published point: the two numbers that tell a different
     * instance from a different answer.  (It exists because this generator once
     * forgot to seed its RNG and measured a different model than its Python
     * twin; the objective check in the runner is what caught it.) */
    if (getenv("PEXP_MATRIX_DUMP")) {
        int nv2 = 0, nc2 = 0, nk2 = 0;
        PRIMAL_getnumvar(task, &nv2); PRIMAL_getnumcon(task, &nc2);
        PRIMAL_getnumcone(task, &nk2);
        double *x = (double *)malloc((size_t)NV * sizeof(double));
        double worst = 0.0;
        if (x) {
            PRIMAL_getxx(task, PRIMAL_SOL_ITR, x);
            for (int k = 0; k < nb; k++) {
                double v = 0.0;
                for (int j = 0; j < d; j++) v += x[k * d + j] * x[k * d + j];
                double sl = (is_soc ? 2.0 * x[dx + k] : 0.0) - v;
                if (sl < worst) worst = sl;
            }
        }
        fprintf(stderr, "model nv=%d nc=%d ncone=%d cx0=%.17g worst=%.3e\n",
                nv2, nc2, nk2, cx[0], worst);
        free(x);
    }
    printf("matrix,%s,%d,%d,%.2f,%d,%d,%d,%.9f,%.12g,%.12g,%.3e,%.3e\n",
           fam, n, d, rho, rep, nb, nrow, t1 - t0, pobj, dobj, pinf, dinf);
    if (rc != PRIMAL_RES_OK)
        fprintf(stderr, "pexp_matrix: rc=%d %s n=%d d=%d rho=%.2f rep=%d\n",
                (int)rc, fam, n, d, rho, rep);

    free(mem); free(A); free(bias); free(idx); free(coef); free(cx);
    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? 0 : 1;
}
