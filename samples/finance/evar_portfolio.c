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

/* evar_portfolio.c - entropic value-at-risk portfolio (MOSEK Portfolio
 * Optimization Cookbook, chapter 8 "EVaR optimization").
 *
 * Source: MOSEK/PortfolioOptimization, python/notebooks/ch8_evar_risk_measure.
 * The cookbook maximizes  m'x - delta * EVaR_alpha(loss)  over x >= 0,
 * sum(x) = 1, where EVaR is the entropic value-at-risk (Ahmadi-Javid),
 *
 *   EVaR_alpha(L) = inf_{s>0} s * ( log( sum_i p_i exp(L_i/s) ) - log(1-alpha) ),
 *
 * a perspective of the log-sum-exp.  The cookbook models it with n exponential
 * cones plus one linear row (persplogsumexp):
 *
 *   (u_i, s, L_i - z) in K_exp   for every scenario i   [ u_i >= s exp((L_i-z)/s) ]
 *   sum_i p_i u_i <= s
 *   EVaR = z - s log(1-alpha)
 *
 * which is exactly this library's PRIMAL_CT_PEXP cone.  The model here is a
 * small deterministic instance (2 assets, 4 equiprobable scenarios), and the
 * conic optimum is checked against a brute force over the weight simplex where
 * EVaR is itself minimized in one dimension (golden section) -- independent of
 * the conic solver.
 *
 * Usage: evar_portfolio   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NA 2   /* assets    */
#define NS 4   /* scenarios */

/* scenario returns R[s][a] and probabilities (equiprobable) */
static const double RET[NS][NA] = {
    { 0.10, 0.02}, {-0.05, 0.04}, { 0.20, 0.01}, {-0.10, 0.03}};
static const double PROB[NS] = {0.25, 0.25, 0.25, 0.25};
static const double ALPHA = 0.95;
static const double DELTA = 1.0;

/* loss in scenario i for weights x: L_i = -sum_a R[i][a] x[a] */
static double loss(int i, const double *x) {
    double r = 0.0;
    for (int a = 0; a < NA; a++) r += RET[i][a] * x[a];
    return -r;
}

/* g(s) = s * ( logsumexp_i(L_i/s) - log(1-alpha)); convex in s > 0.
 * Written stably as max(L) + s*(log(sum p exp((L-max)/s)) - log(1-alpha)). */
static double evar_g(double s, const double *x) {
    double mx = loss(0, x);
    for (int i = 1; i < NS; i++) { double l = loss(i, x); if (l > mx) mx = l; }
    double acc = 0.0;
    for (int i = 0; i < NS; i++) acc += PROB[i] * exp((loss(i, x) - mx) / s);
    return mx + s * (log(acc) - log(1.0 - ALPHA));
}

/* EVaR by golden section on log(s): convex, minimum interior */
static double evar(const double *x) {
    double a = -40.0, b = 12.0;                 /* log(s) bracket (infimum may be at s->0) */
    const double gr = 0.6180339887498949;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = evar_g(exp(c), x), fd = evar_g(exp(d), x);
    for (int it = 0; it < 200; it++) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = evar_g(exp(c), x); }
        else         { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = evar_g(exp(d), x); }
    }
    return evar_g(exp(0.5 * (a + b)), x);
}

/* the conic model: max m'x - delta*EVaR, returns objective and writes x/z/s */
static int evar_sdp(double *obj_out, double *x_out, double *evar_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { X0 = 0, Z = NA, S = NA + 1, U = NA + 2, V = NA + 2 + NS, NV = NA + 2 + 2 * NS };
    int R_BUD = 0, R_ULIN = 1, R_V = 2;
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, 2 + NS);
    for (int a = 0; a < NA; a++) PRIMAL_putvarbound(t, X0 + a, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, Z, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, S, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < NS; i++) PRIMAL_putvarbound(t, U + i, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < NS; i++) PRIMAL_putvarbound(t, V + i, PRIMAL_BK_FR, -INFINITY, INFINITY);

    {   /* budget: sum_a x_a = 1 */
        int sub[NA]; double val[NA];
        for (int a = 0; a < NA; a++) { sub[a] = X0 + a; val[a] = 1.0; }
        PRIMAL_putarow(t, R_BUD, NA, sub, val);
        PRIMAL_putconbound(t, R_BUD, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* sum_i p_i u_i - s <= 0 */
        int sub[NS + 1]; double val[NS + 1];
        for (int i = 0; i < NS; i++) { sub[i] = U + i; val[i] = PROB[i]; }
        sub[NS] = S; val[NS] = -1.0;
        PRIMAL_putarow(t, R_ULIN, NS + 1, sub, val);
        PRIMAL_putconbound(t, R_ULIN, PRIMAL_BK_UP, -INFINITY, 0.0);
    }
    for (int i = 0; i < NS; i++) {
        /* v_i + sum_a R[i][a] x_a + z = 0   (so v_i = -R x - z = loss_i - z) */
        int sub[NA + 2]; double val[NA + 2]; int k = 0;
        sub[k] = V + i; val[k] = 1.0; k++;
        for (int a = 0; a < NA; a++) { sub[k] = X0 + a; val[k] = RET[i][a]; k++; }
        sub[k] = Z; val[k] = 1.0; k++;
        PRIMAL_putarow(t, R_V + i, k, sub, val);
        PRIMAL_putconbound(t, R_V + i, PRIMAL_BK_FX, 0.0, 0.0);
    }
    for (int i = 0; i < NS; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){U + i, S, V + i});

    /* objective: max sum_a m_a x_a - delta z + delta log(1-alpha) s */
    double m[NA];
    for (int a = 0; a < NA; a++) {
        m[a] = 0.0;
        for (int i = 0; i < NS; i++) m[a] += PROB[i] * RET[i][a];
    }
    for (int a = 0; a < NA; a++) PRIMAL_putcj(t, X0 + a, m[a]);
    PRIMAL_putcj(t, Z, -DELTA);
    PRIMAL_putcj(t, S, DELTA * log(1.0 - ALPHA));
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        *obj_out = 0.0;
        for (int a = 0; a < NA; a++) { x_out[a] = x[X0 + a]; *obj_out += m[a] * x[X0 + a]; }
        *obj_out -= DELTA * (x[Z] - x[S] * log(1.0 - ALPHA));
        *evar_out = x[Z] - x[S] * log(1.0 - ALPHA);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double obj = 0.0, x[NA] = {0}, ev = 0.0;
    int ok = evar_sdp(&obj, x, &ev);

    /* reference: brute force over the weight simplex; EVaR by golden section */
    double best = -1e300, bw = 0.0;
    for (int k = 0; k <= 20000; k++) {
        double w = (double)k / 20000.0;
        double xr[NA] = {w, 1.0 - w};
        double ret = 0.0;
        for (int a = 0; a < NA; a++) {
            double mr = 0.0;
            for (int i = 0; i < NS; i++) mr += PROB[i] * RET[i][a];
            ret += mr * xr[a];
        }
        double f = ret - DELTA * evar(xr);
        if (f > best) { best = f; bw = w; }
    }
    double ret_sol = 0.0;
    for (int a = 0; a < NA; a++) {
        double mr = 0.0;
        for (int i = 0; i < NS; i++) mr += PROB[i] * RET[i][a];
        ret_sol += mr * x[a];
    }
    double ev_direct = evar(x);

    int okall = ok && fabs(obj - best) < 1e-3 * (1.0 + fabs(best)) &&
                fabs(ev - ev_direct) < 1e-3 * (1.0 + fabs(ev_direct)) &&
                fabs(x[0] + x[1] - 1.0) < 1e-7 && x[0] >= -1e-9 && x[1] >= -1e-9;

    printf("evar_portfolio  N=%d T=%d alpha=%.2f delta=%.2f\n", NA, NS, ALPHA, DELTA);
    printf("  x = (%.8f, %.8f)   rendimento = %.8f\n", x[0], x[1], ret_sol);
    printf("  EVaR = %.8f   [brute force obj %.8f a w=%.5f]\n", ev, best, bw);
    printf("  obiettivo conico = %.8f   [brute force %.8f]\n", obj, best);
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
