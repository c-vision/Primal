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

/* market_neutral.c - market-neutral long-short portfolio with leverage,
 * turnover and a variance cap (MOSEK conic optimization).
 *
 * Source: fy23-art/Portfolio-Optimization (MOSEK.ipynb, the "Extended
 * MarkowitzReturn Model" cell).  The notebook builds, at each date, a long-short
 * portfolio whose exposure to the market factor is zero:
 *
 *   maximize  m'x - cost_rate * sum_i (d+_i + d-_i)
 *   subject to
 *       factor' x = 0                      (market neutral)
 *       sum_i |x_i| <= leverage_limit      (gross exposure, via z_i >= |x_i|)
 *       x' S x <= gamma^2                  (risk, rotated second-order cone)
 *       x - x0 = d+ - d-,  d+_i, d-_i >= 0 (turnover vs the previous portfolio)
 *
 * with factor = first principal-component loadings.  The risk constraint is the
 * notebook's rotated cone  (gamma^2, 1/2, G' x) in RQCone  with S = G G'; this
 * sample passes the same quadratic form through putqconk.
 *
 * Deterministic two-asset instance with factor = (1,1), so factor'x = 0 means
 * x = (t, -t): the whole model reduces to one scalar t, and every value below
 * is hand-derived.  Two cases exercise the turnover penalty:
 *   A) cost_rate = 0 : the optimum is t = Tmax (the risk/leverage bound binds);
 *   B) cost_rate large: the optimum is t = t0 (no trade), the penalty wins.
 * A dense 1-D brute force over t confirms both.
 *
 * Usage: market_neutral   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NA 2

static const double S[NA][NA] = {{0.04, 0.01}, {0.01, 0.09}};  /* covariance    */
static const double M[NA] = {0.002, 0.0};                      /* alpha         */
static const double GAMMA2 = 0.01;                             /* 0.1^2         */
static const double LEV = 2.0;                                 /* sum |x| <= L  */

/* feasible bound on t from  sum|x| <= L  and  x'Sx <= gamma^2  (x=(t,-t)) */
static double tmax(void) {
    double q = S[0][0] - 2.0 * S[0][1] + S[1][1];   /* x'Sx = t^2 q */
    double tr = LEV / 2.0;
    double tg = sqrt(GAMMA2 / q);
    return tr < tg ? tr : tg;
}

/* conic model; returns objective and x */
static int solve(double c, double t0, double *obj_out, double *x_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    enum { XP = 0, Z = NA, DP = 2 * NA, DM = 3 * NA, NV = 4 * NA };
    int R_MN = 0, R_ZP = 1, R_ZN = 1 + NA, R_LEV = 1 + 2 * NA, R_TL = 2 + 2 * NA, R_RSK = 2 + 3 * NA;
    PRIMAL_appendvars(t, NV);
    PRIMAL_appendcons(t, 3 * NA + 3);
    for (int i = 0; i < NA; i++) {
        PRIMAL_putvarbound(t, XP + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, Z + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(t, DP + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(t, DM + i, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    /* factor' x = 0, factor = (1,1) */
    PRIMAL_putarow(t, R_MN, NA, (int[]){XP, XP + 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, R_MN, PRIMAL_BK_FX, 0.0, 0.0);
    for (int i = 0; i < NA; i++) {   /* z_i >= x_i  and  z_i >= -x_i */
        PRIMAL_putarow(t, R_ZP + i, 2, (int[]){Z + i, XP + i}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, R_ZP + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putarow(t, R_ZN + i, 2, (int[]){Z + i, XP + i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, R_ZN + i, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMAL_putarow(t, R_LEV, NA, (int[]){Z, Z + 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(t, R_LEV, PRIMAL_BK_UP, -INFINITY, LEV);
    for (int i = 0; i < NA; i++) {   /* x_i - d+_i + d-_i = x0_i */
        PRIMAL_putarow(t, R_TL + i, 3, (int[]){XP + i, DP + i, DM + i}, (double[]){1.0, -1.0, 1.0});
        PRIMAL_putconbound(t, R_TL + i, PRIMAL_BK_FX, (i == 0 ? t0 : -t0), (i == 0 ? t0 : -t0));
    }
    /* risk: 1/2 x'Qx <= gamma^2 with Q = 2S  (Q upper triangle: 2s11, 2s22, 2s12) */
    PRIMAL_putqconk(t, R_RSK, 3, (int[]){0, 1, 0}, (int[]){0, 1, 1},
                    (double[]){2.0 * S[0][0], 2.0 * S[1][1], 2.0 * S[0][1]});
    PRIMAL_putconbound(t, R_RSK, PRIMAL_BK_UP, -INFINITY, GAMMA2);

    for (int i = 0; i < NA; i++) {
        PRIMAL_putcj(t, XP + i, M[i]);
        PRIMAL_putcj(t, DP + i, -c);
        PRIMAL_putcj(t, DM + i, -c);
    }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[NV];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        x_out[0] = x[XP]; x_out[1] = x[XP + 1];
        *obj_out = M[0] * x[XP] + M[1] * x[XP + 1] - c * (x[DP] + x[DP + 1] + x[DM] + x[DM + 1]);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* true objective at x=(t,-t): a*t - c*sum_i |x_i - x0_i| */
static double obj_at(double c, double t0, double t) {
    double a = M[0] - M[1];
    double turn = fabs(t - t0) + fabs(-t + t0);
    return a * t - c * turn;
}

int main(void) {
    double T = tmax(), a = M[0] - M[1];
    int all = 1;
    printf("market_neutral  N=%d  Tmax=%.8f  a=m1-m2=%.6f\n", NA, T, a);
    double cs[2] = {0.0, 0.002}, t0s[2] = {0.1, 0.1};
    for (int k = 0; k < 2; k++) {
        double c = cs[k], t0 = t0s[k], obj = 0, x[NA] = {0};
        int ok = solve(c, t0, &obj, x);
        /* hand value: t* = Tmax if a > 2c, -Tmax if a < -2c, else clamp(t0) */
        double tstar;
        if (a > 2.0 * c) tstar = T;
        else if (a < -2.0 * c) tstar = -T;
        else tstar = t0 < -T ? -T : (t0 > T ? T : t0);
        double ref = obj_at(c, t0, tstar);
        /* dense 1-D brute force */
        double best = -1e300;
        for (int j = 0; j <= 200000; j++) {
            double tt = -T + 2.0 * T * j / 200000.0;
            double f = obj_at(c, t0, tt);
            if (f > best) best = f;
        }
        int okk = ok && fabs(obj - ref) < 1e-6 && fabs(obj - best) < 1e-6 &&
                  fabs(x[0] + x[1]) < 1e-7 && fabs(x[0] - tstar) < 1e-5;
        all = all && okk;
        printf("  caso %c: c=%.4f t0=%.2f  x=(%.8f, %.8f)  obj=%.10f  [chiusa %.10f]\n",
               'A' + k, c, t0, x[0], x[1], obj, ref);
    }
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
