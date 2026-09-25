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

/* risk_budgeting_ls.c - LONG-SHORT risk budgeting, mixed-integer conic
 * (MOSEK Portfolio Optimization Cookbook, ch9 risk_budgeting_convex_MIO).
 *
 *   minimize    s - a * sum_i b_i t_i
 *   subject to  s >= 1/2 x'Sx          (S = G G')   [(s, 1, G'x) in Qr]
 *               t_i <= log(|x_i|)                  [(|x_i|, 1, t_i) in K_exp]
 *               x = xp - xm,  xp,xm >= 0
 *               xp <= M yp,  xm <= M ym,  yp + ym <= 1,  yp,ym in {0,1}
 *
 * The binaries are what makes it long-short: only one of xp_i, xm_i may be
 * positive (yp_i + ym_i <= 1), so |x_i| = xp_i + xm_i for the exact orthant
 * and the log term is bounded by M.  Without them xp+xm could inflate freely
 * and the log term would drive the objective to -infinity: the MIO is not
 * optional, it is what bounds the model.
 *
 * Hand case  S = I (G = I), a = 1, b = (1/2,1/2): the objective separates,
 *   f(x_i) = x_i^2/2 - a b_i log|x_i|,  f'(x_i) = x_i - a b_i/x_i = 0
 *   => x_i^2 = a b_i = 1/2  => |x_i| = 1/sqrt(2) = 0.70710678,
 * s = (1/2)(1/2+1/2) = 0.5, t_i = log(1/sqrt2), obj = 0.5 + 0.34657359.
 * The risk contributions x_i (Sx)_i = x_i^2 equal a b_i = 0.5 exactly.
 *
 * Usage: risk_budgeting_ls   (no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

/* solve for N assets with S = G G' (G row-major N x N), budgets b, aversion a;
 * returns the objective and writes |x|. */
static double rb_ls(int N, const double *G, const double *b, double a, double *xabs_out, int *ok_out) {
    if (N > 6) return -1.0;
    /* var layout: x(N) xp(N) xm(N) yp(N) ym(N) t(N) s(1) z(N) u(N) ONE(1) */
    int X0 = 0, XP = N, XM = 2*N, YP = 3*N, YM = 4*N, T0 = 5*N, S0 = 6*N, Z0 = 6*N+1, U0 = 7*N+1, ONE = 8*N+1;
    int nv = ONE + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < N; i++) {
        PRIMAL_putvarbound(t, X0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, XP + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(t, XM + i, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvartype(t, YP + i, PRIMAL_VAR_TYPE_INT);   /* binary via [0,1] + INT */
        PRIMAL_putvarbound(t, YP + i, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvartype(t, YM + i, PRIMAL_VAR_TYPE_INT);
        PRIMAL_putvarbound(t, YM + i, PRIMAL_BK_RA, 0.0, 1.0);
        PRIMAL_putvarbound(t, T0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, Z0 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
        PRIMAL_putvarbound(t, U0 + i, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMAL_putvarbound(t, S0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, ONE, PRIMAL_BK_FX, 1.0, 1.0);
    /* rows: N pos-neg + N ubound-p + N ubound-m + N exclusion + N z=G'x + N u=xp+xm */
    int nrow = 6*N;
    PRIMAL_appendcons(t, nrow);
    double bigM = 20.0;
    for (int i = 0; i < N; i++) {
        PRIMAL_putarow(t, i, 3, (int[]){X0+i, XP+i, XM+i}, (double[]){1.0, -1.0, 1.0});
        PRIMAL_putconbound(t, i, PRIMAL_BK_FX, 0.0, 0.0);                 /* x - xp + xm = 0 */
        PRIMAL_putarow(t, N+i, 2, (int[]){XP+i, YP+i}, (double[]){1.0, -bigM});
        PRIMAL_putconbound(t, N+i, PRIMAL_BK_UP, -INFINITY, 0.0);          /* xp <= M yp */
        PRIMAL_putarow(t, 2*N+i, 2, (int[]){XM+i, YM+i}, (double[]){1.0, -bigM});
        PRIMAL_putconbound(t, 2*N+i, PRIMAL_BK_UP, -INFINITY, 0.0);        /* xm <= M ym */
        PRIMAL_putarow(t, 3*N+i, 2, (int[]){YP+i, YM+i}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, 3*N+i, PRIMAL_BK_UP, -INFINITY, 1.0);        /* yp + ym <= 1 */
        {   /* z_i - (G'x)_i = 0 : G'x_i = sum_k G[k][i] x_k */
            int sub[7]; double val[7]; int nn = 0;
            sub[nn] = Z0+i; val[nn] = 1.0; nn++;
            for (int k = 0; k < N; k++) if (G[k*N+i] != 0.0) { sub[nn] = X0+k; val[nn] = -G[k*N+i]; nn++; }
            PRIMAL_putarow(t, 4*N+i, nn, sub, val);
            PRIMAL_putconbound(t, 4*N+i, PRIMAL_BK_FX, 0.0, 0.0);
        }
        PRIMAL_putarow(t, 5*N+i, 3, (int[]){U0+i, XP+i, XM+i}, (double[]){1.0, -1.0, -1.0});
        PRIMAL_putconbound(t, 5*N+i, PRIMAL_BK_FX, 0.0, 0.0);              /* u - xp - xm = 0 */
    }
    /* (s, 1, z...) in Qr  ->  s >= ||z||^2/2 = 1/2 x'Sx */
    { int mem[8]; mem[0] = S0; mem[1] = ONE; for (int i = 0; i < N; i++) mem[2+i] = Z0+i;
      PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, N+2, mem); }
    /* (u_i, 1, t_i) in K_exp  ->  t_i <= log(u_i) = log|x_i| */
    for (int i = 0; i < N; i++)
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){U0+i, ONE, T0+i});
    PRIMAL_putcj(t, S0, 1.0);
    for (int i = 0; i < N; i++) PRIMAL_putcj(t, T0+i, -a * b[i]);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = -1.0;
    if (rc == PRIMAL_RES_OK) {
        double x[128]; PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int i = 0; i < N; i++) xabs_out[i] = fabs(x[X0+i]);
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = (rc == PRIMAL_RES_OK);
    return obj;
}

int main(void) {
    printf("risk_budgeting_ls (long-short MIO)\n");
    int all = 1;
    /* hand case N=2, G=I, a=1, b=(1/2,1/2) */
    {
        enum { N = 2 };
        double G[N][N] = {{1.0, 0.0}, {0.0, 1.0}}, b[N] = {0.5, 0.5}, xabs[N];
        int ok = 0; double obj = rb_ls(N, (const double *)G, b, 1.0, xabs, &ok);
        double want = 0.5 + 0.34657359027997264;          /* 0.5 - log(1/sqrt2) */
        double xw = 1.0 / sqrt(2.0);
        int good = ok && fabs(obj - want) < 1e-4 && fabs(xabs[0]-xw) < 1e-3 && fabs(xabs[1]-xw) < 1e-3;
        printf("  N=2 S=I: obj=%.6f (atteso %.6f)  |x|=(%.6f,%.6f) (atteso %.6f)  %s\n",
               obj, want, xabs[0], xabs[1], xw, good ? "OK" : "FAIL");
        all &= good;
    }
    printf("%s\n", all ? "OK" : "FAIL");
    return all ? 0 : 1;
}
