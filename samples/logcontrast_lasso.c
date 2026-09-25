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

/* logcontrast_lasso.c - log-contrast LASSO logistic regression.
 *
 * Source: audreyolmsted/admm-portfolio ("Log-Contrast LASSO Logistic
 * Regression via ADMM", benchmarked against MOSEK/CVXR).  Given log-scaled
 * compositional features X (CLR-transformed: each row has zero mean), binary
 * labels Y and a penalty lambda, the model is
 *
 *   minimize  (1/n) sum_i log(1 + exp(-y_i eta_i))  +  lambda * sum_j s_j |b_j|
 *   s.t.      eta = [1 X] beta ,   sum_j b_j = 0         (log-contrast)
 *
 * with beta = (intercept, b_1..b_p), y_i in {+1,-1}, and s_j the column scale
 * (sqrt((n-1)/n) * sd) that the repo folds into the L1 term instead of
 * standardizing X.
 *
 * Two independent solvers run on the SAME deterministic instance:
 *
 *   1. PRIMALSOLVER - the exact convex program.  The logistic loss is a
 *      softplus, t_i >= log(1+exp(z_i))  <=>  exp(-t_i)+exp(z_i-t_i) <= 1,
 *      i.e. two exponential cones per sample (PRIMAL_CT_PEXP); the L1 term is
 *      a linear epigraph and the log-contrast constraint one equality.  So the
 *      whole model is PEXP + linear -- the interior-point side of the repo's
 *      MOSEK comparison.
 *
 *   2. ADMM - the repo's first-order method, reimplemented here: the beta-step
 *      is the smooth logistic loss plus a quadratic penalty (solved by Newton
 *      with backtracking), the z-step is the closed-form joint proximal of
 *      sparsity AND zero-sum (a monotone scalar root, found by bisection).
 *
 * The two must agree (coefficients and objective), and both are checked
 * against a dense brute force over the reduced space: p=3 features with
 * sum(b)=0 leave only (intercept, b_1, b_2), a 3-D grid of the TRUE objective.
 *
 * Deterministic instance: n=16, p=3, lambda=0.3.  The LASSO soft-thresholds
 * feature 0 to exactly zero, so b = (0, -0.161, 0.161), intercept 0.276, and
 * the objective is 0.68265373 for BOTH the conic solve and the ADMM (72
 * iterations, max coefficient difference 3.7e-5).
 *
 * Usage: logcontrast_lasso   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NF 3       /* features            */
#define NS 16      /* samples             */
#define NL 4       /* p + 1 unknowns      */

static double X[NS][NF];      /* CLR-transformed features */
static double Y[NS];          /* labels +1 / -1           */
static double SIG[NF];        /* L1 column scale          */
static double LAM = 0.3;      /* LASSO penalty            */

/* ---- deterministic data (LCG), CLR-transformed ------------------------- */
static unsigned st = 20260924u;
static double rnd(void) {
    st = st * 1103515245u + 12345u;
    return (double)((st >> 16) & 0x7fff) / 32767.0;
}
/* true log-contrast signal (sum zero), used to label the samples */
static const double WTRUE[NF] = {-0.5, -0.5, 1.0};

static void make_instance(void) {
    for (int i = 0; i < NS; i++) {
        double raw[NF], m = 0.0;
        for (int j = 0; j < NF; j++) { raw[j] = (rnd() * 2.0 - 1.0); m += raw[j]; }
        m /= NF;
        for (int j = 0; j < NF; j++) X[i][j] = raw[j] - m;      /* CLR: row mean 0 */
    }
    for (int i = 0; i < NS; i++) {
        double eta = 0.0;
        for (int j = 0; j < NF; j++) eta += X[i][j] * WTRUE[j];
        eta += (rnd() - 0.5);                                   /* label noise */
        Y[i] = (eta > 0.0) ? 1.0 : -1.0;
    }
    for (int j = 0; j < NF; j++) {
        double mu = 0.0, v = 0.0;
        for (int i = 0; i < NS; i++) mu += X[i][j];
        mu /= NS;
        for (int i = 0; i < NS; i++) v += (X[i][j] - mu) * (X[i][j] - mu);
        SIG[j] = sqrt(((double)(NS - 1) / NS) * (v / (NS - 1)));
        if (SIG[j] == 0.0) SIG[j] = 1.0;
    }
}

/* stable softplus and sigmoid */
static double softplus(double z) { return z >= 0.0 ? z + log1p(exp(-z)) : log1p(exp(z)); }
static double sigmoid(double z) { return z >= 0.0 ? 1.0/(1.0+exp(-z)) : exp(z)/(1.0+exp(z)); }

/* objective at beta (intercept first) */
static double objective(const double *b) {
    double s = 0.0;
    for (int i = 0; i < NS; i++) {
        double eta = b[0];
        for (int j = 0; j < NF; j++) eta += X[i][j] * b[1 + j];
        s += softplus(-Y[i] * eta);
    }
    s /= NS;
    for (int j = 0; j < NF; j++) s += LAM * SIG[j] * fabs(b[1 + j]);
    return s;
}

/* ======================= 1. conic model (PrimalSolver) =================== */
/* vars: b0 | b_j | g_j | per sample (t,m,s,a,b) | one                       */
static int BJ(int j) { return 1 + j; }
static int GJ(int j) { return 1 + NF + j; }
static int SB(int i) { return 1 + 2 * NF + 5 * i; }   /* t,m,s,a,b */
static int ONE(void) { return 1 + 2 * NF + 5 * NS; }
#define NVAR (1 + 2 * NF + 5 * NS + 1)
#define NR (1 + 2 * NF + 3 * NS)

static int conic_solve(double *beta_out, double *obj_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, NR, NVAR, &t);
    for (int j = 0; j < NVAR; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int j = 0; j < NF; j++) PRIMAL_putvarbound(t, GJ(j), PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < NS; i++) {
        PRIMAL_putvarbound(t, SB(i) + 3, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putvarbound(t, SB(i) + 4, PRIMAL_BK_LO, 0.0, INFINITY);
    }
    PRIMAL_putvarbound(t, ONE(), PRIMAL_BK_FX, 1.0, 1.0);

    int row = 0;
    {   /* log-contrast: sum_j b_j = 0 */
        int sub[NF]; double val[NF];
        for (int j = 0; j < NF; j++) { sub[j] = BJ(j); val[j] = 1.0; }
        PRIMAL_putarow(t, row, NF, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0);
        row++;
    }
    for (int j = 0; j < NF; j++) {           /* g_j >= |b_j| */
        PRIMAL_putarow(t, row, 2, (int[]){GJ(j), BJ(j)}, (double[]){1.0, -1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY); row++;
        PRIMAL_putarow(t, row, 2, (int[]){GJ(j), BJ(j)}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY); row++;
    }
    for (int i = 0; i < NS; i++) {
        int T = SB(i), M = SB(i) + 1, S = SB(i) + 2, A = SB(i) + 3, B = SB(i) + 4;
        /* m + t = 0   (m = -t) */
        PRIMAL_putarow(t, row, 2, (int[]){M, T}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
        /* s + t + y_i eta_i = 0,  eta_i = b0 + sum_j X_ij b_j */
        {   int sub[3 + NF]; double val[3 + NF]; int k = 0;
            sub[k] = S; val[k] = 1.0; k++;
            sub[k] = T; val[k] = 1.0; k++;
            sub[k] = 0; val[k] = Y[i]; k++;
            for (int j = 0; j < NF; j++) { sub[k] = BJ(j); val[k] = Y[i] * X[i][j]; k++; }
            PRIMAL_putarow(t, row, k, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
        }
        /* a + b <= 1 */
        PRIMAL_putarow(t, row, 2, (int[]){A, B}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 1.0); row++;
    }

    for (int i = 0; i < NS; i++) {
        int M = SB(i) + 1, S = SB(i) + 2, A = SB(i) + 3, B = SB(i) + 4;
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){A, ONE(), M});
        PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){B, ONE(), S});
    }
    for (int i = 0; i < NS; i++) PRIMAL_putcj(t, SB(i), 1.0 / NS);
    for (int j = 0; j < NF; j++) PRIMAL_putcj(t, GJ(j), LAM * SIG[j]);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[NVAR];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        beta_out[0] = x[0];
        for (int j = 0; j < NF; j++) beta_out[1 + j] = x[BJ(j)];
        *obj_out = objective(beta_out);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* ======================= 2. ADMM (the repo's method) ===================== */
/* minimises softplus loss + lambda*sum sigma|b| s.t. sum b_j = 0.  Newton
 * (backtracking) for the beta-step, closed-form joint proximal for z. */
static void admm_solve(double *beta_out, int *iters_out, int *conv_out) {
    const double rho = 1.0, abs_tol = 1e-6, rel_tol = 1e-4;
    const int max_iter = 5000;
    double beta[NL] = {0}, z[NF] = {0}, u[NF] = {0};
    int conv = 0, it = 0;

    for (it = 0; it < max_iter; it++) {
        double zold[NF];
        for (int j = 0; j < NF; j++) zold[j] = z[j];

        /* --- beta-step: min f(b) + (rho/2)||b[1:]-target||^2  (Newton) --- */
        double target[NF];
        for (int j = 0; j < NF; j++) target[j] = z[j] - u[j];
        for (int nt = 0; nt < 20; nt++) {
            double g[NL], H[NL][NL];
            for (int a = 0; a < NL; a++) { g[a] = 0.0; for (int b = 0; b < NL; b++) H[a][b] = 0.0; }
            for (int i = 0; i < NS; i++) {
                double eta = beta[0], xi[NL];
                xi[0] = 1.0;
                for (int j = 0; j < NF; j++) { eta += X[i][j] * beta[1 + j]; xi[1 + j] = X[i][j]; }
                double p = sigmoid(Y[i] * eta);          /* P(correct) */
                double gi = -Y[i] * (1.0 - p) / NS;      /* d loss / d eta */
                double hi = p * (1.0 - p) / NS;
                for (int a = 0; a < NL; a++) {
                    g[a] += gi * xi[a];
                    for (int b = 0; b < NL; b++) H[a][b] += hi * xi[a] * xi[b];
                }
            }
            for (int j = 0; j < NF; j++) {               /* penalty gradient/Hessian */
                g[1 + j] += rho * (beta[1 + j] - target[j]);
                H[1 + j][1 + j] += rho;
            }
            /* solve H d = -g (Gauss elimination, NL<=4) */
            double A[NL][NL + 1], d[NL];
            for (int a = 0; a < NL; a++) {
                for (int b = 0; b < NL; b++) A[a][b] = H[a][b];
                A[a][NL] = -g[a];
            }
            for (int c = 0; c < NL; c++) {
                int piv = c;
                for (int r = c + 1; r < NL; r++) if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
                for (int b = 0; b <= NL; b++) { double tmp = A[c][b]; A[c][b] = A[piv][b]; A[piv][b] = tmp; }
                if (fabs(A[c][c]) < 1e-14) { d[c] = 0.0; continue; }
                for (int r = c + 1; r < NL; r++) {
                    double f = A[r][c] / A[c][c];
                    for (int b = c; b <= NL; b++) A[r][b] -= f * A[c][b];
                }
            }
            for (int c = NL - 1; c >= 0; c--) {
                double s = A[c][NL];
                for (int b = c + 1; b < NL; b++) s -= A[c][b] * d[b];
                d[c] = (fabs(A[c][c]) < 1e-14) ? 0.0 : s / A[c][c];
            }
            {   double dn = 0.0;
                for (int a = 0; a < NL; a++) dn += d[a] * d[a];
                if (dn < 1e-18) break;
            }
            /* backtracking on the objective */
            double step = 1.0, f0;
            {   double s = 0.0;
                for (int i = 0; i < NS; i++) {
                    double eta = beta[0];
                    for (int j = 0; j < NF; j++) eta += X[i][j] * beta[1 + j];
                    s += softplus(-Y[i] * eta);
                }
                f0 = s / NS;
                for (int j = 0; j < NF; j++) {
                    double r = beta[1 + j] - target[j];
                    f0 += 0.5 * rho * r * r;
                }
            }
            for (int ls = 0; ls < 30; ls++) {
                double bn[NL], fn = 0.0;
                for (int a = 0; a < NL; a++) bn[a] = beta[a] + step * d[a];
                for (int i = 0; i < NS; i++) {
                    double eta = bn[0];
                    for (int j = 0; j < NF; j++) eta += X[i][j] * bn[1 + j];
                    fn += softplus(-Y[i] * eta);
                }
                fn /= NS;
                for (int j = 0; j < NF; j++) { double r = bn[1 + j] - target[j]; fn += 0.5 * rho * r * r; }
                if (fn <= f0 - 1e-12 * step * step) { for (int a = 0; a < NL; a++) beta[a] = bn[a]; break; }
                step *= 0.5;
            }
            if (step < 1e-9) break;
        }
        double w[NF];
        for (int j = 0; j < NF; j++) w[j] = beta[1 + j];

        /* --- z-step: soft-threshold, shifted so that sum(z)=0 (monotone root) --- */
        double thr[NF], v[NF];
        for (int j = 0; j < NF; j++) { thr[j] = LAM * SIG[j] / rho; v[j] = w[j] + u[j]; }
        double lo = v[0], hi = v[0];
        for (int j = 0; j < NF; j++) {
            if (v[j] - thr[j] < lo) lo = v[j] - thr[j];
            if (v[j] + thr[j] < lo) lo = v[j] + thr[j];
            if (v[j] + thr[j] > hi) hi = v[j] + thr[j];
            if (v[j] - thr[j] > hi) hi = v[j] - thr[j];
        }
        double c = 0.0;
        for (int bs = 0; bs < 80; bs++) {
            c = 0.5 * (lo + hi);
            double s = 0.0;
            for (int j = 0; j < NF; j++) {
                double zz = v[j] - c;
                if (zz > thr[j]) s += zz - thr[j];
                else if (zz < -thr[j]) s += zz + thr[j];
            }
            if (s > 0.0) lo = c; else hi = c;
        }
        for (int j = 0; j < NF; j++) {
            double zz = v[j] - c;
            if (zz > thr[j]) z[j] = zz - thr[j];
            else if (zz < -thr[j]) z[j] = zz + thr[j];
            else z[j] = 0.0;
        }
        /* --- dual update --- */
        for (int j = 0; j < NF; j++) u[j] += w[j] - z[j];

        /* --- convergence --- */
        double rn = 0.0, sn = 0.0, wn = 0.0, zn = 0.0, un = 0.0;
        for (int j = 0; j < NF; j++) {
            rn += (w[j] - z[j]) * (w[j] - z[j]);
            sn += (z[j] - zold[j]) * (z[j] - zold[j]);
            wn += w[j] * w[j]; zn += z[j] * z[j]; un += u[j] * u[j];
        }
        rn = sqrt(rn); sn = rho * sqrt(sn); 
        double eps_pri = sqrt((double)NF) * abs_tol + rel_tol * fmax(sqrt(wn), sqrt(zn));
        double eps_dual = sqrt((double)NF) * abs_tol + rel_tol * rho * sqrt(un);
        if (rn < eps_pri && sn < eps_dual) { conv = 1; it++; break; }
    }
    beta_out[0] = beta[0];
    for (int j = 0; j < NF; j++) beta_out[1 + j] = z[j];
    *iters_out = it;
    *conv_out = conv;
}

/* ======================= brute force (reduced space) ==================== */
static void brute_force(double *obj_out, double *beta_out) {
    double span = 6.0; int NG = 120;
    double best = 1e300, bb[NL] = {0};
    for (int a = 0; a <= NG; a++) {
        double b0 = -span + 2 * span * a / NG;
        for (int c = 0; c <= NG; c++) {
            double t1 = -span + 2 * span * c / NG;
            for (int e = 0; e <= NG; e++) {
                double t2 = -span + 2 * span * e / NG;
                double b[NL] = {b0, t1, t2, -t1 - t2};
                double f = objective(b);
                if (f < best) { best = f; for (int k = 0; k < NL; k++) bb[k] = b[k]; }
            }
        }
    }
    *obj_out = best;
    for (int k = 0; k < NL; k++) beta_out[k] = bb[k];
}

int main(void) {
    make_instance();
    double bc[NL], ba[NL], bb[NL], oc = 0, oa = 0, ob = 0;
    int ok = 1;

    if (!conic_solve(bc, &oc)) { printf("logcontrast_lasso  conic rc!=OK FAIL\n"); return 1; }
    int iters = 0, conv = 0;
    admm_solve(ba, &iters, &conv);
    oa = objective(ba);
    brute_force(&ob, bb);

    double sumc = 0, suma = 0, dbmax = 0;
    for (int j = 0; j < NF; j++) {
        sumc += bc[1 + j]; suma += ba[1 + j];
        if (fabs(ba[1 + j] - bc[1 + j]) > dbmax) dbmax = fabs(ba[1 + j] - bc[1 + j]);
    }

    int ok_conic = fabs(oc - ob) < 1e-3 * (1.0 + fabs(ob));
    int ok_admm = conv && fabs(oa - oc) < 1e-4 * (1.0 + fabs(oc)) && dbmax < 1e-3;
    int ok_sum = fabs(sumc) < 1e-6 && fabs(suma) < 1e-6;
    int ok_sparse = fabs(bc[1]) < 1e-6 && fabs(ba[1]) < 1e-6;   /* feature 0 penalised to 0 */

    ok = ok_conic && ok_admm && ok_sum && ok_sparse;
    printf("logcontrast_lasso  n=%d p=%d lambda=%.3f\n", NS, NF, LAM);
    printf("  conic : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)  sum=%.2e\n",
           oc, bc[0], bc[1], bc[2], bc[3], sumc);
    printf("  admm  : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)  sum=%.2e  iters=%d %s\n",
           oa, ba[0], ba[1], ba[2], ba[3], suma, iters, conv ? "converged" : "NOT-converged");
    printf("  brute : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)\n", ob, bb[0], bb[1], bb[2], bb[3]);
    printf("  [conic vs brute %s] [admm vs conic: obj %s, max|db| %.2e] [zero-sum %s] [sparse %s]\n",
           ok_conic ? "OK" : "FAIL", ok_admm ? "OK" : "FAIL", dbmax,
           ok_sum ? "OK" : "FAIL", ok_sparse ? "OK" : "FAIL");
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
