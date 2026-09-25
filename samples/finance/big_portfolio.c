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

/* big_portfolio.c - the full mean-variance + correlation-stress pipeline of
 * "BIG Portfolio Optimisation" (https://github.com/tbc81/big-portfolio-optimisation,
 * MIT), solved with this library on the REAL market data.
 *
 * The project converts weekly market prices to GBP, estimates annual arithmetic
 * expected returns mu and covariance Sigma, solves long-only mean-variance
 * problems with MOSEK Fusion, and stresses within-group correlations.  The
 * data below is the actual estimates produced by that pipeline (Yahoo adjusted
 * closes converted to GBP, weekly W-FRI log returns, CSH2 backfilled with the
 * Bank of England series IUDZOS2); reference week 2021-09-09..2026-09-09.
 *
 *   Variables x (risky), c (cash): 1'x + c = 1, x >= 0, c >= 0 (cash 1.5%).
 *   Volatility = ||G' x||_2 with Sigma = G G' (Cholesky) -> quadratic cone.
 *
 *   1) observed snapshot      R0 = mu'x0, vol0 = sqrt(x0' Sigma x0)
 *   2) min risk at R0         min t  s.t. mu'x >= R0, ||G'x|| <= t
 *   3) max return at vol0     max mu'x s.t. ||G'x|| <= vol0
 *   4) correlation stress     nearest valid C* with floors C*_ij >= 0.85 on the
 *        min ||C* - C||_F  s.t.  diag(C*) = 1, C* >= 0, C*_ij >= rho (group)
 *      then Sigma* = D C* D (marginal volatilities preserved)
 *   5) multiplier (x0'Sigma*x0)/(x0'Sigma x0)
 *   6) min risk at R0 under Sigma*
 *
 * Reference values (case study / independent computation):
 *   vol0 = 0.229369 (22.94%), stressed vol = 0.257033 (25.70%), multiplier
 *   1.255767; min-risk base = 0.135508, stressed = 0.137191; max-return = 0.400353.
 *
 * Usage: big_portfolio   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "primal.h"

#define N 18
#define CASH0 0.015
#define RHO 0.85

/* --- real annual arithmetic moments (18 held assets) --------------------- */
static const char *ASSET[N] = {
    "Rheinmetall", "Parkmead", "ASML", "Roku", "CSH2", "Gold", "Duke Energy", "BAE Systems", "Nucor", "Diageo", "Legal & General", "Apollo", "Cocoa", "Salesforce", "Cameco", "Marvell", "Micron", "Volatus Aerospace"
};
static const double MU[N] = {
    0.88505759, 0.01637996, 0.25760384, 0.07062742, 0.03748831, 0.21091841, 0.09583463, 0.33645087, 0.29112897, -0.12002662, 0.04296791, 0.28051820, 0.45077128, 0.07685441, 0.48956336, 0.52495055, 0.96592058, 0.13350019
};
static const double X0[N] = {
    0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.07500000, 0.06500000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.02500000, 0.01000000, 0.02500000, 0.02500000, 0.01000000
};
static const double SIG[N][N] = {
    {7.8693177181e-01, 2.6803537860e-02, 5.6538149846e-02, -1.0901486761e-02, 3.2487564874e-06, 2.9919522273e-02, 8.0130282213e-03, 2.1544668156e-01, 3.9915837454e-02, -8.7715806493e-04, 2.1050625025e-02, 2.6609613869e-02, -5.3455522446e-02, 4.8313824939e-02, 1.0499183179e-01, -3.6500946465e-02, 2.4902557447e-02, 1.3222599613e-01},
    {2.6803537860e-02, 4.5928309389e-01, 1.9088877991e-02, -1.0061816532e-02, -1.7632197774e-05, 7.4229946741e-03, -7.0077694055e-03, 1.4311364548e-02, 4.2593072806e-02, 5.3360311065e-03, 2.0762686147e-03, 1.9391039637e-02, 2.1180310268e-02, 1.5037878828e-02, 2.0166003415e-02, 3.5629773995e-02, 2.9537667670e-02, -1.3758533343e-02},
    {5.6538149846e-02, 1.9088877991e-02, 2.5066126592e-01, 1.1813502943e-01, 9.6443250392e-05, 1.3240649054e-03, -1.0765482295e-02, 7.0614340589e-03, 6.0627095032e-02, 2.1869705589e-02, 3.7913336317e-02, 6.5549905806e-02, 1.5194369470e-02, 5.8940340094e-02, 9.5462362195e-02, 2.1598452995e-01, 2.6458674616e-01, 6.4507143109e-02},
    {-1.0901486761e-02, -1.0061816532e-02, 1.1813502943e-01, 6.0904867749e-01, 2.7241999281e-04, -1.1933723754e-02, -9.7657364319e-03, 1.3790643091e-03, 7.6301742916e-02, 2.0772719604e-02, 4.7275342882e-02, 1.3402884736e-01, 3.7153299320e-02, 1.3623821088e-01, 1.2386162419e-01, 2.1315465175e-01, 2.2927720652e-01, 6.8311932735e-02},
    {3.2487564874e-06, -1.7632197774e-05, 9.6443250392e-05, 2.7241999281e-04, 7.5685315635e-06, 3.0942343054e-05, -1.9767915538e-06, 4.8237201133e-05, -1.6353524879e-05, -6.7791802392e-05, 3.5668333897e-05, 4.6410160705e-05, 7.7073673639e-05, 1.0602503564e-04, 6.1340301552e-05, 1.5667590176e-04, 1.6094265546e-04, 1.0938845624e-05},
    {2.9919522273e-02, 7.4229946741e-03, 1.3240649054e-03, -1.1933723754e-02, 3.0942343054e-05, 3.6356132597e-02, 7.3390627470e-03, 1.3515940803e-02, -9.0856025023e-04, 1.9878094451e-03, -2.7543099646e-03, -1.0684186842e-02, -1.1816937025e-02, 2.0263475533e-03, 2.8071062208e-02, -6.0621001423e-03, 2.5661584954e-02, 2.1354342979e-02},
    {8.0130282213e-03, -7.0077694055e-03, -1.0765482295e-02, -9.7657364319e-03, -1.9767915538e-06, 7.3390627470e-03, 4.0976349236e-02, 1.0543056505e-02, 9.5639122061e-03, 3.3001355599e-03, 8.4537581703e-03, 1.9695009493e-03, -1.6412670252e-02, -8.3389210171e-03, 1.5335439692e-02, -1.1462993782e-02, -2.7969778114e-02, 1.2924825501e-02},
    {2.1544668156e-01, 1.4311364548e-02, 7.0614340589e-03, 1.3790643091e-03, 4.8237201133e-05, 1.3515940803e-02, 1.0543056505e-02, 1.4817381301e-01, 1.1582694026e-02, 1.7261760613e-03, 1.1402050587e-02, 1.2590886788e-02, -3.6625754020e-02, 1.4364919987e-02, 4.3143066606e-02, -2.2381650207e-02, 2.9376699243e-02, 4.6318187274e-02},
    {3.9915837454e-02, 4.2593072806e-02, 6.0627095032e-02, 7.6301742916e-02, -1.6353524879e-05, -9.0856025023e-04, 9.5639122061e-03, 1.1582694026e-02, 2.5772581003e-01, 1.4019697025e-02, 3.1909570880e-02, 9.1302221879e-02, 2.0810635168e-02, 5.1315070545e-02, 1.0541171559e-01, 1.1946151056e-01, 1.4960500545e-01, 3.4878112774e-02},
    {-8.7715806493e-04, 5.3360311065e-03, 2.1869705589e-02, 2.0772719604e-02, -6.7791802392e-05, 1.9878094451e-03, 3.3001355599e-03, 1.7261760613e-03, 1.4019697025e-02, 4.1651678600e-02, 1.5098335082e-02, 1.2791454191e-02, 1.6338995724e-02, 1.0370576683e-02, 1.5773922448e-02, 1.3630426101e-02, 3.2894027535e-02, 1.2690667367e-03},
    {2.1050625025e-02, 2.0762686147e-03, 3.7913336317e-02, 4.7275342882e-02, 3.5668333897e-05, -2.7543099646e-03, 8.4537581703e-03, 1.1402050587e-02, 3.1909570880e-02, 1.5098335082e-02, 6.2684950141e-02, 3.6932759499e-02, -5.8850168677e-04, 2.0251527105e-02, 3.9401137957e-02, 3.6294288556e-02, 4.9918556905e-02, 1.1330921374e-02},
    {2.6609613869e-02, 1.9391039637e-02, 6.5549905806e-02, 1.3402884736e-01, 4.6410160705e-05, -1.0684186842e-02, 1.9695009493e-03, 1.2590886788e-02, 9.1302221879e-02, 1.2791454191e-02, 3.6932759499e-02, 2.4372320134e-01, 6.7285867733e-02, 7.5349253616e-02, 1.0744939020e-01, 1.7602006115e-01, 1.6081480146e-01, 1.5054071340e-02},
    {-5.3455522446e-02, 2.1180310268e-02, 1.5194369470e-02, 3.7153299320e-02, 7.7073673639e-05, -1.1816937025e-02, -1.6412670252e-02, -3.6625754020e-02, 2.0810635168e-02, 1.6338995724e-02, -5.8850168677e-04, 6.7285867733e-02, 5.7209925474e-01, 2.8984962596e-02, 4.1329872305e-02, 4.8367745132e-02, 1.0054468065e-01, -1.2663102393e-02},
    {4.8313824939e-02, 1.5037878828e-02, 5.8940340094e-02, 1.3623821088e-01, 1.0602503564e-04, 2.0263475533e-03, -8.3389210171e-03, 1.4364919987e-02, 5.1315070545e-02, 1.0370576683e-02, 2.0251527105e-02, 7.5349253616e-02, 2.8984962596e-02, 1.8154987864e-01, 7.4519454632e-02, 1.1827749460e-01, 9.0697380314e-02, 4.7172627159e-02},
    {1.0499183179e-01, 2.0166003415e-02, 9.5462362195e-02, 1.2386162419e-01, 6.1340301552e-05, 2.8071062208e-02, 1.5335439692e-02, 4.3143066606e-02, 1.0541171559e-01, 1.5773922448e-02, 3.9401137957e-02, 1.0744939020e-01, 4.1329872305e-02, 7.4519454632e-02, 5.3192394146e-01, 1.8840523736e-01, 2.2317811290e-01, 1.1312375259e-01},
    {-3.6500946465e-02, 3.5629773995e-02, 2.1598452995e-01, 2.1315465175e-01, 1.5667590176e-04, -6.0621001423e-03, -1.1462993782e-02, -2.2381650207e-02, 1.1946151056e-01, 1.3630426101e-02, 3.6294288556e-02, 1.7602006115e-01, 4.8367745132e-02, 1.1827749460e-01, 1.8840523736e-01, 8.3973674794e-01, 4.6250714851e-01, 6.5406670102e-02},
    {2.4902557447e-02, 2.9537667670e-02, 2.6458674616e-01, 2.2927720652e-01, 1.6094265546e-04, 2.5661584954e-02, -2.7969778114e-02, 2.9376699243e-02, 1.4960500545e-01, 3.2894027535e-02, 4.9918556905e-02, 1.6081480146e-01, 1.0054468065e-01, 9.0697380314e-02, 2.2317811290e-01, 4.6250714851e-01, 1.3161522295e+00, 1.2823281199e-01},
    {1.3222599613e-01, -1.3758533343e-02, 6.4507143109e-02, 6.8311932735e-02, 1.0938845624e-05, 2.1354342979e-02, 1.2924825501e-02, 4.6318187274e-02, 3.4878112774e-02, 1.2690667367e-03, 1.1330921374e-02, 1.5054071340e-02, -1.2663102393e-02, 4.7172627159e-02, 1.1312375259e-01, 6.5406670102e-02, 1.2823281199e-01, 1.0492689215e+00}
};

/* technology/growth stress pairs among the held assets */
static const int PAIRS[][2] = {
    {2, 3}, {2, 13}, {2, 15}, {2, 16}, {3, 13}, {3, 15}, {3, 16}, {13, 15}, {13, 16}, {15, 16}
};
#define NPAIR ((int)(sizeof(PAIRS)/sizeof(PAIRS[0])))

static void corr_from_cov(const double S[N][N], double C[N][N]) {
    double d[N];
    for (int i = 0; i < N; i++) d[i] = sqrt(S[i][i]);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) C[i][j] = S[i][j] / (d[i] * d[j]);
}
static void cov_from_corr(const double S[N][N], const double C[N][N], double Ss[N][N]) {
    double d[N];
    for (int i = 0; i < N; i++) d[i] = sqrt(S[i][i]);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) Ss[i][j] = d[i] * d[j] * C[i][j];
}
static int chol(const double S[N][N], double G[N][N]) {
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) G[i][j] = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++) {
            double s = S[i][j];
            for (int k = 0; k < j; k++) s -= G[i][k] * G[j][k];
            if (i == j) { if (s <= 0.0) return 0; G[i][j] = sqrt(s); }
            else G[i][j] = s / G[j][j];
        }
    return 1;
}
static double port_var(const double x[N], const double S[N][N]) {
    double v = 0.0;
    for (int i = 0; i < N; i++) { double t = 0.0;
        for (int j = 0; j < N; j++) t += S[i][j] * x[j];
        v += x[i] * t; }
    return v;
}

/* --- min risk at a target return (SOCP) ---------------------------------- */
static int min_risk(const double G[N][N], double target, double *vol, double xo[N + 1]) {
    PRIMALenv_t env; PRIMALtask_t t;
    /* vars 0..N-1 x, N c, N+1 t, N+2..2N+1 y = G'x */
    int nv = 2 * N + 2, nc = N + 2;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, nc, nv, &t);
    for (int j = 0; j <= N; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, N + 1, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, N + 2 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, N + 1, 1.0);
    { int sub[N+1]; double v[N+1]; for (int j=0;j<=N;j++) { sub[j]=j; v[j]=1.0; }
       PRIMAL_putarow(t, 0, N+1, sub, v); }
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    { int sub[N]; double v[N]; for (int j=0;j<N;j++) { sub[j]=j; v[j]=MU[j]; }
       PRIMAL_putarow(t, 1, N, sub, v); }
    PRIMAL_putconbound(t, 1, PRIMAL_BK_LO, target, INFINITY);
    for (int i = 0; i < N; i++) {
        int sub[N + 1]; double v[N + 1];
        for (int j = 0; j < N; j++) { sub[j] = j; v[j] = -G[j][i]; }
        sub[N] = N + 2 + i; v[N] = 1.0;
        PRIMAL_putarow(t, 2 + i, N + 1, sub, v);
        PRIMAL_putconbound(t, 2 + i, PRIMAL_BK_FX, 0.0, 0.0);
    }
    { int mem[N + 1]; mem[0] = N + 1; for (int i = 0; i < N; i++) mem[1 + i] = N + 2 + i;
       PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, N + 1, mem); }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) { double z, x[2 * N + 2];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z); PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        *vol = z; for (int j = 0; j <= N; j++) xo[j] = x[j]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* --- max return at a risk limit (SOCP) ----------------------------------- */
static int max_return(const double G[N][N], double risk, double *ret) {
    PRIMALenv_t env; PRIMALtask_t t;
    /* vars 0..N-1 x, N c, N+1..2N y, 2N+1 rb */
    int nv = 2 * N + 2, nc = N + 1;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, nc, nv, &t);
    for (int j = 0; j <= N; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < N; i++) PRIMAL_putvarbound(t, N + 1 + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, 2 * N + 1, PRIMAL_BK_FX, risk, risk);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    for (int j = 0; j < N; j++) PRIMAL_putcj(t, j, MU[j]);
    { int sub[N+1]; double v[N+1]; for (int j=0;j<=N;j++) { sub[j]=j; v[j]=1.0; }
       PRIMAL_putarow(t, 0, N+1, sub, v); }
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    for (int i = 0; i < N; i++) {
        int sub[N + 1]; double v[N + 1];
        for (int j = 0; j < N; j++) { sub[j] = j; v[j] = -G[j][i]; }
        sub[N] = N + 1 + i; v[N] = 1.0;
        PRIMAL_putarow(t, 1 + i, N + 1, sub, v);
        PRIMAL_putconbound(t, 1 + i, PRIMAL_BK_FX, 0.0, 0.0);
    }
    { int mem[N + 1]; mem[0] = 2 * N + 1; for (int i = 0; i < N; i++) mem[1 + i] = N + 1 + i;
       PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, N + 1, mem); }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) { double z; PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z); *ret = z; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

/* --- nearest valid stressed correlation matrix (SDP) --------------------- */
static int stress_corr(const double C[N][N], double Cs[N][N]) {
    PRIMALenv_t env; PRIMALtask_t t;
    int NP = N * (N - 1) / 2;
    int *mark = (int *)malloc((size_t)N * N * sizeof(int));
    int *pi = (int *)malloc((size_t)NP * sizeof(int));
    int *pj = (int *)malloc((size_t)NP * sizeof(int));
    int *esym = (int *)malloc((size_t)N * sizeof(int));
    int *psym = (int *)malloc((size_t)NP * sizeof(int));
    int *isgrp = (int *)calloc((size_t)NP, sizeof(int));
    int k = 0;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { pi[k] = i; pj[k] = j; k++; }
    for (int g = 0; g < NPAIR; g++)
        for (int q = 0; q < NP; q++)
            if ((pi[q] == PAIRS[g][0] && pj[q] == PAIRS[g][1]) ||
                (pi[q] == PAIRS[g][1] && pj[q] == PAIRS[g][0])) isgrp[q] = 1;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    /* vars 0..NP-1 C*, NP t, NP+1..2NP d */
    PRIMAL_appendvars(t, 2 * NP + 1);
    for (k = 0; k < NP; k++) {
        if (isgrp[k]) PRIMAL_putvarbound(t, k, PRIMAL_BK_LO, RHO, INFINITY);
        else PRIMAL_putvarbound(t, k, PRIMAL_BK_RA, -1.0, 1.0);
    }
    PRIMAL_putvarbound(t, NP, PRIMAL_BK_LO, 0.0, INFINITY);
    for (k = 0; k < NP; k++) PRIMAL_putvarbound(t, NP + 1 + k, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, NP, 1.0);
    for (int i = 0; i < N; i++) {
        int si[1] = {i}, sj[1] = {i}; double sv[1] = {1.0};
        PRIMAL_appendsparsesymmat(t, N, 1, si, sj, sv, &esym[i]);
    }
    for (k = 0; k < NP; k++) {
        int si[1] = {pi[k]}, sj[1] = {pj[k]}; double sv[1] = {1.0};
        PRIMAL_appendsparsesymmat(t, N, 1, si, sj, sv, &psym[k]);
    }
    { int dim = N; PRIMAL_appendbarvars(t, 1, &dim); }
    PRIMAL_appendcons(t, NP + N + NP);
    for (k = 0; k < NP; k++) {                    /* d_k - sqrt2 C*_k = -sqrt2 C_k */
        int sub[2] = {NP + 1 + k, k}; double v[2] = {1.0, -sqrt(2.0)};
        PRIMAL_putarow(t, k, 2, sub, v);
        PRIMAL_putconbound(t, k, PRIMAL_BK_FX, -sqrt(2.0) * C[pi[k]][pj[k]], -sqrt(2.0) * C[pi[k]][pj[k]]);
    }
    for (int i = 0; i < N; i++) {                 /* <E_ii,X> = 1 */
        PRIMAL_putbaraij(t, NP + i, 0, 1, (int[]){esym[i]}, (double[]){1.0});
        PRIMAL_putconbound(t, NP + i, PRIMAL_BK_FX, 1.0, 1.0);
    }
    for (k = 0; k < NP; k++) {                    /* <E_ij,X> - 2 C*_k = 0 */
        PRIMAL_putbaraij(t, NP + N + k, 0, 1, (int[]){psym[k]}, (double[]){1.0});
        PRIMAL_putarow(t, NP + N + k, 1, (int[]){k}, (double[]){-2.0});
        PRIMAL_putconbound(t, NP + N + k, PRIMAL_BK_FX, 0.0, 0.0);
    }
    { int *mem = (int *)malloc((size_t)(NP + 1) * sizeof(int));
       mem[0] = NP; for (k = 0; k < NP; k++) mem[1 + k] = NP + 1 + k;
       PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, NP + 1, mem); free(mem); }
    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double *X = (double *)malloc((size_t)N * N * sizeof(double));
        PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, X);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) Cs[i][j] = X[i * N + j];
        free(X);
    }
    free(mark); free(pi); free(pj); free(esym); free(psym); free(isgrp);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double G[N][N], C[N][N], Cs[N][N], Ss[N][N], Gs[N][N];
    double R0, var0, vol0, vols_base, Rrisk, volstr, mult, vols_str;
    double xb[N + 1], xs[N + 1];
    int ok = 1;

    R0 = 0.0; for (int j = 0; j < N; j++) R0 += MU[j] * X0[j];
    var0 = port_var(X0, SIG); vol0 = sqrt(var0);

    ok &= chol(SIG, G);
    ok &= min_risk(G, R0, &vols_base, xb);
    ok &= max_return(G, vol0, &Rrisk);

    corr_from_cov(SIG, C);
    ok &= stress_corr(C, Cs);
    cov_from_corr(SIG, Cs, Ss);
    volstr = sqrt(port_var(X0, Ss));
    mult = port_var(X0, Ss) / var0;

    ok &= chol(Ss, Gs);
    ok &= min_risk(Gs, R0, &vols_str, xs);

    if (ok) {
        int ok_c = 1;
        for (int g = 0; g < NPAIR; g++) {
            int a = PAIRS[g][0], b = PAIRS[g][1];
            if (fabs(Cs[a][b] - RHO) > 1e-3) ok_c = 0;
        }
        ok = ok_c &&
             fabs(vol0 - 0.229369) < 1e-4 && fabs(mult - 1.255767) < 2e-3 &&
             fabs(volstr - 0.257033) < 2e-3 &&
             fabs(vols_base - 0.135508) < 2e-3 && fabs(vols_str - 0.137191) < 2e-3 &&
             fabs(Rrisk - 0.400353) < 2e-3;
        printf("big_portfolio  N=%d (%s..%s)  R0=%.6f vol0=%.6f (case 0.229369)\n", N, ASSET[0], ASSET[N-1], R0, vol0);
        printf("               group C* entries=0.85  multiplier=%.6f (case 1.255767) stressed vol=%.6f (case 0.257033)\n",
               mult, volstr);
        printf("               min-risk@R0 base=%.6f stressed=%.6f (case 0.135508 / 0.137191)\n",
               vols_base, vols_str);
        printf("               max-return@vol0=%.6f (case 0.400353)  %s\n", Rrisk, ok ? "OK" : "FAIL");
    } else {
        printf("big_portfolio  solve FAIL (rc)\n");
    }
    return ok ? 0 : 1;
}
