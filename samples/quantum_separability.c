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

/* quantum_separability.c - PPT and CCNR criteria for bipartite entanglement.
 *
 * Source: y1-zhu/quantum-correlations, the code of "Certifying quantum
 * separability with adaptive polytopes" (SciPost Phys. 16, 063).  That repo
 * estimates the ROBUSTNESS of a bipartite state to a set of non-entangled
 * (or bound-entangled) states with SDPs, using SCS in place of MOSEK.  This
 * sample ports the two basic criteria of the file
 * src/entanglement/RobustnessToPPT.jl and RobustnessToCCNR.jl:
 *
 *   - PPT (Peres-Horodecki): a mixture  t*rho + (1-t)/d^2 I  stays PPT while
 *     t*rho^G + (1-t)/d^2 I >= 0, where rho^G is the partial transpose.  The
 *     largest such t is an SDP,  max t  s.t.  (1/d^2) I + t C >= 0  with
 *     C = rho^G - (1/d^2) I, and equals the closed form of the repository,
 *     t* = 1 / (1 - d^2 * lambda_min(rho^G)).
 *
 *   - CCNR (realignment): for a separable state the realigned matrix R(rho),
 *     R[(i,j),(k,l)] = rho[(i,k),(j,l)], has nuclear norm <= 1, so the same
 *     white-noise robustness is where ||R(t rho + (1-t) I/d^2)||_1 = 1.
 *
 * Instance: the qutrit isotropic state rho_p = p |Phi+><Phi+| + (1-p) I/9,
 * |Phi+> = (|00>+|11>+|22>)/sqrt(3).  Its partial transpose has eigenvalues
 * (1+2p)/9 (symmetric subspace) and (1-4p)/9 (antisymmetric), so
 * lambda_min = (1-4p)/9 and t*_PPT = 1/(4p); the realignment norm is
 * (1+8p)/3 and the CCNR robustness is also 1/(4p).  At p=1/4 (the PPT
 * separability threshold) both criteria give exactly 1.
 *
 * The SDP is solved by PRIMALSOLVER over one 9x9 semidefinite bar variable
 * (the entrywise equality  S = (1/9) I + t C  plus  S >= 0), and every number
 * is cross-checked against the closed forms and Jacobi eigenvalues.
 *
 * Usage: quantum_separability   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"
#include "linalg.h"

#define D 3
#define N (D * D)          /* 9 */

static double RHO[N * N];      /* state                    */
static double RHOT[N * N];     /* partial transpose on B   */

/* rho_p = p |Phi+><Phi+| + (1-p) I/9, then partial transpose on the second
 * subsystem:  rho^G[(i,k),(j,l)] = rho[(i,l),(j,k)]. */
static void build_state(double p) {
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++)
            for (int c = 0; c < D; c++)
                for (int d = 0; d < D; d++) {
                    double v = (a == c && b == d) ? (1.0 - p) / (double)(D * D) : 0.0;
                    if (a == b && c == d) v += p / (double)D;
                    RHO[(a * D + b) * N + (c * D + d)] = v;
                }
    for (int a = 0; a < D; a++)
        for (int k = 0; k < D; k++)
            for (int j = 0; j < D; j++)
                for (int l = 0; l < D; l++)
                    RHOT[(a * D + k) * N + (j * D + l)] = RHO[(a * D + l) * N + (j * D + k)];
}

/* smallest eigenvalue of a real symmetric N x N matrix (Jacobi) */
static double lambda_min(const double *A) {
    double ev[N], evec[N * N];
    dmat_eig_jacobi(N, A, ev, evec);
    double m = ev[0];
    for (int i = 1; i < N; i++) if (ev[i] < m) m = ev[i];
    return m;
}

/* nuclear norm (sum of singular values) of a real N x N matrix, via the
 * eigenvalues of M M^T */
static double nuclear_norm(const double *A) {
    double M[N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * A[j * N + k];
            M[i * N + j] = s;
        }
    double ev[N], evec[N * N];
    dmat_eig_jacobi(N, M, ev, evec);
    double s = 0.0;
    for (int i = 0; i < N; i++) s += sqrt(ev[i] > 0.0 ? ev[i] : 0.0);
    return s;
}

/* CCNR: realigned state R[(i,j),(k,l)] = rho[(i,k),(j,l)] */
static void realign(const double *rho, double *R) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            for (int k = 0; k < D; k++)
                for (int l = 0; l < D; l++)
                    R[(i * D + j) * N + (k * D + l)] = rho[(i * D + k) * N + (j * D + l)];
}

/* SDP: max t  s.t.  S = (1/9) I + t C >= 0,  C = rho^G - (1/9) I.
 * Encoded with one bar variable S (dim 9) and the scalar t; the lower
 * triangle of S - t C = (1/9) I is one equality row each. */
static int ppt_robustness_sdp(double *t_out) {
    PRIMALenv_t env; PRIMALtask_t task;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 1);                  /* scalar t */
    PRIMAL_putvarbound(task, 0, PRIMAL_BK_RA, 0.0, 1.0);
    int dim = N;
    PRIMAL_appendbarvars(task, 1, &dim);
    PRIMAL_appendcons(task, N * (N + 1) / 2);    /* lower triangle of S */
    int row = 0;
    for (int L1 = 0; L1 < N; L1++)
        for (int L2 = 0; L2 <= L1; L2++) {
            double C = RHOT[L1 * N + L2] - (L1 == L2 ? 1.0 / (double)(D * D) : 0.0);
            int m;
            int si = L1, sj = L2;
            double sv = (L1 == L2) ? 1.0 : 0.5;   /* <m,S> = S_L1L2 */
            PRIMAL_appendsparsesymmat(task, dim, 1, &si, &sj, &sv, &m);
            PRIMAL_putbaraij(task, row, 0, 1, &m, (double[]){1.0});
            PRIMAL_putarow(task, row, 1, (int[]){0}, (double[]){-C});
            double rhs = (L1 == L2) ? 1.0 / (double)(D * D) : 0.0;
            PRIMAL_putconbound(task, row, PRIMAL_BK_FX, rhs, rhs);
            row++;
        }
    PRIMAL_putcj(task, 0, 1.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMALrescodee rc = PRIMAL_optimize(task);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) { double x[1]; PRIMAL_getxx(task, PRIMAL_SOL_ITR, x); *t_out = x[0]; }
    PRIMAL_deletetask(&task); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    const double P = 0.5;                 /* entangled isotropic state */
    const double PCT = 0.25;              /* PPT separability threshold  */
    build_state(P);

    /* --- eigenvalues of rho^G: closed form vs Jacobi -------------------- */
    double lmin = lambda_min(RHOT);
    double lmin_cf = (1.0 - 4.0 * P) / (double)(D * D);
    /* --- PPT robustness: SDP (PrimalSolver) vs closed form --------------- */
    double t_sdp = 0.0;
    int ok_sdp = ppt_robustness_sdp(&t_sdp);
    double t_cf = 1.0 / (1.0 - (double)(D * D) * lmin);
    /* --- CCNR: realignment norm and robustness -------------------------- */
    double Rmat[N * N];
    realign(RHO, Rmat);
    double ccnr = nuclear_norm(Rmat);
    double ccnr_cf = (1.0 + 8.0 * P) / 3.0;
    /* CCNR robustness: mix with I/9 until ||R(mixture)||_1 = 1 (the scan of
     * RobustnessToCCNR.jl).  R(t rho + (1-t) I/9) = t Rmat + (1-t)/9 e e^T,
     * with e e^T[(i,j),(k,l)] = delta_ij delta_kl. */
    double eeT[N * N];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            for (int k = 0; k < D; k++)
                for (int l = 0; l < D; l++)
                    eeT[(i * D + j) * N + (k * D + l)] = (i == j && k == l) ? 1.0 : 0.0;
    double t_ccnr = 1.0;
    for (int k = 0; k <= 10000; k++) {
        double t = k * 1e-4;
        double RM[N * N];
        for (int m = 0; m < N * N; m++)
            RM[m] = t * Rmat[m] + (1.0 - t) * (1.0 / (double)(D * D)) * eeT[m];
        if (nuclear_norm(RM) > 1.0) break;
        t_ccnr = t;
    }
    /* --- witness:  W = I - |Phi~><Phi~|,  |Phi~> = sum |mm> -------------- *
     * W^G = I - SWAP >= 0, and tr(W rho) = 2/3 - 8p/3 < 0 certifies that
     * rho is entangled. */
    double W[N * N];
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++)
            for (int c = 0; c < D; c++)
                for (int d = 0; d < D; d++)
                    W[(a * D + b) * N + (c * D + d)] =
                        ((a == c && b == d) ? 1.0 : 0.0) - ((a == b && c == d) ? 1.0 : 0.0);
    double WG[N * N];
    for (int a = 0; a < D; a++)
        for (int k = 0; k < D; k++)
            for (int j = 0; j < D; j++)
                for (int l = 0; l < D; l++)
                    WG[(a * D + k) * N + (j * D + l)] = W[(a * D + l) * N + (j * D + k)];
    double wmin = lambda_min(WG);
    double wtr = 0.0;
    for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) wtr += W[a * N + b] * RHO[b * N + a];
    double wtr_cf = 2.0 / 3.0 - 8.0 * P / 3.0;

    /* --- thresholds: at p=1/4 both criteria are exactly 1 ---------------- */
    build_state(PCT);
    double lmin_t = lambda_min(RHOT);
    double t_thr = 1.0 / (1.0 - (double)(D * D) * lmin_t);
    double Rm_t[N * N];
    realign(RHO, Rm_t);
    double ccnr_t = nuclear_norm(Rm_t);

    int ok = ok_sdp && fabs(t_sdp - t_cf) < 1e-5 &&
             fabs(lmin - lmin_cf) < 1e-9 &&
             fabs(ccnr - ccnr_cf) < 1e-8 &&
             fabs(t_ccnr - t_cf) < 1e-3 &&
             (wmin > -1e-9) && (wtr < -1e-6) && fabs(wtr - wtr_cf) < 1e-9 &&
             fabs(t_thr - 1.0) < 1e-8 && fabs(ccnr_t - 1.0) < 1e-8;

    printf("quantum_separability  qutrit isotropic  p=%.2f  (p_thr=%.2f)\n", P, PCT);
    printf("  lambda_min(rho^T) = %.10f   (chiusa %.10f)\n", lmin, lmin_cf);
    printf("  PPT robustness: SDP t*=%.10f   chiusa 1/(4p)=%.10f   ok=%d\n", t_sdp, t_cf, ok_sdp);
    printf("  CCNR ||R(rho)||_1 = %.10f   (chiusa (1+8p)/3=%.10f)\n", ccnr, ccnr_cf);
    printf("  CCNR robustness t*=%.10f   PPT robustness t*=%.10f\n", t_ccnr, t_cf);
    printf("  witness W=I-|Phi~><Phi~|: lambda_min(W^T)=%.3e, tr(W rho)=%.10f (chiusa %.10f)\n",
           wmin, wtr, wtr_cf);
    printf("  soglia p=1/4: PPT t*=%.10f, CCNR ||R||=%.10f (entrambi = 1)\n", t_thr, ccnr_t);
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
