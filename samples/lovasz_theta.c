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

/* lovasz_theta.c - maximum-clique upper bound via the Lovasz theta SDP.
 *
 * Source: Rudolfovoorg/Improving_Upper_Bounds_of_MCP_using_Reduction_Rules
 * (C++17, MOSEK 11.1), which improves maximum-clique upper bounds with
 * truss-core reduction rules and several combinatorial/SDP bounds.  The SDP
 * bound is the Lovasz theta number of the COMPLEMENT,
 *
 *   theta(Gbar) = max <J, X>   s.t.  tr(X) = 1,
 *                                    X_ij = 0  for every non-edge (i,j) of G,
 *                                    X >= 0,
 *
 * which is an upper bound on the clique number omega(G): omega(G) <=
 * theta(Gbar) <= chi(G).  The repository also has an equivalent vector-coloring
 * formulation (used on sparse graphs); this sample ports the theta model,
 * which is exactly the semidefinite-bar path of this library.
 *
 * Hand-derived instances (all exact):
 *   C5 (5-cycle, self-complementary): theta = sqrt(5) = 2.2360679..., omega = 2
 *   K3 (triangle):                    theta = 3,                 omega = 3
 *   E5 (empty graph, no edges):       theta = 1,                 omega = 1
 *
 * The sample solves each with PrimalSolver and checks the value against the
 * closed form, the primal X (tr = 1, X >= 0, X_ij = 0 on non-edges), and the
 * integer bound floor(theta) against omega.
 *
 * Usage: lovasz_theta   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"
#include "linalg.h"

#define NMAX 5

/* theta(Gbar) for a graph on n vertices given by symmetric adj[n][n] */
static double theta_sdp(int n, const int adj[NMAX][NMAX], int *ok_out, double *X_out) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    int dim = n;
    PRIMAL_appendbarvars(t, 1, &dim);
    int ne = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (!adj[i][j]) ne++;
    PRIMAL_appendcons(t, 1 + ne);

    {   /* objective matrix J (all ones), lower triangle */
        int si[NMAX * NMAX], sj[NMAX * NMAX]; double sv[NMAX * NMAX]; int k = 0;
        for (int i = 0; i < n; i++) {
            si[k] = i; sj[k] = i; sv[k] = 1.0; k++;
            for (int j = 0; j < i; j++) { si[k] = i; sj[k] = j; sv[k] = 1.0; k++; }
        }
        int idx;
        PRIMAL_appendsparsesymmat(t, dim, k, si, sj, sv, &idx);
        PRIMAL_putbarcj(t, 0, 1, &idx, (double[]){1.0});
    }
    {   /* row 0: <I, X> = tr(X) = 1 */
        int si[NMAX], sj[NMAX]; double sv[NMAX];
        for (int i = 0; i < n; i++) { si[i] = i; sj[i] = i; sv[i] = 1.0; }
        int idx;
        PRIMAL_appendsparsesymmat(t, dim, n, si, sj, sv, &idx);
        PRIMAL_putbaraij(t, 0, 0, 1, &idx, (double[]){1.0});
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    int row = 1;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (adj[i][j]) continue;
            int si[1] = {j}, sj[1] = {i}; double sv[1] = {0.5};   /* <E_ij,X> = X_ij */
            int idx;
            PRIMAL_appendsparsesymmat(t, dim, 1, si, sj, sv, &idx);
            PRIMAL_putbaraij(t, row, 0, 1, &idx, (double[]){1.0});
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0);
            row++;
        }
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double th = 0.0;
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &th);
        if (X_out) PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, X_out);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = ok;
    return th;
}

/* min eigenvalue of the n x n solution matrix (PSD check) */
static double min_eig(int n, const double *X) {
    double ev[NMAX], evec[NMAX * NMAX];
    dmat_eig_jacobi(n, X, ev, evec);
    double m = ev[0];
    for (int i = 1; i < n; i++) if (ev[i] < m) m = ev[i];
    return m;
}

/* verify tr(X)=1, X⪰0, X_ij=0 on non-edges; returns 1 if all hold */
static int check_X(int n, const int adj[NMAX][NMAX], const double *X, double tol) {
    double tr = 0.0;
    for (int i = 0; i < n; i++) tr += X[i * n + i];
    if (fabs(tr - 1.0) > 1e-6) return 0;
    if (min_eig(n, X) < -tol) return 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (!adj[i][j] && fabs(X[i * n + j]) > tol) return 0;
    return 1;
}

int main(void) {
    int C5[NMAX][NMAX] = {{0}};
    for (int i = 0; i < 5; i++) { C5[i][(i + 1) % 5] = 1; C5[(i + 1) % 5][i] = 1; }
    int K3[NMAX][NMAX] = {{0}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (i != j) K3[i][j] = 1;
    int E5[NMAX][NMAX] = {{0}};

    double th5 = 0, the = 0, X[NMAX * NMAX];
    int ok5 = 0, ok3 = 0, oke = 0;
    th5 = theta_sdp(5, C5, &ok5, X); int x5 = ok5 && check_X(5, C5, X, 1e-6);
    double th3v = theta_sdp(3, K3, &ok3, X); int x3 = ok3 && check_X(3, K3, X, 1e-6);
    the = theta_sdp(5, E5, &oke, X); int xe = oke && check_X(5, E5, X, 1e-6);

    int ok = x5 && x3 && xe &&
             fabs(th5 - sqrt(5.0)) < 1e-6 && fabs(th3v - 3.0) < 1e-6 &&
             fabs(the - 1.0) < 1e-6 &&
             (int)floor(th5 + 1e-6) == 2 && (int)floor(th3v + 1e-6) == 3 &&
             (int)floor(the + 1e-6) == 1;

    printf("lovasz_theta  max-clique upper bound (theta of the complement)\n");
    printf("  C5 (omega=2): theta=%.10f   atteso sqrt(5)=%.10f   X ok=%d\n",
           th5, sqrt(5.0), x5);
    printf("  K3 (omega=3): theta=%.10f   atteso 3              X ok=%d\n", th3v, x3);
    printf("  E5 (omega=1): theta=%.10f   atteso 1              X ok=%d\n", the, xe);
    printf("  bound intero floor(theta): C5->%d, K3->%d, E5->%d\n",
           (int)floor(th5 + 1e-6), (int)floor(th3v + 1e-6), (int)floor(the + 1e-6));
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
