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

/* lmi_lyapunov.c - Lyapunov LMI for the stability of a linear system
 * (andel-vitorio/linear-matrix-inequality, "LMIs & Linear System Analysis and
 * Control", Problem 1):
 *
 *     minimize   trace(P)
 *     subject to P >= 0,     A' P + P A + Q <= 0
 *
 * with Q = I.  Lyapunov's theorem: a P solution exists iff A is stable, and the
 * minimum-trace solution saturates the inequality, i.e. solves A'P + PA + Q = 0.
 *
 * Case A = [[0,-2],[1,-3]], Q = I.  With P = [[p,q],[q,r]],
 *     A'P + PA + I = [[2q+1, r-2p-3q], [r-2p-3q, -4q-6r+1]] = 0
 * gives q = -1/2, r = 1/2, p = 1, so P* = [[1,-1/2],[-1/2,1/2]] (PSD, eigenvalues
 * 0.691, 0.809) and trace(P*) = 3/2.
 *
 * The LMI A'P+PA+I <= 0 is written with a second PSD bar variable S and the
 * three entrywise equalities S = -(A'P+PA+I).  The entries of a bar variable are
 * addressed through SYMMETRIC MATRICES (appendsparsesymmat + putbaraij), exactly
 * as samples/mosek_comparison/sdo2.c does; <M,X> counts the off-diagonal of M
 * twice, so a unit E_ij with i!=j brings 2*X_ij.
 *
 * Usage: lmi_lyapunov   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

int main(void) {
    /* A = [[0,-2],[1,-3]] */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    int d2 = 2, m00, m01, m11;
    PRIMAL_appendsparsesymmat(t, d2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &m00);
    PRIMAL_appendsparsesymmat(t, d2, 1, (int[]){0}, (int[]){1}, (double[]){1.0}, &m01);
    PRIMAL_appendsparsesymmat(t, d2, 1, (int[]){1}, (int[]){1}, (double[]){1.0}, &m11);
    PRIMAL_appendbarvars(t, 1, &d2);   /* barvar 0 : P (2x2 PSD) */
    PRIMAL_appendbarvars(t, 1, &d2);   /* barvar 1 : S (2x2 PSD) = -(A'P+PA+I) */
    PRIMAL_appendcons(t, 3);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    /* Row 0: S00 + 2 P01 = -1  ->  <E00,S> + <E01,P> = -1 */
    PRIMAL_putbaraij(t, 0, 1, 1, &m00, (double[]){1.0});
    PRIMAL_putbaraij(t, 0, 0, 1, &m01, (double[]){1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, -1.0, -1.0);
    /* Row 1: S01 + P11 - 2 P00 - 3 P01 = 0
     *   -> 0.5<E01,S> + <E11,P> - 2<E00,P> - 1.5<E01,P> = 0 */
    PRIMAL_putbaraij(t, 1, 1, 1, &m01, (double[]){0.5});
    PRIMAL_putbaraij(t, 1, 0, 1, &m11, (double[]){1.0});
    PRIMAL_putbaraij(t, 1, 0, 1, &m00, (double[]){-2.0});
    PRIMAL_putbaraij(t, 1, 0, 1, &m01, (double[]){-1.5});
    PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    /* Row 2: S11 - 4 P01 - 6 P11 = -1  ->  <E11,S> - 2<E01,P> - 6<E11,P> = -1 */
    PRIMAL_putbaraij(t, 2, 1, 1, &m11, (double[]){1.0});
    PRIMAL_putbaraij(t, 2, 0, 1, &m01, (double[]){-2.0});
    PRIMAL_putbaraij(t, 2, 0, 1, &m11, (double[]){-6.0});
    PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, -1.0, -1.0);
    /* objective: minimize trace(P) = <E00,P> + <E11,P> */
    PRIMAL_putbarcj(t, 0, 1, &m00, (double[]){1.0});
    PRIMAL_putbarcj(t, 0, 1, &m11, (double[]){1.0});

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, P[4] = {0}, S[4] = {0};
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, P);
        PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 1, S);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    double trP = P[0] + P[3], detP = P[0]*P[3] - P[1]*P[2];
    double disc = sqrt(fmax(0.0, trP*trP/4 - detP));
    double lmin = trP/2 - disc;
    int good = ok && fabs(obj - 1.5) < 1e-6 && fabs(P[0]-1.0) < 1e-6 &&
               fabs(P[1]+0.5) < 1e-6 && fabs(P[2]+0.5) < 1e-6 && fabs(P[3]-0.5) < 1e-6 && lmin > -1e-9;
    printf("lmi_lyapunov  obj=%.8f (atteso 1.5 = 3/2)  P=[[%.6f,%.6f],[%.6f,%.6f]]  lmin=%.2e\n",
           obj, P[0], P[1], P[2], P[3], lmin);
    printf("  trace(P)=%.8f  S (=-(A'P+PA+I)) = [[%.2e,%.2e],[.,%.2e]]  %s\n",
           trP, S[0], S[1], S[2], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
