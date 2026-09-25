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

/* mpc_linear.c - finite-horizon linear MPC as a QP (the linearized branch of
 * jonaylton/mpc-missile; the full missile NLP is out of scope for a convex
 * solver).
 *
 * Discretized double integrator  p'' = u,  h = 1:
 *     p_{k+1} = p_k + v_k + u_k/2,   v_{k+1} = v_k + u_k
 *   i.e.  x_{k+1} = A x_k + B u_k,  A = [[1,1],[0,1]],  B = (1/2,1).
 * Minimum-energy regulation to a terminal state over N steps:
 *     minimize  sum_k u_k^2
 *     s.t.      x_0 = x_ini,  x_{k+1} = A x_k + B u_k,  x_N = x_term.
 *
 * Layout: state k occupies (2k, 2k+1) = (p_k, v_k), controls start at 2(N+1).
 *
 * Hand instance: N = 2, x_ini = (0,0), x_term = (1,0).  Then
 *   x_2 = (3/2 u_0 + 1/2 u_1, u_0 + u_1) = (1,0)  =>  u = (1,-1),
 *   sum u^2 = 2.
 *
 * Usage: mpc_linear   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NSTEP 2
#define NST (NSTEP + 1)
#define P(k) (2 * (k))
#define V(k) (2 * (k) + 1)
#define U(k) (2 * NST + (k))

int main(void) {
    const int NV = 2 * NST + NSTEP;
    const int NROW = 2 + 2 * NSTEP + 2;
    const double A[2][2] = {{1, 1}, {0, 1}};
    const double B[2] = {0.5, 1.0};
    const double xini[2] = {0, 0};

    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, NROW, NV, &t);
    for (int v = 0; v < NV; v++) PRIMAL_putvarbound(t, v, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int k = 0; k < NSTEP; k++) PRIMAL_putqobjij(t, U(k), U(k), 2.0);   /* u_k^2 */
    int row = 0;
    for (int c = 0; c < 2; c++) {                    /* x_0 = x_ini */
        PRIMAL_putarow(t, row, 1, (int[]){c}, (double[]){1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, xini[c], xini[c]); row++;
    }
    for (int k = 0; k < NSTEP; k++)                  /* x_{k+1} - A x_k - B u_k = 0 */
        for (int c = 0; c < 2; c++) {
            int sub[4]; double val[4]; int nn = 0;
            sub[nn] = c == 0 ? P(k + 1) : V(k + 1); val[nn] = 1.0; nn++;
            for (int d = 0; d < 2; d++) if (A[c][d] != 0.0) {
                sub[nn] = d == 0 ? P(k) : V(k); val[nn] = -A[c][d]; nn++;
            }
            if (B[c] != 0.0) { sub[nn] = U(k); val[nn] = -B[c]; nn++; }
            PRIMAL_putarow(t, row, nn, sub, val);
            PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;
        }
    PRIMAL_putarow(t, row, 1, (int[]){P(NSTEP)}, (double[]){1.0});   /* p_N = 1 */
    PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    PRIMAL_putarow(t, row, 1, (int[]){V(NSTEP)}, (double[]){1.0});   /* v_N = 0 */
    PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 0.0, 0.0); row++;

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0, u[NSTEP] = {0}, x[16] = {0};
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        for (int k = 0; k < NSTEP; k++) u[k] = x[U(k)];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    int good = ok && fabs(obj - 2.0) < 1e-7 && fabs(u[0] - 1.0) < 1e-6 &&
               fabs(u[1] + 1.0) < 1e-6 &&
               fabs(x[P(NSTEP)] - 1.0) < 1e-6 && fabs(x[V(NSTEP)]) < 1e-6;
    printf("mpc_linear: obj=%.8f  u=(%.6f, %.6f)  x_N=(%.6f, %.6f)  (atteso 2, u=(1,-1), x_N=(1,0))  %s\n",
           obj, u[0], u[1], x[P(NSTEP)], x[V(NSTEP)], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
