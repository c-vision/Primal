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

/* unit_commitment.c - a tiny thermal unit-commitment MILP.
 *
 * Source: MOSEK Modeling Cookbook / MOSEK Tutorials, "unit commitment".
 * Two generators over two periods; commitment and startup are binary, the
 * dispatch is continuous:
 *
 *   min  sum_{i,t} ( no_load_i u_{i,t} + marginal_i p_{i,t} + startup_i v_{i,t} )
 *   s.t. sum_i p_{i,t} = d_t                         (demand, both periods)
 *        min_i u_{i,t} <= p_{i,t} <= max_i u_{i,t}   (dispatch bounds)
 *        v_{i,t} >= u_{i,t} - u_{i,t-1},  v >= 0     (startup, u_{i,-1} = 0)
 *        u_{i,t}, v_{i,t} in {0,1}
 *
 * Instance (hand-derived optimum 36):
 *   gen 0: min 1, max 3, no-load 5, marginal 2, startup 10
 *   gen 1: min 0, max 4, no-load 0, marginal 6, startup 0
 *   demand: d_0 = 2, d_1 = 4
 *
 * Derivation. Period 0 (d=2): gen 0 costs 5+2*2 = 9, gen 1 costs 6*2 = 12, so
 * gen 0 serves it. Period 1 (d=4): gen 0 at its max 3 costs 5+6 = 11 and the
 * remaining 1 from gen 1 costs 6, total 17 (gen 1 alone would be 24). Gen 0
 * starts once (v=1 at t=0, from off) = 10; gen 1 has zero startup. Total
 * 9 + 17 + 10 = 36. The all-gen-1 plan (12 + 24 = 36) is the other optimum.
 *
 * Usage: unit_commitment   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NG 2
#define NT 2

/* variable layout: u = i*NT+t, v = NG*NT + i*NT+t, p = 2*NG*NT + i*NT+t */
#define U(i,t) ((i) * NT + (t))
#define V(i,t) (NG * NT + (i) * NT + (t))
#define P(i,t) (2 * NG * NT + (i) * NT + (t))
#define NVAR (3 * NG * NT)

static const double lo[NG]   = {1, 0};
static const double up[NG]   = {3, 4};
static const double noload[NG] = {5, 0};
static const double marg[NG] = {2, 6};
static const double startc[NG] = {10, 0};
static const double demand[NT] = {2, 4};

int main(void) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 2 * NT + NG * NT + NG * NT, NVAR, &t);

    for (int i = 0; i < NG; i++)
        for (int k = 0; k < NT; k++) {
            PRIMAL_putvarbound(t, U(i, k), PRIMAL_BK_RA, 0.0, 1.0);
            PRIMAL_putvartype(t, U(i, k), PRIMAL_VAR_TYPE_INT_BIN);
            PRIMAL_putvarbound(t, V(i, k), PRIMAL_BK_RA, 0.0, 1.0);
            PRIMAL_putvartype(t, V(i, k), PRIMAL_VAR_TYPE_INT_BIN);
            PRIMAL_putvarbound(t, P(i, k), PRIMAL_BK_LO, 0.0, INFINITY);
            PRIMAL_putcj(t, U(i, k), noload[i]);
            PRIMAL_putcj(t, V(i, k), startc[i]);
            PRIMAL_putcj(t, P(i, k), marg[i]);
        }

    int row = 0;
    for (int k = 0; k < NT; k++) {          /* demand: sum_i p_i,k = d_k */
        int sub[NG]; double val[NG];
        for (int i = 0; i < NG; i++) { sub[i] = P(i, k); val[i] = 1.0; }
        PRIMAL_putarow(t, row, NG, sub, val);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, demand[k], demand[k]);
        row++;
    }
    for (int i = 0; i < NG; i++)
        for (int k = 0; k < NT; k++) {
            /* p - max*u <= 0 */
            PRIMAL_putarow(t, row, 2, (int[]){P(i, k), U(i, k)},
                           (double[]){1.0, -up[i]});
            PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 0.0);
            row++;
            /* p - min*u >= 0 */
            PRIMAL_putarow(t, row, 2, (int[]){P(i, k), U(i, k)},
                           (double[]){1.0, -lo[i]});
            PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY);
            row++;
            /* v - u_k + u_{k-1} >= 0  (u_{i,-1} = 0) */
            if (k == 0) {
                PRIMAL_putarow(t, row, 2, (int[]){V(i, k), U(i, k)},
                               (double[]){1.0, -1.0});
            } else {
                PRIMAL_putarow(t, row, 3, (int[]){V(i, k), U(i, k), U(i, k - 1)},
                               (double[]){1.0, -1.0, 1.0});
            }
            PRIMAL_putconbound(t, row, PRIMAL_BK_LO, 0.0, INFINITY);
            row++;
        }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double obj = 0.0;
    if (ok) {
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
        double x[NVAR];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        ok = fabs(obj - 36.0) < 1e-6;
        printf("unit_commitment  d=[2,4]  p0=(%.1f,%.1f) p1=(%.1f,%.1f) obj=%.4f (atteso 36) %s\n",
               x[P(0, 0)], x[P(0, 1)], x[P(1, 0)], x[P(1, 1)], obj, ok ? "OK" : "FAIL");
    } else {
        printf("unit_commitment  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
