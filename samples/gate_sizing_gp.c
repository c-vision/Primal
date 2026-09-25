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

/* gate_sizing_gp.c - gate sizing of a digital circuit as a geometric program
 * (lrakai/gate-sizing, from Boyd's "A Tutorial on Geometric Programming").
 *
 * Variables: the gate scale factors s_i, the arrival times T_i, the "max"
 * auxiliaries V_i and the objective D (the critical path delay).  The cell
 * library gives, per gate, an area a, a unit output resistance gamma, an
 * intrinsic input capacitance alpha and a per-unit input capacitance beta, so
 * gate i's delay is the posynomial
 *     D_i(s) = gamma_i * s_i^-1 * sum_{j driven by i} (beta_j s_j + alpha_j).
 * The formulation minimises D subject to  D_i <= T_i,  T_anc <= V_i,
 * T_i <= D, and a constraint on the total area  sum_i a_i s_i <= Amax.
 *
 * GP under log transform y = log x (same encoder as samples/gp_toolbox.c):
 * each posynomial  sum_k c_k exp(a_k . y) <= 1  becomes one exponential cone
 * per monomial plus a linear row sum p_k <= 1; a single-monomial constraint is
 * linear; the objective min D is linear (D itself is a variable).
 *
 * Hand case: a chain of two inverters INV -> INV -> primary output, with the
 * library INV = (area 1, gamma 1, alpha 1, beta 1) and a primary output of
 * intrinsic capacitance 10.  Then
 *     D(s) = (s2 + 1)/s1 + 10/s2,   area = s1 + s2 <= Amax = 10.
 * The area binds (larger gates are faster), so s1 = 10 - s2 and
 *     dD/ds2 = 11/(10-s2)^2 - 10/s2^2 = 0  =>  sqrt(11) s2 = sqrt(10)(10-s2)
 *     => s2 = 10 sqrt(10)/(sqrt(11)+sqrt(10)) = 4.88088482,  s1 = 5.11911518,
 *     D* = 3.19761770.
 *
 * Usage: gate_sizing_gp   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

typedef struct { double coef; int n; int var[4]; double alpha[4]; } Mono;
static Mono mon(double c, int n, const double *p) {
    Mono r; r.coef = c; r.n = n;
    for (int i = 0; i < n; i++) { r.var[i] = (int)p[2*i]; r.alpha[i] = p[2*i+1]; }
    return r;
}
#define MO(c, ...) mon((c), (int)(sizeof((double[]){__VA_ARGS__})/sizeof(double)/2), (double[]){__VA_ARGS__})

#define NV 6                       /* s1 s2 T1 T2 V2 D */
enum { S1=0, S2, T1, T2, V2, D };
#define NPF 5                      /* number of posynomial constraints */
#define NMAXM 2                    /* max monomials per posynomial */

int main(void) {
    const double Amax = 10.0;
    /* posynomials */
    Mono P[NPF][NMAXM];
    int nm[NPF];
    P[0][0] = MO(1.0, S2,1, S1,-1, T1,-1); P[0][1] = MO(1.0, S1,-1, T1,-1);              nm[0]=2;
    P[1][0] = MO(10.0, S2,-1, T2,-1);     P[1][1] = MO(1.0, V2,1, T2,-1);               nm[1]=2;
    P[2][0] = MO(1.0, T1,1, V2,-1);                                                      nm[2]=1;
    P[3][0] = MO(1.0, T2,1, D,-1);                                                       nm[3]=1;
    P[4][0] = MO(1.0/Amax, S1,1);         P[4][1] = MO(1.0/Amax, S2,1);                 nm[4]=2;

    int nmulti = 0; for (int i = 0; i < NPF; i++) if (nm[i] > 1) nmulti += nm[i];
    int YBASE = 0, EBASE = NV, PBASE = EBASE + nmulti, ONEV = PBASE + nmulti;
    int NRQ = 0; for (int i = 0; i < NPF; i++) if (nm[i] == 1) NRQ++;
    int nv = ONEV + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int i = 0; i < nv; i++) PRIMAL_putvarbound(t, i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nmulti; i++) PRIMAL_putvarbound(t, PBASE + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ONEV, PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_appendcons(t, NPF /* sum rows (only the multi ones used, plus room) */ + 2*nmulti + NRQ);
    int r = 0, ecur = 0, pcur = 0;
    for (int i = 0; i < NPF; i++) {
        if (nm[i] == 1) {                      /* linear: a.y + ln c <= 0 */
            int sub[4]; double val[4];
            for (int q = 0; q < P[i][0].n; q++) { sub[q] = YBASE + P[i][0].var[q]; val[q] = P[i][0].alpha[q]; }
            PRIMAL_putarow(t, r, P[i][0].n, sub, val);
            PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, -log(P[i][0].coef)); r++;
        } else {
            int p0 = pcur;
            for (int q = 0; q < nm[i]; q++) {
                int sub[5]; double val[5]; int nn = 0;
                sub[nn] = EBASE + ecur; val[nn] = 1.0; nn++;
                for (int z = 0; z < P[i][q].n; z++) { sub[nn] = YBASE + P[i][q].var[z]; val[nn] = -P[i][q].alpha[z]; nn++; }
                PRIMAL_putarow(t, r, nn, sub, val);
                PRIMAL_putconbound(t, r, PRIMAL_BK_FX, log(P[i][q].coef), log(P[i][q].coef)); r++;
                PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){PBASE + pcur, ONEV, EBASE + ecur});
                ecur++; pcur++;
            }
            int sub[4]; double val[4];
            for (int q = 0; q < nm[i]; q++) { sub[q] = PBASE + p0 + q; val[q] = 1.0; }
            PRIMAL_putarow(t, r, nm[i], sub, val);
            PRIMAL_putconbound(t, r, PRIMAL_BK_UP, -INFINITY, 1.0); r++;
        }
    }
    PRIMAL_putcj(t, YBASE + D, 1.0);           /* minimize log D */

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double y[NV] = {0}, obj = 0.0, v[64] = {0};
    if (ok) {
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        for (int i = 0; i < NV; i++) y[i] = exp(v[YBASE + i]);
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* independent checks: the delay and the area at the recovered sizes */
    double s1 = y[S1], s2 = y[S2];
    double delay = (s2 + 1.0)/s1 + 10.0/s2;         /* D(s) = D1 + D2 */
    double area = s1 + s2;
    double want = 10.0*sqrt(10.0)/(sqrt(11.0)+sqrt(10.0));   /* s2* */
    int good = ok && fabs(delay - 3.1976177) < 1e-4 && fabs(area - Amax) < 1e-6 &&
               fabs(s2 - want) < 1e-2 && fabs(y[D] - delay) < 1e-4;
    printf("gate_sizing_gp  delay=%.8f (atteso 3.19761770)  s1=%.6f s2=%.6f (atteso %.6f)\n",
           delay, s1, s2, want);
    printf("  area=%.6f (<= %.0f)  D(var)=%.8f  %s\n", area, Amax, y[D], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
