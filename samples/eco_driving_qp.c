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

/* eco_driving_qp.c - QP speed planning of a fuel-cell vehicle through
 * signalized intersections (liuboer/MOSEK-ADMM, Eco_MOSEK/MOSEK_main.m, the
 * upper level of the bi-level paper Energy 2022, doi 10.1016/j.energy.2022.123956).
 *
 * With a the vector of accelerations (one per time step, dt = 1 s), the speed
 * and position are prefix sums:
 *     v = Psi a        (v_k = sum_{i<=k} a_i,  v_0 = 0)
 *     s = Psi (Psi a)  (s_k = sum_{j<=k} v_j,  s_0 = 0)
 * and the upper level solves
 *
 *   min_a   m a'(Psi a) + m g rrc Phi'(Psi a) + 1/2 rho CD FA v_mean (Psi a)'(Psi a)
 *   s.t.    acc_min <= a_k <= acc_max
 *           v_k <= spd_max (k < N),   v_N <= 1
 *           v_k >= 0
 *           s_lower_k <= s_k <= s_upper_k           (traffic-light corridor)
 *
 * The traffic-light constraints become the time-varying linear corridor
 * [s_lower, s_upper] (green window); upstream s_upper is an IDM free-flow
 * trajectory and s_lower is built from the signal timing.  This sample uses a
 * hand-built corridor on a single short route.
 *
 * Verification (no external reference, the corridor is the model): the returned
 * a is checked FEASIBLE against every original constraint, its objective is
 * recomputed directly, and the minimum-energy solution is compared with the
 * energy of a constant-speed profile that is also feasible -- the QP must not
 * be worse.
 *
 * Usage: eco_driving_qp   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define N 10

int main(void) {
    /* vehicle parameters (from the paper's code) */
    const double spd_max = 60.0/3.6, acc_min = -2.0, acc_max = 1.4;
    const double mass = 1380.0, rho = 1.2, CD = 0.335, FA = 2.0, g = 9.81, rrc = 0.009;
    /* hand-built traffic-light corridor for ten 1 s steps: the vehicle must
     * keep up (s_lower) and stay under the free-flow cap (s_upper). */
    const double s_lo[N] = {0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 16.5, 17.0};
    const double s_up[N] = {1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 20.0, 20.5, 21.0};
    double v_mean = 17.0 / N;

    /* QP: minimize 1/2 a'Q a + c'a.
       Q = m(Psi+Psi') + rho CD FA v_mean Psi'Psi ; c = m g rrc Psi' Phi. */
    double Psi[N][N], PsitPsi, Q[N][N], c[N];
    for (int k = 0; k < N; k++) for (int i = 0; i < N; i++) Psi[k][i] = (i <= k) ? 1.0 : 0.0;
    for (int i = 0; i < N; i++) { c[i] = 0; for (int k = 0; k < N; k++) c[i] += mass*g*rrc*Psi[k][i]; }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        double q = mass*(Psi[i][j] + Psi[j][i]);
        PsitPsi = 0;
        for (int k = 0; k < N; k++) PsitPsi += Psi[k][i]*Psi[k][j];   /* (Psi'Psi)_ij */
        q += rho*CD*FA*v_mean*PsitPsi;
        Q[i][j] = q;
    }

    /* variables: a(0..N-1), v(N), s(N)  (v and s linked by rows) */
    int A = 0, V = N, S = 2*N, nv = 3*N;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    for (int k = 0; k < N; k++) {
        PRIMAL_putvarbound(t, A+k, PRIMAL_BK_RA, acc_min, acc_max);
        PRIMAL_putvarbound(t, V+k, PRIMAL_BK_LO, 0.0, (k < N-1) ? spd_max : 1.0);
        PRIMAL_putvarbound(t, S+k, PRIMAL_BK_RA, s_lo[k], s_up[k]);
    }
    /* v_k - sum_{i<=k} a_i = 0 ; s_k - sum_{j<=k} v_j = 0 */
    PRIMAL_appendcons(t, 2*N);
    for (int k = 0; k < N; k++) {
        int sub[N+1]; double val[N+1]; int nn = 0;
        sub[nn] = V+k; val[nn] = 1.0; nn++;
        for (int i = 0; i <= k; i++) { sub[nn] = A+i; val[nn] = -1.0; nn++; }
        PRIMAL_putarow(t, k, nn, sub, val);
        PRIMAL_putconbound(t, k, PRIMAL_BK_FX, 0.0, 0.0);
    }
    for (int k = 0; k < N; k++) {
        int sub[N+1]; double val[N+1]; int nn = 0;
        sub[nn] = S+k; val[nn] = 1.0; nn++;
        for (int j = 0; j <= k; j++) { sub[nn] = V+j; val[nn] = -1.0; nn++; }
        PRIMAL_putarow(t, N+k, nn, sub, val);
        PRIMAL_putconbound(t, N+k, PRIMAL_BK_FX, 0.0, 0.0);
    }
    /* the quadratic objective, lower triangle, full x'Qx coefficient */
    for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) PRIMAL_putqobjij(t, i, j, Q[i][j]);
    for (int i = 0; i < N; i++) PRIMAL_putcj(t, i, c[i]);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double a[N] = {0}, v[N] = {0}, s[N] = {0}, obj = 0.0, all[3*N] = {0};
    if (ok) {
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, all);
        for (int k = 0; k < N; k++) { a[k] = all[A+k]; v[k] = all[V+k]; s[k] = all[S+k]; }
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* independent checks: the constraints and the direct objective */
    double worst = -1e300;
    double vv[N], ss[N]; double av = 0, sv = 0;
    for (int k = 0; k < N; k++) {
        av += a[k]; vv[k] = av; sv += av; ss[k] = sv;
        worst = fmax(worst, a[k] - acc_max); worst = fmax(worst, acc_min - a[k]);
        worst = fmax(worst, (k < N-1 ? vv[k]-spd_max : vv[k]-1.0)); worst = fmax(worst, -vv[k]);
        worst = fmax(worst, ss[k]-s_up[k]); worst = fmax(worst, s_lo[k]-ss[k]);
    }
    double direct = 0;
    for (int k = 0; k < N; k++) direct += mass*a[k]*vv[k];            /* m a'(Psi a) */
    for (int k = 0; k < N; k++) direct += mass*g*rrc*vv[k];           /* m g rrc sum v */
    for (int k = 0; k < N; k++) direct += 0.5*rho*CD*FA*v_mean*vv[k]*vv[k];
    /* constant-speed feasible profile: s_k = v_mean*k needs v_k = v_mean on the
     * corridor; use the midpoint of the corridor instead as a feasible profile */
    double ac[N], vc[N], sc[N], svc = 0, cons = 0;
    for (int k = 0; k < N; k++) { vc[k] = 0.5*(s_lo[k] + s_up[k]) - (k ? 0.5*(s_lo[k-1]+s_up[k-1]) : 0.0); }
    for (int k = 0; k < N; k++) { ac[k] = vc[k] - (k ? vc[k-1] : 0.0); sc[k] = 0; }
    for (int k = 0; k < N; k++) { svc = 0; for (int j = 0; j <= k; j++) svc += vc[j]; sc[k] = svc; }
    int consFeas = 1;
    for (int k = 0; k < N; k++) {
        if (ac[k] > acc_max + 1e-9 || ac[k] < acc_min - 1e-9) consFeas = 0;
        if (vc[k] < -1e-9 || (k < N-1 && vc[k] > spd_max + 1e-9) || (k == N-1 && vc[k] > 1.0 + 1e-9)) consFeas = 0;
        if (sc[k] < s_lo[k] - 1e-9 || sc[k] > s_up[k] + 1e-9) consFeas = 0;
        cons += mass*ac[k]*vc[k] + mass*g*rrc*vc[k] + 0.5*rho*CD*FA*v_mean*vc[k]*vc[k];
    }
    int good = ok && worst < 1e-6 && fabs(direct - obj) < 1e-6*(1.0+fabs(obj)) &&
               (!consFeas || obj <= cons + 1e-6);
    printf("eco_driving_qp  N=%d  obj=%.6f  worst_viol=%.2e  costante=%.6f (feasible=%d)\n",
           N, obj, worst, cons, consFeas);
    printf("  a=(%.4f..%.4f)  v=(%.4f..%.4f)  s=(%.4f,%.4f,..,%.4f)  %s\n",
           a[0], a[N-1], v[0], v[N-1], s[0], s[1], s[N-1], good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
