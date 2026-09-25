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

/* limit_analysis_dp.c - Drucker-Prager yield cone of a rigid-plastic limit
 * analysis, as a second-order cone program (a0082273/limit-analysis-with-
 * node-based-element-and-MOSEK, which reduces mixed rigid-plastic FEM limit
 * analysis to an SOCP solved by MOSEK).
 *
 * The yield function is the Drucker-Prager surface
 *     yf(sigma) = sqrt(J2) + 3 alpha m - k <= 0,   m = tr(sigma)/3,
 * with J2 the second invariant of the deviatoric stress.  Writing the
 * deviatoric stress as the 6-vector s = sigma - m I (s:s = 2 J2), the surface
 * is the second-order cone
 *     || s ||  <=  sqrt(2) (k - alpha tr(sigma)).
 * The material parameters follow the repo:
 *     alpha = sin(phi) / sqrt(9 + 3 sin^2 phi),
 *     k     = sqrt(3) c cos(phi) / sqrt(3 + sin^2 phi).
 *
 * A load PROPORTIONAL stress state sigma = t * sigma0 is admissible up to the
 * load factor at which the cone is hit:
 *     t* = k / ( sqrt(J2(sigma0)) + alpha tr(sigma0) ).
 *
 * This ports the CONE (the model's core), not the full node-based FEM mesh.
 *
 * Case: uniaxial sigma0 = (1,0,0,0,0,0), cohesion c = 1, friction phi = 30 deg.
 *   alpha = 0.5/sqrt(9.75) = 0.160128,
 *   k     = sqrt(3) cos30 / sqrt(3.25) = 0.832050,
 *   sqrt(J2(sigma0)) = 1/sqrt(3) = 0.577350,  tr(sigma0) = 1,
 *   t* = 0.832050 / (0.577350 + 0.160128) = 1.128237.
 *
 * Usage: limit_analysis_dp   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define PI 3.14159265358979323846

int main(void) {
    /* material (materials.py) */
    const double c = 1.0, phi = 30.0 * PI / 180.0;
    const double alpha = sin(phi) / sqrt(9.0 + 3.0*sin(phi)*sin(phi));
    const double kk = sqrt(3.0)*c*cos(phi) / sqrt(3.0 + sin(phi)*sin(phi));
    /* proportional reference stress: uniaxial */
    const double s0[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int T = 0, SIG = 1, SDEV = 7, RHS = 13, nv = 14;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);
    PRIMAL_putvarbound(t, T, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < 6; i++) PRIMAL_putvarbound(t, SIG+i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < 6; i++) PRIMAL_putvarbound(t, SDEV+i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, RHS, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_appendcons(t, 6 + 6 + 1);
    int r = 0;
    /* sigma_i - s0_i * t = 0 */
    for (int i = 0; i < 6; i++) {
        PRIMAL_putarow(t, r, 2, (int[]){SIG+i, T}, (double[]){1.0, -s0[i]});
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0.0, 0.0); r++;
    }
    /* sdev_i - sigma_i + (sigma0+sigma1+sigma2)/3 = 0   (deviatoric) */
    for (int i = 0; i < 6; i++) {
        /* sdev_i - sigma_i + (sigma0+sigma1+sigma2)/3 = 0, no duplicate columns */
        int sub[5]; double val[5]; int nn = 0;
        sub[nn] = SDEV+i; val[nn] = 1.0; nn++;
        if (i < 3) {
            sub[nn] = SIG+i; val[nn] = -2.0/3.0; nn++;
            sub[nn] = SIG+(i+1)%3; val[nn] = 1.0/3.0; nn++;
            sub[nn] = SIG+(i+2)%3; val[nn] = 1.0/3.0; nn++;
        } else {
            /* shear components: sdev_i = sigma_i (the mean stress does not
             * change the off-diagonal entries) */
            sub[nn] = SIG+i; val[nn] = -1.0; nn++;
        }
        PRIMAL_putarow(t, r, nn, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0.0, 0.0); r++;
    }
    /* RHS - sqrt2*k + sqrt2*alpha*(sigma0+sigma1+sigma2) = 0 */
    {
        int sub[4] = {RHS, SIG+0, SIG+1, SIG+2};
        double val[4] = {1.0, sqrt(2.0)*alpha, sqrt(2.0)*alpha, sqrt(2.0)*alpha};
        PRIMAL_putarow(t, r, 4, sub, val);
        PRIMAL_putconbound(t, r, PRIMAL_BK_FX, sqrt(2.0)*kk, sqrt(2.0)*kk); r++;
    }
    /* (RHS, sdev0..sdev5) in Q_7  :  RHS >= ||s|| */
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 7, (int[]){RHS, SDEV+0, SDEV+1, SDEV+2, SDEV+3, SDEV+4, SDEV+5});
    PRIMAL_putcj(t, T, 1.0);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    { PRIMALprostae ps=PRIMAL_PRO_STA_UNKNOWN; PRIMALsolstae ss=PRIMAL_SOL_STA_UNKNOWN; PRIMAL_getprosta(t,PRIMAL_SOL_ITR,&ps); PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss); fprintf(stderr,"rc=%d prosta=%d solsta=%d\n",(int)rc,(int)ps,(int)ss); }
    double tstar = 0.0, v[32] = {0};
    if (ok) { PRIMAL_getxx(t, PRIMAL_SOL_ITR, v); tstar = v[T]; }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* hand value and independent check of the cone at the returned stress */
    double want = kk / (1.0/sqrt(3.0) + alpha*1.0);
    double s0v = v[SIG+0], s1v = v[SIG+1], s2v = v[SIG+2];
    double m = (s0v + s1v + s2v)/3.0;
    double d[6] = {s0v-m, s1v-m, s2v-m, v[SIG+3], v[SIG+4], v[SIG+5]};
    double sq = 0; for (int i = 0; i < 6; i++) sq += d[i]*d[i];
    double yf = sqrt(sq/2.0) + 3.0*alpha*m - kk;             /* must be ~ 0 at yield */
    int good = ok && fabs(tstar - want) < 1e-7 && fabs(tstar - 1.128237) < 1e-5 &&
               fabs(yf) < 1e-7;
    printf("limit_analysis_dp  alpha=%.8f k=%.8f  t*=%.8f (atteso %.8f)\n", alpha, kk, tstar, want);
    printf("  sigma_x=%.6f  sqrt(J2)=%.6f  m=%.6f  yf=%.2e  %s\n",
           s0v, sqrt(sq/2.0), m, yf, good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
