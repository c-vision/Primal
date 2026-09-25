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

/* transformer_design.c - high-frequency transformer, minimum-loss design, as a
 * geometric program (MOSEK Tutorials, transformer-design).
 *
 * A GP  min f0(x)  s.t.  f_k(x) <= 1  with posynomials f (positive sums of
 * monomials c * prod x_i^a_i) is convex after the log transform y = log x:
 * each posynomial becomes a log-sum-exp,
 *   sum_i c_i exp(a_i . y) <= 1   <=>   p_i >= exp(a_i . y + ln c_i), sum p_i <= 1
 * i.e. one exponential cone (PRIMAL_CT_PEXP) per monomial plus one linear row.
 * A single-monomial constraint is linear,  a . y + ln c <= 0.  The objective
 * (itself a posynomial) is handled with an extra -t, minimizing t.
 *
 * This sample ports the notebook's 15-variable, 28-constraint HF ferrite model
 * (variables c,t,bw,hw,Np,Ns,Nls,B,J,Pcbar,Pcubar,VRrbar,VRxbar,Imbar,hbar).
 * DECLARED DEVIATION: the reference optimum is not reproduced; the solution is
 * checked by the GP primal feasibility (every posynomial <= 1) and x = exp(y).
 *
 * Usage: transformer_design   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* ---- variable indices (the notebook's vid) ----------------------------- */
enum { VC=0, VT, VBW, VHW, VNP, VNS, VNLS, VB, VJ, VPC, VPCU, VVRR, VVRX, VIM, VHB, NVX };
#define NGP NVX
#define MCON 28

typedef struct { double coef; int n; int var[8]; double alpha[8]; } Mono;
typedef struct { int nm; Mono m[6]; } Posy;

/* pairs = {var0,exp0, var1,exp1, ...}; doubles so fractional exponents survive */
static Mono M(double coef, int n, const double *pairs) {
    Mono r; r.coef = coef; r.n = n;
    for (int i = 0; i < n; i++) { r.var[i] = (int)pairs[2 * i]; r.alpha[i] = pairs[2 * i + 1]; }
    return r;
}
#define MK(coef, ...) M((coef), (int)(sizeof((double[]){__VA_ARGS__}) / sizeof(double) / 2), \
    (double[]){__VA_ARGS__})

int main(void) {
    /* ---- constants ---- */
    const double c0 = 0.005, t0 = 0.015, bw0 = 0.010, hw0 = 0.025;
    const double S = 1200.0, Ep = 300.0, Es = 75.0, f = 100e3, pf = 0.80;
    const double rf = sqrt(1.0 - pf * pf);
    const double VRm = 0.03, k_phi = 0.02, dT = 65.0;
    const double Ip = S / Ep;
    const double m_layers = 2, d_foil = 0.238e-3, kCu = 0.6;
    const double Kc = 1.9e-3, alpha = 1.24, beta = 2.0, Bsat = 0.4, rho_c = 4800.0;
    const double rho_w = 2.26e-8;
    const double mu0 = 4e-7 * M_PI, mu = mu0 * 2000.0;
    const double kf = 0.95, ka = 8.0 * pow(100.0, 1.75), Fr = 1.32, Fl = 0.9;
    const double PI = M_PI;

    /* derived coefficients */
    double c33 = pow(2.0, 1.5) * PI * f * kf / Ep;
    double c34 = Es / Ep;
    double c35 = 2.0 * Ip / kCu;
    double c36 = (1.0 / Fl) * (1.0 / d_foil) * Ip;
    double denomT = 1.42 * sqrt(2.0) * ka * pow(dT, 1.25);
    double Ccore = kf * Kc * rho_c * pow(f, alpha);
    double Ccu = kCu * Fr * rho_w;
    double CIm = sqrt(2.0) / mu;
    double Cnl = (1.0 / (k_phi * k_phi)) * (1.0 / (Ip * Ip));
    double Cr = Fr * rho_w * pf;
    double inv_p2 = (m_layers * m_layers) / 4.0;
    double Cx = Ip * rf * f * mu0 * inv_p2;
    double coeff_vr = (1.0 / Ep) * (1.0 / VRm);

    /* ---- the 28 constraint slots (each a posynomial) ---- */
    Posy P[MCON]; for (int i = 0; i < MCON; i++) P[i].nm = 0;
    int k = 0;
    /* fix_var c,t,bw,hw : ONE monomial per slot (x/v <= 1 and v/x <= 1) */
    P[k].m[0] = MK(1.0 / c0, VC, 1); P[k++].nm = 1;
    P[k].m[0] = MK(c0, VC, -1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0 / t0, VT, 1); P[k++].nm = 1;
    P[k].m[0] = MK(t0, VT, -1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0 / bw0, VBW, 1); P[k++].nm = 1;
    P[k].m[0] = MK(bw0, VBW, -1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0 / hw0, VHW, 1); P[k++].nm = 1;
    P[k].m[0] = MK(hw0, VHW, -1); P[k++].nm = 1;
    /* (33)-(36) equality as two single-monomial slots each */
    P[k].m[0] = MK(c33, VNP,1, VB,1, VC,1, VT,1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0/c33, VNP,-1, VB,-1, VC,-1, VT,-1); P[k++].nm = 1;
    P[k].m[0] = MK(c34, VNP,1, VNS,-1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0/c34, VNP,-1, VNS,1); P[k++].nm = 1;
    P[k].m[0] = MK(c35, VNP,1, VJ,-1, VBW,-1, VHW,-1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0/c35, VNP,-1, VJ,1, VBW,1, VHW,1); P[k++].nm = 1;
    P[k].m[0] = MK(c36, VJ,-1, VNP,1, VNS,-1, VNLS,1, VHW,-1); P[k++].nm = 1;
    P[k].m[0] = MK(1.0/c36, VJ,1, VNP,-1, VNS,1, VNLS,-1, VHW,1); P[k++].nm = 1;
    /* (37) temperature rise */
    P[k].m[0] = MK(1.0/denomT, VPC,1, VHB,0.25, VC,-0.5, VT,-0.5, VBW,-0.5, VHW,-0.5);
    P[k].m[1] = MK(1.0/denomT, VPCU,1, VHB,0.25, VC,-0.5, VT,-0.5, VBW,-0.5, VHW,-0.5); P[k++].nm = 2;
    /* (38) hw + 2c <= hbar */
    P[k].m[0] = MK(1.0, VHW,1, VHB,-1); P[k].m[1] = MK(2.0, VC,1, VHB,-1); P[k++].nm = 2;
    /* (39) core loss */
    P[k].m[0] = MK(4.0*Ccore, VB,beta, VC,1, VT,1, VHW,1, VPC,-1);
    P[k].m[1] = MK(8.0*Ccore, VB,beta, VC,2, VT,1, VPC,-1);
    P[k].m[2] = MK(4.0*Ccore, VB,beta, VC,1, VT,1, VBW,1, VPC,-1); P[k++].nm = 3;
    /* (40) copper loss */
    P[k].m[0] = MK(4.0*Ccu, VC,1, VBW,1, VHW,1, VJ,2, VPCU,-1);
    P[k].m[1] = MK(2.0*Ccu, VT,1, VBW,1, VHW,1, VJ,2, VPCU,-1);
    P[k].m[2] = MK(PI*Ccu, VBW,2, VHW,1, VJ,2, VPCU,-1); P[k++].nm = 3;
    /* (42) magnetizing current */
    P[k].m[0] = MK(CIm, VB,1, VNP,-1, VHW,1, VIM,-1);
    P[k].m[1] = MK(CIm, VB,1, VNP,-1, VBW,1, VIM,-1);
    P[k].m[2] = MK(2.0*CIm, VB,1, VNP,-1, VC,1, VIM,-1); P[k++].nm = 3;
    /* (43) saturation : single */
    P[k].m[0] = MK(1.0/Bsat, VB,1); P[k++].nm = 1;
    /* (44) no-load current ratio */
    P[k].m[0] = MK(Cnl, VIM,2); P[k].m[1] = MK(Cnl/(Ep*Ep), VPC,2); P[k++].nm = 2;
    /* (45) resistive regulation */
    P[k].m[0] = MK(8.0*Cr, VNP,1, VC,1, VJ,1, VVRR,-1);
    P[k].m[1] = MK(4.0*Cr, VNP,1, VT,1, VJ,1, VVRR,-1);
    P[k].m[2] = MK(2.0*PI*Cr, VNP,1, VBW,1, VJ,1, VVRR,-1); P[k++].nm = 3;
    /* (46) reactive regulation */
    P[k].m[0] = MK((8.0*PI/3.0)*Cx, VNP,2, VC,1, VBW,1, VHW,-1, VVRX,-1, VNLS,2, VNS,-2);
    P[k].m[1] = MK((4.0*PI/3.0)*Cx, VNP,2, VT,1, VBW,1, VHW,-1, VVRX,-1, VNLS,2, VNS,-2);
    P[k].m[2] = MK((2.0*PI*PI/3.0)*Cx, VNP,2, VBW,2, VHW,-1, VVRX,-1, VNLS,2, VNS,-2); P[k++].nm = 3;
    /* (47) Nls >= 1 and Ns >= m*Nls : single each */
    P[k].m[0] = MK(1.0, VNLS,-1); P[k++].nm = 1;
    P[k].m[0] = MK(m_layers, VNLS,1, VNS,-1); P[k++].nm = 1;
    /* (48) total regulation */
    P[k].m[0] = MK(coeff_vr, VVRR,1); P[k].m[1] = MK(coeff_vr, VVRX,1); P[k++].nm = 2;
    if (k != MCON) { printf("transformer_design  slot count %d != %d FAIL\n", k, MCON); return 1; }

    /* ---- objective posynomial ---- */
    Mono OBJ[2] = { MK(1.0, VPC,1), MK(1.0, VPCU,1) };

    /* ---- build the conic model ---- */
    /* vars: y[NGP] | t | e_i (per multi-mono monomial) | p_i (per mono) | ONE */
    int nmulti = 0;
    for (int i = 0; i < MCON; i++) if (P[i].nm > 1) nmulti += P[i].nm;
    nmulti += 2;                                   /* objective */
    int YBASE = 0, TV = NGP, EBASE = NGP + 1, PBASE = EBASE + nmulti, ONEV = PBASE + nmulti;
    int NVAR = ONEV + 1;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, NVAR);
    for (int i = 0; i < NGP; i++) PRIMAL_putvarbound(t, YBASE + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, TV, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nmulti; i++) PRIMAL_putvarbound(t, EBASE + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    for (int i = 0; i < nmulti; i++) PRIMAL_putvarbound(t, PBASE + i, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, ONEV, PRIMAL_BK_FX, 1.0, 1.0);

    int ecur = 0, pcur = 0, nrow = 0;
    PRIMAL_appendcons(t, MCON + 1 + 2 * nmulti);
    for (int i = 0; i < MCON; i++) {
        if (P[i].nm == 1) {                       /* linear: a.y + ln c <= 0 */
            int sub[8]; double val[8]; int nn = 0;
            for (int q = 0; q < P[i].m[0].n; q++) { sub[nn] = YBASE + P[i].m[0].var[q]; val[nn] = P[i].m[0].alpha[q]; nn++; }
            PRIMAL_putarow(t, nrow, nn, sub, val);
            PRIMAL_putconbound(t, nrow, PRIMAL_BK_UP, -INFINITY, -log(P[i].m[0].coef));
            nrow++;
        } else {
            int e0 = ecur, p0 = pcur;
            for (int q = 0; q < P[i].nm; q++) {
                /* e - a.y = ln c ; p = p_i ; PEXP(p, ONE, e) */
                int sub[8]; double val[8]; int nn = 0;
                sub[nn] = EBASE + ecur; val[nn] = 1.0; nn++;
                for (int z = 0; z < P[i].m[q].n; z++) { sub[nn] = YBASE + P[i].m[q].var[z]; val[nn] = -P[i].m[q].alpha[z]; nn++; }
                PRIMAL_putarow(t, nrow, nn, sub, val);
                PRIMAL_putconbound(t, nrow, PRIMAL_BK_FX, log(P[i].m[q].coef), log(P[i].m[q].coef));
                nrow++;
                PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){PBASE + pcur, ONEV, EBASE + ecur});
                ecur++; pcur++;
            }
            int sub[6]; double val[6];
            for (int q = 0; q < P[i].nm; q++) { sub[q] = PBASE + p0 + q; val[q] = 1.0; }
            PRIMAL_putarow(t, nrow, P[i].nm, sub, val);
            PRIMAL_putconbound(t, nrow, PRIMAL_BK_UP, -INFINITY, 1.0); nrow++;
            (void)e0;
        }
    }
    /* objective: min t, p_k >= exp(a.y + ln c - t), sum p <= 1 */
    {
        int p0 = pcur;
        for (int q = 0; q < 2; q++) {
            int sub[9]; double val[9]; int nn = 0;
            sub[nn] = EBASE + ecur; val[nn] = 1.0; nn++;
            for (int z = 0; z < OBJ[q].n; z++) { sub[nn] = YBASE + OBJ[q].var[z]; val[nn] = -OBJ[q].alpha[z]; nn++; }
            sub[nn] = TV; val[nn] = 1.0; nn++;
            PRIMAL_putarow(t, nrow, nn, sub, val);
            PRIMAL_putconbound(t, nrow, PRIMAL_BK_FX, log(OBJ[q].coef), log(OBJ[q].coef)); nrow++;
            PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){PBASE + pcur, ONEV, EBASE + ecur});
            ecur++; pcur++;
        }
        PRIMAL_putarow(t, nrow, 2, (int[]){PBASE + p0, PBASE + p0 + 1}, (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, nrow, PRIMAL_BK_UP, -INFINITY, 1.0); nrow++;
    }
    PRIMAL_putcj(t, TV, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    double y[NGP] = {0}, x[NGP] = {0}, topt = 0.0;
    if (ok) {
        double v[512] = {0};
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, v);
        for (int i = 0; i < NGP; i++) { y[i] = v[YBASE + i]; x[i] = exp(y[i]); }
        topt = v[TV];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);

    /* independent check: every posynomial is <= 1 at x = exp(y) */
    int worst_i = -1; double worst = -1e300;
    for (int i = 0; i < MCON; i++) {
        double s = 0.0;
        for (int q = 0; q < P[i].nm; q++) {
            double v = P[i].m[q].coef;
            for (int z = 0; z < P[i].m[q].n; z++) v *= pow(x[P[i].m[q].var[z]], P[i].m[q].alpha[z]);
            s += v;
        }
        if (s - 1.0 > worst) { worst = s - 1.0; worst_i = i; }
    }
    double objf = 0.0;
    for (int q = 0; q < 2; q++) { double v = OBJ[q].coef; for (int z = 0; z < OBJ[q].n; z++) v *= pow(x[OBJ[q].var[z]], OBJ[q].alpha[z]); objf += v; }
    int okall = ok && worst < 1e-6 && fabs(objf - exp(topt)) < 1e-6 * (1.0 + objf);
    printf("transformer_design  n=%d m=%d  loss = %.6f\n", NGP, MCON, objf);
    printf("  Np=%.4f Ns=%.4f Nls=%.4f B=%.6f J=%.4e hbar=%.4f\n",
           x[VNP], x[VNS], x[VNLS], x[VB], x[VJ], x[VHB]);
    printf("  worst posynomial <= 1: %.3e (slot %d)   exp(t)=%.6f\n", worst, worst_i, exp(topt));
    printf("%s\n", okall ? "OK" : "FAIL");
    return okall ? 0 : 1;
}
