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

/* secure_ee_sdma.c - secure energy efficiency of a downlink SDMA beamformer.
 *
 * Source: rezarhp/Secure-EE-SDMA (Hashempour et al., "Secure Rate-Splitting
 * and RIS Beamforming with Untrusted Energy Harvesting Receivers", ICC
 * Workshops 2026, arXiv:2603.21889).  The repository solves, with CVXPY+MOSEK,
 * the transmit-precoder subproblem (its "Part B") by
 *
 *   - an OUTER Dinkelbach loop on the secure energy efficiency
 *         SEE = zeta / (P0 + ||p||^2),
 *     where zeta = min_k [ log2(1+gamma_user_k) - log2(1+gamma_eve_k) ] is the
 *     minimum secrecy rate, and
 *   - an INNER successive-convex-approximation (SCA) loop for the two
 *     non-convex ratios (the user's |h^H p|^2 in the numerator and the
 *     eavesdropper's in the denominator).
 *
 * This sample is a FAITHFUL, deterministic instance of that subproblem:
 * ONE legitimate user (K=1) and ONE untrusted energy-harvesting receiver /
 * eavesdropper (J=1) on a two-antenna BS, real channels.  The RIS is held at
 * the zero-phase reflection s=1 (Theta=I), so the effective channels follow
 * the repository's Part-C-free case: h_u = gk and h_e = gbj + gj.
 *
 * The SCA surrogates are exactly the ones of sca_funcs.py:
 *   psi_cvx   -> affine in (p, rho_u):  2 Re(hp_old^H h^H p)/rho_old
 *                                        - (|hp_old|^2/rho_old^2) rho_u
 *   phi_cvx   -> affine in p:  2 Re(ph_old^H h^H p) - |ph_old|^2
 *   gamma_cvx -> affine in f_e:  2^f_e_old (1 + ln2 (f_e - f_e_old))
 * and the two exact cones:
 *   |h_e^H p|^2 <= sigma^2 rho_e            (rotated SOC, RQUAD)
 *   1 + rho_u >= 2^f_p                       (exponential cone, PEXP)
 * plus the power SOC ||p||^2 <= Pmax and the harvesting constraint
 * |h_e^H p|^2 >= E_req.  (E_req is the repository's sai_inverse(Eh), the
 * RF-power inverse of the non-linear harvesting function; here it is computed
 * in closed form from b0,b1,phi as in sai_inverse().)
 *
 * Reference: the precoder is a real 2-vector, so the problem lives in the
 * plane.  The sample validates the cone solution against a dense brute-force
 * search over (p0,p1) (power disk, harvesting feasible) of the TRUE objective
 * -- independent of the surrogates.  At the optimum the power bound is NOT
 * active: the Dinkelbach ratio trades rate against the circuit power P0, so
 * ||p||^2 < Pmax.
 *
 * Usage: secure_ee_sdma   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* ---- deterministic instance --------------------------------------------- */
#define NT 2
static const double HU[NT] = {1.0, 0.5};   /* legitimate-user channel      */
static const double HE[NT] = {0.3, 1.0};   /* eavesdropper/harvester channel */
static const double SIGMA2 = 1.0;
static const double PMAX   = 2.0;
static const double PCIRC  = 1.0;          /* circuit power P0             */
static const double EH     = 0.01;         /* harvested DC energy [W]      */

/* sai_inverse(): the repository's inverse of the non-linear harvesting map. */
static double sai_inverse(double x) {
    const double b0 = 150.0, b1 = 0.014, phi = 0.024, k1 = 1.0;
    double k2 = phi / (k1 * (1.0 + exp(b0 * b1)));
    return b1 - log(phi / (k1 * (x + k2)) - 1.0) / b0;
}

static double dot(const double *a, const double *b) { return a[0]*b[0] + a[1]*b[1]; }

/* true SINRs / secrecy rate / EE at a precoder p */
static void evaluate(const double *p, double *rate, double *ee) {
    double hu = dot(HU, p), he = dot(HE, p);
    double gu = hu * hu / SIGMA2;
    double ge = he * he / SIGMA2;
    double r = log(1.0 + gu) / log(2.0) - log(1.0 + ge) / log(2.0);
    *rate = r;
    *ee = r / (PCIRC + dot(p, p));
}

/* ---- one SCA/Dinkelbach inner subproblem ---------------------------------
 * Variables:  p0,p1, y, zeta, f_p, f_e, rho_u, rho_e, onepru, lfp, z_e,
 *             pw, half(=0.5), one(=1), tps(=sqrt(Pmax)), halfsig(=sigma^2/2)
 * Returns 1 on success and writes the solved p and zeta. */
enum { PV0=0, PV1, Y, Z, FP, FE, RU, RE, OPRU, LFP, ZE, PW, HALF, ONE, TPS, HALFSIG, NVAR };
enum { R_ZE, R_OBJ, R_SEC, R_EXP, R_GAM, R_USR, R_HARV, R_LFP, NR };

static int solve_sca(const double *p_old, double rho_u_old, double f_e_old,
                     double eta, double out_p[2], double *out_zeta) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, NR, NVAR, &t);
    for (int j = 0; j < NVAR; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, Z,  PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, FP, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, FE, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, RU, PRIMAL_BK_LO, 1e-9, INFINITY);
    PRIMAL_putvarbound(t, RE, PRIMAL_BK_LO, 1e-9, INFINITY);
    PRIMAL_putvarbound(t, PW, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, HALF, PRIMAL_BK_FX, 0.5, 0.5);
    PRIMAL_putvarbound(t, ONE,  PRIMAL_BK_FX, 1.0, 1.0);
    PRIMAL_putvarbound(t, TPS,  PRIMAL_BK_FX, sqrt(PMAX), sqrt(PMAX));
    PRIMAL_putvarbound(t, HALFSIG, PRIMAL_BK_FX, 0.5 * SIGMA2, 0.5 * SIGMA2);

    double hu_old = dot(HU, p_old), he_old = dot(HE, p_old);
    double A = pow(2.0, f_e_old), LN2 = log(2.0);

    /* z_e - (h_e^T p) = 0 */
    PRIMAL_putarow(t, R_ZE, 3, (int[]){ZE, PV0, PV1}, (double[]){1.0, -HE[0], -HE[1]});
    PRIMAL_putconbound(t, R_ZE, PRIMAL_BK_FX, 0.0, 0.0);
    /* y - zeta + eta*pw <= -eta*P0  (EE surrogate, pw >= ||p||^2) */
    PRIMAL_putarow(t, R_OBJ, 3, (int[]){Y, Z, PW}, (double[]){1.0, -1.0, eta});
    PRIMAL_putconbound(t, R_OBJ, PRIMAL_BK_UP, -INFINITY, -eta * PCIRC);
    /* zeta - f_p + f_e <= 0  (zeta is the minimum secrecy rate) */
    PRIMAL_putarow(t, R_SEC, 3, (int[]){Z, FP, FE}, (double[]){1.0, -1.0, 1.0});
    PRIMAL_putconbound(t, R_SEC, PRIMAL_BK_UP, -INFINITY, 0.0);
    /* onepru - rho_u = 1 */
    PRIMAL_putarow(t, R_EXP, 2, (int[]){OPRU, RU}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, R_EXP, PRIMAL_BK_FX, 1.0, 1.0);
    /* 1 + rho_e - A(1 + ln2 (f_e - f_e_old)) <= 0 :
     *   rho_e - A ln2 f_e <= A - 1 - A ln2 f_e_old */
    PRIMAL_putarow(t, R_GAM, 2, (int[]){RE, FE}, (double[]){1.0, -A * LN2});
    PRIMAL_putconbound(t, R_GAM, PRIMAL_BK_UP, -INFINITY, A - 1.0 - A * LN2 * f_e_old);
    /* user SCA: 2 hu_old (h_u^T p)/rho_u_old - (hu_old^2/rho_u_old^2) rho_u >= sigma^2 */
    PRIMAL_putarow(t, R_USR, 3, (int[]){PV0, PV1, RU},
                   (double[]){2.0 * hu_old * HU[0] / rho_u_old,
                              2.0 * hu_old * HU[1] / rho_u_old,
                              -hu_old * hu_old / (rho_u_old * rho_u_old)});
    PRIMAL_putconbound(t, R_USR, PRIMAL_BK_LO, SIGMA2, INFINITY);
    /* harvesting SCA: 2 he_old (h_e^T p) - he_old^2 >= E_req */
    PRIMAL_putarow(t, R_HARV, 2, (int[]){PV0, PV1},
                   (double[]){2.0 * he_old * HE[0], 2.0 * he_old * HE[1]});
    PRIMAL_putconbound(t, R_HARV, PRIMAL_BK_LO, sai_inverse(EH) + he_old * he_old, INFINITY);
    /* lfp - ln2*f_p = 0 (exponential-cone argument) */
    PRIMAL_putarow(t, R_LFP, 2, (int[]){LFP, FP}, (double[]){1.0, -LN2});
    PRIMAL_putconbound(t, R_LFP, PRIMAL_BK_FX, 0.0, 0.0);

    PRIMAL_appendcone(t, PRIMAL_CT_PEXP, 0.0, 3, (int[]){OPRU, ONE, LFP});
    PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 3, (int[]){RE, HALFSIG, ZE});
    PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 4, (int[]){PW, HALF, PV0, PV1});
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){TPS, PV0, PV1});

    PRIMAL_putcj(t, Y, 1.0);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MAXIMIZE);

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double x[NVAR];
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        out_p[0] = x[PV0]; out_p[1] = x[PV1];
        *out_zeta = x[Z];
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok;
}

int main(void) {
    double ereq = sai_inverse(EH);
    double p[2], zeta = 0.0;
    /* MRT initialization toward the legitimate user (repository's pp init). */
    double nrm = sqrt(dot(HU, HU));
    p[0] = sqrt(PMAX) * HU[0] / nrm;
    p[1] = sqrt(PMAX) * HU[1] / nrm;

    /* outer Dinkelbach on SEE, inner SCA on the surrogates */
    double eta = 0.0, ee_actual = 0.0, rate_actual = 0.0;
    int outer, inner;
    for (outer = 0; outer < 50; outer++) {
        double y_prev = -INFINITY;
        for (inner = 0; inner < 50; inner++) {
            double hu = dot(HU, p), he = dot(HE, p);
            double rho_u_old = hu * hu / SIGMA2;
            double f_e_old = log(1.0 + he * he / SIGMA2) / log(2.0);
            double pn[2]; double z;
            if (!solve_sca(p, rho_u_old, f_e_old, eta, pn, &z)) {
                printf("secure_ee_sdma  SCA subproblem rc!=OK FAIL\n");
                return 1;
            }
            p[0] = pn[0]; p[1] = pn[1]; zeta = z;
            double y = zeta - eta * (PCIRC + dot(p, p));
            if (fabs(y - y_prev) < 1e-9) break;
            y_prev = y;
        }
        evaluate(p, &rate_actual, &ee_actual);
        if (fabs(ee_actual - eta) < 1e-9) { eta = ee_actual; break; }
        eta = ee_actual;
    }
    double power = dot(p, p);
    double harvest = dot(HE, p) * dot(HE, p);

    /* reference: dense brute force over (p0,p1) of the TRUE objective */
    double best_ee = -1e300, best_p0 = 0, best_p1 = 0;
    double span = sqrt(PMAX);
    int NG = 1200;
    for (int a = 0; a <= NG; a++) {
        double p0 = -span + 2.0 * span * a / NG;
        for (int b = 0; b <= NG; b++) {
            double p1 = -span + 2.0 * span * b / NG;
            double q[2] = {p0, p1};
            if (dot(q, q) > PMAX) continue;
            double he = dot(HE, q);
            if (he * he < ereq) continue;
            double r, ee;
            evaluate(q, &r, &ee);
            if (ee > best_ee) { best_ee = ee; best_p0 = p0; best_p1 = p1; }
        }
    }

    int ok = (fabs(ee_actual - best_ee) < 1e-3 * (1.0 + fabs(best_ee))) &&
             (power <= PMAX + 1e-7) && (harvest >= ereq - 1e-7);
    printf("secure_ee_sdma  K=1 J=1 Nt=2\n");
    printf("  E_req = sai_inverse(%.3f) = %.6f\n", EH, ereq);
    printf("  p = (%.6f, %.6f)  ||p||^2 = %.6f <= %.3f\n", p[0], p[1], power, PMAX);
    printf("  secrecy rate = %.6f bit/s/Hz   harvested = %.6f >= %.6f\n",
           rate_actual, harvest, ereq);
    printf("  SEE = %.6f bit/J/Hz   [brute force %.6f at (%.4f,%.4f)]\n",
           ee_actual, best_ee, best_p0, best_p1);
    printf("  outer=%d inner=%d\n", outer + 1, inner + 1);
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
