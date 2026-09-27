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
 * License excerpt (Apache License 2.0, §2 "Grant of Copyright License"):
 *   "Subject to the terms and conditions of this License, each Contributor
 *    hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
 *    royalty-free, irrevocable copyright license to reproduce, prepare
 *    Derivative Works of, publicly display, publicly perform, sublicense, and
 *    distribute the Work and such Derivative Works in Source or Object form."
 * 
 * Disclaimer of liability and absence of warranty (Apache License 2.0, §7-§8):
 *   [§7] Unless required by applicable law or agreed to in writing, Licensor
 *   provides the Work (and each Contributor provides its Contributions) on an
 *   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 *   or implied, including, without limitation, any warranties or conditions of
 *   TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
 *   PURPOSE.  You are solely responsible for determining the appropriateness of
 *   using or redistributing the Work.
 *   [§8] In no event and under no legal theory, whether in tort (including
 *   negligence), contract, or otherwise, unless required by applicable law or
 *   agreed to in writing, shall any Contributor be liable to You for damages,
 *   including any direct, indirect, special, incidental, or consequential
 *   damages arising as a result of this License or out of the use or inability
 *   to use the Work.  This software is provided without any guarantee that it
 *   will operate correctly or be free of defects.
 */

/* expcone_ipm_probe.c - diagnostic for the NATIVE exp/power interior point.
 *
 * Formulates the reference problems of test T79 directly in the sdp_ipm()
 * standard form (no bars, no SOC: one exp/power block plus the two equality
 * rows that fix its last members) and reports whether the native path
 * converges.  Expected values are the T79 oracles:
 *
 *   PEXP  u=1, v=1 -> t = e            PEXP  u=2, v=1 -> t = 2 sqrt(e)
 *   PPOW(.3) u=1, v=2 -> t = 2^(1/.3)  RPOW(.4) u=1, v=1 -> t = 2^(-1.25)
 *
 * Build: gcc -std=c99 -Wall -Wextra -pedantic -O2 -I. \
 *          -o out/expcone_ipm_probe bench/expcone_ipm_probe.c the library objects from out/ -lm
 * Run:   GMB_DBG=1 ./out/expcone_ipm_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sdp.h"
#include "expcone.h"

/* Solve one T79 reference case through sdp_ipm and check its value. */
static int case_run(const char *name, int kind, double alpha,
                    double uu, double vv, double want) {
    /* min z0  s.t.  z in K,  z1 = uu,  z2 = vv */
    double b[2] = { uu, vv };
    double Aexpv[2][3] = { { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    double Cexpv[3] = { 1.0, 0.0, 0.0 };
    const double *Cexp[1] = { Cexpv };
    const double *Aexp[2] = { Aexpv[0], Aexpv[1] };
    int ekind[1] = { kind };
    double ealpha[1] = { alpha };
    double z[3] = { 0, 0, 0 }, s[3] = { 0, 0, 0 }, y[2] = { 0, 0 };
    double *Zexp[1] = { z }, *Sexp[1] = { s };
    int st = sdp_ipm(2, 0, NULL, b, NULL, 0, NULL,
                     NULL, NULL, 0, NULL, NULL, NULL,
                     1, ekind, ealpha,
                     (const double *const *)Cexp, (const double *const *)Aexp,
                     200, 1e-9, 1e-8, 1e-8, 1e-8,
                     1.0,      /* nep>0: the route is chosen on the declared tolerances,
                                * so the factor plays no part here -- 1 keeps the printed
                                * status exactly that verdict. */
                     NULL, NULL, y, NULL, NULL, NULL, Zexp, Sexp, NULL, NULL, NULL);
    double diff = fabs(z[0] - want);
    int din = expcone_dual_in(kind, alpha, s);
    double zs = z[0] * s[0] + z[1] * s[1] + z[2] * s[2];
    printf("%-24s status=%d t=%-15.10g want=%-15.10g |diff|=%.3g  "
           "s in K*=%d  <z,s>=%.3g  y=(%.5g,%.5g)\n",
           name, st, z[0], want, diff, din, zs, y[0], y[1]);
    return st == 0 && diff < 1e-3 * (1.0 + fabs(want)) && din;
}

/* Run the scaling unit check plus the four T79 reference problems. */
int main(void) {
    int ok = 1;
    /* ---- unit check: the initialization point of sdp_ipm ---- */
    {
        struct { const char *nm; int kind; double alpha; double z[3]; } u[3] = {
            { "PEXP", EXPCONE_PEXP, 0.0, { 2.0, 1.0, 0.0 } },
            { "PPOW", EXPCONE_PPOW, 0.3, { 1.0, 1.0, 0.0 } },
            { "RPOW", EXPCONE_RPOW, 0.4, { 1.0, 1.0, 0.0 } }
        };
        for (int i = 0; i < 3; i++) {
            double g[3], s[3], W[3], Th[9], Tin[9], rt[3], sc = 0.0;
            int rc_s, rc_p, din;
            expcone_grad(u[i].kind, u[i].alpha, u[i].z, g);
            for (int a = 0; a < 3; a++) s[a] = -g[a];
            din = expcone_dual_in(u[i].kind, u[i].alpha, s);
            rc_p = expcone_dual_point(u[i].kind, u[i].alpha, s, W);
            rc_s = expcone_scaling(u[i].kind, u[i].alpha, u[i].z, s, 0.0, Th, Tin, rt, &sc);
            printf("[unit] %-5s s=(%.6g,%.6g,%.6g) inK*=%d dual_point=%d W=(%.6g,%.6g,%.6g) row=%d scale=%.6g\n",
                   u[i].nm, s[0], s[1], s[2], din, rc_p, W[0], W[1], W[2], rc_s, sc);
        }
    }
    /* ---- the T79 reference problems through the native path ---- */
    ok &= case_run("PEXP u=1 v=1",      EXPCONE_PEXP, 0.0, 1.0, 1.0, exp(1.0));
    ok &= case_run("PEXP u=2 v=1",      EXPCONE_PEXP, 0.0, 2.0, 1.0, 2.0 * exp(0.5));
    ok &= case_run("PPOW(0.3) u=1 v=2", EXPCONE_PPOW, 0.3, 1.0, 2.0, pow(2.0, 1.0 / 0.3));
    ok &= case_run("RPOW(0.4) u=1 v=1", EXPCONE_RPOW, 0.4, 1.0, 1.0, pow(2.0, -1.25));
    printf("%s\n", ok ? "ALL CONVERGED" : "NOT CONVERGED");
    return ok ? 0 : 1;
}
