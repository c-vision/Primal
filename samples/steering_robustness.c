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

/* steering_robustness.c - steering robustness of the two-qubit Werner state
 * (slmaquedano/semidefinite_programming_workbench, steering_robustness/src).
 *
 * Quantum steering: given a state rho_AB and measurements {M_{a|x}} on A the
 * assemblage is sigma_{a|x} = tr_A[(M_{a|x} (x) I) rho_AB].  It is unsteerable
 * iff sigma_{a|x} = sum_lambda D[a,x,lambda] tau_lambda for some deterministic
 * strategies D and states tau_lambda >= 0.  The steering ROBUSTNESS is the
 * minimal t with
 *     tau~_{a|x} = sum_lambda D[a,x,lambda] tau~_lambda,   tau~_lambda >= 0,
 *     pi~_{a|x}  = tau~_{a|x} - sigma_{a|x} >= 0,          t = sum_lambda tr(tau~_lambda) - 1.
 *
 * The workbench declares every tau~/pi~ a 2x2 Hermitian PSD variable.  Here
 * each is written EXACTLY as a 4-dimensional rotated cone: for
 *     H = [[a, c+id],[c-id, b]]  <->  (a, b, sqrt2 c, sqrt2 d) in Qr^4
 * the cone 2 a b >= (sqrt2 c)^2 + (sqrt2 d)^2 is exactly H >= 0.  So the SDP is
 * an SOCP (a coefficient-free linear map preserves the cone).
 *
 * TWO DECLARED DEVIATIONS from the reference:
 *  - the no-signaling rows sum_a pi~_{a|x} = (x-independent) are OMITTED: they
 *    are automatically satisfied, because sum_a D[a,x,lambda] = 1 for every
 *    (x,lambda) and sum_a sigma_{a|x} = tr_A(rho) = I/2, so
 *    sum_a pi~_{a|x} = sum_lambda tau~_lambda - I/2 independent of x.  As an
 *    equality block they are redundant and make the Newton system rank-deficient.
 *  - the objective is the explicit expression sum(alpha_l+beta_l)-1 rather than
 *    an extra free variable: a declared-but-unused variable is a zero column in
 *    the KKT matrix and the interior-point diverges on it.
 * Both removals leave the model equivalent -- the optimum matches the reference
 * dat/SR_werner_state.dat (SR=0 below w~0.56, 0.0017/0.1412/0.2046/0.2679 at
 * w=0.58/0.8/0.9/1.0).
 *
 * Usage: steering_robustness   (no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

typedef struct { double re, im; } C;
static C cmul(C a, C b) { C r; r.re = a.re*b.re - a.im*b.im; r.im = a.re*b.im + a.im*b.re; return r; }
static C cadd(C a, C b) { C r; r.re = a.re+b.re; r.im = a.im+b.im; return r; }
static C cconj(C a) { C r; r.re = a.re; r.im = -a.im; return r; }

/* measurement vector M(dim=2, x=1..3, a=1..2): the three mutually unbiased bases */
static void meas(int x, int a, C m[2]) {
    double s = 1.0 / sqrt(2.0);
    if (x == 1) { m[0].re = s; m[0].im = 0; m[1].re = a == 1 ? s : -s; m[1].im = 0; }
    else if (x == 2) { m[0].re = s; m[0].im = 0; m[1].re = 0; m[1].im = a == 1 ? s : -s; }
    else { m[0].re = a == 1 ? 1 : 0; m[0].im = 0; m[1].re = a == 2 ? 1 : 0; m[1].im = 0; }
}
/* rho_2qbits(0,0,0,0,0,0,-w,-w,-w) : Werner state p|Phi+><Phi+|+(1-p)I/4, p=w */
static void werner(double w, C r[4][4]) {
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { r[i][j].re = 0; r[i][j].im = 0; }
    r[0][0].re = (1.0-w)/4; r[3][3].re = (1.0-w)/4;
    r[1][1].re = (1.0+w)/4; r[2][2].re = (1.0+w)/4;
    r[1][2].re = -w/2; r[2][1].re = -w/2;
}
/* sigma_{a|x} = tr_A[(M M^dag (x) I) rho] = sum_{iA,mA} M[iA][mA] rho[(mA,iB)][(iA,jB)] */
static void assemblage(double w, int x, int a, C sig[2][2]) {
    C rho[4][4], m[2], P[2][2];
    werner(w, rho); meas(x + 1, a + 1, m);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) P[i][j] = cmul(m[i], cconj(m[j]));
    for (int iB = 0; iB < 2; iB++) for (int jB = 0; jB < 2; jB++) {
        C s; s.re = 0; s.im = 0;
        for (int iA = 0; iA < 2; iA++) for (int mA = 0; mA < 2; mA++)
            s = cadd(s, cmul(P[iA][mA], rho[2*mA + iB][2*iA + jB]));
        sig[iB][jB] = s;
    }
}
static int Dstrategy(int a, int x, int l) { return (((l >> x) & 1) == a); }

#define OA 2
#define MA 3
#define L  8

/* steering robustness at Werner parameter w, as an SOCP over Qr^4 */
static double sr(double w, int *ok_out) {
    /* constants sigma_{a,x} as 4-vectors (a, b, sqrt2 Re c, sqrt2 Im c) */
    double sig[OA][MA][4];
    for (int x = 0; x < MA; x++) for (int a = 0; a < OA; a++) {
        C s[2][2]; assemblage(w, x, a, s);
        sig[a][x][0] = s[0][0].re; sig[a][x][1] = s[1][1].re;
        sig[a][x][2] = sqrt(2.0) * s[0][1].re; sig[a][x][3] = sqrt(2.0) * s[0][1].im;
    }
    int TAU = 0, PI = 4 * L, nv = PI + 4 * OA * MA;
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, nv);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    PRIMAL_appendcons(t, 4 * OA * MA);          /* pi link rows only */
    for (int l = 0; l < L; l++) PRIMAL_putvarbound(t, TAU + 4*l, PRIMAL_BK_LO, 0.0, INFINITY);
    for (int i = 0; i < 4 * OA * MA; i++) PRIMAL_putvarbound(t, PI + i, PRIMAL_BK_FR, -INFINITY, INFINITY);
    int r = 0;
    for (int x = 0; x < MA; x++) for (int a = 0; a < OA; a++) {
        int piv = 4 * (a * MA + x);
        for (int cp = 0; cp < 4; cp++) {
            int sub[1 + L]; double val[1 + L]; int nn = 0;
            sub[nn] = PI + piv + cp; val[nn] = 1.0; nn++;
            for (int l = 0; l < L; l++) if (Dstrategy(a, x, l)) { sub[nn] = TAU + 4*l + cp; val[nn] = -1.0; nn++; }
            PRIMAL_putarow(t, r, nn, sub, val);
            PRIMAL_putconbound(t, r, PRIMAL_BK_FX, -sig[a][x][cp], -sig[a][x][cp]); r++;
        }
    }
    for (int x = 0; x < MA; x++) for (int a = 0; a < OA; a++)
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 4,
            (int[]){PI + 4*(a*MA+x), PI + 4*(a*MA+x) + 1, PI + 4*(a*MA+x) + 2, PI + 4*(a*MA+x) + 3});
    for (int l = 0; l < L; l++)
        PRIMAL_appendcone(t, PRIMAL_CT_RQUAD, 0.0, 4, (int[]){TAU + 4*l, TAU + 4*l + 1, TAU + 4*l + 2, TAU + 4*l + 3});
    for (int l = 0; l < L; l++) { PRIMAL_putcj(t, TAU + 4*l, 1.0); PRIMAL_putcj(t, TAU + 4*l + 1, 1.0); }
    PRIMAL_putcfix(t, -1.0);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double obj = -1.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    *ok_out = (rc == PRIMAL_RES_OK);
    return obj;
}

int main(void) {
    printf("steering_robustness (Werner state)\n");
    const double ws[7]   = {0.0, 0.4, 0.5, 0.58, 0.8, 0.9, 1.0};
    const double want[7] = {0.0000, 0.0000, 0.0000, 0.0017, 0.1412, 0.2046, 0.2679};
    int bad = 0;
    for (int i = 0; i < 7; i++) {
        int ok = 0; double v = sr(ws[i], &ok);
        int good = ok && v > -1e-6 && fabs(v - want[i]) < 1e-3;
        if (!good) bad++;
        printf("  w=%.2f  SR=%.4f  (rif %.4f)  %s\n", ws[i], v, want[i], good ? "OK" : "FAIL");
    }
    printf("%s\n", bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 1;
}
