/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 ... (see LICENSE).
 */

/* qcqp_sdo_relaxation.c - the SDO relaxation of a mixed-integer QCQP
 * (MOSEK Cookbook 11.10, qcqp_sdo_relaxation.cc).
 *
 * The relaxation drops x in {integer} and relaxes X = x x' to the PSD block
 *     Z = [[X, x],[x', 1]] >= 0        with   diag(X) >= x,
 * minimizing  <P, X> + 2 q'x.  The Cookbook compares its rounded x with the
 * integer least-squares  min |A x - b|  over integer x (a QUAD cone).
 *
 * The Cookbook draws A, c and the seed from std::default_random_engine seeded
 * with time(0) -- not reproducible at all -- so here the same two models are
 * solved on a hand instance: A = I (so P = A'A = I, b = A c = c, q = -P c = -c)
 * and c = (2/5, 3/5).  The constraint diag(X) >= x is a VALID inequality of the
 * integer problem (at an integer x, x_i^2 >= x_i), and it is what makes the
 * relaxation useful: it cuts the continuous optimum.  At x = c the rank-one
 * block has X00 = x0^2 = 4/25 < x0 = 2/5, so x = c is infeasible; the optimum
 * is the integral point x = (0,1) (X = x x', Z >= 0), where
 *     <I,X> - 2 c'x = 1 - 2*(3/5) = -1/5 = -0.2.
 * The integer least-squares (the model the relaxation relaxes) is
 *     min_{x in Z^2} |x - c| = |(0,1) - c| = sqrt(4/25 + 4/25) = sqrt8/5
 *     = 0.565685..., i.e. the relaxation's already-integral x is optimal.
 *
 * Usage: qcqp_sdo_relaxation   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

static int unit_sym(PRIMALtask_t t, int n, int i, int j) {
    int id;
    PRIMAL_appendsparsesymmat(t, n, 1, (int[]){i}, (int[]){j}, (double[]){1.0}, &id);
    return id;
}

/* the SDO relaxation; returns its value in *val (rc via the return). */
static PRIMALrescodee sdo_relax(const double c[2], double *val) {
    enum { X00 = 0, X01, X11, X0, X1, NV };
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 8, NV, &t);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    int d3 = 3;
    PRIMAL_appendbarvars(t, 1, &d3);           /* Z : 3x3 PSD */
    for (int v = 0; v < NV; v++) PRIMAL_putvarbound(t, v, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putcj(t, X00, 1.0); PRIMAL_putcj(t, X11, 1.0);          /* <I,X> */
    PRIMAL_putcj(t, X0, -2.0 * c[0]); PRIMAL_putcj(t, X1, -2.0 * c[1]);  /* 2 q'x, q=-c */
    int E00 = unit_sym(t, 3, 0, 0), E01 = unit_sym(t, 3, 0, 1), E11 = unit_sym(t, 3, 1, 1),
        E02 = unit_sym(t, 3, 0, 2), E12 = unit_sym(t, 3, 1, 2), E22 = unit_sym(t, 3, 2, 2);
    int r = 0;   /* Z entries <- scalar variables (off-diagonal counted twice) */
    PRIMAL_putarow(t, r, 1, (int[]){X00}, (double[]){-1.0}); PRIMAL_putbaraij(t, r, 0, 1, &E00, (double[]){1.0});
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0, 0); r++;
    PRIMAL_putarow(t, r, 1, (int[]){X01}, (double[]){-1.0}); PRIMAL_putbaraij(t, r, 0, 1, &E01, (double[]){0.5});
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0, 0); r++;
    PRIMAL_putarow(t, r, 1, (int[]){X11}, (double[]){-1.0}); PRIMAL_putbaraij(t, r, 0, 1, &E11, (double[]){1.0});
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0, 0); r++;
    PRIMAL_putarow(t, r, 1, (int[]){X0}, (double[]){-1.0});  PRIMAL_putbaraij(t, r, 0, 1, &E02, (double[]){0.5});
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0, 0); r++;
    PRIMAL_putarow(t, r, 1, (int[]){X1}, (double[]){-1.0});  PRIMAL_putbaraij(t, r, 0, 1, &E12, (double[]){0.5});
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 0, 0); r++;
    PRIMAL_putbaraij(t, r, 0, 1, &E22, (double[]){1.0});     /* Z22 = 1 */
    PRIMAL_putconbound(t, r, PRIMAL_BK_FX, 1.0, 1.0); r++;
    PRIMAL_putarow(t, r, 2, (int[]){X00, X0}, (double[]){1.0, -1.0});   /* diag(X) >= x */
    PRIMAL_putconbound(t, r, PRIMAL_BK_LO, 0.0, INFINITY); r++;
    PRIMAL_putarow(t, r, 2, (int[]){X11, X1}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, r, PRIMAL_BK_LO, 0.0, INFINITY); r++;
    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, val);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc;
}

/* the integer least-squares min |x - c| over integer x, via (t, x-c) in QUAD. */
static PRIMALrescodee int_ls(const double c[2], double *val) {
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 1, 3, &t);
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    int X0 = 0, X1 = 1, TT = 2;
    for (int j = 0; j < 2; j++) {
        PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, -100.0, INFINITY);
        PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT);
    }
    PRIMAL_putvarbound(t, TT, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(t, TT, 1.0);
    /* (t, x0-c0, x1-c1) in QUAD: t >= ||x-c|| ; make the cone members auxiliary */
    int E0 = 3, E1 = 4;
    PRIMAL_appendvars(t, 2);
    PRIMAL_putvarbound(t, E0, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putvarbound(t, E1, PRIMAL_BK_FR, -INFINITY, INFINITY);
    PRIMAL_putarow(t, 0, 2, (int[]){E0, X0}, (double[]){1.0, -1.0});
    PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, -c[0], -c[0]);
    { int sub[2] = {E1, X1}; double v[2] = {1.0, -1.0};
      PRIMAL_appendcons(t, 1); PRIMAL_putarow(t, 1, 2, sub, v);
      PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, -c[1], -c[1]); }
    PRIMAL_appendcone(t, PRIMAL_CT_QUAD, 0.0, 3, (int[]){TT, E0, E1});
    PRIMALrescodee rc = PRIMAL_optimize(t);
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, val);
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return rc;
}

int main(void) {
    const double c[2] = {2.0 / 5.0, 3.0 / 5.0};
    double relax = 0.0, ils = 0.0;
    PRIMALrescodee r1 = sdo_relax(c, &relax), r2 = int_ls(c, &ils);
    const double expect_relax = -1.0 / 5.0;        /* diag(X)>=x cuts x=c */
    const double expect_ils = sqrt(8.0) / 5.0;     /* sqrt(8/25) */
    int good = r1 == PRIMAL_RES_OK && r2 == PRIMAL_RES_OK &&
               fabs(relax - expect_relax) < 1e-6 && fabs(ils - expect_ils) < 1e-6;
    printf("qcqp_sdo_relaxation: relax=%.8f (atteso -1/5=%.8f)  int_ls=%.8f (atteso sqrt8/5=%.8f)  %s\n",
           relax, expect_relax, ils, expect_ils, good ? "OK" : "FAIL");
    return good ? 0 : 1;
}
