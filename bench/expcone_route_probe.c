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

/* bench/expcone_route_probe.c — diagnostic for the native exp/power route
 * through the public API: solves each reference case twice, with the unified
 * conic interior point and with the tangent-cut outer approximation forced by
 * GMB_NO_EXP_IPM, and prints both answers side by side.
 *
 * build (from the repo root):
 *   make bench
 *   # or, against the objects make already produced:
 *   gcc -std=c99 -Wall -Wextra -pedantic -O2 -I. -o out/bench/expcone_route_probe \
 *       bench/expcone_route_probe.c <the Makefile's $(LIBSRCS)> -lm
 */
#define _POSIX_C_SOURCE 200809L  /* clock_gettime/setenv under -std=c99 on glibc */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "primal.h"
#include "cbf.h"

static const char *only = NULL;   /* argv[1]: substring of the case name to run */

static int saw_native;            /* 1 when the unified conic IPM answered */
/* Log callback recording whether the unified conic IPM answered. */
static void onlog(void *h, const char *msg) {
    (void)h;
    if (strstr(msg, "SDP IPM")) saw_native = 1;
}
/* label of the route that actually answered: 'ipm' asked for the native
 * blocks, 'NATV' is the native route, 'fall' the tangent cuts after a stall */
static const char *answered(int cuts) { return cuts ? "cuts" : (saw_native ? "NATV" : "fall"); }

/* Select the native route (cuts=0) or force the tangent-cut route. */
static void set_route(int cuts) {
    if (cuts) setenv("GMB_NO_EXP_IPM", "1", 1);
    else      unsetenv("GMB_NO_EXP_IPM");
}

/* Solve one exp/power reference case and print native vs cuts answers. */
static void run(const char *name, int cuts, PRIMALconetypee ct, double param,
                double bu, double vv, double want) {
    PRIMALenv_t env; PRIMALtask_t t;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) { printf("env fail\n"); return; }
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 3);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FR, 0, 0);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_FX, bu, bu);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_FX, vv, vv);
    PRIMAL_putcj(t, 0, 1.0);
    PRIMAL_appendcone(t, ct, param, 3, (int[]){0, 1, 2});
    set_route(cuts);
    saw_native = 0; PRIMAL_setlogcb(t, onlog, NULL);
    PRIMAL_putintparam(t, PRIMAL_IPAR_LOG, 1);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double x[3], po, dob;
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dob);
    printf("%-20s %-4s rc=%4d  t=%-15.12g |t-want|=%-9.3g pobj=%-15.12g dobj=%-15.12g u=%.12g v=%.12g\n",
           name, answered(cuts), (int)rc, x[0], fabs(x[0] - want), po, dob, x[1], x[2]);
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
}

/* Solve the T36 DEXP case and print native vs cuts answers. */
static void run_dexp(const char *name, int cuts) {
    PRIMALenv_t env; PRIMALtask_t t;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) { printf("env fail\n"); return; }
    PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, 3);
    PRIMAL_putvarbound(t, 0, PRIMAL_BK_FR, 0, 0);
    PRIMAL_putvarbound(t, 1, PRIMAL_BK_FX, -1.0, -1.0);
    PRIMAL_putvarbound(t, 2, PRIMAL_BK_FR, 0, 0);
    PRIMAL_putcj(t, 0, -1.0);
    PRIMAL_putcj(t, 2, 1.0);
    PRIMAL_appendcone(t, PRIMAL_CT_DEXP, 0.0, 3, (int[]){0, 1, 2});
    set_route(cuts);
    saw_native = 0; PRIMAL_setlogcb(t, onlog, NULL);
    PRIMAL_putintparam(t, PRIMAL_IPAR_LOG, 1);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double x[3], po, dob;
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dob);
    printf("%-20s %-4s rc=%4d  a=%-15.12g b=%.12g c=%-15.12g pobj=%-15.12g dobj=%-15.12g (want a=-1 c=0 obj=1)\n",
           name, answered(cuts), (int)rc, x[0], x[1], x[2], po, dob);
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
}

static const char *T47_CBF =
    "# variant example C.2 (format4)\n"
    "VER\n4\n"
    "OBJSENSE\nMAX\n"
    "VAR\n4 1\nF 4\n"
    "CON\n7 3\nL= 1\nQ 3\nEXP 3\n"
    "OBJACOORD\n2\n0 -1.0\n3 1.0\n"
    "ACOORD\n7\n0 0 1.0\n0 1 2.0\n0 2 -1.0\n"
    "2 0 1.0\n3 1 1.0\n4 2 1.0\n6 3 1.0\n"
    "BCOORD\n2\n1 5.0\n5 1.0\n";

/* Solve the T47 CBF case and print native vs cuts answers. */
static void run_t47(int cuts) {
    PRIMALenv_t env; PRIMALtask_t t;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) { printf("env fail\n"); return; }
    PRIMAL_maketask(env, 0, 0, &t);
    FILE *f = fopen("/tmp/probe_t47.cbf", "w");
    if (!f) { printf("cbf write fail\n"); return; }
    fputs(T47_CBF, f); fclose(f);
    FILE *fr = fopen("/tmp/probe_t47.cbf", "r");
    if (!fr) { printf("T47 cbf open fail\n"); return; }
    PRIMALrescodee rdc = cbf_read(t, fr);
    fclose(fr);
    if (rdc != PRIMAL_RES_OK) { printf("T47 cbf_read fail\n"); return; }
    /* Same iteration limit as test T47, so the cuts comparison is fair. */
    PRIMAL_putintparam(t, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 1000);
    set_route(cuts);
    saw_native = 0; PRIMAL_setlogcb(t, onlog, NULL);
    PRIMAL_putintparam(t, PRIMAL_IPAR_LOG, 1);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    double x[10], po, dob; int nv;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dob);
    double s = x[0] + 2.0 * x[1];
    printf("%-20s %-4s rc=%4d nv=%d  x0=%.12g x1=%.12g x2=%.12g x3=%.12g  s=%.12g ln s=%.12g\n"
           "%24s  radial KKT=%-9.3g (tol 1e-4)  pobj=%.12g dobj=%.12g\n",
           "T47 QUAD+PEXP", answered(cuts), (int)rc, nv, x[0], x[1], x[2], x[3], s,
           s > 0 ? log(s) : 0.0, "",
           s > 0.0 ? (1.0 / s - 1.0) * x[1] - (2.0 / s) * x[0] : 0.0, po, dob);
    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
}

/* True when the case name passes the argv[1] substring filter. */
static int want(const char *name) { return !only || strstr(name, only) != NULL; }

/* Run the selected reference cases on both routes and compare. */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) only = argv[1];
    if (want("PEXP")) {
        run("PEXP u=1 v=1", 0, PRIMAL_CT_PEXP, 0.0, 1.0, 1.0, exp(1.0));
        run("PEXP cuts u=1 v=1", 1, PRIMAL_CT_PEXP, 0.0, 1.0, 1.0, exp(1.0));
        run("PEXP u=2 v=1", 0, PRIMAL_CT_PEXP, 0.0, 2.0, 1.0, 2.0 * exp(0.5));
        run("PEXP cuts u=2 v=1", 1, PRIMAL_CT_PEXP, 0.0, 2.0, 1.0, 2.0 * exp(0.5));
    }
    if (want("T36")) {
        run_dexp("T36 DEXP min -a+c", 0);
        run_dexp("T36 cuts DEXP min -a+c", 1);
    }
    if (want("PPOW")) {
        run("PPOW(0.3) u=1 v=2", 0, PRIMAL_CT_PPOW, 0.3, 1.0, 2.0, pow(2.0, 1.0 / 0.3));
        run("PPOW cuts(0.3) u=1 v=2", 1, PRIMAL_CT_PPOW, 0.3, 1.0, 2.0, pow(2.0, 1.0 / 0.3));
    }
    if (want("RPOW")) {
        run("RPOW(0.4) u=1 v=1", 0, PRIMAL_CT_RPOW, 0.4, 1.0, 1.0, pow(2.0, -1.25));
        run("RPOW cuts(0.4) u=1 v=1", 1, PRIMAL_CT_RPOW, 0.4, 1.0, 1.0, pow(2.0, -1.25));
    }
    if (want("T47")) { run_t47(0); run_t47(1); }
    return 0;
}
