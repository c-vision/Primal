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

/* solve_mps.c - PrimalSolver benchmark driver.
 * Reads an MPS / CPLEX LP / CBF file, solves it, prints one CSV line:
 *   rc,nvar,ncon,obj,seconds
 * Timing covers PRIMAL_optimize only (model I/O excluded). */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../primal.h"

/* Current wall-clock time in seconds. */
static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/* Read the model file, solve it and print one CSV result line. */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 2; }
    PRIMALenv_t env = NULL;
    if (PRIMAL_makeenv(&env, NULL) != PRIMAL_RES_OK) return 2;
    PRIMALtask_t t = NULL;
    if (PRIMAL_maketask(env, 0, 0, &t) != PRIMAL_RES_OK) return 2;

    PRIMALrescodee rc = PRIMAL_readdata(t, argv[1]);
    if (rc != PRIMAL_RES_OK) {
        printf("READ_ERROR,%d,0,0,0\n", (int)rc);
        PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
        return 1;
    }
    int nv = 0, nc = 0;
    PRIMAL_getnumvar(t, &nv);
    PRIMAL_getnumcon(t, &nc);
    /* diagnostic hooks (bench only): toggle the LP pipeline from the env */
    if (getenv("GMB_BENCH_NOPRESOLVE")) PRIMAL_putintparam(t, PRIMAL_IPAR_PRESOLVE, 0);
    if (getenv("GMB_BENCH_NOSCALING"))  PRIMAL_putintparam(t, PRIMAL_IPAR_SCALING, 0);

    double a = now();
    rc = PRIMAL_optimize(t);
    double sec = now() - a;

    double obj = 0.0;
    if (rc == PRIMAL_RES_OK) PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &obj);
    printf("%d,%d,%d,%.10g,%.6f\n", (int)rc, nv, nc, obj, sec);
    if (getenv("GMB_BENCH_DIAG")) {
        double pinf = -1.0, dobj = 0.0;
        PRIMAL_getprimalinfeas(t, PRIMAL_SOL_ITR, &pinf);
        PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dobj);
        fprintf(stderr, "[diag] pobj=%.10g dobj=%.10g pinfeas=%.3e\n", obj, dobj, pinf);
    }

    PRIMAL_deletetask(&t);
    PRIMAL_deleteenv(&env);
    return rc == PRIMAL_RES_OK ? 0 : 1;
}
