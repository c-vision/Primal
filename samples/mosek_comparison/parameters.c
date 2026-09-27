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
 *
 * Port of a MOSEK example (see the comment below), rewritten against
 * the PrimalSolver (PRIMAL_*) API.  The MOSEK examples are Copyright (c)
 * MOSEK ApS; this port re-implements the same optimization problem and is
 * distributed under the Apache License, Version 2.0.  PrimalSolver is not
 * affiliated with, or endorsed by, MOSEK.
 */

/* parameters.c — port of the MOSEK Julia API "parameters.jl" example:
 * reading/writing parameters (putintparam/putdouparam and the getters).
 *
 * Check: every parameter set is read back with the same value; an
 * invalid parameter is rejected (ERR_ARG); changing
 * INTPNT_MAX_ITERATIONS really limits the IPM iterations (TRM).
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* Round-trip parameters, reject invalid ones and check the iteration limit. */
int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);

    PRIMALtask_t task;
    PRIMAL_maketask(env, 0, 0, &task);

    /* int params */
    int rc = 0;
    PRIMAL_putintparam(task, PRIMAL_IPAR_LOG, 0);
    PRIMAL_putintparam(task, PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS, 5000);
    PRIMAL_putintparam(task, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 123);
    int vi = -1;
    PRIMAL_getintparam(task, PRIMAL_IPAR_LOG, &vi);
    rc += (vi != 0);
    PRIMAL_getintparam(task, PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS, &vi);
    rc += (vi != 5000);
    PRIMAL_getintparam(task, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, &vi);
    rc += (vi != 123);

    /* double params */
    PRIMAL_putdouparam(task, PRIMAL_DPAR_INTPNT_TOL_PFEAS, 1e-7);
    PRIMAL_putdouparam(task, PRIMAL_DPAR_INTPNT_TOL_DFEAS, 2e-7);
    PRIMAL_putdouparam(task, PRIMAL_DPAR_INTPNT_TOL_REL_GAP, 3e-7);
    double vd = -1;
    PRIMAL_getdouparam(task, PRIMAL_DPAR_INTPNT_TOL_PFEAS, &vd);
    rc += (fabs(vd - 1e-7) > 1e-15);
    PRIMAL_getdouparam(task, PRIMAL_DPAR_INTPNT_TOL_DFEAS, &vd);
    rc += (fabs(vd - 2e-7) > 1e-15);
    PRIMAL_getdouparam(task, PRIMAL_DPAR_INTPNT_TOL_REL_GAP, &vd);
    rc += (fabs(vd - 3e-7) > 1e-15);

    /* invalid parameters rejected: validation comes from the ranges of the
     * declarative table (PRIMAL_getparaminfo), so a boolean outside
     * 0/1 or a tolerance <= 0 are API errors; 9999 exists in
     * neither namespace */
    rc += (PRIMAL_putintparam(task, 9999, 1) == PRIMAL_RES_OK ? 1 : 0);
    rc += (PRIMAL_putdouparam(task, 9999, 1.0) == PRIMAL_RES_OK ? 1 : 0);
    /* the range of TOL_PFEAS is [0,1] (reconciled with the reference): a value
     * OUTSIDE the range is the error, not 0.0 which is now valid. */
    rc += (PRIMAL_putdouparam(task, PRIMAL_DPAR_INTPNT_TOL_PFEAS, -1.0) == PRIMAL_RES_OK ? 1 : 0);
    rc += (PRIMAL_getintparam(task, 9999, &vi) == PRIMAL_RES_OK ? 1 : 0);

    /* INTPNT_MAX_ITERATIONS as a real limit: QP with 1 iteration ->
     * PRIMAL_RES_TRM_MAX_ITER (1007) or anyway non-OK if the QP does not converge */
    PRIMALtask_t q;
    PRIMAL_maketask(env, 0, 0, &q);
    PRIMAL_appendvars(q, 2);
    PRIMAL_appendcons(q, 1);
    PRIMAL_putvarbound(q, 0, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(q, 1, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putqobj(q, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putarow(q, 0, 2, (int[]){0, 1}, (double[]){1.0, 1.0});
    PRIMAL_putconbound(q, 0, PRIMAL_BK_LO, 1.0, INFINITY);
    PRIMAL_putintparam(q, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 1);
    PRIMALrescodee rq = PRIMAL_optimize(q);
    printf("QP con max_iter=1: rc=%d (atteso non-OK: 1007 TRM o 1002)\n", rq);
    int rc_trm = (rq != PRIMAL_RES_OK);   /* the limit really interrupts */
    PRIMAL_deletetask(&q);

    printf("conteggio errori parametri: %d\n", rc);
    int ok = (rc == 0) && rc_trm;
    printf("%s (parametri round-trip + limite iterazioni reale)\n",
           ok ? "OK" : "FAIL");

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
