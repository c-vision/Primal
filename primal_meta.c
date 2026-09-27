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
/* primal_meta.c - version/build, parameter getters, enum-name tables.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* ---- version/build/error utilities (reference checkversion/getbuildinfo/
 * getcodedesc/getlasterror/get,putatruncatetol) ---- */
#define PRIMAL_VER_MAJOR 0
#define PRIMAL_VER_MINOR 1
#define PRIMAL_VER_REVISION 0

/**
 * Checks if the given version matches this solver's version.
 *
 * @param env      [in] Environment handle (unused).
 * @param major    [in] Major version to check.
 * @param minor    [in] Minor version to check.
 * @param revision [in] Revision to check.
 *
 * @return PRIMAL_RES_OK if version matches (0.1.0),
 *         PRIMAL_RES_ERR_ARG if version doesn't match.
 *
 * @note Only accepts THIS solver's exact version (0.1.0).
 */
PRIMALrescodee PRIMAL_checkversion(PRIMALenv_t env, int major, int minor, int revision) {
    (void)env;
    /* Accepts a request for THIS solver's version; anything else is a mismatch. */
    if (major == PRIMAL_VER_MAJOR && minor == PRIMAL_VER_MINOR && revision == PRIMAL_VER_REVISION)
        return PRIMAL_RES_OK;
    return PRIMAL_RES_ERR_ARG;
}

/**
 * Retrieves build information strings.
 *
 * @param buildstate [out] Optional buffer receiving "PrimalSolver" (size PRIMAL_MAX_STR_LEN).
 * @param builddate  [out] Optional buffer receiving build date (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note The buffers must be at least PRIMAL_MAX_STR_LEN characters.
 *
 * @example
 * char state[PRIMAL_MAX_STR_LEN];
 * char date[PRIMAL_MAX_STR_LEN];
 * PRIMAL_getbuildinfo(state, date);
 * printf("Build: %s, Date: %s\n", state, date);
 */
PRIMALrescodee PRIMAL_getbuildinfo(char *buildstate, char *builddate) {
    if (buildstate) snprintf(buildstate, PRIMAL_MAX_STR_LEN, "%s", "PrimalSolver");
    if (builddate)  snprintf(builddate, PRIMAL_MAX_STR_LEN, "%s", __DATE__);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name and description of a result code.
 *
 * @param code   [in]  Result code.
 * @param symname [out] Optional buffer receiving symbolic name (size PRIMAL_MAX_STR_LEN).
 * @param str     [out] Optional buffer receiving description (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note The buffers must be at least PRIMAL_MAX_STR_LEN characters.
 *       Unknown codes return "PRIMAL_RES_UNKNOWN" / "Unknown error code".
 *
 * @example
 * char sym[PRIMAL_MAX_STR_LEN];
 * char desc[PRIMAL_MAX_STR_LEN];
 * PRIMAL_getcodedesc(rc, sym, desc);
 * printf("Code: %s - %s\n", sym, desc);
 */
PRIMALrescodee PRIMAL_getcodedesc(PRIMALrescodee code, char *symname, char *str) {
    const char *nm;
    switch (code) {
        case PRIMAL_RES_OK:             nm = "PRIMAL_RES_OK"; break;
        case PRIMAL_RES_ERR_ARG:        nm = "PRIMAL_RES_ERR_ARG"; break;
        case PRIMAL_RES_ERR_INFEASIBLE: nm = "PRIMAL_RES_ERR_INFEASIBLE"; break;
        case PRIMAL_RES_ERR_UNBOUNDED:  nm = "PRIMAL_RES_ERR_UNBOUNDED"; break;
        case PRIMAL_RES_ERR_ALLOC:      nm = "PRIMAL_RES_ERR_ALLOC"; break;
        case PRIMAL_RES_ERR_FILE:       nm = "PRIMAL_RES_ERR_FILE"; break;
        case PRIMAL_RES_ERR_NULL:       nm = "PRIMAL_RES_ERR_NULL"; break;
        case PRIMAL_RES_TRM_MAX_ITER:   nm = "PRIMAL_RES_TRM_MAX_ITER"; break;
        default:                        nm = "PRIMAL_RES_UNKNOWN"; break;
    }
    if (symname) snprintf(symname, PRIMAL_MAX_STR_LEN, "%s", nm);
    if (str) { char d[PRIMAL_MAX_STR_LEN]; PRIMAL_rescodetostr(code, d);
               snprintf(str, PRIMAL_MAX_STR_LEN, "%s", d); }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the last error code and message for a task.
 *
 * @param t            [in]  Task handle.
 * @param lastrescode  [out] Pointer to receive the last result code. Must not be NULL.
 * @param sizelastmsg  [in]  Size of lastmsg buffer (must include null terminator).
 * @param lastmsglen   [out] Pointer to int receiving message length.
 * @param lastmsg      [out] Optional buffer receiving error message (size sizelastmsg).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, lastrescode, or lastmsglen is NULL,
 *         PRIMAL_RES_ERR_ARG if lastmsg provided but buffer too small.
 *
 * @note The message is the string representation of the last result code.
 *
 * @example
 * PRIMALrescodee rc; int len; char msg[256];
 * PRIMAL_getlasterror(task, &rc, sizeof(msg), &len, msg);
 * printf("Last error: %s\n", msg);
 */
PRIMALrescodee PRIMAL_getlasterror(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    int sizelastmsg, int *lastmsglen, char *lastmsg) {
    if (!t || !lastrescode || !lastmsglen) return PRIMAL_RES_ERR_NULL;
    *lastrescode = t->last_rc;
    char d[PRIMAL_MAX_STR_LEN];
    PRIMAL_rescodetostr(t->last_rc, d);
    int len = (int)strlen(d);
    if (lastmsg) {
        if (sizelastmsg < len + 1) return PRIMAL_RES_ERR_ARG;
        memcpy(lastmsg, d, (size_t)len + 1);
    }
    *lastmsglen = len;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the last error code and message (64-bit size variant).
 *
 * @param t            [in]  Task handle.
 * @param lastrescode  [out] Pointer to receive the last result code. Must not be NULL.
 * @param sizelastmsg  [in]  Size of lastmsg buffer (64-bit).
 * @param lastmsglen   [out] Pointer to 64-bit int receiving message length.
 * @param lastmsg      [out] Optional buffer receiving error message.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, lastrescode, or lastmsglen is NULL,
 *         PRIMAL_RES_ERR_ARG if lastmsg provided but buffer too small.
 *
 * @note 64-bit variant of getlasterror. The sizes use PRIMALint64t.
 */
PRIMALrescodee PRIMAL_getlasterror64(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    PRIMALint64t sizelastmsg, PRIMALint64t *lastmsglen, char *lastmsg) {
    if (!t || !lastrescode || !lastmsglen) return PRIMAL_RES_ERR_NULL;
    int slen = 0;
    PRIMALrescodee rc = PRIMAL_getlasterror(t, lastrescode, 0, &slen, NULL);
    if (rc != PRIMAL_RES_OK) return rc;
    *lastmsglen = slen;
    if (lastmsg) {
        if (sizelastmsg < (PRIMALint64t)slen + 1) return PRIMAL_RES_ERR_ARG;
        char d[PRIMAL_MAX_STR_LEN];
        PRIMAL_rescodetostr(*lastrescode, d);
        memcpy(lastmsg, d, (size_t)slen + 1);
    }
    return PRIMAL_RES_OK;
}

/* Stream summaries (reference solutionsummary/optimizersummary/
 * onesolutionsummary): print to stdout. */
PRIMALrescodee PRIMAL_solutionsummary(PRIMALtask_t t, int whichstream) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream;
    if (!t->has_sol) { printf("No solution.\n"); return PRIMAL_RES_OK; }
    int sta = -1;
    PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, (PRIMALsolstae *)&sta);
    double po = 0.0, dob = 0.0;
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dob);
    printf("Solution status: %d\n", sta);
    printf("Primal objective: %.10g\n", po);
    printf("Dual objective: %.10g\n", dob);
    return PRIMAL_RES_OK;
}
/* Prints one solution summary (status and objectives) to stdout. */
PRIMALrescodee PRIMAL_onesolutionsummary(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream; (void)whichsol;
    return PRIMAL_solutionsummary(t, whichstream);
}
/* Prints the optimizer termination code and problem status to stdout. */
PRIMALrescodee PRIMAL_optimizersummary(PRIMALtask_t t, int whichstream) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream;
    int pro = -1;
    PRIMAL_getprosta(t, PRIMAL_SOL_ITR, (PRIMALprostae *)&pro);
    printf("Optimizer terminated with code %d, problem status %d\n", t->last_rc, pro);
    return PRIMAL_RES_OK;
}

/* Stream diagnostics (reference analyzeproblem/analyzesolution/
 * infeasibilityreport/sensitivityreport): print to stdout. */
PRIMALrescodee PRIMAL_analyzeproblem(PRIMALtask_t t, int whichstream) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream;
    printf("Problem: %d variables, %d constraints, %d cones, %d bar variables, "
           "%d domains, %d AFE, %d ACC, %d DJC\n",
           t->numvar, t->numcon, t->numcones, t->numbarvar, t->numdomain,
           t->numafe, t->numacc, t->numdjc);
    return PRIMAL_RES_OK;
}
/* Prints objectives and primal/dual infeasibilities to stdout. */
PRIMALrescodee PRIMAL_analyzesolution(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream; (void)whichsol;
    if (!t->has_sol) { printf("No solution.\n"); return PRIMAL_RES_OK; }
    double po = 0, dob = 0, pi = 0, di = 0;
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    PRIMAL_getdualobj(t, PRIMAL_SOL_ITR, &dob);
    PRIMAL_getprimalinfeas(t, PRIMAL_SOL_ITR, &pi);
    PRIMAL_getdualinfeas(t, PRIMAL_SOL_ITR, &di);
    printf("pobj=%.10g dobj=%.10g primal_infeas=%.3g dual_infeas=%.3g\n", po, dob, pi, di);
    return PRIMAL_RES_OK;
}
/* Prints the primal and dual infeasibility figures to stdout. */
PRIMALrescodee PRIMAL_infeasibilityreport(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream; (void)whichsol;
    if (!t->has_sol) { printf("No solution: no infeasibility to report.\n"); return PRIMAL_RES_OK; }
    double pi = 0, di = 0;
    PRIMAL_getprimalinfeas(t, PRIMAL_SOL_ITR, &pi);
    PRIMAL_getdualinfeas(t, PRIMAL_SOL_ITR, &di);
    printf("Primal infeasibility: %.10g\nDual infeasibility: %.10g\n", pi, di);
    return PRIMAL_RES_OK;
}
/* Prints a stub sensitivity report pointing at the range getters. */
PRIMALrescodee PRIMAL_sensitivityreport(PRIMALtask_t t, int whichstream) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)whichstream;
    printf("Sensitivity report: use PRIMAL_costsensitivity/PRIMAL_rhssensitivity "
           "for the ranges.\n");
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the A matrix truncation tolerance.
 *
 * @param t      [in]  Task handle.
 * @param tolzero [out] Pointer to double receiving the stored tolerance.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or tolzero is NULL.
 *
 * @note This value is STORED but NOT APPLIED. This solver does not truncate
 *       the A matrix; the parameter exists for API compatibility only.
 *
 * @example
 * double tol;
 * PRIMAL_getatruncatetol(task, &tol);
 * printf("A truncation tolerance: %g\n", tol);
 */
PRIMALrescodee PRIMAL_getatruncatetol(PRIMALtask_t t, PRIMALrealt *tolzero) {
    if (!t || !tolzero) return PRIMAL_RES_ERR_NULL;
    *tolzero = t->atruncatetol;
    return PRIMAL_RES_OK;
}

/**
 * Sets the A matrix truncation tolerance (stored only, not applied).
 *
 * @param t      [in] Task handle.
 * @param tolzero [in] Tolerance value. Must be >= 0 and not NaN.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if tolzero is NaN or negative.
 *
 * @note This value is STORED but NOT APPLIED. This solver does not truncate
 *       the A matrix; the parameter exists for API compatibility only.
 */
PRIMALrescodee PRIMAL_putatruncatetol(PRIMALtask_t t, PRIMALrealt tolzero) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (tolzero != tolzero || tolzero < 0.0) return PRIMAL_RES_ERR_ARG;
    t->atruncatetol = tolzero;   /* stored, not applied: this solver does not truncate A */
    return PRIMAL_RES_OK;
}

/**
 * Loads the entire linear problem data in one call.
 *
 * @param t         [in] Task handle (must be empty).
 * @param maxnumcon [in] Maximum constraints the task can hold.
 * @param maxnumvar [in] Maximum variables the task can hold.
 * @param numcon    [in] Number of constraints to load.
 * @param numvar    [in] Number of variables to load.
 * @param c         [in] Linear objective coefficients (array of size numvar, or NULL for zeros).
 * @param cfix      [in] Objective constant term.
 * @param aptrb     [in] Column start pointers (size numvar+1) for CSC format A.
 * @param aptre     [in] Column end pointers (size numvar+1) for CSC format A.
 * @param asub      [in] Row indices for non-zeros (size aptre[numvar-1]).
 * @param aval      [in] Coefficient values for non-zeros (size aptre[numvar-1]).
 * @param bkc       [in] Constraint bound keys (size numcon, or NULL for free).
 * @param blc       [in] Constraint lower bounds (size numcon, or NULL for -INF).
 * @param buc       [in] Constraint upper bounds (size numcon, or NULL for +INF).
 * @param bkx       [in] Variable bound keys (size numvar, or NULL for free).
 * @param blx       [in] Variable lower bounds (size numvar, or NULL for -INF).
 * @param bux       [in] Variable upper bounds (size numvar, or NULL for +INF).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if dimensions invalid or task not empty.
 *
 * @note The task MUST be empty (numvar=0, numcon=0). The data is loaded
 *       via appendvars/appendcons followed by individual put* calls.
 *       The matrix A is provided in CSC format (aptrb/aptre/asub/aval).
 *
 * @example
 * int aptrb[] = {0, 2, 4};
 * int aptre[] = {2, 4, 6};
 * int asub[] = {0, 1, 0, 1};
 * double aval[] = {1.0, 2.0, 3.0, 4.0};
 * PRIMAL_inputdata(task, 2, 2, 2, 2, c, 0.0, aptrb, aptre, asub, aval, ...);
 */
PRIMALrescodee PRIMAL_inputdata(PRIMALtask_t t, int maxnumcon, int maxnumvar,
    int numcon, int numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (numcon < 0 || numvar < 0 || numcon > maxnumcon || numvar > maxnumvar)
        return PRIMAL_RES_ERR_ARG;
    if (t->numvar > 0 || t->numcon > 0) return PRIMAL_RES_ERR_ARG;   /* must be empty */
    PRIMALrescodee rc;
    if ((rc = PRIMAL_appendvars(t, numvar)) != PRIMAL_RES_OK) return rc;
    if ((rc = PRIMAL_appendcons(t, numcon)) != PRIMAL_RES_OK) return rc;
    PRIMAL_putcfix(t, cfix);
    for (int j = 0; j < numvar; j++) {
        PRIMAL_putcj(t, j, c ? c[j] : 0.0);
        if (bkx) PRIMAL_putvarbound(t, j, bkx[j], blx ? blx[j] : 0.0, bux ? bux[j] : 0.0);
        if (aptrb && aptre && asub && aval && aptre[j] > aptrb[j])
            PRIMAL_putacol(t, j, aptre[j] - aptrb[j], asub + aptrb[j], aval + aptrb[j]);
    }
    for (int i = 0; i < numcon; i++)
        if (bkc) PRIMAL_putconbound(t, i, bkc[i], blc ? blc[i] : 0.0, buc ? buc[i] : 0.0);
    return PRIMAL_RES_OK;
}

/**
 * Loads the entire linear problem data in one call (64-bit dimension variant).
 *
 * @param t         [in] Task handle (must be empty).
 * @param maxnumcon [in] Maximum constraints (64-bit, must fit in int).
 * @param maxnumvar [in] Maximum variables (64-bit, must fit in int).
 * @param numcon    [in] Number of constraints to load (64-bit, must fit in int).
 * @param numvar    [in] Number of variables to load (64-bit, must fit in int).
 * @param c         [in] Linear objective coefficients (array of size numvar, or NULL).
 * @param cfix      [in] Objective constant term.
 * @param aptrb     [in] Column start pointers (size numvar+1) for CSC format A.
 * @param aptre     [in] Column end pointers (size numvar+1) for CSC format A.
 * @param asub      [in] Row indices for non-zeros (size aptre[numvar-1]).
 * @param aval      [in] Coefficient values for non-zeros.
 * @param bkc       [in] Constraint bound keys (size numcon, or NULL).
 * @param blc       [in] Constraint lower bounds (size numcon, or NULL).
 * @param buc       [in] Constraint upper bounds (size numcon, or NULL).
 * @param bkx       [in] Variable bound keys (size numvar, or NULL).
 * @param blx       [in] Variable lower bounds (size numvar, or NULL).
 * @param bux       [in] Variable upper bounds (size numvar, or NULL).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if dimensions exceed INT_MAX or task not empty.
 *
 * @note 64-bit variant of inputdata. All dimension parameters use PRIMALint64t
 *       but must fit in int. The matrix format and semantics are identical
 *       to inputdata.
 */
PRIMALrescodee PRIMAL_inputdata64(PRIMALtask_t t, PRIMALint64t maxnumcon, PRIMALint64t maxnumvar,
    PRIMALint64t numcon, PRIMALint64t numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux) {
    if (numcon > INT_MAX || numvar > INT_MAX || maxnumcon > INT_MAX || maxnumvar > INT_MAX)
        return PRIMAL_RES_ERR_ARG;
    return PRIMAL_inputdata(t, (int)maxnumcon, (int)maxnumvar, (int)numcon, (int)numvar,
        c, cfix, aptrb, aptre, asub, aval, bkc, blc, buc, bkx, blx, bux);
}

/**
 * Estimates the memory usage of the task.
 *
 * @param t        [in]  Task handle.
 * @param meminuse [out] Optional pointer to int64 receiving estimated memory in use (bytes).
 * @param maxmemuse [out] Optional pointer to int64 receiving estimated max memory (bytes).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This is THIS solver's own accounting (sum of allocated arrays),
 *       not the reference's memory tracking. Both pointers are optional.
 *
 * @example
 * PRIMALint64t inuse, maxuse;
 * PRIMAL_getmemusagetask(task, &inuse, &maxuse);
 * printf("Memory in use: %lld bytes\n", inuse);
 */
PRIMALrescodee PRIMAL_getmemusagetask(PRIMALtask_t t, PRIMALint64t *meminuse, PRIMALint64t *maxmemuse) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t b = (PRIMALint64t)sizeof(struct PRIMAL_task_s);
    if (t->cols) for (int j = 0; j < t->numvar; j++)
        b += (PRIMALint64t)t->cols[j].cap * (PRIMALint64t)(sizeof(int) + sizeof(double));
    b += (PRIMALint64t)t->numvar * (PRIMALint64t)(3 * sizeof(double) + sizeof(PRIMALboundkeye) +
         sizeof(PRIMALvariabletypee) + sizeof(char *));
    b += (PRIMALint64t)t->numcon * (PRIMALint64t)(3 * sizeof(double) + sizeof(PRIMALboundkeye) + sizeof(char *));
    b += (PRIMALint64t)t->qt_cap * (PRIMALint64t)(2 * sizeof(int) + sizeof(double));
    if (t->qcon) for (int i = 0; i < t->qcon_cap; i++)
        if (t->qcon[i]) b += (PRIMALint64t)t->numvar * t->numvar * (PRIMALint64t)sizeof(double);
    for (int k = 0; k < t->numcones; k++)
        b += (PRIMALint64t)t->cone_nmem[k] * (PRIMALint64t)sizeof(int);
    for (int j = 0; j < t->numbarvar; j++)
        b += 2 * (PRIMALint64t)t->barDim[j] * t->barDim[j] * (PRIMALint64t)sizeof(double);
    if (meminuse) *meminuse = b;
    if (maxmemuse) *maxmemuse = b;
    return PRIMAL_RES_OK;
}

/* Reference getprobtype: the problem class, with MOSEK's MSKproblemtypee values.
 * Rule (documented): quadratic constraints mixed with conic blocks -> MIXED
 * ("general nonlinear + conic", which the reference cannot solve either); then
 * quadratic constraints -> QCQO, quadratic objective -> QO, conic blocks ->
 * CONIC, otherwise LO. Integer variables do not enter the class (as in the
 * reference, where the type describes the conic/quadratic structure). */
PRIMALrescodee PRIMAL_getprobtype(PRIMALtask_t t, PRIMALproblemtypee *probtype) {
    if (!t || !probtype) return PRIMAL_RES_ERR_NULL;
    int conic = (t->numcones > 0 || t->numbarvar > 0);
    if (t->has_qcon > 0 && conic)       *probtype = PRIMAL_PROBTYPE_MIXED;
    else if (t->has_qcon > 0)           *probtype = PRIMAL_PROBTYPE_QCQO;
    else if (t->has_qobj)               *probtype = PRIMAL_PROBTYPE_QO;
    else if (conic)                     *probtype = PRIMAL_PROBTYPE_CONIC;
    else                                *probtype = PRIMAL_PROBTYPE_LO;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a problem status enum value.
 *
 * @param t      [in]  Task handle (unused).
 * @param prosta [in]  Problem status enum value.
 * @param str    [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note The strings are THIS solver's own -- not read from the reference.
 *       Unknown values return "UNKNOWN".
 *
 * @example
 * char buf[PRIMAL_MAX_STR_LEN];
 * PRIMAL_prostatostr(task, PRIMAL_PRO_STA_PRIM_AND_DUAL_FEAS, buf);
 * printf("Problem status: %s\n", buf);
 */
/* Writes the string s into the fixed-size output buffer str. */
static inline void set_str(char *str, const char *s) {
    if (str) snprintf(str, PRIMAL_MAX_STR_LEN, "%s", s);
}

/* Symbolic name of a problem status enum value (reference prostatostr). */
PRIMALrescodee PRIMAL_prostatostr(PRIMALtask_t t, PRIMALprostae prosta, char *str) {
    (void)t;
    switch (prosta) {
        case PRIMAL_PRO_STA_UNKNOWN:                  set_str(str, "UNKNOWN"); break;
        case PRIMAL_PRO_STA_PRIM_AND_DUAL_FEAS:       set_str(str, "PRIM_AND_DUAL_FEAS"); break;
        case PRIMAL_PRO_STA_PRIM_FEAS:                set_str(str, "PRIM_FEAS"); break;
        case PRIMAL_PRO_STA_DUAL_FEAS:                set_str(str, "DUAL_FEAS"); break;
        case PRIMAL_PRO_STA_PRIM_INFEAS:              set_str(str, "PRIM_INFEAS"); break;
        case PRIMAL_PRO_STA_DUAL_INFEAS:              set_str(str, "DUAL_INFEAS"); break;
        case PRIMAL_PRO_STA_PRIM_AND_DUAL_INFEAS:     set_str(str, "PRIM_AND_DUAL_INFEAS"); break;
        case PRIMAL_PRO_STA_ILL_POSED:                set_str(str, "ILL_POSED"); break;
        case PRIMAL_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED: set_str(str, "PRIM_INFEAS_OR_UNBOUNDED"); break;
        default:                                      set_str(str, "UNKNOWN"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a solution status enum value.
 *
 * @param t      [in]  Task handle (unused).
 * @param solsta [in]  Solution status enum value.
 * @param str    [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note The strings are THIS solver's own -- not read from the reference.
 *       The numeric values match MSKsolsta (0=UNKNOWN, 1=OPTIMAL, 5=PRIM_INFEAS_CER, 
 *       6=DUAL_INFEAS_CER, 9=INTEGER_OPTIMAL). Unknown values return "UNKNOWN".
 *
 * @example
 * char buf[PRIMAL_MAX_STR_LEN];
 * PRIMAL_solstatostr(task, PRIMAL_SOL_STA_OPTIMAL, buf);
 * printf("Solution status: %s\n", buf);
 */
PRIMALrescodee PRIMAL_solstatostr(PRIMALtask_t t, PRIMALsolstae solsta, char *str) {
    (void)t;
    switch ((int)solsta) {
        case 0: set_str(str, "UNKNOWN"); break;
        case 1: set_str(str, "OPTIMAL"); break;
        case 2: set_str(str, "PRIM_FEAS"); break;
        case 3: set_str(str, "DUAL_FEAS"); break;
        case 4: set_str(str, "PRIM_AND_DUAL_FEAS"); break;
        case 5: set_str(str, "PRIM_INFEAS_CER"); break;
        case 6: set_str(str, "DUAL_INFEAS_CER"); break;
        case 7: set_str(str, "PRIM_ILLPOSED_CER"); break;
        case 8: set_str(str, "DUAL_ILLPOSED_CER"); break;
        case 9: set_str(str, "INTEGER_OPTIMAL"); break;
        default: set_str(str, "UNKNOWN"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a bound key enum value.
 *
 * @param t  [in]  Task handle (unused).
 * @param bk [in]  Bound key enum value.
 * @param str [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note The strings are THIS solver's own: LO, UP, FX, FR, RA.
 *
 * @example
 * char buf[PRIMAL_MAX_STR_LEN];
 * PRIMAL_bktostr(task, PRIMAL_BK_FX, buf);
 * printf("Bound key: %s\n", buf);
 */
PRIMALrescodee PRIMAL_bktostr(PRIMALtask_t t, PRIMALboundkeye bk, char *str) {
    (void)t;
    switch (bk) {
        case PRIMAL_BK_LO: set_str(str, "LO"); break;
        case PRIMAL_BK_UP: set_str(str, "UP"); break;
        case PRIMAL_BK_FX: set_str(str, "FX"); break;
        case PRIMAL_BK_FR: set_str(str, "FR"); break;
        case PRIMAL_BK_RA: set_str(str, "RA"); break;
        default:           set_str(str, "?"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a cone type enum value.
 *
 * @param t  [in]  Task handle (unused).
 * @param ct [in]  Cone type enum value.
 * @param str [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note Strings: QUAD, RQUAD, PEXP, DEXP, PPOW, RPOW. Unknown: "?".
 *
 * @example
 * char buf[PRIMAL_MAX_STR_LEN];
 * PRIMAL_conetypetostr(task, PRIMAL_CT_QUAD, buf);
 * printf("Cone type: %s\n", buf);
 */
PRIMALrescodee PRIMAL_conetypetostr(PRIMALtask_t t, PRIMALconetypee ct, char *str) {
    (void)t;
    switch ((int)ct) {
        case PRIMAL_CT_QUAD:  set_str(str, "QUAD"); break;
        case PRIMAL_CT_RQUAD: set_str(str, "RQUAD"); break;
        case PRIMAL_CT_PEXP:  set_str(str, "PEXP"); break;
        case PRIMAL_CT_DEXP:  set_str(str, "DEXP"); break;
        case PRIMAL_CT_PPOW:  set_str(str, "PPOW"); break;
        case PRIMAL_CT_RPOW:  set_str(str, "RPOW"); break;
        default:              set_str(str, "?"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a problem type enum value.
 *
 * @param t  [in]  Task handle (unused).
 * @param pt [in]  Problem type enum value.
 * @param str [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note Strings: LO, QO, QCQO, CONIC, MIXED. Unknown: "?".
 */
PRIMALrescodee PRIMAL_probtypetostr(PRIMALtask_t t, PRIMALproblemtypee pt, char *str) {
    (void)t;
    switch (pt) {
        case PRIMAL_PROBTYPE_LO:    set_str(str, "LO"); break;
        case PRIMAL_PROBTYPE_QO:    set_str(str, "QO"); break;
        case PRIMAL_PROBTYPE_QCQO:  set_str(str, "QCQO"); break;
        case PRIMAL_PROBTYPE_CONIC: set_str(str, "CONIC"); break;
        case PRIMAL_PROBTYPE_MIXED: set_str(str, "MIXED"); break;
        default:                    set_str(str, "?"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a solution basis status key enum value.
 *
 * @param t  [in]  Task handle (unused).
 * @param sk [in]  Basis status key enum value.
 * @param str [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note Strings: BS, UP, LO, FX, SB, SN, UNDEF. Unknown: "?".
 */
PRIMALrescodee PRIMAL_sktostr(PRIMALtask_t t, PRIMALstakeye sk, char *str) {
    (void)t;
    switch (sk) {
        case PRIMAL_SK_UNDEF:  set_str(str, "UNK"); break;
        case PRIMAL_SK_BAS:    set_str(str, "BAS"); break;
        case PRIMAL_SK_SUPBAS: set_str(str, "SUPBAS"); break;
        case PRIMAL_SK_LOW:    set_str(str, "LOW"); break;
        case PRIMAL_SK_UPR:    set_str(str, "UPR"); break;
        default:               set_str(str, "UNK"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the symbolic name of a result code enum value.
 *
 * @param res [in]  Result code enum value.
 * @param str [out] Optional buffer receiving the name (size PRIMAL_MAX_STR_LEN).
 *
 * @return PRIMAL_RES_OK on success.
 *
 * @note Strings: OK, ERR_ARG, ERR_INFEASIBLE, ERR_UNBOUNDED, ERR_ALLOC, 
 *       ERR_FILE, ERR_NULL, TRM_MAX_ITER. Unknown: "UNKNOWN".
 *
 * @example
 * char buf[PRIMAL_MAX_STR_LEN];
 * PRIMAL_rescodetostr(PRIMAL_RES_OK, buf);
 * printf("Result: %s\n", buf);
 */
PRIMALrescodee PRIMAL_rescodetostr(PRIMALrescodee res, char *str) {
    switch (res) {
        case PRIMAL_RES_OK:             set_str(str, "OK"); break;
        case PRIMAL_RES_ERR_ARG:        set_str(str, "ERR_ARG"); break;
        case PRIMAL_RES_ERR_INFEASIBLE: set_str(str, "ERR_INFEASIBLE"); break;
        case PRIMAL_RES_ERR_UNBOUNDED:  set_str(str, "ERR_UNBOUNDED"); break;
        case PRIMAL_RES_ERR_ALLOC:      set_str(str, "ERR_ALLOC"); break;
        case PRIMAL_RES_ERR_FILE:       set_str(str, "ERR_FILE"); break;
        case PRIMAL_RES_ERR_NULL:       set_str(str, "ERR_NULL"); break;
        case PRIMAL_RES_TRM_MAX_ITER:   set_str(str, "TRM_MAX_ITER"); break;
        default:                        set_str(str, "UNKNOWN"); break;
    }
    return PRIMAL_RES_OK;
}

/**
 * Converts a cone type string to its enum value (inverse of conetypetostr).
 *
 * @param t  [in]  Task handle (unused).
 * @param str [in]  Cone type string (e.g., "QUAD", "RQUAD", "PEXP", "DEXP", "PPOW", "RPOW").
 * @param ct  [out] Pointer to cone type enum receiving the value. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if str or ct is NULL,
 *         PRIMAL_RES_ERR_ARG if string not recognized.
 *
 * @note This is the inverse of PRIMAL_conetypetostr. Accepts: "QUAD", "RQUAD",
 *       "PEXP", "DEXP", "PPOW", "RPOW". Case-sensitive.
 *
 * @example
 * PRIMALconetypee ct;
 * PRIMAL_strtoconetype(task, "PEXP", &ct);
 * // ct = PRIMAL_CT_PEXP
 */
PRIMALrescodee PRIMAL_strtoconetype(PRIMALtask_t t, const char *str, PRIMALconetypee *ct) {
    (void)t;
    if (!str || !ct) return PRIMAL_RES_ERR_NULL;
    if (!strcmp(str, "QUAD")) *ct = PRIMAL_CT_QUAD;
    else if (!strcmp(str, "RQUAD")) *ct = PRIMAL_CT_RQUAD;
    else if (!strcmp(str, "PEXP")) *ct = PRIMAL_CT_PEXP;
    else if (!strcmp(str, "DEXP")) *ct = PRIMAL_CT_DEXP;
    else if (!strcmp(str, "PPOW")) *ct = PRIMAL_CT_PPOW;
    else if (!strcmp(str, "RPOW")) *ct = PRIMAL_CT_RPOW;
    else return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/**
 * Converts a solution status key string to its enum value (inverse of sktostr).
 *
 * @param t  [in]  Task handle (unused).
 * @param str [in]  Status key string (e.g., "UNK", "BAS", "SUPBAS", "LOW", "UPR").
 * @param sk  [out] Pointer to status key enum receiving the value. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if str or sk is NULL,
 *         PRIMAL_RES_ERR_ARG if string not recognized.
 *
 * @note This is the inverse of PRIMAL_sktostr. Accepts: "UNK", "BAS", "SUPBAS",
 *       "LOW", "UPR". Case-sensitive.
 */
PRIMALrescodee PRIMAL_strtosk(PRIMALtask_t t, const char *str, PRIMALstakeye *sk) {
    (void)t;
    if (!str || !sk) return PRIMAL_RES_ERR_NULL;
    if (!strcmp(str, "UNK")) *sk = PRIMAL_SK_UNDEF;
    else if (!strcmp(str, "BAS")) *sk = PRIMAL_SK_BAS;
    else if (!strcmp(str, "SUPBAS")) *sk = PRIMAL_SK_SUPBAS;
    else if (!strcmp(str, "LOW")) *sk = PRIMAL_SK_LOW;
    else if (!strcmp(str, "UPR")) *sk = PRIMAL_SK_UPR;
    else return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/* Reference getnumparam: the number of parameters of a given type in this
 * solver's declarative table (kind: 1 = double, 2 = int, 3 = string). */
PRIMALrescodee PRIMAL_getnumparam(PRIMALtask_t t, int partype, int *numparam) {
    if (!t || !numparam) return PRIMAL_RES_ERR_NULL;
    int n = 0;
    if (partype == 1) {          /* double (includes the double API alias) */
        for (int i = 0; i < PRIMAL_NPARAM; i++)
            if (PRIMAL_PARAMS[i].kind == P_DOU || PRIMAL_PARAMS[i].kind == P_DOUI) n++;
    } else if (partype == 2) {   /* int */
        for (int i = 0; i < PRIMAL_NPARAM; i++)
            if (PRIMAL_PARAMS[i].kind == P_INT) n++;
    } else if (partype == 3) {   /* string: none here */
        n = 0;
    } else {
        return PRIMAL_RES_ERR_ARG;
    }
    *numparam = n;
    return PRIMAL_RES_OK;
}

/* Parameter names (reference getparamname/whichparam/getnaintparam/
 * getnadouparam/getnastrparam/getparammax/isdouparname/isintparname/
 * isstrparname). The name is our own enum's (`PRIMAL_IPAR_*`), as the
 * status symbolic names are ours (T129). There is no string
 * parameter: `getnastrparam`/`isstrparname` answer ERR_ARG. */
static const PrimalParam *param_find_name(const char *name, int kind) {
    if (!name) return NULL;
    for (int i = 0; i < PRIMAL_NPARAM; i++)
        if (PRIMAL_PARAMS[i].name && strcmp(PRIMAL_PARAMS[i].name, name) == 0 &&
            PRIMAL_PARAMS[i].kind == kind)
            return &PRIMAL_PARAMS[i];
    return NULL;
}
/* Reads the declared enum name of one parameter of the given kind. */
PRIMALrescodee PRIMAL_getparamname(PRIMALtask_t t, int partype, int param, char *parname) {
    if (!t || !parname) return PRIMAL_RES_ERR_NULL;
    if (partype != PRIMAL_PARAM_KIND_DOU && partype != PRIMAL_PARAM_KIND_INT)
        return PRIMAL_RES_ERR_ARG;
    const PrimalParam *d = param_find(partype == PRIMAL_PARAM_KIND_DOU ? P_DOU : P_INT, param);
    if (!d || !d->name) return PRIMAL_RES_ERR_ARG;
    strcpy(parname, d->name);
    return PRIMAL_RES_OK;
}
/* Reads the parameter count of the given kind (same as getnumparam). */
PRIMALrescodee PRIMAL_getparammax(PRIMALtask_t t, int partype, int *parammax) {
    return PRIMAL_getnumparam(t, partype, parammax);
}
/* Resolves a parameter name to its kind and id in the table. */
PRIMALrescodee PRIMAL_whichparam(PRIMALtask_t t, const char *parname, int *partype, int *param) {
    if (!t || !parname || !partype || !param) return PRIMAL_RES_ERR_NULL;
    for (int i = 0; i < PRIMAL_NPARAM; i++)
        if (PRIMAL_PARAMS[i].name && strcmp(PRIMAL_PARAMS[i].name, parname) == 0) {
            *partype = (PRIMAL_PARAMS[i].kind == P_INT) ? 2 : 1;
            *param = PRIMAL_PARAMS[i].id;
            return PRIMAL_RES_OK;
        }
    return PRIMAL_RES_ERR_ARG;
}
/* Resolves a double-parameter name to its id, or ERR_ARG. */
PRIMALrescodee PRIMAL_isdouparname(PRIMALtask_t t, const char *parname, int *param) {
    if (!t || !parname || !param) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(parname, P_DOU);
    if (!d) d = param_find_name(parname, P_DOUI);
    if (!d) return PRIMAL_RES_ERR_ARG;
    *param = d->id;
    return PRIMAL_RES_OK;
}
/* Resolves an int-parameter name to its id, or ERR_ARG. */
PRIMALrescodee PRIMAL_isintparname(PRIMALtask_t t, const char *parname, int *param) {
    if (!t || !parname || !param) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(parname, P_INT);
    if (!d) return PRIMAL_RES_ERR_ARG;
    *param = d->id;
    return PRIMAL_RES_OK;
}
/* String-parameter lookup: always ERR_ARG, no string parameter exists. */
PRIMALrescodee PRIMAL_isstrparname(PRIMALtask_t t, const char *parname, int *param) {
    if (!t || !parname || !param) return PRIMAL_RES_ERR_NULL;
    (void)parname;
    return PRIMAL_RES_ERR_ARG;   /* no string parameter */
}
/* Reads one int parameter addressed by its enum name. */
PRIMALrescodee PRIMAL_getnaintparam(PRIMALtask_t t, const char *paramname, int *parvalue) {
    if (!t || !paramname || !parvalue) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(paramname, P_INT);
    if (!d) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_getintparam(t, d->id, parvalue);
}
/* Reads one double parameter addressed by its enum name. */
PRIMALrescodee PRIMAL_getnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt *parvalue) {
    if (!t || !paramname || !parvalue) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(paramname, P_DOU);
    if (!d) d = param_find_name(paramname, P_DOUI);
    if (!d) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_getdouparam(t, d->id, parvalue);
}
/* String-parameter read: always ERR_ARG, no string parameter exists. */
PRIMALrescodee PRIMAL_getnastrparam(PRIMALtask_t t, const char *paramname,
                                    int sizeparamname, int *len, char *parvalue) {
    if (!t || !paramname || !len || !parvalue) return PRIMAL_RES_ERR_NULL;
    (void)sizeparamname;
    return PRIMAL_RES_ERR_ARG;   /* no string parameter */
}
/* The by-name setters and the string family. There is no string parameter,
 * so get/put/resetstrparam* answer ERR_ARG; the int and double by-name
 * setters find the row and delegate to the by-id setter. */
PRIMALrescodee PRIMAL_putnaintparam(PRIMALtask_t t, const char *paramname, int parvalue) {
    if (!t || !paramname) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(paramname, P_INT);
    if (!d) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_putintparam(t, d->id, parvalue);
}
/* Sets one double parameter addressed by its enum name. */
PRIMALrescodee PRIMAL_putnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt parvalue) {
    if (!t || !paramname) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find_name(paramname, P_DOU);
    if (!d) d = param_find_name(paramname, P_DOUI);
    if (!d) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_putdouparam(t, d->id, parvalue);
}
/* String-parameter write: always ERR_ARG, no string parameter exists. */
PRIMALrescodee PRIMAL_putnastrparam(PRIMALtask_t t, const char *paramname, const char *parvalue) {
    if (!t || !paramname || !parvalue) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_RES_ERR_ARG;   /* no string parameter */
}
/* String-parameter read by id: always ERR_ARG, none exists. */
PRIMALrescodee PRIMAL_getstrparam(PRIMALtask_t t, int param, int maxlen, int *len, char *parvalue) {
    if (!t || !len || !parvalue) return PRIMAL_RES_ERR_NULL;
    (void)param; (void)maxlen;
    return PRIMAL_RES_ERR_ARG;
}
/* String-parameter length by id: always ERR_ARG, none exists. */
PRIMALrescodee PRIMAL_getstrparamlen(PRIMALtask_t t, int param, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    (void)param;
    return PRIMAL_RES_ERR_ARG;
}
/* String-parameter write by id: always ERR_ARG, none exists. */
PRIMALrescodee PRIMAL_putstrparam(PRIMALtask_t t, int param, const char *parvalue) {
    if (!t || !parvalue) return PRIMAL_RES_ERR_NULL;
    (void)param;
    return PRIMAL_RES_ERR_ARG;
}
/* String-parameter reset by id: always ERR_ARG, none exists. */
PRIMALrescodee PRIMAL_resetstrparam(PRIMALtask_t t, int param) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)param;
    return PRIMAL_RES_ERR_ARG;
}

