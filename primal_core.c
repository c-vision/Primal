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
/* primal_core.c - task/env lifecycle, declarative parameter table, streams/callbacks, model growth.
 * Shares primal_priv.h.
 */
#include "primal_priv.h"

void result_changed(PRIMALtask_t t) {
    if (!t || t->derived_syncing) return;
    t->result_stale = 1;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN; t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    t->has_pray = t->has_dray = 0;
    t->mip_bound_defined = 0; t->mip_result = 0;
}

void model_changed(PRIMALtask_t t) {
    if (!t || t->derived_syncing) return;
    t->derived_dirty = 1;
    result_changed(t);
}

/* ---------------- declarative parameter table ----------------
 * One row per accepted parameter: kind, where it lives in the task, its
 * default and its inclusive range.  PRIMAL_putintparam/PRIMAL_putdouparam and
 * their getters, the defaults applied by PRIMAL_maketask and
 * PRIMAL_getparaminfo all read this table, so adding a parameter is one row
 * plus the field it points at — no switch to keep in sync.
 * P_DOUI is the double-valued API alias of an integer slot
 * (PRIMAL_DPAR_INTPNT_MAX_ITER is an alias of PRIMAL_IPAR_INTPNT_MAX_ITERATIONS).
 * int ids and double ids are separate namespaces: an id is only visible to the
 * API of its own kind. */


#define P_OFF(f) offsetof(struct PRIMAL_task_s, f)

const PrimalParam PRIMAL_PARAMS[] = {
    { PRIMAL_IPAR_OPTIMIZER,              P_INT, P_OFF(optimizer),
      /* the reference's default is FREE, and with MSKoptimizertypee's numbering
       * FREE is 2: keeping 0 here would publish CONIC on a fresh task */
      2.0, 0.0, 8.0, "PRIMAL_IPAR_OPTIMIZER" },
    { PRIMAL_IPAR_LOG,                    P_INT, P_OFF(log),
      0.0, 0.0, 1.0, "PRIMAL_IPAR_LOG" },
    { PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS, P_INT, P_OFF(max_iter_simplex),
      10000000.0, 0.0, INT_MAX, "PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS" },
    { PRIMAL_IPAR_INTPNT_MAX_ITERATIONS,  P_INT, P_OFF(max_iter_intpnt),
      400.0, 0.0, INT_MAX, "PRIMAL_IPAR_INTPNT_MAX_ITERATIONS" },
    { PRIMAL_IPAR_INTPNT_MAX_NUM_COR,     P_INT, P_OFF(intpnt_max_cor),
      -1.0, -1.0, INT_MAX, "PRIMAL_IPAR_INTPNT_MAX_NUM_COR" },
    { PRIMAL_IPAR_PRESOLVE,               P_INT, P_OFF(presolve),
      1.0, 0.0, 1.0, "PRIMAL_IPAR_PRESOLVE" },
    { PRIMAL_IPAR_PRESOLVE_LEVEL,         P_INT, P_OFF(presolve_level),
      1.0, 0.0, 2.0, "PRIMAL_IPAR_PRESOLVE_LEVEL" },
    { PRIMAL_IPAR_CONCURRENT_TIME,        P_INT, P_OFF(concurrent_time),
      0.0, 0.0, 1.0, "PRIMAL_IPAR_CONCURRENT_TIME" },
    { PRIMAL_IPAR_SCALING,                P_INT, P_OFF(scaling),
      1.0, 0.0, 1.0, "PRIMAL_IPAR_SCALING" },
    { PRIMAL_IPAR_MIP_MAX_NODES,          P_INT, P_OFF(mip_max_nodes),
      100000.0, 1.0, INT_MAX, "PRIMAL_IPAR_MIP_MAX_NODES" },
    { PRIMAL_IPAR_NUM_THREADS,            P_INT, P_OFF(num_threads),
      1.0, 0.0, INT_MAX, "PRIMAL_IPAR_NUM_THREADS" },
    /* Accepted ranges are the reference's (parameters.html 11.2.4):
     *   INTPNT_TOL_PFEAS/DFEAS  -> [0.0; 1.0]
     *   INTPNT_TOL_REL_GAP      -> [1.0e-14; +inf]
     * The previous table widened all three to [DBL_MIN, DBL_MAX]; that was a
     * misread of the reference (only REL_GAP is unbounded below 1), corrected
     * 2026-09-22. T82 still exercises 1e-15 on the *conic* set (CO_TOL_*), whose
     * range is [0,1] and which the conic route reads. */
    { PRIMAL_DPAR_INTPNT_TOL_PFEAS,       P_DOU, P_OFF(tol_pfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_TOL_PFEAS" },
    { PRIMAL_DPAR_INTPNT_TOL_DFEAS,       P_DOU, P_OFF(tol_dfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_TOL_DFEAS" },
    { PRIMAL_DPAR_INTPNT_TOL_REL_GAP,     P_DOU, P_OFF(tol_gap),
      1e-8, 1e-14, DBL_MAX, "PRIMAL_DPAR_INTPNT_TOL_REL_GAP" },
    { PRIMAL_DPAR_INTPNT_MAX_ITER,        P_DOUI, P_OFF(max_iter_intpnt),
      400.0, 0.0, INT_MAX, "PRIMAL_DPAR_INTPNT_MAX_ITER" },
    { PRIMAL_DPAR_INTPNT_TOL_NEAR_REL,    P_DOU, P_OFF(tol_near_rel),
      1000.0, 1.0, DBL_MAX, "PRIMAL_DPAR_INTPNT_TOL_NEAR_REL" },
    /* Conic interior-point tolerance set (reference MSK_DPAR_INTPNT_CO_TOL_*,
     * parameters.html 11.2.4: default 1e-8, accepted [0,1]) -- read by the conic
     * route (sdp_ipm / socp_solve). */
    { PRIMAL_DPAR_INTPNT_CO_TOL_PFEAS,    P_DOU, P_OFF(tol_co_pfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_CO_TOL_PFEAS" },
    { PRIMAL_DPAR_INTPNT_CO_TOL_DFEAS,    P_DOU, P_OFF(tol_co_dfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_CO_TOL_DFEAS" },
    { PRIMAL_DPAR_INTPNT_CO_TOL_REL_GAP,  P_DOU, P_OFF(tol_co_gap),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_CO_TOL_REL_GAP" },
    /* Quadratic interior-point tolerance set (reference MSK_DPAR_INTPNT_QO_TOL_*,
     * same defaults/ranges) -- read by the quadratic route (ipm_solve_qp_csc and
     * the dense QP ipm_solve_std calls). */
    { PRIMAL_DPAR_INTPNT_QO_TOL_PFEAS,    P_DOU, P_OFF(tol_qo_pfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_QO_TOL_PFEAS" },
    { PRIMAL_DPAR_INTPNT_QO_TOL_DFEAS,    P_DOU, P_OFF(tol_qo_dfeas),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_QO_TOL_DFEAS" },
    { PRIMAL_DPAR_INTPNT_QO_TOL_REL_GAP,  P_DOU, P_OFF(tol_qo_gap),
      1e-8, 0.0, 1.0, "PRIMAL_DPAR_INTPNT_QO_TOL_REL_GAP" },
    /* Reference MSK_DPAR_OPTIMIZER_MAX_TIME, default -1.0 (no limit), accepted
     * [-inf,+inf]: a wall-clock cap on the whole optimization. Read by
     * PRIMAL_optimize (deadline) and enforced in the IPM and B&B loops. */
    { PRIMAL_DPAR_OPTIMIZER_MAX_TIME,     P_DOU, P_OFF(optimizer_max_time),
      -1.0, -DBL_MAX, DBL_MAX, "PRIMAL_DPAR_OPTIMIZER_MAX_TIME" },
    /* Reference MSK_DPAR_MIO_MAX_TIME, default -1.0 (no limit), accepted
     * [-inf,+inf]: a wall-clock cap on the mixed-integer phase only. The B&B
     * obeys the tighter of this and OPTIMIZER_MAX_TIME. */
    { PRIMAL_DPAR_MIO_MAX_TIME,           P_DOU, P_OFF(mio_max_time),
      -1.0, -DBL_MAX, DBL_MAX, "PRIMAL_DPAR_MIO_MAX_TIME" },
    /* Reference MSK_DPAR_LOWER_OBJ_CUT, default -inf (no cut), accepted
     * [-inf,+inf]: if a primal-FEASIBLE point has objective below this, the
     * optimum is proven below the cut and the solve terminates with
     * PRIMAL_RES_TRM_OBJECTIVE_RANGE. Wired for minimization; a maximization
     * would need the mirrored cut on the dual side (not implemented). */
    { PRIMAL_DPAR_LOWER_OBJ_CUT,          P_DOU, P_OFF(lower_obj_cut),
      -DBL_MAX, -DBL_MAX, DBL_MAX, "PRIMAL_DPAR_LOWER_OBJ_CUT" },
    /* Reference MSK_DPAR_UPPER_OBJ_CUT, default +inf (no cut), accepted
     * [-inf,+inf]: the dual-side twin -- a dual-FEASIBLE point whose dual
     * objective is above the cut proves the optimum is above it. Enforced on the
     * LP route (b'y); the QP/conic routes do not compute that bound. */
    { PRIMAL_DPAR_UPPER_OBJ_CUT,          P_DOU, P_OFF(upper_obj_cut),
      DBL_MAX, -DBL_MAX, DBL_MAX, "PRIMAL_DPAR_UPPER_OBJ_CUT" },
    /* Reference MSK_DPAR_SEMIDEFINITE_TOL_APPROX, default 1.0e-10, accepted
     * [1e-15,+inf]: tolerance to define a matrix PSD. Read as the RELATIVE factor
     * of the encoder's convexity threshold (bad_thr = semi_tol*max(lmax,1)) and of
     * the witness PSD check (e >= -semi_tol*(1+emax)) -- a declared deviation:
     * the reference's is an absolute tolerance. */
    { PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX, P_DOU, P_OFF(semi_tol_approx),
      1e-10, 1e-15, DBL_MAX, "PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX" },
      /* MSK_DPAR_INTPNT_CO_TOL_NEAR_REL, default 1000, accepted [1.0; +inf] --
       * read from parameters.html.  The reference's own words: "if MOSEK cannot
       * compute a solution that has the prescribed accuracy then it will check
       * if the solution found satisfies the termination criteria with all
       * tolerances multiplied by the value of this parameter.  If yes, then the
       * solution is also declared optimal."  So a point at 1.3x the declared
       * rel_gap is not a "near optimal" verdict -- MSKsolsta has no NEAR_ member
       * -- it is OPTIMAL, judged against an effective tolerance 1000x the
       * nominal one.  Applied here to the unified conic IPM's gate (sdp.c),
       * which is where this solver decides "solved" from a measured triple. */
    { PRIMAL_DPAR_MIP_TOL_ABS_GAP,        P_DOU, P_OFF(mip_tol_abs_gap),
      0.0, 0.0, DBL_MAX, "PRIMAL_DPAR_MIP_TOL_ABS_GAP" },
    /* Reference ranges (parameters.html): MIO_TOL_REL_GAP [0.0; +inf],
     * MIO_TOL_ABS_RELAX_INT [1e-9; +inf] -- the old [0,1]/[DBL_MIN,1] were
     * narrower than the reference. */
    { PRIMAL_DPAR_MIP_TOL_REL_GAP,        P_DOU, P_OFF(mip_tol_rel_gap),
      1e-4, 0.0, DBL_MAX, "PRIMAL_DPAR_MIP_TOL_REL_GAP" },
    { PRIMAL_DPAR_MIP_TOL_INTHER,         P_DOU, P_OFF(mip_tol_inther),
      1e-5, 1e-9, DBL_MAX, "PRIMAL_DPAR_MIP_TOL_INTHER" },
    { PRIMAL_DPAR_MIP_TOL_FEAS,           P_DOU, P_OFF(mip_tol_feas),
      1e-6, 1e-9, 1e-3, "PRIMAL_DPAR_MIP_TOL_FEAS" },
      /* MSK_DPAR_MIO_TOL_FEAS, "feasibility tolerance for mixed integer solver",
       * default 1e-6, accepted [1e-9; 1e-3] -- read from parameters.html. This is
       * the tolerance an INCUMBENT is re-verified against (mip_point_measures):
       * integrality may be declared within ABS_RELAX_INT, but no point that
       * violates the model's own bounds, rows, cones, semi or SOS sets by more
       * than this is published as an integer solution. Before this row existed the
       * solver had no such check, which is why ABS_RELAX_INT had to stay at 1e-6
       * here -- at the reference's 1e-5, djc1's max case published x0 = 10,
       * outside both disjuncts (x0 <= 2 OR 6 <= x0 <= 7), with rc = OK. */
};

/* row count for PRIMAL_NPARAM (see primal_priv.h) */
const int primal_nparam_count = (int)(sizeof PRIMAL_PARAMS / sizeof PRIMAL_PARAMS[0]);



/* The reference accepts 0 for the iteration limits, meaning "no limit"; this
 * solver's loops take a positive cap, so 0 is mapped to the largest int at every
 * use site and the stored value stays 0 (what a getter must read back). */
int iter_cap(int v) { return v > 0 ? v : INT_MAX; }

/* the row of `id` for the API of `kind` (P_DOUI answers to a double query) */
const PrimalParam *param_find(int kind, int id) {
    for (int i = 0; i < PRIMAL_NPARAM; i++)
        if (PRIMAL_PARAMS[i].id == id &&
            (PRIMAL_PARAMS[i].kind == kind ||
             (kind == P_DOU && PRIMAL_PARAMS[i].kind == P_DOUI)))
            return &PRIMAL_PARAMS[i];
    return NULL;
}

/* Address of the parameter slot in the task, from the table row offset. */
void *param_slot(const PrimalParam *d, struct PRIMAL_task_s *t) {
    return (void *)((char *)t + d->off);
}

/* write every declared default into a freshly allocated task */
void param_defaults(struct PRIMAL_task_s *t) {
    for (int i = 0; i < PRIMAL_NPARAM; i++) {
        const PrimalParam *d = &PRIMAL_PARAMS[i];
        if (d->kind == P_DOU) *(double *)param_slot(d, t) = d->dflt;
        else *(int *)param_slot(d, t) = (int)d->dflt;
    }
}

/* carry every declared parameter from a model to the shadow task that solves
 * it: a new parameter is one table row, not a new line at each shadow site. */
void param_copy(struct PRIMAL_task_s *dst, struct PRIMAL_task_s *src) {
    for (int i = 0; i < PRIMAL_NPARAM; i++) {
        const PrimalParam *d = &PRIMAL_PARAMS[i];
        if (d->kind == P_DOU)
            *(double *)param_slot(d, dst) = *(double *)param_slot(d, src);
        else
            *(int *)param_slot(d, dst) = *(int *)param_slot(d, src);
    }
}

/* ---------------- env/task lifecycle ---------------- */

/**
 * Creates a new solver environment.
 *
 * @param env      [out] Pointer to a PRIMALenv_t variable that will receive the
 *                   handle to the newly created environment. Must not be NULL.
 * @param usercb   [in]  Reserved for future use (user-defined callback context).
 *                       Currently ignored; pass NULL.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if env is NULL,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note The environment holds global settings such as stream callbacks
 *       (linkfunctoenvstream, linkfiletoenvstream) and the exit function
 *       (putexitfunc). It is the parent of all tasks created via PRIMAL_maketask.
 *       The usercb parameter is reserved for future extensibility and is not
 *       currently used by the solver.
 *
 * @example
 * PRIMALenv_t env;
 * PRIMALrescodee rc = PRIMAL_makeenv(&env, NULL);
 * if (rc != PRIMAL_RES_OK) { / * handle error * / }
 */
PRIMALrescodee PRIMAL_makeenv(PRIMALenv_t *env, void *usercb) {
    (void)usercb;
    if (!env) return PRIMAL_RES_ERR_NULL;
    *env = (PRIMALenv_t)calloc(1, sizeof(struct PRIMAL_env_s));
    return *env ? PRIMAL_RES_OK : PRIMAL_RES_ERR_ALLOC;
}

/**
 * Registers an exit callback function to be called on fatal errors.
 *
 * @param env        [in] Environment handle returned by PRIMAL_makeenv.
 * @param exitfunc   [in] Callback function with signature:
 *                        void exitfunc(void *handle, int exitcode).
 *                        Called when the solver encounters an unrecoverable error.
 * @param handle     [in] User-defined pointer passed to exitfunc on invocation.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if env is NULL.
 *
 * @note The exit function is invoked before the process terminates on fatal
 *       internal errors (e.g., assertion failures in debug builds, out-of-memory
 *       conditions that cannot be returned as error codes). It allows the host
 *       application to perform cleanup or logging.
 *
 * @example
 * void my_exit(void *handle, int code) {
 *     fprintf(stderr, "Solver fatal exit: %d\n", code);
 * }
 * PRIMAL_putexitfunc(env, my_exit, NULL);
 */
PRIMALrescodee PRIMAL_putexitfunc(PRIMALenv_t env, PRIMALexitfunc exitfunc, void *handle) {
    if (!env) return PRIMAL_RES_ERR_NULL;
    env->exitfunc = exitfunc;
    env->exithandle = handle;
    return PRIMAL_RES_OK;
}

/**
 * Creates a new optimization task within an environment.
 *
 * @param env      [in]  Environment handle from PRIMAL_makeenv.
 * @param maxcon   [in]  Initial number of constraints (rows). The task will
 *                       pre-allocate storage for this many constraints.
 * @param maxvar   [in]  Initial number of variables (columns). The task will
 *                       pre-allocate storage for this many variables.
 * @param task     [out] Pointer to a PRIMALtask_t variable that will receive
 *                       the handle to the newly created task. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if task is NULL,
 *         PRIMAL_RES_ERR_ARG if env is NULL,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note This function pre-allocates the constraint and variable arrays to the
 *       specified sizes. You do NOT need to call PRIMAL_appendvars/appendcons
 *       afterward for the same sizes -- doing so would double the dimensions.
 *       All solver parameters are initialized to their default values from the
 *       declarative PRIMAL_PARAMS table. The task inherits the environment's
 *       stream callback (if set) as its initial log stream.
 *
 * @warning If you call PRIMAL_appendvars or PRIMAL_appendcons after this
 *          function with the same maxvar/maxcon values, you will double the
 *          allocated sizes, leading to buffer overflows in solution retrieval.
 *
 * @example
 * PRIMALtask_t task;
 * PRIMALrescodee rc = PRIMAL_maketask(env, 100, 50, &task);
 * // Task now has space for 100 constraints and 50 variables
 */
PRIMALrescodee PRIMAL_maketask(PRIMALenv_t env, int maxcon, int maxvar, PRIMALtask_t *task) {
    (void)maxcon; (void)maxvar;
    if (!task) return PRIMAL_RES_ERR_NULL;
    if (!env || maxcon < 0 || maxvar < 0) return PRIMAL_RES_ERR_ARG;
    PRIMALtask_t t = (PRIMALtask_t)calloc(1, sizeof(struct PRIMAL_task_s));
    if (!t) return PRIMAL_RES_ERR_ALLOC;
    t->env = env;
    t->sense = PRIMAL_OPTIMIZE_MINIMIZE;
    param_defaults(t);   /* every parameter default comes from the table */
    if (env->streamfunc) {   /* inherit the env stream (link*functoenvstream),
                               * after the defaults because param_defaults zeroes log */
        t->logcb = env->streamfunc; t->loghandle = env->streamhandle; t->log = 1;
    }
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    t->last_rc = PRIMAL_RES_ERR_ARG;
    *task = t;
    /* pre-allocate the declared sizes (draft- compatible: the example never
     * calls PRIMAL_appendvars/PRIMAL_appendcons explicitly) */
    PRIMALrescodee rc = PRIMAL_appendvars(t,maxvar);
    if (rc == PRIMAL_RES_OK) rc = PRIMAL_appendcons(t,maxcon);
    if (rc != PRIMAL_RES_OK) { PRIMAL_deletetask(task); return rc; }
    return PRIMAL_RES_OK;
}

/**
 * Creates an empty optimization task with no pre-allocated constraints or variables.
 *
 * @param env    [in]  Environment handle from PRIMAL_makeenv.
 * @param task   [out] Pointer to a PRIMALtask_t variable that will receive
 *                     the handle to the newly created task. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success, or an error code from PRIMAL_maketask.
 *
 * @note This is a convenience wrapper equivalent to PRIMAL_maketask(env, 0, 0, task).
 *       Use this when you want to build the model incrementally by calling
 *       PRIMAL_appendvars and PRIMAL_appendcons yourself.
 *
 * @example
 * PRIMALtask_t task;
 * PRIMAL_makeemptytask(env, &task);
 * PRIMAL_appendvars(task, 50);
 * PRIMAL_appendcons(task, 100);
 */
PRIMALrescodee PRIMAL_makeemptytask(PRIMALenv_t env, PRIMALtask_t *task) {
    return PRIMAL_maketask(env, 0, 0, task);
}
/**
 * Retrieves the environment handle associated with a task.
 *
 * @param t    [in]  Task handle.
 * @param env  [out] Pointer to a PRIMALenv_t variable that will receive
 *                   the environment handle. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or env is NULL.
 *
 * @example
 * PRIMALenv_t env;
 * PRIMAL_getenv(task, &env);
 */
PRIMALrescodee PRIMAL_getenv(PRIMALtask_t t, PRIMALenv_t *env) {
    if (!t || !env) return PRIMAL_RES_ERR_NULL;
    *env = t->env;
    return PRIMAL_RES_OK;
}
/**
 * Commits pending changes to the task (no-op in this implementation).
 *
 * @param t    [in] Task handle.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This solver does not cache modifications; all changes to the model
 *       (bounds, matrix entries, cones, etc.) take effect immediately.
 *       This function exists for API compatibility with the reference solver
 *       where deferred changes may be batched.
 *
 * @example
 * PRIMAL_commitchanges(task); // No effect in this solver
 */
PRIMALrescodee PRIMAL_commitchanges(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    return PRIMAL_RES_OK;
}
/**
 * Resizes the task's internal storage (no-op in this implementation).
 *
 * @param t             [in] Task handle.
 * @param maxnumcon     [in] New maximum number of constraints (ignored).
 * @param maxnumvar     [in] New maximum number of variables (ignored).
 * @param maxnumcone    [in] New maximum number of cones (ignored).
 * @param maxnumanz     [in] New maximum number of non-zeros in A (ignored).
 * @param maxnumqnz     [in] New maximum number of non-zeros in Q (ignored).
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This solver grows all arrays automatically as needed when new
 *       constraints, variables, or matrix entries are added. This function
 *       exists for API compatibility and performs no action.
 */
PRIMALrescodee PRIMAL_resizetask(PRIMALtask_t t, int maxnumcon, int maxnumvar,
                                 int maxnumcone, PRIMALint64t maxnumanz,
                                 PRIMALint64t maxnumqnz) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)maxnumcon; (void)maxnumvar; (void)maxnumcone;
    (void)maxnumanz; (void)maxnumqnz;
    return PRIMAL_RES_OK;
}
/**
 * Updates solution information after a solve (no-op in this implementation).
 *
 * @param t    [in] Task handle.
 * @param which [in] Which solution to update (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS,
 *                    PRIMAL_SOL_ITG). Ignored.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This solver does not maintain a separate solution information cache;
 *       all getters read directly from the task's solution arrays.
 */
PRIMALrescodee PRIMAL_updatesolutioninfo(PRIMALtask_t t, PRIMALsolt which) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)which;
    return PRIMAL_RES_OK;
}
/**
 * Deletes a solution from the task, clearing the published point and status.
 *
 * @param t    [in] Task handle.
 * @param which [in] Which solution to delete (PRIMAL_SOL_ITR, PRIMAL_SOL_BAS,
 *                    PRIMAL_SOL_ITG). All solutions are cleared regardless.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This resets has_sol, has_pray, has_dray, solsta, and prosta to their
 *       initial states (UNKNOWN), effectively making the task appear as if it
 *       has never been solved. The next PRIMAL_optimize call will reinitialize
 *       the solution buffers via opt_prepare.
 *
 * @example
 * PRIMAL_deletesolution(task, PRIMAL_SOL_ITR);
 * PRIMAL_optimize(task); // Will recompute solution from scratch
 */
PRIMALrescodee PRIMAL_deletesolution(PRIMALtask_t t, PRIMALsolt which) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    (void)which;
    t->has_sol = 0;
    t->has_pray = 0;
    t->has_dray = 0;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    return PRIMAL_RES_OK;
}

/**
 * Deletes a task and frees all associated memory.
 *
 * @param task [in/out] Pointer to the task handle. On return, *task is set to NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if task or *task is NULL.
 *
 * @note This function frees all memory allocated for the task, including:
 *       - Constraint matrix columns (A)
 *       - Objective vectors (c, qobj, quadratic objective triplets)
 *       - Bounds (bkx, blx, bux, bkc, blc, buc)
 *       - Variable types, names
 *       - Quadratic constraints (qcon)
 *       - AFE (affine expressions) storage
 *       - Conic domains, ACCs, DJCs
 *       - SOS constraints
 *       - Cones and their parameters
 *       - Bar variables, symmetric matrix store, barA/barC
 *       - Solution vectors (x, y, slc, suc, slx, sux, snx, xc)
 *       - Basis keys (skc, skx)
 *       - Farkas rays (pray, dray)
 *       - Basis solve factorization
 *       - Warm start vectors
 *       - Log file handle (if owned)
 *
 *       After this call, the task handle is invalid and must not be used.
 *
 * @example
 * PRIMAL_deletetask(&task); // task is now NULL
 * PRIMAL_deleteenv(&env);   // Then delete the environment
 */
PRIMALrescodee PRIMAL_deletetask(PRIMALtask_t *task) {
    if (!task || !*task) return PRIMAL_RES_ERR_NULL;
    PRIMALtask_t t = *task;
    if (t->logfile) fclose(t->logfile);
    if (t->cols)
        for (int j = 0; j < t->numvar; j++) { free(t->cols[j].sub); free(t->cols[j].val); }
    if (t->afe_sub)
        for (int i = 0; i < t->numafe; i++) { free(t->afe_sub[i]); free(t->afe_val[i]); }
    free(t->afe_sub); free(t->afe_val); free(t->afe_nz); free(t->afe_cap); free(t->afeg);
    if (t->afe_baridx)
        for (int i = 0; i < t->numafe; i++) {
            free(t->afe_baridx[i]); free(t->afe_barsym[i]); free(t->afe_barcoef[i]);
        }
    free(t->afe_baridx); free(t->afe_barsym); free(t->afe_barcoef);
    free(t->afe_barnz); free(t->afe_barcap);
    free(t->dom_type); free(t->dom_n); free(t->dom_param);
    if (t->domname) {
        for (int i = 0; i < t->numdomain; i++) free(t->domname[i]);
        free(t->domname);
    }
    if (t->acc_afe) for (int i = 0; i < t->numacc; i++) { free(t->acc_afe[i]); free(t->acc_b[i]); }
    free(t->acc_afe); free(t->acc_b); free(t->acc_dom); free(t->acc_nafe);
    free(t->acc_rowbase); free(t->acc_varbase);
    if (t->accname) {
        for (int i = 0; i < t->numacc; i++) free(t->accname[i]);
        free(t->accname);
    }
    if (t->djc_dom) for (int i = 0; i < t->numdjc; i++) {
        free(t->djc_dom[i]); free(t->djc_afe[i]); free(t->djc_b[i]); free(t->djc_termsize[i]);
    }
    free(t->djc_dom); free(t->djc_afe); free(t->djc_b); free(t->djc_termsize);
    free(t->djc_ndom); free(t->djc_nafe); free(t->djc_numterm);
    free(t->djc_rowbase); free(t->djc_varbase); free(t->djc_nrow);
    if (t->djcname) {
        for (int i = 0; i < t->numdjc; i++) free(t->djcname[i]);
        free(t->djcname);
    }
    free(t->cols);
    free(t->c); free(t->qobj);
    free(t->qt_i); free(t->qt_j); free(t->qt_v);
    free(t->bkx); free(t->blx); free(t->bux);
    free(t->bkc); free(t->blc); free(t->buc);
    free(t->vartype);
    if (t->varname) {
        for (int j = 0; j < t->numvar; j++) free(t->varname[j]);
        free(t->varname);
    }
    if (t->conname) {
        for (int i = 0; i < t->numcon; i++) free(t->conname[i]);
        free(t->conname);
    }
    if (t->barname) {
        for (int j = 0; j < t->numbarvar; j++) free(t->barname[j]);
        free(t->barname);
    }
    if (t->conename) {
        for (int k = 0; k < t->numcones; k++) free(t->conename[k]);
        free(t->conename);
    }
    free(t->objname);
    free(t->taskname);
    if (t->cone_mem)
        for (int k = 0; k < t->numcones; k++) free(t->cone_mem[k]);
    free(t->cone_type); free(t->cone_nmem); free(t->cone_mem); free(t->cone_param);
    free(t->barDim);
    if (t->sym_subi)
        for (int k = 0; k < t->nsym; k++) {
            free(t->sym_subi[k]); free(t->sym_subj[k]); free(t->sym_val[k]);
        }
    free(t->sym_dim); free(t->sym_nnz); free(t->sym_cap);
    free(t->sym_subi); free(t->sym_subj); free(t->sym_val);
    free(t->barA_con); free(t->barA_bar); free(t->barA_sym); free(t->barA_coef);
    free(t->barC_bar); free(t->barC_sym); free(t->barC_coef);
    if (t->barx)
        for (int j = 0; j < t->numbarvar; j++) free(t->barx[j]);
    free(t->barx);
    if (t->barsj)
        for (int j = 0; j < t->numbarvar; j++) free(t->barsj[j]);
    free(t->barsj);
    free(t->x); free(t->y); free(t->slc); free(t->suc); free(t->slx); free(t->sux);
    free(t->soc_dual); t->soc_dual = NULL; t->nsoc_dual = 0;
    free(t->snx); free(t->xc);
    if (t->basis_lu) dmat_lu_free((LuFact *)t->basis_lu);
    free(t->basis_vec);
    free(t->pray); free(t->dray);
    free(t->warm_x); free(t->warm_y);
    free(t->skc); free(t->skx);
    if (t->sos_mem)
        for (int k = 0; k < t->numsos; k++) { free(t->sos_mem[k]); free(t->sos_w[k]); }
    free(t->sos_type); free(t->sos_n); free(t->sos_mem); free(t->sos_w);
    if (t->qcon)
        for (int i = 0; i < t->numcon; i++) free(t->qcon[i]);
    free(t->qcon);
    free(t);
    *task = NULL;
    return PRIMAL_RES_OK;
}

/**
 * Deletes an environment and frees its memory.
 *
 * @param env [in/out] Pointer to the environment handle. On return, *env is set to NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if env or *env is NULL.
 *
 * @note This closes the environment's log file (if opened via
 *       linkfiletoenvstream) and frees the environment structure.
 *       All tasks created from this environment must be deleted first
 *       via PRIMAL_deletetask before calling this function.
 *
 * @example
 * PRIMAL_deletetask(&task);
 * PRIMAL_deleteenv(&env); // env is now NULL
 */
PRIMALrescodee PRIMAL_deleteenv(PRIMALenv_t *env) {
    if (!env || !*env) return PRIMAL_RES_ERR_NULL;
    if ((*env)->streamfile) fclose((*env)->streamfile);
    free(*env);
    *env = NULL;
    return PRIMAL_RES_OK;
}

/* Memory helpers: plain calloc/free bound to env/task. No internal pool exists
 * here, so the debug variants coincide with the plain ones and the checks are
 * vacuously OK (declared deviation, see primal.h). */
/* Allocate a zeroed block bound to the environment (plain calloc). */
void *PRIMAL_callocenv(PRIMALenv_t env, size_t number, size_t size) {
    (void)env;
    return calloc(number, size);
}
/* Debug variant of PRIMAL_callocenv; no pool exists, so file/line are ignored. */
void *PRIMAL_callocdbgenv(PRIMALenv_t env, size_t number, size_t size,
                          const char *file, unsigned line) {
    (void)env; (void)file; (void)line;
    return calloc(number, size);
}
/* Free a block bound to the environment. */
void PRIMAL_freeenv(PRIMALenv_t env, void *buffer) { (void)env; free(buffer); }
/* Debug variant of PRIMAL_freeenv; file/line are ignored. */
void PRIMAL_freedbgenv(PRIMALenv_t env, void *buffer, const char *file, unsigned line) {
    (void)env; (void)file; (void)line;
    free(buffer);
}
/* Allocate a zeroed block bound to the task (plain calloc). */
void *PRIMAL_calloctask(PRIMALtask_t task, size_t number, size_t size) {
    (void)task;
    return calloc(number, size);
}
/* Debug variant of PRIMAL_calloctask; no pool exists, so file/line are ignored. */
void *PRIMAL_callocdbgtask(PRIMALtask_t task, size_t number, size_t size,
                           const char *file, unsigned line) {
    (void)task; (void)file; (void)line;
    return calloc(number, size);
}
/* Free a block bound to the task. */
void PRIMAL_freetask(PRIMALtask_t task, void *buffer) { (void)task; free(buffer); }
/* Debug variant of PRIMAL_freetask; file/line are ignored. */
void PRIMAL_freedbgtask(PRIMALtask_t task, void *buffer, const char *file, unsigned line) {
    (void)task; (void)file; (void)line;
    free(buffer);
}
/* Global environment init: no internal pool, so this is a no-op returning OK. */
PRIMALrescodee PRIMAL_globalenvinitialize(PRIMALint64t maxnumalloc, const char *dbgfile) {
    (void)maxnumalloc; (void)dbgfile;
    return PRIMAL_RES_OK;
}
/* Global environment finalize: no-op (there is no pool to release). */
PRIMALrescodee PRIMAL_globalenvfinalize(void) { return PRIMAL_RES_OK; }
/* Check environment memory: vacuous success, no allocator state is tracked. */
PRIMALrescodee PRIMAL_checkmemenv(PRIMALenv_t env, const char *file, int line) {
    (void)env; (void)file; (void)line;
    return PRIMAL_RES_OK;
}
/* Check task memory: vacuous success, no allocator state is tracked. */
PRIMALrescodee PRIMAL_checkmemtask(PRIMALtask_t task, const char *file, int line) {
    (void)task; (void)file; (void)line;
    return PRIMAL_RES_OK;
}

/* Emit a message on the task log stream, when logging is enabled. */
void tlog(PRIMALtask_t t, const char *msg) {
    if (t && t->log && t->logcb) t->logcb(t->loghandle, msg);
}

/* progress info string: "<infoname>: <msg>" via the progress callback
 * (only when a callback is set; independent of the log stream) */
void tprog(PRIMALtask_t t, const char *msg) {
    if (t && t->progcb) {
        char buf[192];
        if (t->infoname[0])
            snprintf(buf, sizeof buf, "%s: %s", t->infoname, msg);
        else
            snprintf(buf, sizeof buf, "%s", msg);
        t->progcb(t->proghandle, buf);
    }
}

/**
 * Sets a progress callback for per-iteration information.
 *
 * @param t      [in] Task handle.
 * @param cb     [in] Callback function with signature:
 *                   void cb(void *handle, const char *msg).
 *                   Called with formatted progress messages during optimization.
 * @param handle [in] User-defined pointer passed to cb on each invocation.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This callback is independent of the log stream (set via
 *       linkfunctotaskstream). It receives concise progress messages
 *       (e.g., "IPM iter 5: pobj=1.234 dobj=1.233 rel_gap=1e-4") and is
 *       suitable for GUI updates or progress bars.
 *
 * @example
 * void progress(void *h, const char *msg) { printf("PROGRESS: %s\n", msg); }
 * PRIMAL_setprogresscb(task, progress, NULL);
 */
PRIMALrescodee PRIMAL_setprogresscb(PRIMALtask_t t, PRIMALprogresscb cb, void *handle) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    t->progcb = cb;
    t->proghandle = handle;
    return PRIMAL_RES_OK;
}

/**
 * Sets the info name prefix for progress callback messages.
 *
 * @param t        [in] Task handle.
 * @param connname [in] String to prefix progress messages with (e.g., "SOLVER").
 *                       If NULL, clears the prefix. Maximum 63 characters.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note When a progress callback is set (PRIMAL_setprogresscb), each message
 *       will be formatted as "<connname>: <message>" if connname is non-empty.
 *       This helps identify which solver instance produced the message when
 *       multiple tasks share a callback.
 *
 * @example
 * PRIMAL_setinfoconnname(task, "PRIMAL");
 * // Progress messages will now be: "PRIMAL: IPM iter 5: ..."
 */
PRIMALrescodee PRIMAL_setinfoconnname(PRIMALtask_t t, const char *connname) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (connname) {
        size_t k = 0;
        for (; k < sizeof t->infoname - 1 && connname[k]; k++) t->infoname[k] = connname[k];
        t->infoname[k] = '\0';
    } else t->infoname[0] = '\0';
    return PRIMAL_RES_OK;
}

/**
 * Links a callback function to a task's output stream.
 *
 * @param t     [in] Task handle.
 * @param which [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param handle [in] User-defined pointer passed to func on each write.
 * @param func   [in] Callback function with signature:
 *                   void func(void *handle, const char *msg).
 *                   Called for each log message produced by the solver.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note This replaces any existing log callback or file stream on the task.
 *       The log flag is enabled automatically.
 *
 * @example
 * void my_log(void *h, const char *msg) { fputs(msg, stderr); }
 * PRIMAL_linkfunctotaskstream(task, PRIMAL_STREAM_LOG, NULL, my_log);
 */
PRIMALrescodee PRIMAL_linkfunctotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                      void *handle, PRIMALstreamfunc func) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    t->logcb = func;
    t->loghandle = handle;
    t->log = 1;
    return PRIMAL_RES_OK;
}

/* ---- file streams, env streams, echo (reference linkfiletotaskstream,
 * linkfiletoenvstream, linkfunctoenvstream, unlinkfuncfrom*stream, echo*) ----
 * A file stream is a callback writing to a FILE*; the FILE* is owned by
 * whoever opened it (task or env) and closed on its destruction. The env
 * stream is inherited by tasks created afterwards. */
/* File-backed log callback: appends the message to the open FILE*. */
static void file_stream_cb(void *handle, const char *msg) {
    FILE *f = (FILE *)handle;
    if (f && msg) fputs(msg, f);
}

/**
 * Links a file to a task's output stream.
 *
 * @param t        [in] Task handle.
 * @param which    [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param filename [in] Path to the file to open. Must not be NULL.
 * @param append   [in] If non-zero, open in append mode ("a"); otherwise truncate ("w").
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or filename is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG,
 *         PRIMAL_RES_ERR_FILE if the file cannot be opened.
 *
 * @note The file is owned by the task and will be closed when the task is
 *       deleted or when the stream is unlinked. Any existing file stream is
 *       closed first. The log flag is enabled automatically.
 *
 * @example
 * PRIMAL_linkfiletotaskstream(task, PRIMAL_STREAM_LOG, "solver.log", 0);
 * PRIMAL_optimize(task); // Log goes to solver.log
 */
PRIMALrescodee PRIMAL_linkfiletotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                            const char *filename, int append) {
    if (!t || !filename) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, append ? "a" : "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    if (t->logfile) fclose(t->logfile);
    t->logfile = f;
    t->logcb = file_stream_cb;
    t->loghandle = f;
    t->log = 1;
    return PRIMAL_RES_OK;
}

/**
 * Unlinks the callback/file from a task's output stream.
 *
 * @param t     [in] Task handle.
 * @param which [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note Closes the file stream (if any), clears the callback, and disables
 *       the log flag. Subsequent log messages will be discarded.
 *
 * @example
 * PRIMAL_unlinkfuncfromtaskstream(task, PRIMAL_STREAM_LOG);
 */
PRIMALrescodee PRIMAL_unlinkfuncfromtaskstream(PRIMALtask_t t, PRIMALstreamtypee which) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    if (t->logfile) { fclose(t->logfile); t->logfile = NULL; }
    t->logcb = NULL; t->loghandle = NULL; t->log = 0;
    return PRIMAL_RES_OK;
}

/**
 * Writes a formatted message to a task's log stream.
 *
 * @param t      [in] Task handle.
 * @param which  [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param format [in] printf-style format string.
 * @param ...    [in] Arguments for the format string.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or format is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note The message is passed to the task's log callback (if set via
 *       linkfunctotaskstream or linkfiletotaskstream). If no log callback
 *       is set, the message is discarded.
 *
 * @example
 * PRIMAL_echotask(task, PRIMAL_STREAM_LOG, "Starting optimization at %s\n", timestamp);
 */
PRIMALrescodee PRIMAL_echotask(PRIMALtask_t t, PRIMALstreamtypee which, const char *format, ...) {
    if (!t || !format) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    char buf[1024];
    va_list ap; va_start(ap, format);
    vsnprintf(buf, sizeof buf, format, ap);
    va_end(ap);
    tlog(t, buf);
    return PRIMAL_RES_OK;
}

/**
 * Links a file to an environment's output stream.
 *
 * @param env      [in] Environment handle.
 * @param which    [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param filename [in] Path to the file to open. Must not be NULL.
 * @param append   [in] If non-zero, open in append mode ("a"); otherwise truncate ("w").
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if env or filename is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG,
 *         PRIMAL_RES_ERR_FILE if the file cannot be opened.
 *
 * @note The file is owned by the environment and will be closed when the
 *       environment is deleted or the stream is unlinked. Any existing file
 *       stream is closed first. Tasks created AFTER this call inherit this
 *       stream as their initial log stream.
 *
 * @example
 * PRIMAL_linkfiletoenvstream(env, PRIMAL_STREAM_LOG, "global.log", 1);
 * // All future tasks will log to global.log by default
 */
PRIMALrescodee PRIMAL_linkfiletoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                           const char *filename, int append) {
    if (!env || !filename) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    FILE *f = fopen(filename, append ? "a" : "w");
    if (!f) return PRIMAL_RES_ERR_FILE;
    if (env->streamfile) fclose(env->streamfile);
    env->streamfile = f;
    env->streamfunc = file_stream_cb;
    env->streamhandle = f;
    return PRIMAL_RES_OK;
}

/**
 * Links a callback function to an environment's output stream.
 *
 * @param env    [in] Environment handle.
 * @param which  [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param handle [in] User-defined pointer passed to func on each write.
 * @param func   [in] Callback function with signature:
 *                   void func(void *handle, const char *msg).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if env is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note Replaces any existing file stream or callback on the environment.
 *       Tasks created after this call will inherit this callback.
 *
 * @example
 * void my_log(void *h, const char *msg) { fprintf(stderr, "[ENV] %s", msg); }
 * PRIMAL_linkfunctoenvstream(env, PRIMAL_STREAM_LOG, NULL, my_log);
 */
PRIMALrescodee PRIMAL_linkfunctoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                           void *handle, PRIMALstreamfunc func) {
    if (!env) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    if (env->streamfile) { fclose(env->streamfile); env->streamfile = NULL; }
    env->streamfunc = func;
    env->streamhandle = handle;
    return PRIMAL_RES_OK;
}

/**
 * Unlinks the callback/file from an environment's output stream.
 *
 * @param env   [in] Environment handle.
 * @param which [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if env is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note Closes the file stream (if any) and clears the callback.
 *       Does NOT affect tasks already created from this environment.
 *
 * @example
 * PRIMAL_unlinkfuncfromenvstream(env, PRIMAL_STREAM_LOG);
 */
PRIMALrescodee PRIMAL_unlinkfuncfromenvstream(PRIMALenv_t env, PRIMALstreamtypee which) {
    if (!env) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    if (env->streamfile) { fclose(env->streamfile); env->streamfile = NULL; }
    env->streamfunc = NULL;
    env->streamhandle = NULL;
    return PRIMAL_RES_OK;
}

/**
 * Writes a formatted message to an environment's log stream.
 *
 * @param env    [in] Environment handle.
 * @param which  [in] Stream type. Currently only PRIMAL_STREAM_LOG is supported.
 * @param format [in] printf-style format string.
 * @param ...    [in] Arguments for the format string.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if env or format is NULL,
 *         PRIMAL_RES_ERR_ARG if which is not PRIMAL_STREAM_LOG.
 *
 * @note The message is passed to the environment's log callback (if set via
 *       linkfunctoenvstream or linkfiletoenvstream). If no log callback
 *       is set, the message is discarded.
 *
 * @example
 * PRIMAL_echoenv(env, PRIMAL_STREAM_LOG, "Starting batch of %d solves\n", n);
 */
PRIMALrescodee PRIMAL_echoenv(PRIMALenv_t env, PRIMALstreamtypee which, const char *format, ...) {
    if (!env || !format) return PRIMAL_RES_ERR_NULL;
    if (which != PRIMAL_STREAM_LOG) return PRIMAL_RES_ERR_ARG;
    char buf[1024];
    va_list ap; va_start(ap, format);
    vsnprintf(buf, sizeof buf, format, ap);
    va_end(ap);
    if (env->streamfunc) env->streamfunc(env->streamhandle, buf);
    return PRIMAL_RES_OK;
}

/**
 * Writes an introductory banner to an environment's log stream.
 *
 * @param env     [in] Environment handle.
 * @param longver [in] If non-zero, include detailed version/build information;
 *                     if zero, write a short one-line banner.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if env is NULL.
 *
 * @note This is typically called at the start of a solving session to identify
 *       the solver version in the log output. The message is written to the
 *       environment's log stream (if configured).
 *
 * @example
 * PRIMAL_echointro(env, 1); // Writes: "PrimalSolver (longver=1)\n"
 */
PRIMALrescodee PRIMAL_echointro(PRIMALenv_t env, int longver) {
    if (!env) return PRIMAL_RES_ERR_NULL;
    char buf[256];
    snprintf(buf, sizeof buf, "PrimalSolver (longver=%d)\n", longver);
    if (env->streamfunc) env->streamfunc(env->streamhandle, buf);
    return PRIMAL_RES_OK;
}

/**
 * Sets a legacy log callback on a task (deprecated, use linkfunctotaskstream).
 *
 * @param t       [in] Task handle.
 * @param logcb   [in] Callback function with signature:
 *                     void logcb(void *handle, const char *msg).
 * @param loghandle [in] User-defined pointer passed to logcb.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note This is a legacy API maintained for compatibility. It sets the same
 *       internal callback as PRIMAL_linkfunctotaskstream but does not enable
 *       the log flag automatically. Prefer linkfunctotaskstream for new code.
 *
 * @example
 * PRIMAL_setlogcb(task, my_log, NULL);
 */
PRIMALrescodee PRIMAL_setlogcb(PRIMALtask_t t, PRIMALlogcb logcb, void *loghandle) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    t->logcb = logcb;
    t->loghandle = loghandle;
    return PRIMAL_RES_OK;
}

/**
 * Registers a general callback function to receive solver events.
 *
 * @param t      [in] Task handle.
 * @param cb     [in] Callback function with signature:
 *                   void cb(PRIMALtask_t task, void *handle,
 *                           PRIMALcallbackcodee code,
 *                           const double *dinf, const int *iinf, const PRIMALint64t *linf).
 *                   The dinf, iinf, linf arrays are currently NULL (declared deviation:
 *                   the reference populates these with solver statistics).
 * @param handle [in] User-defined pointer passed to cb on each event.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note The callback is invoked at the following events (PRIMALcallbackcodee):
 *       - PRIMAL_CALLBACK_BEGIN_OPTIMIZER: optimization started
 *       - PRIMAL_CALLBACK_END_OPTIMIZER: optimization finished
 *       - PRIMAL_CALLBACK_BEGIN_READ: reading model file started
 *       - PRIMAL_CALLBACK_END_READ: reading model file finished
 *       - PRIMAL_CALLBACK_BEGIN_WRITE: writing model file started
 *       - PRIMAL_CALLBACK_END_WRITE: writing model file finished
 *
 *       The detail vectors (dinf, iinf, linf) are reserved for future use and
 *       are currently passed as NULL. This is a known deviation from the reference.
 *
 * @example
 * void my_callback(PRIMALtask_t t, void *h, PRIMALcallbackcodee code,
 *                  const double *d, const int *i, const PRIMALint64t *l) {
 *     if (code == PRIMAL_CALLBACK_END_OPTIMIZER) printf("Solve finished\n");
 * }
 * PRIMAL_putcallbackfunc(task, my_callback, NULL);
 */
PRIMALrescodee PRIMAL_putcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb cb, void *handle) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    t->cbfn = cb;
    t->cbhandle = handle;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the currently registered general callback function and handle.
 *
 * @param t      [in]  Task handle.
 * @param cb     [out] Pointer to receive the callback function pointer. Must not be NULL.
 * @param handle [out] Optional pointer to receive the user handle (may be NULL).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or cb is NULL.
 *
 * @example
 * PRIMALcallbackcb cb; void *h;
 * PRIMAL_getcallbackfunc(task, &cb, &h);
 * if (cb) cb(task, h, PRIMAL_CALLBACK_END_OPTIMIZER, NULL, NULL, NULL);
 */
PRIMALrescodee PRIMAL_getcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb *cb, void **handle) {
    if (!t || !cb) return PRIMAL_RES_ERR_NULL;
    *cb = t->cbfn;
    if (handle) *handle = t->cbhandle;
    return PRIMAL_RES_OK;
}

/**
 * Registers a response callback to receive error/warning notifications.
 *
 * @param t      [in] Task handle.
 * @param cb     [in] Callback function with signature:
 *                   void cb(void *handle, PRIMALrescodee code, const char *msg).
 *                   Called when the solver encounters an error or warning condition.
 * @param handle [in] User-defined pointer passed to cb.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note The response callback is invoked for error conditions that don't cause
 *       immediate termination (e.g., invalid parameter values, file read warnings).
 *       The code parameter is the error code, and msg is a human-readable message.
 *
 * @example
 * void my_response(void *h, PRIMALrescodee code, const char *msg) {
 *     fprintf(stderr, "Solver response: code=%d msg=%s\n", code, msg);
 * }
 * PRIMAL_putresponsefunc(task, my_response, NULL);
 */
PRIMALrescodee PRIMAL_putresponsefunc(PRIMALtask_t t, PRIMALresponsecb cb, void *handle) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    t->respfn = cb;
    t->resphandle = handle;
    return PRIMAL_RES_OK;
}

/* Fire the general callback with an event code; the detail vectors are not
 * populated (declared deviation, see primal.h). */
void cb_fire(PRIMALtask_t t, PRIMALcallbackcodee code) {
    if (t && t->cbfn) t->cbfn(t, t->cbhandle, code, NULL, NULL, NULL);
}
/* Internal hook so the I/O readers in mpsio.c (which cannot see cb_fire) can
 * emit the OPF data callbacks (READ_OPF/READ_OPF_SECTION/WRITE_OPF). Not public
 * API: declared in mpsio.h only. */
void primal_cb_notify(PRIMALtask_t t, int code) {
    cb_fire(t, (PRIMALcallbackcodee)code);
}
/* Per-iteration callback hook for the engine loops (ipm.c/simplex.c/socp.c/
 * sdp.c), which do not receive the task.  A flag keeps the common case (no
 * callback) a plain load+branch, so the hot loops are not perturbed.  The
 * engines declare `extern int primal_cb_iter_on; void primal_cb_iter(int);`.
 * Deviation: with PRIMAL_IPAR_NUM_THREADS > 1 the concurrent LP path runs two
 * engines in parallel and the per-iteration events interleave. */
static PRIMALtask_t g_iter_task = NULL;
int primal_cb_iter_on = 0;
/* Forward a per-iteration event code to the registered task callback. */
void primal_cb_iter(int code) {
    if (g_iter_task && g_iter_task->cbfn)
        g_iter_task->cbfn(g_iter_task, g_iter_task->cbhandle,
                          (PRIMALcallbackcodee)code, NULL, NULL, NULL);
}
/* Arm the per-iteration hook for task t for the duration of a solve. */
void iter_cb_begin(PRIMALtask_t t) { g_iter_task = t; primal_cb_iter_on = (t && t->cbfn) ? 1 : 0; }
/* Disarm the per-iteration hook after the solve finishes. */
void iter_cb_end(void) { primal_cb_iter_on = 0; g_iter_task = NULL; }

/* Allocate the per-variable and per-constraint arrays for the current
 * model size, with free bounds and empty columns; no-op when present. */
PRIMALrescodee ensure_size(PRIMALtask_t t) {
    if (t->numvar > 0 && !t->c) {
        t->c = (double *)calloc((size_t)t->numvar, sizeof(double));
        t->bkx = (PRIMALboundkeye *)calloc((size_t)t->numvar, sizeof(PRIMALboundkeye));
        t->blx = (double *)calloc((size_t)t->numvar, sizeof(double));
        t->bux = (double *)calloc((size_t)t->numvar, sizeof(double));
        t->cols = (Col *)calloc((size_t)t->numvar, sizeof(Col));
        t->vartype = (PRIMALvariabletypee *)calloc((size_t)t->numvar, sizeof(PRIMALvariabletypee));
        t->varname = (char **)calloc((size_t)t->numvar, sizeof(char *));
        if (!t->c || !t->bkx || !t->blx || !t->bux || !t->cols || !t->vartype ||
            !t->varname) return PRIMAL_RES_ERR_ALLOC;
        for (int j = 0; j < t->numvar; j++) { t->bkx[j] = PRIMAL_BK_FR; t->blx[j] = -INF; t->bux[j] = INF; }
    }
    if (t->numcon > 0 && !t->bkc) {
        t->bkc = (PRIMALboundkeye *)calloc((size_t)t->numcon, sizeof(PRIMALboundkeye));
        t->blc = (double *)calloc((size_t)t->numcon, sizeof(double));
        t->buc = (double *)calloc((size_t)t->numcon, sizeof(double));
        t->conname = (char **)calloc((size_t)t->numcon, sizeof(char *));
        if (!t->bkc || !t->blc || !t->buc || !t->conname) return PRIMAL_RES_ERR_ALLOC;
        for (int i = 0; i < t->numcon; i++) { t->bkc[i] = PRIMAL_BK_FR; t->blc[i] = -INF; t->buc[i] = INF; }
    }
    return PRIMAL_RES_OK;
}

/* An answer is an answer ABOUT the model that was solved. When the model
 * changes shape — a vector the answer lives in no longer has its length —
 * there is no point to deliver and no verdict to read: the solve that produced
 * them never saw this model. This is exactly the state opt_prepare puts the
 * task in before a solve, and every getter of the point is guarded by has_sol
 * (T99), which is what turns the clearing into the fix rather than a polite
 * refusal to read memory that cannot hold the answer. */
void model_resized(PRIMALtask_t t) {
    t->has_sol = 0;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    t->has_pray = 0; t->has_dray = 0;
    t->pobj = 0.0; t->dobj = 0.0;
    t->mip_bound_defined = 0; t->mip_result = 0;
}

/* One realloc for the lazy per-variable / per-constraint tables. The new tail
 * is zeroed and zero is the value that means something there: 0.0 is "no
 * opinion" for a warm start (the incumbent still has to measure), and
 * PRIMAL_SK_UNDEF == 0 is "never set" for a basis status, which is exactly the
 * state of a coordinate that did not exist when the basis was written.
 * Returns the block, possibly moved, or NULL on allocation failure: realloc
 * leaves the original live and *cap unchanged, so the caller can refuse before
 * the model has moved. */
void *lazy_grow(void *tbl, int *cap, int need, size_t esz) {
    if (need <= *cap) return tbl;
    void *p = realloc(tbl, (size_t)need * esz);
    if (!p) return NULL;
    memset((char *)p + (size_t)(*cap) * esz, 0, (size_t)(need - *cap) * esz);
    *cap = need;
    return p;
}

/* The quadratic-constraint blocks are numvar x numvar dense, stored at the
 * stride the model had when the block was allocated. Growing the model changes
 * the stride, so every allocated block is rebuilt at the new one — row-wise,
 * because a row of Q is the same row of the operator under either stride, and
 * the new rows and columns are zero because no term touches them yet.
 * Call while t->numvar is still the old stride (on), before the model moves.
 * Transactional: all the new blocks are allocated and filled before any old
 * one is freed, so a failure leaves the store exactly as it was. */
static PRIMALrescodee qcon_reshape(PRIMALtask_t t, int on, int nn) {
    if (!t->qcon || t->qcon_cap == 0 || on == nn) return PRIMAL_RES_OK;
    int nb = 0;
    for (int k = 0; k < t->qcon_cap; k++) if (t->qcon[k]) nb++;
    if (nb == 0) return PRIMAL_RES_OK;
    double **fresh = (double **)calloc((size_t)t->qcon_cap, sizeof(double *));
    if (!fresh) return PRIMAL_RES_ERR_ALLOC;
    for (int k = 0; k < t->qcon_cap; k++) {
        if (!t->qcon[k]) continue;
        double *blk = (double *)calloc((size_t)nn * (size_t)nn, sizeof(double));
        if (!blk) {
            for (int q = 0; q < k; q++) free(fresh[q]);
            free(fresh);
            return PRIMAL_RES_ERR_ALLOC;
        }
        for (int i = 0; i < on; i++)
            memcpy(blk + (size_t)i * nn, t->qcon[k] + (size_t)i * on,
                   (size_t)on * sizeof(double));
        fresh[k] = blk;
    }
    for (int k = 0; k < t->qcon_cap; k++) {
        if (!fresh[k]) continue;
        free(t->qcon[k]);
        t->qcon[k] = fresh[k];
    }
    free(fresh);
    return PRIMAL_RES_OK;
}

/**
 * Appends variables (columns) to the task.
 *
 * @param t   [in] Task handle.
 * @param num [in] Number of variables to append. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note This function grows all per-variable arrays (objective coefficients,
 *       bounds, variable types, names, column storage for A, warm start,
 *       basis status keys, quadratic objective cache). New variables are
 *       initialized as:
 *       - Free bounds (PRIMAL_BK_FR, -INF to +INF)
 *       - Zero objective coefficient
 *       - Continuous type (PRIMAL_VAR_TYPE_CONT)
 *       - Unnamed
 *       - Empty column in A
 *
 *       If the task already has constraints, the quadratic constraint blocks
 *       (qcon) are reshaped to the new variable count. This is done
 *       transactionally: all new blocks are allocated and filled before any
 *       old block is freed, so an allocation failure leaves the model unchanged.
 *
 *       IMPORTANT: If you created the task with PRIMAL_maketask(env, maxcon, maxvar),
 *       the variables are already pre-allocated. Do NOT call appendvars again
 *       with the same maxvar, or you will double the allocation.
 *
 *       The lazy tables (warm_x, skx) are grown via lazy_grow, which preserves
 *       existing data and zeroes new entries. Zero is the sentinel value meaning
 *       "no opinion" for warm start and "never set" for basis status.
 *
 * @warning Calling this function after a solve invalidates any existing solution
 *          (has_sol is cleared). The next solve will recompute from scratch.
 *
 * @example
 * PRIMALtask_t task;
 * PRIMAL_maketask(env, 0, 0, &task); // Empty task
 * PRIMAL_appendvars(task, 10); // Add 10 free continuous variables
 * // Variables 0-9 now exist with default bounds
 */
PRIMALrescodee PRIMAL_appendvars(PRIMALtask_t t, int num) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX-t->numvar) return PRIMAL_RES_ERR_ARG;
    int nv = t->numvar;
    int nn = nv + num;
    if (num == 0) return PRIMAL_RES_OK;
    /* Grow the lazy tables and the quadratic blocks to the NEW length before
     * the model moves: if an allocation fails here, numvar is still the old one
     * and nothing downstream can read a table shorter than the model. */
    if (num > 0) {
        if (t->warm_x) {
            double *w = (double *)lazy_grow(t->warm_x, &t->warmxcap, nn, sizeof(double));
            if (!w) return PRIMAL_RES_ERR_ALLOC;
            t->warm_x = w;
        }
        if (t->skx) {
            PRIMALstakeye *s = (PRIMALstakeye *)lazy_grow(t->skx, &t->skxcap, nn, sizeof(PRIMALstakeye));
            if (!s) return PRIMAL_RES_ERR_ALLOC;
            t->skx = s;
        }
    }
    {
        double *c2 = (double *)calloc((size_t)nn, sizeof(double));
        PRIMALboundkeye *k2 = (PRIMALboundkeye *)calloc((size_t)nn, sizeof(PRIMALboundkeye));
        double *l2 = (double *)calloc((size_t)nn, sizeof(double));
        double *u2 = (double *)calloc((size_t)nn, sizeof(double));
        Col *co2 = (Col *)calloc((size_t)nn, sizeof(Col));
        PRIMALvariabletypee *vt2 = (PRIMALvariabletypee *)calloc((size_t)nn, sizeof(PRIMALvariabletypee));
        char **nm2 = (char **)calloc((size_t)nn, sizeof(char *));
        if (!c2 || !k2 || !l2 || !u2 || !co2 || !vt2 || !nm2) {
            free(c2); free(k2); free(l2); free(u2); free(co2); free(vt2); free(nm2);
            return PRIMAL_RES_ERR_ALLOC;
        }
        PRIMALrescodee rrc = qcon_reshape(t,nv,nn);
        if (rrc != PRIMAL_RES_OK) {
            free(c2); free(k2); free(l2); free(u2); free(co2); free(vt2); free(nm2);
            return rrc;
        }
        if (nv) memcpy(c2, t->c, (size_t)nv * sizeof(double));
        if (nv) memcpy(k2, t->bkx, (size_t)nv * sizeof(PRIMALboundkeye));
        if (nv) memcpy(l2, t->blx, (size_t)nv * sizeof(double));
        if (nv) memcpy(u2, t->bux, (size_t)nv * sizeof(double));
        if (nv) memcpy(co2, t->cols, (size_t)nv * sizeof(Col));
        if (nv) memcpy(vt2, t->vartype, (size_t)nv * sizeof(PRIMALvariabletypee));
        if (t->varname) memcpy(nm2, t->varname, (size_t)nv * sizeof(char *));
        for (int j = 0; j < nv; j++) { c2[j] = t->c[j]; }
        for (int j = nv; j < nn; j++) { k2[j] = PRIMAL_BK_FR; l2[j] = -INF; u2[j] = INF; }
        model_changed(t);
        free(t->c); free(t->bkx); free(t->blx); free(t->bux); free(t->cols); free(t->vartype);
        free(t->varname);
        t->c = c2; t->bkx = k2; t->blx = l2; t->bux = u2; t->cols = co2; t->vartype = vt2;
        t->varname = nm2;
    }
    t->numvar = nn;
    PRIMALrescodee rc = ensure_size(t);
    if (rc != PRIMAL_RES_OK) return rc;
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/**
 * Appends constraints (rows) to the task.
 *
 * @param t   [in] Task handle.
 * @param num [in] Number of constraints to append. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0,
 *         PRIMAL_RES_ERR_ALLOC if memory allocation fails.
 *
 * @note This function grows all per-constraint arrays (bounds, names,
 *       quadratic constraint row-pointer array, warm start, basis status keys).
 *       New constraints are initialized as:
 *       - Free bounds (PRIMAL_BK_FR, -INF to +INF)
 *       - Unnamed
 *       - No quadratic terms (qcon row pointer is NULL, block allocated on first use)
 *
 *       If the task already has variables, the quadratic constraint row-pointer
 *       array (qcon) is grown to the new constraint count. New rows start as
 *       NULL; a dense numvar x numvar block is allocated lazily when the first
 *       quadratic term is added to that row via PRIMAL_putqconk.
 *
 *       IMPORTANT: If you created the task with PRIMAL_maketask(env, maxcon, maxvar),
 *       the constraints are already pre-allocated. Do NOT call appendcons again
 *       with the same maxcon, or you will double the allocation.
 *
 *       The lazy tables (warm_y, skc) are grown via lazy_grow, preserving
 *       existing data and zeroing new entries. Zero means "no opinion" for
 *       warm start and "never set" for basis status.
 *
 * @warning Calling this function after a solve invalidates any existing solution
 *          (has_sol is cleared). The next solve will recompute from scratch.
 *
 * @example
 * PRIMALtask_t task;
 * PRIMAL_maketask(env, 0, 0, &task);
 * PRIMAL_appendvars(task, 5);
 * PRIMAL_appendcons(task, 3); // Add 3 free constraints
 * // Constraints 0-2 now exist with default bounds
 */
PRIMALrescodee PRIMAL_appendcons(PRIMALtask_t t, int num) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || num > INT_MAX-t->numcon) return PRIMAL_RES_ERR_ARG;
    int nc = t->numcon, nn = nc + num;
    if (num == 0) return PRIMAL_RES_OK;
    if (num > 0) {
        if (t->warm_y) {
            double *w = (double *)lazy_grow(t->warm_y, &t->warmycap, nn, sizeof(double));
            if (!w) return PRIMAL_RES_ERR_ALLOC;
            t->warm_y = w;
        }
        if (t->skc) {
            PRIMALstakeye *s = (PRIMALstakeye *)lazy_grow(t->skc, &t->skccap, nn, sizeof(PRIMALstakeye));
            if (!s) return PRIMAL_RES_ERR_ALLOC;
            t->skc = s;
        }
        /* The row-pointer array is indexed by constraint: PRIMAL_putqconk
         * allocates it at the numcon of the first call, and PRIMAL_deletetask
         * frees one entry per constraint, so it has to reach the new numcon
         * before anything else does. New rows hold NULL: a block is allocated
         * when a term lands on that row. */
        if (t->qcon) {
            double **q = (double **)lazy_grow(t->qcon, &t->qcon_cap, nn, sizeof(double *));
            if (!q) return PRIMAL_RES_ERR_ALLOC;
            t->qcon = q;
        }
    }
    {
        PRIMALboundkeye *k2 = (PRIMALboundkeye *)calloc((size_t)nn, sizeof(PRIMALboundkeye));
        double *l2 = (double *)calloc((size_t)nn, sizeof(double));
        double *u2 = (double *)calloc((size_t)nn, sizeof(double));
        char **nm2 = (char **)calloc((size_t)nn, sizeof(char *));
        if (!k2 || !l2 || !u2 || !nm2) { free(k2); free(l2); free(u2); free(nm2); return PRIMAL_RES_ERR_ALLOC; }
        if (nc) memcpy(k2, t->bkc, (size_t)nc * sizeof(PRIMALboundkeye));
        if (nc) memcpy(l2, t->blc, (size_t)nc * sizeof(double));
        if (nc) memcpy(u2, t->buc, (size_t)nc * sizeof(double));
        if (t->conname) memcpy(nm2, t->conname, (size_t)nc * sizeof(char *));
        for (int i = nc; i < nn; i++) { k2[i] = PRIMAL_BK_FR; l2[i] = -INF; u2[i] = INF; }
        model_changed(t);
        free(t->bkc); free(t->blc); free(t->buc);
        free(t->conname);
        t->bkc = k2; t->blc = l2; t->buc = u2;
        t->conname = nm2;
    }
    t->numcon = nn;
    PRIMALrescodee rc = ensure_size(t);
    if (rc != PRIMAL_RES_OK) return rc;
    if (num > 0) model_resized(t);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the current number of variables in the task.
 *
 * @param t      [in]  Task handle.
 * @param numvar [out] Pointer to an int that will receive the variable count. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or numvar is NULL.
 *
 * @example
 * int n;
 * PRIMAL_getnumvar(task, &n);
 * printf("Model has %d variables\n", n);
 */
PRIMALrescodee PRIMAL_getnumvar(PRIMALtask_t t, int *numvar) {
    if (!t || !numvar) return PRIMAL_RES_ERR_NULL;
    *numvar = t->numvar; return PRIMAL_RES_OK;
}
/**
 * Retrieves the current number of constraints in the task.
 *
 * @param t     [in]  Task handle.
 * @param numcon [out] Pointer to an int that will receive the constraint count. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or numcon is NULL.
 *
 * @example
 * int m;
 * PRIMAL_getnumcon(task, &m);
 * printf("Model has %d constraints\n", m);
 */
PRIMALrescodee PRIMAL_getnumcon(PRIMALtask_t t, int *numcon) {
    if (!t || !numcon) return PRIMAL_RES_ERR_NULL;
    *numcon = t->numcon; return PRIMAL_RES_OK;
}
/* Preallocated counts (reference: getmaxnumvar/getmaxnumcon/getmaxnumcone/
 * getmaxnumbarvar). In the reference these say how much room is reserved before
 * a reallocation; this solver grows its arrays in place, so the number reserved
 * IS the current one -- a declared deviation, not a wrong answer. The conic and
 * bar capacities are real (cone_cap/barcap), so those two are exact. */
/**
 * Retrieves the maximum number of variables the task can hold without reallocation.
 *
 * @param t [in]  Task handle.
 * @param n [out] Pointer to an int that will receive the capacity. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t or n is NULL.
 *
 * @note In this solver, arrays grow in place as needed, so the reserved capacity
 *       equals the current size (numvar). This is a declared deviation from the
 *       reference, where getmaxnumvar reports the pre-allocated capacity which
 *       may be larger than the current numvar. For cones and bar variables,
 *       the capacities (cone_cap, barcap) are real and distinct from the counts.
 *
 * @example
 * int cap;
 * PRIMAL_getmaxnumvar(task, &cap);
 * printf("Variable capacity: %d (current: %d)\n", cap, task->numvar);
 */
PRIMALrescodee PRIMAL_getmaxnumvar(PRIMALtask_t t, int *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    *n = t->numvar; return PRIMAL_RES_OK;
}
/**
 * Retrieves the maximum number of constraints the task can hold without reallocation.
 *
 * @param t [in]  Task handle.
 * @param n [out] Pointer to an int that will receive the capacity. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t or n is NULL.
 *
 * @note In this solver, arrays grow in place as needed, so the reserved capacity
 *       equals the current size (numcon). This is a declared deviation from the
 *       reference. For cones and bar variables, the capacities are real.
 */
PRIMALrescodee PRIMAL_getmaxnumcon(PRIMALtask_t t, int *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    *n = t->numcon; return PRIMAL_RES_OK;
}
/**
 * Retrieves the capacity of the cone table (maximum number of cones).
 *
 * @param t [in]  Task handle.
 * @param n [out] Pointer to an int that will receive the cone capacity. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t or n is NULL.
 *
 * @note Unlike variables and constraints, the cone capacity (cone_cap) is a
 *       separate allocation limit. The actual number of cones is numcones.
 *       This capacity is the exact pre-allocated size of the cone arrays.
 *
 * @example
 * int cap;
 * PRIMAL_getmaxnumcone(task, &cap);
 * printf("Cone capacity: %d (current: %d)\n", cap, task->numcones);
 */
PRIMALrescodee PRIMAL_getmaxnumcone(PRIMALtask_t t, int *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    *n = t->cone_cap; return PRIMAL_RES_OK;
}
/**
 * Retrieves the capacity of the bar variable table (maximum number of bar variables).
 *
 * @param t [in]  Task handle.
 * @param n [out] Pointer to an int that will receive the bar variable capacity. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t or n is NULL.
 *
 * @note The bar capacity (barcap) is a separate allocation limit for symmetric
 *       matrix variables (PSD variables). The actual number is numbarvar.
 *       This capacity is shared with the bar variable name table.
 *
 * @example
 * int cap;
 * PRIMAL_getmaxnumbarvar(task, &cap);
 * printf("Bar variable capacity: %d (current: %d)\n", cap, task->numbarvar);
 */
PRIMALrescodee PRIMAL_getmaxnumbarvar(PRIMALtask_t t, int *n) {
    if (!t || !n) return PRIMAL_RES_ERR_NULL;
    *n = t->barcap; return PRIMAL_RES_OK;
}
