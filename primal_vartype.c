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
/* primal_vartype.c - variable types, SOS, int/double parameter setters.
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* ---------------- variable types (MIP) ---------------- */
/**
 * Validates a variable type enum value.
 *
 * @param vt [in] Variable type to validate.
 * @return 1 if valid, 0 otherwise.
 *
 * @note Valid types are: CONT (continuous), INT (general integer),
 *       INT_BIN (binary), SEMI_CONT (semi-continuous), SEMI_INT (semi-integer).
 *       Used by putvartype and putvartypelist.
 */
static int vartype_ok(PRIMALvariabletypee vt) {
    return vt == PRIMAL_VAR_TYPE_CONT || vt == PRIMAL_VAR_TYPE_INT ||
           vt == PRIMAL_VAR_TYPE_INT_BIN || vt == PRIMAL_VAR_TYPE_SEMI_CONT ||
           vt == PRIMAL_VAR_TYPE_SEMI_INT;
}

/**
 * Sets the variable type for a single variable (MIP).
 *
 * @param t  [in] Task handle.
 * @param j  [in] Variable index, 0 <= j < numvar.
 * @param vt [in] Variable type:
 *                - PRIMAL_VAR_TYPE_CONT: continuous (default)
 *                - PRIMAL_VAR_TYPE_INT: general integer
 *                - PRIMAL_VAR_TYPE_INT_BIN: binary (0 or 1)
 *                - PRIMAL_VAR_TYPE_SEMI_CONT: semi-continuous (0 or [l,u])
 *                - PRIMAL_VAR_TYPE_SEMI_INT: semi-integer (0 or {l..u})
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds or vt invalid.
 *
 * @note Semi-continuous/semi-integer variables have domain {0} U [l,u] where
 *       l, u are the variable bounds. The solver handles these specially in
 *       the MIP branch-and-bound (branch on x=0 vs x in [l,u]).
 *
 * @example
 * // Make x_5 binary
 * PRIMAL_putvartype(task, 5, PRIMAL_VAR_TYPE_INT_BIN);
 * // Make x_3 semi-continuous
 * PRIMAL_putvartype(task, 3, PRIMAL_VAR_TYPE_SEMI_CONT);
 */
PRIMALrescodee PRIMAL_putvartype(PRIMALtask_t t, int j, PRIMALvariabletypee vt) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    if (!vartype_ok(vt)) return PRIMAL_RES_ERR_ARG;
    t->vartype[j] = vt;
    return PRIMAL_RES_OK;
}

/**
 * Sets variable types for a list of variables (atomic validation).
 *
 * @param t       [in] Task handle.
 * @param num     [in] Number of variables. Must be non-negative.
 * @param subj    [in] Array of variable indices (length num). Each must satisfy
 *                    0 <= subj[k] < numvar.
 * @param vartype [in] Array of variable types (length num).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or required arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0, any index out of bounds, or any type invalid.
 *
 * @note The ENTIRE list is validated first before any modification. If any
 *       entry is invalid, the model is unchanged.
 *
 * @example
 * int idx[] = {0, 1, 2};
 * PRIMALvariabletypee types[] = {PRIMAL_VAR_TYPE_INT_BIN, PRIMAL_VAR_TYPE_INT, PRIMAL_VAR_TYPE_CONT};
 * PRIMAL_putvartypelist(task, 3, idx, types);
 * // x_0 binary, x_1 general integer, x_2 continuous
 */
PRIMALrescodee PRIMAL_putvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, const PRIMALvariabletypee *vartype) {
    model_changed(t);
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && (!subj || !vartype))) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (subj[k] < 0 || subj[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        if (!vartype_ok(vartype[k])) return PRIMAL_RES_ERR_ARG;
    }
    for (int k = 0; k < num; k++) t->vartype[subj[k]] = vartype[k];
    return PRIMAL_RES_OK;
}

/**
 * Retrieves variable types for a list of variables.
 *
 * @param t       [in]  Task handle.
 * @param num     [in]  Number of variables. Must be non-negative.
 * @param subj    [in]  Array of variable indices (length num). Each must satisfy
 *                      0 <= subj[k] < numvar.
 * @param vartype [out] Array receiving variable types (length num). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or vartype is NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0 or any index out of bounds.
 *
 * @note Reads the vartype array for the specified indices. Does not modify
 *       the model.
 *
 * @example
 * int idx[] = {0, 1, 2};
 * PRIMALvariabletypee types[3];
 * PRIMAL_getvartypelist(task, 3, idx, types);
 * // types[0..2] now hold the variable types
 */
PRIMALrescodee PRIMAL_getvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, PRIMALvariabletypee *vartype) {
    if (!t || !vartype) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subj)) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++)
        if (subj[k] < 0 || subj[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) vartype[k] = t->vartype[subj[k]];
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the variable type for a single variable.
 *
 * @param t  [in]  Task handle.
 * @param j  [in]  Variable index, 0 <= j < numvar.
 * @param vt [out] Pointer to receive the variable type.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or vt is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds.
 *
 * @example
 * PRIMALvariabletypee vt;
 * PRIMAL_getvartype(task, 5, &vt);
 * if (vt == PRIMAL_VAR_TYPE_INT_BIN) printf("x_5 is binary\n");
 */
PRIMALrescodee PRIMAL_getvartype(PRIMALtask_t t, int j, PRIMALvariabletypee *vt) {
    if (!t || !vt) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    *vt = t->vartype[j];
    return PRIMAL_RES_OK;
}

/**
 * Counts the number of integer-constrained variables (non-continuous).
 *
 * @param t   [in]  Task handle.
 * @param num [out] Pointer to int receiving the count.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or num is NULL.
 *
 * @note Counts variables with vartype != PRIMAL_VAR_TYPE_CONT (i.e., INT,
 *       INT_BIN, SEMI_CONT, SEMI_INT). Continuous variables are not counted.
 *
 * @example
 * int nint;
 * PRIMAL_getnumintvar(task, &nint);
 * printf("Model has %d integer-constrained variables\n", nint);
 */
PRIMALrescodee PRIMAL_getnumintvar(PRIMALtask_t t, int *num) {
    if (!t || !num) return PRIMAL_RES_ERR_NULL;
    int n = 0;
    for (int j = 0; j < t->numvar; j++)
        if (t->vartype[j] != PRIMAL_VAR_TYPE_CONT) n++;
    *num = n;
    return PRIMAL_RES_OK;
}

/**
 * Internal helper: appends an SOS constraint (SOS1 or SOS2).
 *
 * @param t       [in] Task handle.
 * @param sostype [in] SOS type: 1 (SOS1) or 2 (SOS2).
 * @param num     [in] Number of members. Must be >= 1.
 * @param submem  [in] Array of variable indices (length num).
 * @param weight  [in] Array of weights (length num). Must be finite and distinct;
 *                    input order is preserved, adjacency is by increasing weight.
 *
 * @return PRIMAL_RES_OK on success, error code otherwise.
 *
 * @note SOS1: at most one variable in the set can be non-zero.
 *       SOS2: at most two adjacent variables (by weight order) can be non-zero.
 *       The weights define the adjacency order for SOS2.
 */
static PRIMALrescodee appendsos(PRIMALtask_t t, int sostype, int num,
                             const int *submem, const double *weight) {
    if (!t || !submem || !weight) return PRIMAL_RES_ERR_NULL;
    if (num < 1) return PRIMAL_RES_ERR_ARG;
    for (int k = 0; k < num; k++) {
        if (submem[k] < 0 || submem[k] >= t->numvar || !isfinite(weight[k]))
            return PRIMAL_RES_ERR_ARG;
        for (int j = 0; j < k; j++)
            if (submem[j] == submem[k] || weight[j] == weight[k])
                return PRIMAL_RES_ERR_ARG;
    }
    int *mem = (int *)malloc((size_t)num * sizeof(int));
    double *w = (double *)malloc((size_t)num * sizeof(double));
    if (!mem || !w) { free(mem); free(w); return PRIMAL_RES_ERR_ALLOC; }
    for (int k = 0; k < num; k++) { mem[k] = submem[k]; w[k] = weight[k]; }
    if (t->numsos >= t->soscap) {
        if (t->soscap > INT_MAX / 2) { free(mem); free(w); return PRIMAL_RES_ERR_ALLOC; }
        int nc = t->soscap ? t->soscap * 2 : 4;
        int *a1 = (int *)malloc((size_t)nc * sizeof(int));
        int *a2 = (int *)malloc((size_t)nc * sizeof(int));
        int **a3 = (int **)malloc((size_t)nc * sizeof(int *));
        double **a4 = (double **)malloc((size_t)nc * sizeof(double *));
        if (!a1 || !a2 || !a3 || !a4) {
            free(a1); free(a2); free(a3); free(a4); free(mem); free(w);
            return PRIMAL_RES_ERR_ALLOC;
        }
        for (int k = 0; k < t->numsos; k++) {
            a1[k] = t->sos_type[k]; a2[k] = t->sos_n[k];
            a3[k] = t->sos_mem[k]; a4[k] = t->sos_w[k];
        }
        free(t->sos_type); free(t->sos_n); free(t->sos_mem); free(t->sos_w);
        t->sos_type = a1; t->sos_n = a2; t->sos_mem = a3; t->sos_w = a4;
        t->soscap = nc;
    }
    int k = t->numsos++;
    t->sos_type[k] = sostype;
    t->sos_n[k] = num;
    t->sos_mem[k] = mem;
    t->sos_w[k] = w;
    return PRIMAL_RES_OK;
}

/**
 * Appends an SOS1 constraint (Special Ordered Set type 1).
 *
 * @param t      [in] Task handle.
 * @param num    [in] Number of members. Must be >= 1.
 * @param submem [in] Array of variable indices (length num). Each must satisfy
 *                   0 <= submem[k] < numvar.
 * @param weight [in] Array of weights (length num). Used to order variables;
 *                    not required to be unique for SOS1.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 1 or any index out of bounds.
 *
 * @note SOS1 means at most ONE variable in the set can be non-zero.
 *       The weights are used to break ties if multiple variables are at the
 *       boundary during branching. The variables must be integer-constrained
 *       (typically binary) for the SOS to be effective.
 *
 * @example
 * // SOS1 on x_0, x_1, x_2 with weights 1.0, 2.0, 3.0
 * int idx[] = {0, 1, 2};
 * double w[] = {1.0, 2.0, 3.0};
 * PRIMAL_appendsos1(task, 3, idx, w);
 */
PRIMALrescodee PRIMAL_appendsos1(PRIMALtask_t t, int num, const int *submem, const double *weight) {
    model_changed(t);
    return appendsos(t, 1, num, submem, weight);
}

/**
 * Appends an SOS2 constraint (Special Ordered Set type 2).
 *
 * @param t      [in] Task handle.
 * @param num    [in] Number of members. Must be >= 1.
 * @param submem [in] Array of variable indices (length num). Each must satisfy
 *                   0 <= submem[k] < numvar.
 * @param weight [in] Array of weights (length num). MUST be strictly increasing
 *                    for SOS2 to define the adjacency order.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or arrays are NULL,
 *         PRIMAL_RES_ERR_ARG if num < 1 or any index out of bounds.
 *
 * @note SOS2 means at most TWO ADJACENT variables (by weight order) can be non-zero.
 *       The weights MUST be strictly increasing. This is used for piecewise
 *       linear functions and non-convex separable functions.
 *
 * @example
 * // SOS2 on x_0..x_4 for piecewise linear approximation
 * int idx[] = {0, 1, 2, 3, 4};
 * double w[] = {0.0, 0.25, 0.5, 0.75, 1.0}; // strictly increasing
 * PRIMAL_appendsos2(task, 5, idx, w);
 */
PRIMALrescodee PRIMAL_appendsos2(PRIMALtask_t t, int num, const int *submem, const double *weight) {
    model_changed(t);
    return appendsos(t, 2, num, submem, weight);
}

/**
 * Retrieves the number of SOS constraints in the task.
 *
 * @param t     [in]  Task handle.
 * @param numsos [out] Pointer to int receiving the SOS count.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or numsos is NULL.
 */
PRIMALrescodee PRIMAL_getnumsos(PRIMALtask_t t, int *numsos) {
    if (!t || !numsos) return PRIMAL_RES_ERR_NULL;
    *numsos = t->numsos;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves an SOS constraint by index.
 *
 * @param t       [in]  Task handle.
 * @param k       [in]  SOS index, 0 <= k < numsos.
 * @param sostype [out] Pointer to int receiving SOS type (1 or 2).
 * @param num     [out] Pointer to int receiving number of members.
 * @param submem  [out] Optional array for variable indices (length num).
 * @param weight  [out] Optional array for weights (length num).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, sostype, or num is NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds.
 *
 * @note The optional submem and weight arrays must have capacity >= num
 *       if provided. If NULL, they are skipped.
 *
 * @example
 * int type, n;
 * PRIMAL_getsos(task, 0, &type, &n, NULL, NULL);
 * int *mem = malloc(n * sizeof(int));
 * double *w = malloc(n * sizeof(double));
 * PRIMAL_getsos(task, 0, &type, &n, mem, w);
 */
PRIMALrescodee PRIMAL_getsos(PRIMALtask_t t, int k, int *sostype, int *num,
                        int *submem, double *weight) {
    if (!t || !sostype || !num) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numsos) return PRIMAL_RES_ERR_ARG;
    *sostype = t->sos_type[k];
    *num = t->sos_n[k];
    if (submem)
        for (int i = 0; i < t->sos_n[k]; i++) submem[i] = t->sos_mem[k][i];
    if (weight)
        for (int i = 0; i < t->sos_n[k]; i++) weight[i] = t->sos_w[k][i];
    return PRIMAL_RES_OK;
}

/**
 * Sets an integer parameter.
 *
 * @param t     [in] Task handle.
 * @param param [in] Parameter ID (e.g., PRIMAL_IPAR_INTPNT_MAX_ITERATIONS).
 * @param value [in] Value to set. Must be within the parameter's declared range.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if param unknown or value out of range.
 *
 * @note The parameter table (PRIMAL_PARAMS) defines all valid IDs, ranges,
 *       and defaults. The value is validated against the inclusive range [lo, hi].
 *       NaN is not possible for integers.
 *
 * @example
 * // Set max interior-point iterations to 500
 * PRIMAL_putintparam(task, PRIMAL_IPAR_INTPNT_MAX_ITERATIONS, 500);
 */
PRIMALrescodee PRIMAL_putintparam(PRIMALtask_t t, int param, int value) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_INT, param);
    if (!d || (double)value < d->lo || (double)value > d->hi) return PRIMAL_RES_ERR_ARG;
    *(int *)param_slot(d, t) = value;
    return PRIMAL_RES_OK;
}

/**
 * Sets a double parameter.
 *
 * @param t     [in] Task handle.
 * @param param [in] Parameter ID (e.g., PRIMAL_DPAR_INTPNT_TOL_PFEAS).
 * @param value [in] Value to set. Must be within [lo, hi] and not NaN.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if param unknown, value out of range, or NaN.
 *
 * @note The parameter table defines valid IDs, ranges, and defaults. The
 *       value must satisfy lo <= value <= hi. NaN is rejected (NaN >= lo is
 *       false). For P_DOUI parameters (double API alias of int field, like
 *       PRIMAL_DPAR_INTPNT_MAX_ITER), the value is truncated to int.
 *
 * @example
 * // Set primal feasibility tolerance to 1e-9
 * PRIMAL_putdouparam(task, PRIMAL_DPAR_INTPNT_TOL_PFEAS, 1e-9);
 */
PRIMALrescodee PRIMAL_putdouparam(PRIMALtask_t t, int param, double value) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_DOU, param);
    if (!d || !(value >= d->lo && value <= d->hi)) return PRIMAL_RES_ERR_ARG;  /* NaN fails */
    if (d->kind == P_DOUI) *(int *)param_slot(d, t) = (int)value;
    else *(double *)param_slot(d, t) = value;
    return PRIMAL_RES_OK;
}

/**
 * Resets a single integer parameter to its default value.
 *
 * @param t     [in] Task handle.
 * @param param [in] Parameter ID.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if param unknown.
 *
 * @note Uses the default from the declarative PRIMAL_PARAMS table.
 */
PRIMALrescodee PRIMAL_resetintparam(PRIMALtask_t t, int param) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_INT, param);
    if (!d) return PRIMAL_RES_ERR_ARG;
    *(int *)param_slot(d, t) = (int)d->dflt;
    return PRIMAL_RES_OK;
}

/**
 * Resets a single double parameter to its default value.
 *
 * @param t     [in] Task handle.
 * @param param [in] Parameter ID.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if param unknown.
 *
 * @note For P_DOUI parameters, the int field is set to the default cast to int.
 */
PRIMALrescodee PRIMAL_resetdouparam(PRIMALtask_t t, int param) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    const PrimalParam *d = param_find(P_DOU, param);
    if (!d) return PRIMAL_RES_ERR_ARG;
    if (d->kind == P_DOUI) *(int *)param_slot(d, t) = (int)d->dflt;
    else *(double *)param_slot(d, t) = d->dflt;
    return PRIMAL_RES_OK;
}

/**
 * Resets all parameters to their default values.
 *
 * @param t [in] Task handle.
 *
 * @return PRIMAL_RES_OK on success, PRIMAL_RES_ERR_NULL if t is NULL.
 *
 * @note Calls param_defaults() which writes every entry from the PRIMAL_PARAMS
 *       table into the task. This is equivalent to recreating the task.
 */
PRIMALrescodee PRIMAL_resetparameters(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    param_defaults(t);
    return PRIMAL_RES_OK;
}

