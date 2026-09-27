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
/* primal_names.c - the seven name tables (shared name_put/name_find).
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Stores name at slot idx, enforcing uniqueness within this table. */
PRIMALrescodee name_put(char **names, int n, int idx, const char *name) {
    if (!name) return PRIMAL_RES_ERR_ARG;
    if (name[0] == '\0') {           /* "" is the request to UN-name */
        free(names[idx]);
        names[idx] = NULL;
        return PRIMAL_RES_OK;
    }
    for (int k = 0; k < n; k++)
        if (k != idx && names[k] && strcmp(names[k], name) == 0)
            return PRIMAL_RES_ERR_ARG;   /* the old name survives */
    char *dup = (char *)malloc(strlen(name) + 1);
    if (!dup) return PRIMAL_RES_ERR_ALLOC;
    free(names[idx]);
    names[idx] = strcpy(dup, name);
    return PRIMAL_RES_OK;
}

/* Looks up name in the table, writing its slot index into *idx. */
PRIMALrescodee name_find(const char **names, int n, const char *name, int *idx) {
    /* "" can never be a stored name, so it is never found. */
    for (int k = 0; k < n; k++)
        if (names[k] && strcmp(names[k], name) == 0) { *idx = k; return PRIMAL_RES_OK; }
    return PRIMAL_RES_ERR_ARG;       /* *idx untouched */
}

/**
 * Sets the name of a variable.
 *
 * @param t    [in] Task handle.
 * @param j    [in] Variable index, 0 <= j < numvar.
 * @param name [in] Name string. If empty string (""), removes the existing name.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL or j out of bounds.
 *
 * @note Names are stored per-variable in a table that grows with appendvars.
 *       An empty string "" removes the name (stored as NULL). Duplicate names
 *       are rejected: the old name survives. The pointer returned by
 *       getvarnameidx/getallvarname points to the stored string (borrowed,
 *       valid until deletetask).
 *
 * @example
 * PRIMAL_putvarname(task, 3, "x_production");
 * // Variable 3 is now named "x_production"
 */
PRIMALrescodee PRIMAL_putvarname(PRIMALtask_t t, int j, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    return name_put(t->varname, t->numvar, j, name);
}

/**
 * Sets the name of a constraint.
 *
 * @param t    [in] Task handle.
 * @param i    [in] Constraint index, 0 <= i < numcon.
 * @param name [in] Name string. If empty string (""), removes the existing name.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL or i out of bounds.
 *
 * @note Same semantics as putvarname. The constraint and variable name tables
 *       are independent: the same string can name a variable AND a constraint.
 */
PRIMALrescodee PRIMAL_putconname(PRIMALtask_t t, int i, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    PRIMALrescodee rc = ensure_size(t); if (rc != PRIMAL_RES_OK) return rc;
    return name_put(t->conname, t->numcon, i, name);
}

/**
 * Retrieves the name of a variable by index.
 *
 * @param t    [in]  Task handle.
 * @param j    [in]  Variable index, 0 <= j < numvar.
 * @param name [out] Pointer to const char* receiving the name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or name is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds or name table not allocated.
 *
 * @note Returns "" (empty string) for unnamed variables. The returned pointer
 *       points to the stored string (borrowed, valid until deletetask).
 */
PRIMALrescodee PRIMAL_getvarnameidx(PRIMALtask_t t, int j, const char **name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar || !t->varname) return PRIMAL_RES_ERR_ARG;
    *name = t->varname[j] ? t->varname[j] : "";
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the name of a constraint by index.
 *
 * @param t    [in]  Task handle.
 * @param i    [in]  Constraint index, 0 <= i < numcon.
 * @param name [out] Pointer to const char* receiving the name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or name is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds or name table not allocated.
 *
 * @note Returns "" for unnamed constraints.
 */
PRIMALrescodee PRIMAL_getconnameidx(PRIMALtask_t t, int i, const char **name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon || !t->conname) return PRIMAL_RES_ERR_ARG;
    *name = t->conname[i] ? t->conname[i] : "";
    return PRIMAL_RES_OK;
}

/**
 * Finds a variable index by name.
 *
 * @param t    [in]  Task handle.
 * @param name [in]  Variable name to search for. Must not be NULL.
 * @param j    [out] Pointer to int receiving the variable index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, name, or j is NULL,
 *         PRIMAL_RES_ERR_ARG if name table not allocated or name not found.
 *
 * @note An empty string "" is never a stored name, so searching for ""
 *       always returns ERR_ARG.
 *
 * @example
 * int idx;
 * PRIMAL_getvarname(task, "x_production", &idx);
 * // idx now holds the variable index
 */
PRIMALrescodee PRIMAL_getvarname(PRIMALtask_t t, const char *name, int *j) {
    if (!t || !name || !j) return PRIMAL_RES_ERR_NULL;
    if (!t->varname) return PRIMAL_RES_ERR_ARG;
    return name_find((const char **)t->varname, t->numvar, name, j);
}

/**
 * Finds a constraint index by name.
 *
 * @param t    [in]  Task handle.
 * @param name [in]  Constraint name to search for. Must not be NULL.
 * @param i    [out] Pointer to int receiving the constraint index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, name, or i is NULL,
 *         PRIMAL_RES_ERR_ARG if name table not allocated or name not found.
 */
PRIMALrescodee PRIMAL_getconname(PRIMALtask_t t, const char *name, int *i) {
    if (!t || !name || !i) return PRIMAL_RES_ERR_NULL;
    if (!t->conname) return PRIMAL_RES_ERR_ARG;
    return name_find((const char **)t->conname, t->numcon, name, i);
}

/**
 * Alias for getvarname (reference's lookup-by-name spelling).
 *
 * @param t   [in]  Task handle.
 * @param vname [in] Variable name to search for.
 * @param var   [out] Pointer to int receiving the variable index.
 *
 * @return Same as PRIMAL_getvarname.
 *
 * @note This is a delegate to getvarname, not a second copy of the rule.
 */
PRIMALrescodee PRIMAL_getidxvar(PRIMALtask_t t, const char *vname, int *var) {
    return PRIMAL_getvarname(t, vname, var);
}

/**
 * Alias for getconname (reference's lookup-by-name spelling).
 *
 * @param t    [in]  Task handle.
 * @param cname [in] Constraint name to search for.
 * @param con   [out] Pointer to int receiving the constraint index.
 *
 * @return Same as PRIMAL_getconname.
 *
 * @note This is a delegate to getconname, not a second copy of the rule.
 */
PRIMALrescodee PRIMAL_getidxcon(PRIMALtask_t t, const char *cname, int *con) {
    return PRIMAL_getconname(t, cname, con);
}

/**
 * Retrieves all variable names in one call.
 *
 * @param t     [in]  Task handle.
 * @param names [out] Pre-allocated array of const char* (size numvar). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or names is NULL.
 *
 * @note Writes exactly numvar pointers. Unnamed slots receive "" (empty string).
 *       The pointers are the SAME objects returned by getvarnameidx (borrowed).
 *       This is the SAME table read, not a copy. A poison value after the last
 *       element can be used to verify exact write length.
 *
 * @example
 * int n; PRIMAL_getnumvar(task, &n);
 * const char **names = malloc(n * sizeof(char*));
 * PRIMAL_getallvarname(task, names);
 * for (int i = 0; i < n; i++) printf("var %d: %s\n", i, names[i]);
 */
PRIMALrescodee PRIMAL_getallvarname(PRIMALtask_t t, const char **names) {
    if (!t || !names) return PRIMAL_RES_ERR_NULL;
    for (int j = 0; j < t->numvar; j++)
        names[j] = t->varname[j] ? t->varname[j] : "";
    return PRIMAL_RES_OK;
}

/**
 * Retrieves all constraint names in one call.
 *
 * @param t     [in]  Task handle.
 * @param names [out] Pre-allocated array of const char* (size numcon). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or names is NULL.
 *
 * @note Same semantics as getallvarname but for constraints.
 */
PRIMALrescodee PRIMAL_getallconname(PRIMALtask_t t, const char **names) {
    if (!t || !names) return PRIMAL_RES_ERR_NULL;
    for (int i = 0; i < t->numcon; i++)
        names[i] = t->conname[i] ? t->conname[i] : "";
    return PRIMAL_RES_OK;
}

/**
 * Sets the name of a bar variable (PSD matrix variable).
 *
 * @param t    [in] Task handle.
 * @param j    [in] Bar variable index, 0 <= j < numbarvar.
 * @param name [in] Name string. If empty string (""), removes the existing name.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL, j out of bounds, or barname table not allocated.
 *
 * @note The bar variable name table shares capacity (barcap) with barDim/barx/barsj.
 *       The namespace is SEPARATE from scalar variables and constraints: the same
 *       string can name a bar variable, a scalar variable, AND a constraint.
 *       Uniqueness is enforced WITHIN the bar variable table only.
 */
PRIMALrescodee PRIMAL_putbarname(PRIMALtask_t t, int j, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || j < 0 || j >= t->numbarvar || !t->barname) return PRIMAL_RES_ERR_ARG;
    return name_put(t->barname, t->numbarvar, j, name);
}

/**
 * Retrieves the name of a bar variable by index.
 *
 * @param t    [in]  Task handle.
 * @param j    [in]  Bar variable index, 0 <= j < numbarvar.
 * @param name [out] Pointer to const char* receiving the name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or name is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds or barname table not allocated.
 *
 * @note Returns "" for unnamed bar variables.
 */
PRIMALrescodee PRIMAL_getbarnameidx(PRIMALtask_t t, int j, const char **name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numbarvar || !t->barname) return PRIMAL_RES_ERR_ARG;
    *name = t->barname[j] ? t->barname[j] : "";
    return PRIMAL_RES_OK;
}

/**
 * Finds a bar variable index by name.
 *
 * @param t    [in]  Task handle.
 * @param name [in]  Bar variable name to search for. Must not be NULL.
 * @param j    [out] Pointer to int receiving the bar variable index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, name, or j is NULL,
 *         PRIMAL_RES_ERR_ARG if barname table not allocated or name not found.
 *
 * @note Empty string "" is never a stored name.
 */
PRIMALrescodee PRIMAL_getbarname(PRIMALtask_t t, const char *name, int *j) {
    if (!t || !name || !j) return PRIMAL_RES_ERR_NULL;
    if (!t->barname) return PRIMAL_RES_ERR_ARG;
    return name_find((const char **)t->barname, t->numbarvar, name, j);
}

/**
 * Alias for getbarname (reference's lookup-by-name spelling).
 *
 * @param t    [in]  Task handle.
 * @param bname [in] Bar variable name to search for.
 * @param bar   [out] Pointer to int receiving the bar variable index.
 *
 * @return Same as PRIMAL_getbarname.
 *
 * @note This is a delegate to getbarname, not a second copy of the rule.
 */
PRIMALrescodee PRIMAL_getidxbarvar(PRIMALtask_t t, const char *bname, int *bar) {
    return PRIMAL_getbarname(t, bname, bar);
}

/**
 * Retrieves all bar variable names in one call.
 *
 * @param t     [in]  Task handle.
 * @param names [out] Pre-allocated array of const char* (size numbarvar). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or names is NULL.
 *
 * @note Writes exactly numbarvar pointers. Same semantics as getallvarname.
 *       The pointers are the SAME objects returned by getbarnameidx (borrowed).
 */
PRIMALrescodee PRIMAL_getallbarname(PRIMALtask_t t, const char **names) {
    if (!t || !names) return PRIMAL_RES_ERR_NULL;
    for (int j = 0; j < t->numbarvar; j++)
        names[j] = t->barname[j] ? t->barname[j] : "";
    return PRIMAL_RES_OK;
}

/**
 * Sets the name of a cone.
 *
 * @param t    [in] Task handle.
 * @param k    [in] Cone index, 0 <= k < numcones.
 * @param name [in] Name string. If empty string (""), removes the existing name.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL, k out of bounds.
 *
 * @note The cone name table shares capacity (cone_cap) with cone_type/cone_nmem/
 *       cone_mem/cone_param. The namespace is SEPARATE from variables, constraints,
 *       and bar variables. The same string can name a cone, a variable, a constraint,
 *       and a bar variable simultaneously. Uniqueness is enforced WITHIN the cone table only.
 *
 * @example
 * PRIMAL_putconename(task, 0, "my_quad_cone");
 */
PRIMALrescodee PRIMAL_putconename(PRIMALtask_t t, int k, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name || k < 0 || k >= t->numcones) return PRIMAL_RES_ERR_ARG;
    return name_put(t->conename, t->numcones, k, name);
}

/**
 * Retrieves the name of a cone by index.
 *
 * @param t    [in]  Task handle.
 * @param k    [in]  Cone index, 0 <= k < numcones.
 * @param name [out] Pointer to const char* receiving the name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or name is NULL,
 *         PRIMAL_RES_ERR_ARG if k out of bounds or conename table not allocated.
 *
 * @note Returns "" for unnamed cones.
 */
PRIMALrescodee PRIMAL_getconenameidx(PRIMALtask_t t, int k, const char **name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    if (k < 0 || k >= t->numcones || !t->conename) return PRIMAL_RES_ERR_ARG;
    *name = t->conename[k] ? t->conename[k] : "";
    return PRIMAL_RES_OK;
}

/**
 * Finds a cone index by name.
 *
 * @param t    [in]  Task handle.
 * @param name [in]  Cone name to search for. Must not be NULL.
 * @param k    [out] Pointer to int receiving the cone index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, name, or k is NULL,
 *         PRIMAL_RES_ERR_ARG if conename table not allocated or name not found.
 *
 * @note Empty string "" is never a stored name.
 */
PRIMALrescodee PRIMAL_getconename(PRIMALtask_t t, const char *name, int *k) {
    if (!t || !name || !k) return PRIMAL_RES_ERR_NULL;
    if (!t->conename) return PRIMAL_RES_ERR_ARG;
    return name_find((const char **)t->conename, t->numcones, name, k);
}

/**
 * Alias for getconename (reference's lookup-by-name spelling).
 *
 * @param t    [in]  Task handle.
 * @param cname [in] Cone name to search for.
 * @param cone  [out] Pointer to int receiving the cone index.
 *
 * @return Same as PRIMAL_getconename.
 *
 * @note This is a delegate to getconename, not a second copy of the rule.
 */
PRIMALrescodee PRIMAL_getidxcone(PRIMALtask_t t, const char *cname, int *cone) {
    return PRIMAL_getconename(t, cname, cone);   /* delegate, not a second copy of the rule */
}

/**
 * Retrieves all cone names in one call.
 *
 * @param t     [in]  Task handle.
 * @param names [out] Pre-allocated array of const char* (size numcones). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or names is NULL.
 *
 * @note Writes exactly numcones pointers. Same semantics as getallvarname.
 */
PRIMALrescodee PRIMAL_getallconename(PRIMALtask_t t, const char **names) {
    if (!t || !names) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < t->numcones; k++)
        names[k] = t->conename[k] ? t->conename[k] : "";
    return PRIMAL_RES_OK;
}

/**
 * Finds a variable index by name (with assignment type).
 *
 * @param t        [in]  Task handle.
 * @param somename [in]  Variable name to search for. Must not be NULL.
 * @param asgn     [out] Optional pointer to int receiving assignment type (always 0 here).
 * @param index    [out] Pointer to int receiving the variable index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, somename, or index is NULL,
 *         PRIMAL_RES_ERR_ARG if name not found.
 *
 * @note This is a delegate to getvarname with an additional asgn output (always 0).
 *       The reference uses this for a different assignment convention; here it's fixed.
 */
PRIMALrescodee PRIMAL_getvarnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index) {
    if (!t || !somename || !index) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_getvarname(t, somename, index);
    if (rc == PRIMAL_RES_OK && asgn) *asgn = 0;
    return rc;
}
/**
 * Finds a constraint index by name (with assignment type).
 *
 * @param t        [in]  Task handle.
 * @param somename [in]  Constraint name to search for. Must not be NULL.
 * @param asgn     [out] Optional pointer to int receiving assignment type (always 0 here).
 * @param index    [out] Pointer to int receiving the constraint index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, somename, or index is NULL,
 *         PRIMAL_RES_ERR_ARG if name not found.
 *
 * @note Delegate to getconname with asgn output (always 0).
 */
PRIMALrescodee PRIMAL_getconnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index) {
    if (!t || !somename || !index) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_getconname(t, somename, index);
    if (rc == PRIMAL_RES_OK && asgn) *asgn = 0;
    return rc;
}
/**
 * Finds a cone index by name (with assignment type).
 *
 * @param t        [in]  Task handle.
 * @param somename [in]  Cone name to search for. Must not be NULL.
 * @param asgn     [out] Optional pointer to int receiving assignment type (always 0 here).
 * @param index    [out] Pointer to int receiving the cone index. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t, somename, or index is NULL,
 *         PRIMAL_RES_ERR_ARG if name not found.
 *
 * @note Delegate to getconename with asgn output (always 0).
 */
PRIMALrescodee PRIMAL_getconenameindex(PRIMALtask_t t, const char *somename,
                                       int *asgn, int *index) {
    if (!t || !somename || !index) return PRIMAL_RES_ERR_NULL;
    PRIMALrescodee rc = PRIMAL_getconename(t, somename, index);
    if (rc == PRIMAL_RES_OK && asgn) *asgn = 0;
    return rc;
}
/**
 * Retrieves the length of a cone name.
 *
 * @param t   [in]  Task handle.
 * @param i   [in]  Cone index, 0 <= i < numcones.
 * @param len [out] Pointer to int receiving the name length (0 for unnamed).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or len is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds.
 */
PRIMALrescodee PRIMAL_getconenamelen(PRIMALtask_t t, int i, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcones) return PRIMAL_RES_ERR_ARG;
    const char *nm = (t->conename && t->conename[i]) ? t->conename[i] : NULL;
    *len = nm ? (int)strlen(nm) : 0;
    return PRIMAL_RES_OK;
}

/* getmaxnumqnz: the stored nonzeros of Q (objective + constraints).
 * getmaxnumqnz64 is the 64-bit variant. */
static PRIMALint64t q_nonzeros(PRIMALtask_t t) {
    PRIMALint64t n = t->qt_n;
    if (t->has_qcon > 0 && t->qcon)
        for (int i = 0; i < t->numcon; i++) {
            if (!t->qcon[i]) continue;
            for (int a = 0; a < t->numvar; a++)
                for (int b = 0; b < t->numvar; b++)
                    if (t->qcon[i][a * t->numvar + b] != 0.0) n++;
        }
    return n;
}
/* Maximum number of nonzeros stored in Q (int variant, clamped to INT_MAX). */
PRIMALrescodee PRIMAL_getmaxnumqnz(PRIMALtask_t t, int *maxnumqnz) {
    if (!t || !maxnumqnz) return PRIMAL_RES_ERR_NULL;
    PRIMALint64t n = q_nonzeros(t);
    *maxnumqnz = (n > INT_MAX) ? INT_MAX : (int)n;
    return PRIMAL_RES_OK;
}
/* Maximum number of nonzeros stored in Q (64-bit variant). */
PRIMALrescodee PRIMAL_getmaxnumqnz64(PRIMALtask_t t, PRIMALint64t *maxnumqnz) {
    if (!t || !maxnumqnz) return PRIMAL_RES_ERR_NULL;
    *maxnumqnz = q_nonzeros(t);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the linear objective coefficients for a list of indices.
 *
 * @param t   [in]  Task handle.
 * @param num [in]  Number of indices. Must be non-negative.
 * @param subj [in] Array of variable indices (length num).
 * @param c    [out] Array receiving coefficients (length num). Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or c is NULL,
 *         PRIMAL_RES_ERR_ARG if num < 0 or any index out of bounds.
 */
PRIMALrescodee PRIMAL_getclist(PRIMALtask_t t, int num, const int *subj, PRIMALrealt *c) {
    if (!t || !c) return PRIMAL_RES_ERR_NULL;
    if (num < 0 || (num > 0 && !subj)) return PRIMAL_RES_ERR_NULL;
    for (int k = 0; k < num; k++) {
        if (subj[k] < 0 || subj[k] >= t->numvar) return PRIMAL_RES_ERR_ARG;
        c[k] = t->c[subj[k]];
    }
    return PRIMAL_RES_OK;
}

/**
 * Sets the maximum number of variables (no-op, this solver grows automatically).
 *
 * @param t         [in] Task handle.
 * @param maxnumvar [in] Suggested capacity. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if maxnumvar < 0.
 *
 * @note This is a no-op for compatibility with the reference API. This solver
 *       grows its arrays automatically as needed.
 */
PRIMALrescodee PRIMAL_putmaxnumvar(PRIMALtask_t t, int maxnumvar) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumvar < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/**
 * Sets the maximum number of constraints (no-op, this solver grows automatically).
 *
 * @param t         [in] Task handle.
 * @param maxnumcon [in] Suggested capacity. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if maxnumcon < 0.
 *
 * @note No-op for compatibility.
 */
PRIMALrescodee PRIMAL_putmaxnumcon(PRIMALtask_t t, int maxnumcon) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumcon < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/**
 * Sets the maximum number of cones (no-op, this solver grows automatically).
 *
 * @param t          [in] Task handle.
 * @param maxnumcone [in] Suggested capacity. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if maxnumcone < 0.
 *
 * @note No-op for compatibility.
 */
PRIMALrescodee PRIMAL_putmaxnumcone(PRIMALtask_t t, int maxnumcone) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumcone < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/**
 * Sets the maximum number of non-zeros in A (no-op).
 *
 * @param t         [in] Task handle.
 * @param maxnumanz [in] Suggested capacity. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if maxnumanz < 0.
 *
 * @note No-op for compatibility.
 */
PRIMALrescodee PRIMAL_putmaxnumanz(PRIMALtask_t t, PRIMALint64t maxnumanz) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumanz < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}
/**
 * Sets the maximum number of non-zeros in Q (no-op).
 *
 * @param t         [in] Task handle.
 * @param maxnumqnz [in] Suggested capacity. Must be non-negative.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if maxnumqnz < 0.
 *
 * @note No-op for compatibility.
 */
PRIMALrescodee PRIMAL_putmaxnumqnz(PRIMALtask_t t, PRIMALint64t maxnumqnz) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (maxnumqnz < 0) return PRIMAL_RES_ERR_ARG;
    return PRIMAL_RES_OK;
}

/**
 * Sets the name of the objective function.
 *
 * @param t    [in] Task handle.
 * @param name [in] Name string. If empty string (""), removes the existing name.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL.
 *
 * @note The objective has a single name slot (a "table of one"). Unlike other
 *       name tables, there is no uniqueness check since there's only one slot.
 *       Renaming simply replaces the previous name.
 *
 * @example
 * PRIMAL_putobjname(task, "maximize_profit");
 */
PRIMALrescodee PRIMAL_putobjname(PRIMALtask_t t, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name) return PRIMAL_RES_ERR_ARG;
    return name_put(&t->objname, 1, 0, name);
}

/**
 * Retrieves the name of the objective function.
 *
 * @param t    [in]  Task handle.
 * @param name [out] Pointer to const char* receiving the name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or name is NULL.
 *
 * @note Returns "" if no name is set. The returned pointer is borrowed
 *       (valid until deletetask).
 */
PRIMALrescodee PRIMAL_getobjname(PRIMALtask_t t, const char **name) {
    if (!t || !name) return PRIMAL_RES_ERR_NULL;
    *name = t->objname ? t->objname : "";
    return PRIMAL_RES_OK;
}

/**
 * Sets the name of the task.
 *
 * @param t    [in] Task handle.
 * @param name [in] Name string. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t is NULL,
 *         PRIMAL_RES_ERR_ARG if name is NULL.
 *
 * @note The name is copied and stored. The previous name is freed.
 *
 * @example
 * PRIMAL_puttaskname(task, "production_planning");
 */
PRIMALrescodee PRIMAL_puttaskname(PRIMALtask_t t, const char *name) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    if (!name) return PRIMAL_RES_ERR_ARG;
    size_t n = strlen(name);
    char *s = (char *)malloc(n + 1);
    if (!s) return PRIMAL_RES_ERR_ALLOC;
    memcpy(s, name, n + 1);
    free(t->taskname);
    t->taskname = s;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the length of the task name (excluding null terminator).
 *
 * @param t   [in]  Task handle.
 * @param len [out] Pointer to int receiving the length (0 if no name).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or len is NULL.
 */
PRIMALrescodee PRIMAL_gettasknamelen(PRIMALtask_t t, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    *len = t->taskname ? (int)strlen(t->taskname) : 0;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the task name into a caller-provided buffer.
 *
 * @param t             [in]  Task handle.
 * @param sizetaskname  [in]  Size of the taskname buffer (must include space for null terminator).
 * @param taskname      [out] Buffer receiving the task name. Must not be NULL.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or taskname is NULL,
 *         PRIMAL_RES_ERR_ARG if buffer too small (sizetaskname < strlen(name) + 1).
 *
 * @note The name is copied with null terminator. The buffer must hold
 *       strlen(name) + 1 characters. If too small, nothing is written
 *       (a rejection writes nothing).
 *
 * @example
 * char buf[256];
 * PRIMAL_gettaskname(task, sizeof(buf), buf);
 * printf("Task name: %s\n", buf);
 */
PRIMALrescodee PRIMAL_gettaskname(PRIMALtask_t t, int sizetaskname, char *taskname) {
    if (!t || !taskname) return PRIMAL_RES_ERR_NULL;
    const char *nm = t->taskname ? t->taskname : "";
    int len = (int)strlen(nm);
    if (sizetaskname < len + 1) return PRIMAL_RES_ERR_ARG;   /* no room for the zero */
    memcpy(taskname, nm, (size_t)len + 1);
    return PRIMAL_RES_OK;
}

/* Length of a name string, or 0 for a NULL (unnamed) slot. */
int name_len_of(const char *s) { return s ? (int)strlen(s) : 0; }

/**
 * Retrieves the length of a variable name (excluding null terminator).
 *
 * @param t   [in]  Task handle.
 * @param j   [in]  Variable index, 0 <= j < numvar.
 * @param len [out] Pointer to int receiving the length (0 if unnamed).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or len is NULL,
 *         PRIMAL_RES_ERR_ARG if j out of bounds.
 */
PRIMALrescodee PRIMAL_getvarnamelen(PRIMALtask_t t, int j, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (j < 0 || j >= t->numvar) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->varname ? t->varname[j] : NULL);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the length of a constraint name (excluding null terminator).
 *
 * @param t   [in]  Task handle.
 * @param i   [in]  Constraint index, 0 <= i < numcon.
 * @param len [out] Pointer to int receiving the length (0 if unnamed).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or len is NULL,
 *         PRIMAL_RES_ERR_ARG if i out of bounds.
 */
PRIMALrescodee PRIMAL_getconnamelen(PRIMALtask_t t, int i, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    if (i < 0 || i >= t->numcon) return PRIMAL_RES_ERR_ARG;
    *len = name_len_of(t->conname ? t->conname[i] : NULL);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the length of the objective name (excluding null terminator).
 *
 * @param t   [in]  Task handle.
 * @param len [out] Pointer to int receiving the length (0 if unnamed).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or len is NULL.
 */
PRIMALrescodee PRIMAL_getobjnamelen(PRIMALtask_t t, int *len) {
    if (!t || !len) return PRIMAL_RES_ERR_NULL;
    *len = name_len_of(t->objname);
    return PRIMAL_RES_OK;
}

/**
 * Retrieves the maximum name length across all named entities.
 *
 * @param t      [in]  Task handle.
 * @param maxlen [out] Pointer to int receiving the maximum length (0 if no names).
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if t or maxlen is NULL.
 *
 * @note Scans variable names, constraint names, cone names, and objective name.
 *       Returns the longest name length (excluding null terminators).
 */
PRIMALrescodee PRIMAL_getmaxnamelen(PRIMALtask_t t, int *maxlen) {
    if (!t || !maxlen) return PRIMAL_RES_ERR_NULL;
    int m = name_len_of(t->objname);
    if (t->varname) for (int j = 0; j < t->numvar; j++) {
        int l = name_len_of(t->varname[j]); if (l > m) m = l; }
    if (t->conname) for (int i = 0; i < t->numcon; i++) {
        int l = name_len_of(t->conname[i]); if (l > m) m = l; }
    if (t->conename) for (int k = 0; k < t->numcones; k++) {
        int l = name_len_of(t->conename[k]); if (l > m) m = l; }
    *maxlen = m;
    return PRIMAL_RES_OK;
}

/**
 * Retrieves this solver's version numbers.
 *
 * @param major    [out] Pointer to int receiving major version.
 * @param minor    [out] Pointer to int receiving minor version.
 * @param revision [out] Pointer to int receiving revision.
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if any output pointer is NULL.
 *
 * @note Returns THIS solver's version (0, 1, 0), not the reference's version.
 *       The reference's version numbers are not interchangeable.
 *
 * @example
 * int maj, min, rev;
 * PRIMAL_getversion(&maj, &min, &rev);
 * printf("PrimalSolver %d.%d.%d\n", maj, min, rev);
 */
PRIMALrescodee PRIMAL_getversion(int *major, int *minor, int *revision) {
    if (!major || !minor || !revision) return PRIMAL_RES_ERR_NULL;
    *major = 0; *minor = 1; *revision = 0;
    return PRIMAL_RES_OK;
}

/**
 * Checks if a value is IEEE infinity.
 *
 * @param value [in] Double value to check.
 *
 * @return 1 if value is +INF or -INF, 0 otherwise.
 *
 * @note Uses the standard isinf() function. This solver's infinity is the
 *       IEEE infinity (`INF` == `INFINITY`).
 *
 * @example
 * if (PRIMAL_isinfinity(value)) printf("Value is infinite\n");
 */
int PRIMAL_isinfinity(PRIMALrealt value) {
    return isinf(value) ? 1 : 0;
}

/**
 * Maps a solver result code to a response class.
 *
 * @param res            [in]  Result code from a solver function.
 * @param responseclass  [out] Pointer to int receiving the class:
 *                              0 = OK (success)
 *                              2 = TRM (termination, e.g., max iterations)
 *                              3 = ERR (error)
 *                              4 = UNK (unknown)
 *
 * @return PRIMAL_RES_OK on success,
 *         PRIMAL_RES_ERR_NULL if responseclass is NULL.
 *
 * @note Maps this solver's codes to the reference's response classes.
 *       1 (WRN) is not used by this solver.
 *
 * @example
 * int cls;
 * PRIMAL_getresponseclass(rc, &cls);
 * if (cls == 0) printf("Success\n");
 */
PRIMALrescodee PRIMAL_getresponseclass(PRIMALrescodee res, int *responseclass) {
    if (!responseclass) return PRIMAL_RES_ERR_NULL;
    switch (res) {
        case PRIMAL_RES_OK:             *responseclass = 0; break;   /* OK  */
        case PRIMAL_RES_TRM_MAX_ITER:   *responseclass = 2; break;   /* TRM */
        case PRIMAL_RES_ERR_ARG:
        case PRIMAL_RES_ERR_INFEASIBLE:
        case PRIMAL_RES_ERR_UNBOUNDED:
        case PRIMAL_RES_ERR_ALLOC:
        case PRIMAL_RES_ERR_FILE:
        case PRIMAL_RES_ERR_NULL:       *responseclass = 3; break;   /* ERR */
        default:                        *responseclass = 4; break;   /* UNK */
    }
    return PRIMAL_RES_OK;
}

