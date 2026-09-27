/* Modified 2026-09-27: numerical and result-contract corrections. */
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

/* primal.h - public API (PRIMAL-compatible subset, extended)
 *
 * DUAL CONVENTIONS (documented, verified by tests):
 *   The reported dual solutions (y, slc, suc, slx, sux) satisfy,
 *   for the ORIGINAL problem as written (either sense):
 *       c + Qx + A'y + z = 0          with z = slx + sux
 *   where (in min-normalized form, i.e. multiplying by s=+1 min / -1 max):
 *       y_i > 0  <=> row i at active UPPER bound
 *       y_i < 0  <=> row i at active LOWER bound
 *       z_j > 0  <=> variable j at active upper bound
 *       z_j < 0  <=> variable j at active lower bound
 *   slc/suc and slx/sux are the by-sign split of y and z:
 *       slc = min(y,0), suc = max(y,0), slx = min(z,0), sux = max(z,0).
 *   Multipliers of inactive rows/variables are 0 (complementarity).
 *   The dual objective is: dobj = -sum_i y_i*b_i(active) - sum_j z_j*xb_j(active)
 *   in min form (strong duality |pobj - s*dobj_min| ~ 0, verified by tests).
 *
 * SOLUTION STATUS (PRIMAL_getsolsta):
 *   PRIMAL_SOL_STA_OPTIMAL, PRIMAL_SOL_STA_PRIM_INFEAS_CER, PRIMAL_SOL_STA_DUAL_INFEAS_CER,
 *   PRIMAL_SOL_STA_INTEGER_OPTIMAL, PRIMAL_SOL_STA_UNKNOWN.
 * PRIMAL_optimize returns PRIMAL_RES_OK with the verdict in solsta; for
 * compatibility with the old draft, infeasibility/unboundedness
 * also return PRIMAL_RES_ERR_INFEASIBLE / PRIMAL_RES_ERR_UNBOUNDED.
 * A *_CER member NAMES a Farkas vector: it is published only where
 * PRIMAL_getdualray / PRIMAL_getprimalray answer (the LP/QP route, and only
 * if the ray measures in the solved form). Where the verdict exists but the
 * vector does not — the tangent-cuts route, the branch-and-bound tree, a
 * non-admissible basis — solsta stays UNKNOWN and the verdict is read in
 * PRIMAL_getprosta.
 * Raw numbers and the prosta+solsta pairing are fixed by T93 and T94.
 *
 * Documented deviations from real PRIMAL:
 *  - a single interior solution, served for both PRIMAL_SOL_ITR and PRIMAL_SOL_BAS;
 *  - PRIMAL_OPTIMIZER_DUAL_SIMPLEX and PRIMAL_SIMPLEX both use the two-phase
 *    primal simplex; QPs always use the interior point (INTPNT).
 */
#ifndef PRIMAL_H
#define PRIMAL_H

#include <stddef.h>
#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PRIMALAPI

typedef int PRIMALint32t;
typedef long long PRIMALint64t;
typedef double PRIMALrealt;

typedef struct PRIMAL_env_s *PRIMALenv_t;
typedef struct PRIMAL_task_s *PRIMALtask_t;

typedef enum {
    PRIMAL_RES_OK = 0,
    PRIMAL_RES_ERR_ARG = 1001,
    PRIMAL_RES_ERR_INFEASIBLE = 1002,
    PRIMAL_RES_ERR_UNBOUNDED = 1003,
    PRIMAL_RES_ERR_ALLOC = 1004,
    PRIMAL_RES_ERR_FILE = 1005,
    PRIMAL_RES_ERR_NULL = 1006,      /* null pointer argument */
    PRIMAL_RES_TRM_MAX_ITER = 1007,  /* iteration limit reached */
    PRIMAL_RES_TRM_MAX_TIME = 1008,  /* time limit reached (PRIMAL_DPAR_OPTIMIZER_MAX_TIME) */
    PRIMAL_RES_TRM_OBJECTIVE_RANGE = 1009   /* optimal value proven outside [LOWER,UPPER]_OBJ_CUT */
} PRIMALrescodee;

/* Values are the reference solver's: MOSEK 11.2.4 lists MSK_SOL_STA_UNKNOWN=0,
 * OPTIMAL=1, PRIM_FEAS=2, DUAL_FEAS=3, PRIM_AND_DUAL_FEAS=4, PRIM_INFEAS_CER=5,
 * DUAL_INFEAS_CER=6, PRIM_ILLPOSED_CER=7, DUAL_ILLPOSED_CER=8, INTEGER_OPTIMAL=9.
 * Only the members this solver produces are declared; a declared one carries the
 * number the reference gives it, so a raw status read against that table means the
 * same thing here. (The numbers used to be ours: INTEGER_OPTIMAL was 2, which the
 * reference calls PRIM_FEAS, and PRIM_INFEAS_CER was 4, which it calls
 * PRIM_AND_DUAL_FEAS.) */
typedef enum {
    PRIMAL_SOL_STA_UNKNOWN = 0,
    PRIMAL_SOL_STA_OPTIMAL = 1,
    PRIMAL_SOL_STA_PRIM_FEAS = 2,        /* feasible point, not proven optimal */
    PRIMAL_SOL_STA_PRIM_INFEAS_CER = 5,
    PRIMAL_SOL_STA_DUAL_INFEAS_CER = 6,
    PRIMAL_SOL_STA_INTEGER_OPTIMAL = 9
} PRIMALsolstae;

/* Problem status, numbered as the reference numbers MSKprostae. It answers a
 * different question from PRIMAL_getsolsta: whether the model is primal/dual
 * feasible, regardless of whether a solution or certificate was reached. The
 * reference pairs the two in two tables (continuous and integer problems); this
 * solver derives the pairing from those tables in PRIMAL_getprosta, so the two
 * numbers published by one solve cannot disagree. */
typedef enum {
    PRIMAL_PRO_STA_UNKNOWN = 0,
    PRIMAL_PRO_STA_PRIM_AND_DUAL_FEAS = 1,
    PRIMAL_PRO_STA_PRIM_FEAS = 2,
    PRIMAL_PRO_STA_DUAL_FEAS = 3,
    PRIMAL_PRO_STA_PRIM_INFEAS = 4,
    PRIMAL_PRO_STA_DUAL_INFEAS = 5,
    PRIMAL_PRO_STA_PRIM_AND_DUAL_INFEAS = 6,
    PRIMAL_PRO_STA_ILL_POSED = 7,
    PRIMAL_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED = 8
} PRIMALprostae;

/* Solution keys, numbered as the reference numbers MSKsoltypee. PRIMAL_SOL_ITG
 * is its key for the integer solution of a mixed-integer problem: this solver
 * stores one solution per task, so ITG reports the same point ITR/BAS do, where
 * the reference keeps the relaxation under the other two -- declared deviation. */
typedef enum {
    PRIMAL_SOL_ITR = 0,
    PRIMAL_SOL_BAS = 1,
    PRIMAL_SOL_ITG = 2
} PRIMALsolt;

/* Solution slice items, numbered as the reference numbers its own
 * (MSK_SOL_ITEM_XC=0, XX=1, Y=2, SLC=3, SUC=4, SLX=5, SUX=6, SNX=7). The
 * numbers used to be ours, and all three served here disagreed. XC ("solution
 * for the constraints") and SNX ("lagrange multipliers corresponding to the
 * conic constraints on the variables") are not produced by this solver -- its
 * conic duals live per cone block, not per variable -- so the getter answers
 * PRIMAL_RES_ERR_ARG for those two items. */
#define PRIMAL_SOL_ITEM_XC  0
#define PRIMAL_SOL_ITEM_XX  1
#define PRIMAL_SOL_ITEM_Y   2
#define PRIMAL_SOL_ITEM_SLC 3
#define PRIMAL_SOL_ITEM_SUC 4
#define PRIMAL_SOL_ITEM_SLX 5
#define PRIMAL_SOL_ITEM_SUX 6
#define PRIMAL_SOL_ITEM_SNX 7

/* Bound keys, numbered as the reference numbers its own (MSK_BK_LO=0, UP=1,
 * FX=2, FR=3, RA=4). The names always matched; the numbers did not: FX, FR and
 * RA sat on 4, 2 and 3, so a caller passing the reference's number for a fixed
 * bound asked this solver for a free variable -- a different model, with no
 * error to read. */
typedef enum {
    PRIMAL_BK_LO = 0,  /* lx <= x (bux ignored)          */
    PRIMAL_BK_UP = 1,  /* x <= ux (blx ignored)          */
    PRIMAL_BK_FX = 2,  /* x = blx                        */
    PRIMAL_BK_FR = 3,  /* free                           */
    PRIMAL_BK_RA = 4   /* blx <= x <= bux                */
} PRIMALboundkeye;

typedef enum { PRIMAL_OPTIMIZE_MINIMIZE = 0, PRIMAL_OPTIMIZE_MAXIMIZE = 1,
               PRIMAL_OBJECTIVE_SENSE_MINIMIZE = 0, PRIMAL_OBJECTIVE_SENSE_MAXIMIZE = 1 } PRIMALobjsensee;

/* variable types (MIP) */
typedef enum {
    PRIMAL_VAR_TYPE_CONT = 0,
    PRIMAL_VAR_TYPE_INT = 1,
    PRIMAL_VAR_TYPE_INT_BIN = 2,
    PRIMAL_VAR_TYPE_SEMI_CONT = 3,   /* x = 0 or l <= x <= u (l = blx) */
    PRIMAL_VAR_TYPE_SEMI_INT = 4     /* x = 0 or l <= x <= u, x integer  */
} PRIMALvariabletypee;

/* SOS constraints: SOS1 = at most one member nonzero, SOS2 = at most two
 * adjacent (by weight order) members nonzero. Weights order the members. */
PRIMALrescodee PRIMAL_appendsos1(PRIMALtask_t t, int num, const int *submem, const PRIMALrealt *weight);
/* Appends an SOS2 constraint; weights order the members (at most two adjacent). */
PRIMALrescodee PRIMAL_appendsos2(PRIMALtask_t t, int num, const int *submem, const PRIMALrealt *weight);
/* Returns the number of SOS constraints in the task. */
PRIMALrescodee PRIMAL_getnumsos(PRIMALtask_t t, int *numsos);
/* Reads SOS constraint k: its type, member count, members and weights. */
PRIMALrescodee PRIMAL_getsos(PRIMALtask_t t, int k, int *sostype, int *num,
                       int *submem, PRIMALrealt *weight);

/* conic types */
typedef enum {
    PRIMAL_CT_QUAD = 0,   /* (t, x1..xk): t >= sqrt(sum xi^2)    */
    PRIMAL_CT_RQUAD = 1,  /* rotated: 2*x1*x2 >= sum_{i>=3} xi^2 */
    PRIMAL_CT_PEXP = 2,   /* (x1,x2,x3): x1 >= x2*exp(x3/x2), x2 >= 0 (param ignored) */
    PRIMAL_CT_DEXP = 3,   /* dual exponential: x1 >= -x3*exp(x2/x3-1), x3<=0 */
    PRIMAL_CT_PPOW = 4,   /* (x1,x2,x3): x1^a*x2^(1-a) >= |x3|, a=param in (0,1) */
    /* 5 and 6 are the reference's DPOW (dual power cone) and ZERO: not
     * implemented here, and deliberately left empty rather than reused, so a
     * caller passing one of those numbers is refused instead of answered with
     * the wrong cone. RPOW is this solver's own cone and has no reference
     * number, so it sits past the reference's list. */
    PRIMAL_CT_RPOW = 7    /* rotated: 2*x1^(2a)*x2^(2(1-a)) >= x3^2; a=1/2 -> RQUAD */
} PRIMALconetypee;

/* File data-format types, numbered as the reference's MSKdataformate. */
typedef enum {
    PRIMAL_DATA_FORMAT_EXTENSION = 0,  /* decide by the file extension  */
    PRIMAL_DATA_FORMAT_MPS = 1,        /* MPS                           */
    PRIMAL_DATA_FORMAT_LP = 2,         /* LP                            */
    PRIMAL_DATA_FORMAT_OP = 3,         /* optimization problem (OPF)    */
    PRIMAL_DATA_FORMAT_FREE_MPS = 4,   /* free-form MPS                 */
    PRIMAL_DATA_FORMAT_TASK = 5,       /* generic task dump (not read)  */
    PRIMAL_DATA_FORMAT_PTF = 6,        /* pretty text format (not read) */
    PRIMAL_DATA_FORMAT_CB = 7,         /* conic benchmark (CBF)         */
    PRIMAL_DATA_FORMAT_JSON_TASK = 8   /* JSON task (not read)          */
} PRIMALdataformate;

/* File compression types, numbered as the reference's MSKcompresstypee. No
 * compression is linked here: NONE and FREE are accepted and treated the same,
 * GZIP and ZSTD are refused (declared deviation). */
typedef enum {
    PRIMAL_COMPRESS_NONE = 0,
    PRIMAL_COMPRESS_FREE = 1,
    PRIMAL_COMPRESS_GZIP = 2,
    PRIMAL_COMPRESS_ZSTD = 3
} PRIMALcompresstypee;

/* Optimizer selection, numbered as the reference numbers its own
 * (MSK_OPTIMIZER_DUAL_SIMPLEX=1, FREE=2, INTPNT=4, PRIMAL_SIMPLEX=8). All four
 * used to disagree. The reference's remaining members (CONIC=0, FREE_SIMPLEX=3,
 * MIXED_INT=5, NEW_DUAL_SIMPLEX=6, NEW_PRIMAL_SIMPLEX=7) are not implemented
 * here and select the free choice, as FREE does. */
typedef enum {
    PRIMAL_OPTIMIZER_FREE = 2,
    PRIMAL_OPTIMIZER_INTPNT = 4,
    PRIMAL_OPTIMIZER_DUAL_SIMPLEX = 1,
    PRIMAL_OPTIMIZER_PRIMAL_SIMPLEX = 8
} PRIMALoptimizer;

/* parameters (int) */
#define PRIMAL_IPAR_OPTIMIZER                 0
#define PRIMAL_IPAR_LOG                       4
#define PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS    10
#define PRIMAL_IPAR_INTPNT_MAX_ITERATIONS     11
#define PRIMAL_IPAR_PRESOLVE                  12   /* 0 off, 1 on (default on, LP) */
#define PRIMAL_IPAR_SCALING                   13   /* 0 off, 1 on (default on, LP/QP) */
#define PRIMAL_IPAR_MIP_MAX_NODES             14   /* B&B node cap (default 100000) */
#define PRIMAL_IPAR_NUM_THREADS               15   /* threads for the parallel parts (default 1) */
#define PRIMAL_IPAR_INTPNT_MAX_NUM_COR        16   /* IPM correctors: -1/1 = 1 (default), >=2 higher-order */
#define PRIMAL_IPAR_PRESOLVE_LEVEL            17   /* 0 off, 1 empty/singleton (default), 2 +duplicates */
#define PRIMAL_IPAR_CONCURRENT_TIME           18   /* concurrent optimizer: 0 tie-break (default), 1 fastest */
/* parameters (double) */
#define PRIMAL_DPAR_INTPNT_TOL_PFEAS          0
#define PRIMAL_DPAR_INTPNT_TOL_DFEAS          1
#define PRIMAL_DPAR_INTPNT_TOL_REL_GAP        2
#define PRIMAL_DPAR_INTPNT_MAX_ITER           3
#define PRIMAL_DPAR_MIP_TOL_ABS_GAP           4    /* prune gap, absolute (0 = off) */
#define PRIMAL_DPAR_MIP_TOL_REL_GAP           5    /* prune gap, relative (1e-4) */
#define PRIMAL_DPAR_MIP_TOL_INTHER            6    /* integrality threshold (1e-5) */
#define PRIMAL_DPAR_MIP_TOL_FEAS              7    /* incumbent feasibility tolerance (1e-6) */
#define PRIMAL_DPAR_INTPNT_TOL_NEAR_REL       8    /* near-optimal acceptance factor (1000) */
/* The reference splits the interior-point tolerances into three sets by problem
 * class: INTPNT_TOL_* (LP), INTPNT_CO_TOL_* (conic), INTPNT_QO_TOL_* (quadratic).
 * This solver has all three; each route reads its own set.  Defaults and
 * accepted ranges are the reference's (parameters.html, 11.2.4). */
#define PRIMAL_DPAR_INTPNT_CO_TOL_PFEAS       16   /* conic primal feasibility (1e-8, [0,1]) */
#define PRIMAL_DPAR_INTPNT_CO_TOL_DFEAS       17   /* conic dual feasibility (1e-8, [0,1]) */
#define PRIMAL_DPAR_INTPNT_CO_TOL_REL_GAP     18   /* conic relative gap (1e-8, [0,1]) */
#define PRIMAL_DPAR_INTPNT_QO_TOL_PFEAS       19   /* quadratic primal feasibility (1e-8, [0,1]) */
#define PRIMAL_DPAR_INTPNT_QO_TOL_DFEAS       20   /* quadratic dual feasibility (1e-8, [0,1]) */
#define PRIMAL_DPAR_INTPNT_QO_TOL_REL_GAP     21   /* quadratic relative gap (1e-8, [0,1]) */
#define PRIMAL_DPAR_OPTIMIZER_MAX_TIME        22   /* wall-clock cap in seconds (-1 = no limit) */
#define PRIMAL_DPAR_MIO_MAX_TIME              23   /* MIP-phase wall-clock cap (-1 = no limit) */
#define PRIMAL_DPAR_LOWER_OBJ_CUT             24   /* lower objective cut (min; -inf = none) */
#define PRIMAL_DPAR_UPPER_OBJ_CUT             25   /* upper objective cut (min; +inf = none) */
#define PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX   26   /* PSD tolerance (1e-10, [1e-15,+inf]) */

/* Parameter table introspection.  Every id accepted by
 * PRIMAL_putintparam/PRIMAL_putdouparam has one row in a declarative table
 * (kind, default, inclusive range) that is the single source of truth for the
 * setters, the getters and the defaults applied by PRIMAL_maketask.  `kind`
 * selects the namespace: int ids and double ids share numbers
 * (PRIMAL_IPAR_OPTIMIZER and PRIMAL_DPAR_INTPNT_TOL_PFEAS are both 0), so an
 * id answered through the other kind is unknown and returns
 * PRIMAL_RES_ERR_ARG.  dflt/lo/hi are optional outputs; the range is inclusive.
 * The kind code itself is the reference's parameter-type enum
 * (MSK_PAR_INVALID_TYPE=0, MSK_PAR_DOU_TYPE=1, MSK_PAR_INT_TYPE=2,
 * MSK_PAR_STR_TYPE=3); the two kinds used to be 0 and 1 here, so the reference's
 * number for a double id selected the int namespace.
 */
#define PRIMAL_PARAM_KIND_DOU 1
#define PRIMAL_PARAM_KIND_INT 2
PRIMALrescodee PRIMAL_getparaminfo(PRIMALtask_t t, int kind, int param,
                                   PRIMALrealt *dflt, PRIMALrealt *lo,
                                   PRIMALrealt *hi);
/* number of parameters of one type in the table (1 dou, 2 int, 3 str). */
PRIMALrescodee PRIMAL_getnumparam(PRIMALtask_t t, int partype, int *numparam);
/* parameter names: getparamname/whichparam/getparammax, the by-name lookups
 * (isdouparname/isintparname/isstrparname) and the by-name read
 * (getnaintparam/getnadouparam/getnastrparam). The name is our own
 * enum's; there is no string parameter. */
PRIMALrescodee PRIMAL_getparamname(PRIMALtask_t t, int partype, int param, char *parname);
/* Highest parameter id of one type; infmax is the table's limit. */
PRIMALrescodee PRIMAL_getparammax(PRIMALtask_t t, int partype, int *parammax);
/* Resolves a parameter name to its type and id; ERR_ARG if unknown. */
PRIMALrescodee PRIMAL_whichparam(PRIMALtask_t t, const char *parname, int *partype, int *param);
/* True when parname is a double parameter; returns its id. */
PRIMALrescodee PRIMAL_isdouparname(PRIMALtask_t t, const char *parname, int *param);
/* True when parname is an int parameter; returns its id. */
PRIMALrescodee PRIMAL_isintparname(PRIMALtask_t t, const char *parname, int *param);
/* True when parname is a string parameter (none exist here); returns its id. */
PRIMALrescodee PRIMAL_isstrparname(PRIMALtask_t t, const char *parname, int *param);
/* Reads an int parameter by name. */
PRIMALrescodee PRIMAL_getnaintparam(PRIMALtask_t t, const char *paramname, int *parvalue);
/* Reads a double parameter by name. */
PRIMALrescodee PRIMAL_getnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt *parvalue);
/* Reads a string parameter by name (no string parameter: ERR_ARG). */
PRIMALrescodee PRIMAL_getnastrparam(PRIMALtask_t t, const char *paramname,
                                    int sizeparamname, int *len, char *parvalue);
/* by-name setters for int/double and the string family (no string parameter:
 * get/put/resetstrparam* answer ERR_ARG). */
PRIMALrescodee PRIMAL_putnaintparam(PRIMALtask_t t, const char *paramname, int parvalue);
/* Sets a double parameter by name. */
PRIMALrescodee PRIMAL_putnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt parvalue);
/* Sets a string parameter by name (no string parameter: ERR_ARG). */
PRIMALrescodee PRIMAL_putnastrparam(PRIMALtask_t t, const char *paramname, const char *parvalue);
/* Reads a string parameter by id (no string parameter: ERR_ARG). */
PRIMALrescodee PRIMAL_getstrparam(PRIMALtask_t t, int param, int maxlen, int *len, char *parvalue);
/* Reads the length of a string parameter (none exist: ERR_ARG). */
PRIMALrescodee PRIMAL_getstrparamlen(PRIMALtask_t t, int param, int *len);
/* Sets a string parameter by id (none exist: ERR_ARG). */
PRIMALrescodee PRIMAL_putstrparam(PRIMALtask_t t, int param, const char *parvalue);
/* Resets a string parameter to its default (none exist: ERR_ARG). */
PRIMALrescodee PRIMAL_resetstrparam(PRIMALtask_t t, int param);

/* stream callbacks (PRIMAL compatible) */
typedef enum { PRIMAL_STREAM_LOG = 0 } PRIMALstreamtypee;
typedef void (*PRIMALstreamfunc)(void *handle, const char *msg);
/* name type for analyzenames (reference MSKnametypee) */
typedef enum {
    PRIMAL_NAME_TYPE_GEN = 0,
    PRIMAL_NAME_TYPE_MPS = 1,
    PRIMAL_NAME_TYPE_LP  = 2
} PRIMALnametypee;
/* Analyzes the names of the task as one of the name types. */
PRIMALrescodee PRIMAL_analyzenames(PRIMALtask_t t, PRIMALstreamtypee whichstream,
                                   PRIMALnametypee nametype);
/* "al" (allocate-and-return) variants of the string parameters (reference
 * getstrparamal/getnastrparamal). This solver has no string parameters:
 * they answer ERR_ARG without allocating. */
PRIMALrescodee PRIMAL_getstrparamal(PRIMALtask_t t, int param, int numaddchr, char **value);
/* Allocate-and-return string parameter by name (no string parameter: ERR_ARG). */
PRIMALrescodee PRIMAL_getnastrparamal(PRIMALtask_t t, const char *paramname, int numaddchr,
                                      char **value);

/* environment/task lifecycle */
/* exit callback on fatal error (reference MSKexitfunc). */
typedef void (*PRIMALexitfunc)(void *handle, const char *msg);
PRIMALrescodee PRIMAL_putexitfunc(PRIMALenv_t env, PRIMALexitfunc exitfunc, void *handle);
/* Creates a solver environment. */
PRIMALrescodee PRIMAL_makeenv(PRIMALenv_t *env, void *usercb);
/* Creates a task with maxcon rows and maxvar columns already preallocated. */
PRIMALrescodee PRIMAL_maketask(PRIMALenv_t env, int maxcon, int maxvar, PRIMALtask_t *task);
/* reference task management: makeemptytask (no declared size),
 * getenv, commitchanges/resizetask/updatesolutioninfo (no-ops), deletesolution
 * (removes the point and the verdict). */
PRIMALrescodee PRIMAL_makeemptytask(PRIMALenv_t env, PRIMALtask_t *task);
/* Returns the environment that owns the task. */
PRIMALrescodee PRIMAL_getenv(PRIMALtask_t t, PRIMALenv_t *env);
/* Commits pending model changes (no-op here). */
PRIMALrescodee PRIMAL_commitchanges(PRIMALtask_t t);
/* Resizes the task's capacity hints (no-op: arrays grow in place). */
PRIMALrescodee PRIMAL_resizetask(PRIMALtask_t t, int maxnumcon, int maxnumvar,
                                 int maxnumcone, PRIMALint64t maxnumanz,
                                 PRIMALint64t maxnumqnz);
/* Updates the solution information (no-op here). */
PRIMALrescodee PRIMAL_updatesolutioninfo(PRIMALtask_t t, PRIMALsolt which);
/* Removes the point and the verdict of a solution key. */
PRIMALrescodee PRIMAL_deletesolution(PRIMALtask_t t, PRIMALsolt which);
/* Destroys a task and its storage. */
PRIMALrescodee PRIMAL_deletetask(PRIMALtask_t *task);
/* Destroys an environment and its storage. */
PRIMALrescodee PRIMAL_deleteenv(PRIMALenv_t *env);

/* Environment/task memory helpers (reference MSK_callocenv/MSK_freeenv and their
 * task variants). This solver keeps no internal memory pool, so the debug
 * variants (callocdbg/freedbg) behave like the plain calloc/free and the memory
 * checks (checkmem) always answer OK: there is no pool to overrun or to inspect
 * (declared deviation). globalenvinitialize/finalize have no global state to set
 * up here. freeenv/freetask free a buffer allocated by these helpers, they are
 * NOT the environment/task destructors (those are deleteenv/deletetask). */
void *PRIMAL_callocenv(PRIMALenv_t env, size_t number, size_t size);
/* Debug variant of PRIMAL_callocenv (file/line are ignored here). */
void *PRIMAL_callocdbgenv(PRIMALenv_t env, size_t number, size_t size,
                          const char *file, unsigned line);
/* Frees a buffer allocated by PRIMAL_callocenv. */
void PRIMAL_freeenv(PRIMALenv_t env, void *buffer);
/* Debug variant of PRIMAL_freeenv (file/line are ignored here). */
void PRIMAL_freedbgenv(PRIMALenv_t env, void *buffer, const char *file, unsigned line);
/* Allocates zeroed memory owned by the task. */
void *PRIMAL_calloctask(PRIMALtask_t task, size_t number, size_t size);
/* Debug variant of PRIMAL_calloctask (file/line are ignored here). */
void *PRIMAL_callocdbgtask(PRIMALtask_t task, size_t number, size_t size,
                           const char *file, unsigned line);
/* Frees a buffer allocated by PRIMAL_calloctask. */
void PRIMAL_freetask(PRIMALtask_t task, void *buffer);
/* Debug variant of PRIMAL_freetask (file/line are ignored here). */
void PRIMAL_freedbgtask(PRIMALtask_t task, void *buffer, const char *file, unsigned line);
/* Initializes the global environment (no global state to set up here). */
PRIMALrescodee PRIMAL_globalenvinitialize(PRIMALint64t maxnumalloc, const char *dbgfile);
/* Finalizes the global environment (no global state to tear down here). */
PRIMALrescodee PRIMAL_globalenvfinalize(void);
/* Checks the environment's memory (always OK: no internal pool). */
PRIMALrescodee PRIMAL_checkmemenv(PRIMALenv_t env, const char *file, int line);
/* Checks the task's memory (always OK: no internal pool). */
PRIMALrescodee PRIMAL_checkmemtask(PRIMALtask_t task, const char *file, int line);
/* informational utilities (reference getversion/isinfinity/getresponseclass):
 * getversion reports THIS solver's version; isinfinity uses our
 * infinity (IEEE, `INF == INFINITY`); getresponseclass maps a code into the
 * reference class (OK 0, WRN 1, TRM 2, ERR 3, UNK 4). */
PRIMALrescodee PRIMAL_getversion(int *major, int *minor, int *revision);
/* True when value is this solver's infinity (IEEE INF). */
int PRIMAL_isinfinity(PRIMALrealt value);
/* Maps a response code to the reference class (OK/WRN/TRM/ERR/UNK). */
PRIMALrescodee PRIMAL_getresponseclass(PRIMALrescodee res, int *responseclass);
/* Problem type, with MSKproblemtypee values. Rule documented in the .c:
 * QC+cones -> MIXED, QC -> QCQO, quadratic objective -> QO, cones/bars -> CONIC,
 * otherwise LO (integer variables do not enter the class). */
typedef enum {
    PRIMAL_PROBTYPE_LO    = 0,
    PRIMAL_PROBTYPE_QO    = 1,
    PRIMAL_PROBTYPE_QCQO  = 2,
    PRIMAL_PROBTYPE_CONIC = 3,
    PRIMAL_PROBTYPE_MIXED = 4
} PRIMALproblemtypee;
/* information items (reference MSKdinfiteme/MSKiinfiteme/MSKliinfiteme and
 * MSKinftypee). Indices are the reference 11.2.4 ones (read from
 * constants.html on 2026-09-19); END is the table limit, not an item. */
typedef enum {
    PRIMAL_INF_DOU_TYPE  = 0,
    PRIMAL_INF_INT_TYPE  = 1,
    PRIMAL_INF_LINT_TYPE = 2
} PRIMALinftypee;
typedef enum {
    PRIMAL_DINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_DENSITY = 0,
    PRIMAL_DINF_BI_CLEAN_TIME = 1,
    PRIMAL_DINF_BI_DUAL_TIME = 2,
    PRIMAL_DINF_BI_PRIMAL_TIME = 3,
    PRIMAL_DINF_BI_TIME = 4,
    PRIMAL_DINF_FOLDING_BI_OPTIMIZE_TIME = 5,
    PRIMAL_DINF_FOLDING_BI_UNFOLD_DUAL_TIME = 6,
    PRIMAL_DINF_FOLDING_BI_UNFOLD_INITIALIZE_TIME = 7,
    PRIMAL_DINF_FOLDING_BI_UNFOLD_PRIMAL_TIME = 8,
    PRIMAL_DINF_FOLDING_BI_UNFOLD_TIME = 9,
    PRIMAL_DINF_FOLDING_FACTOR = 10,
    PRIMAL_DINF_FOLDING_TIME = 11,
    PRIMAL_DINF_INTPNT_DUAL_FEAS = 12,
    PRIMAL_DINF_INTPNT_DUAL_OBJ = 13,
    PRIMAL_DINF_INTPNT_FACTOR_NUM_FLOPS = 14,
    PRIMAL_DINF_INTPNT_OPT_STATUS = 15,
    PRIMAL_DINF_INTPNT_ORDER_TIME = 16,
    PRIMAL_DINF_INTPNT_PRIMAL_FEAS = 17,
    PRIMAL_DINF_INTPNT_PRIMAL_OBJ = 18,
    PRIMAL_DINF_INTPNT_TIME = 19,
    PRIMAL_DINF_MIO_CLIQUE_SELECTION_TIME = 20,
    PRIMAL_DINF_MIO_CLIQUE_SEPARATION_TIME = 21,
    PRIMAL_DINF_MIO_CMIR_SELECTION_TIME = 22,
    PRIMAL_DINF_MIO_CMIR_SEPARATION_TIME = 23,
    PRIMAL_DINF_MIO_CONSTRUCT_SOLUTION_OBJ = 24,
    PRIMAL_DINF_MIO_DUAL_BOUND_AFTER_PRESOLVE = 25,
    PRIMAL_DINF_MIO_GMI_SELECTION_TIME = 26,
    PRIMAL_DINF_MIO_GMI_SEPARATION_TIME = 27,
    PRIMAL_DINF_MIO_IMPLIED_BOUND_SELECTION_TIME = 28,
    PRIMAL_DINF_MIO_IMPLIED_BOUND_SEPARATION_TIME = 29,
    PRIMAL_DINF_MIO_INITIAL_FEASIBLE_SOLUTION_OBJ = 30,
    PRIMAL_DINF_MIO_KNAPSACK_COVER_SELECTION_TIME = 31,
    PRIMAL_DINF_MIO_KNAPSACK_COVER_SEPARATION_TIME = 32,
    PRIMAL_DINF_MIO_LIPRO_SELECTION_TIME = 33,
    PRIMAL_DINF_MIO_LIPRO_SEPARATION_TIME = 34,
    PRIMAL_DINF_MIO_OBJ_ABS_GAP = 35,
    PRIMAL_DINF_MIO_OBJ_BOUND = 36,
    PRIMAL_DINF_MIO_OBJ_INT = 37,
    PRIMAL_DINF_MIO_OBJ_REL_GAP = 38,
    PRIMAL_DINF_MIO_PROBING_TIME = 39,
    PRIMAL_DINF_MIO_ROOT_CUT_SELECTION_TIME = 40,
    PRIMAL_DINF_MIO_ROOT_CUT_SEPARATION_TIME = 41,
    PRIMAL_DINF_MIO_ROOT_OPTIMIZER_TIME = 42,
    PRIMAL_DINF_MIO_ROOT_PRESOLVE_TIME = 43,
    PRIMAL_DINF_MIO_ROOT_TIME = 44,
    PRIMAL_DINF_MIO_SYMMETRY_DETECTION_TIME = 45,
    PRIMAL_DINF_MIO_SYMMETRY_FACTOR = 46,
    PRIMAL_DINF_MIO_TIME = 47,
    PRIMAL_DINF_MIO_USER_OBJ_CUT = 48,
    PRIMAL_DINF_OPTIMIZER_TICKS = 49,
    PRIMAL_DINF_OPTIMIZER_TIME = 50,
    PRIMAL_DINF_PRESOLVE_ELI_TIME = 51,
    PRIMAL_DINF_PRESOLVE_LINDEP_TIME = 52,
    PRIMAL_DINF_PRESOLVE_TIME = 53,
    PRIMAL_DINF_PRESOLVE_TOTAL_PRIMAL_PERTURBATION = 54,
    PRIMAL_DINF_PRIMAL_REPAIR_PENALTY_OBJ = 55,
    PRIMAL_DINF_QCQO_REFORMULATE_MAX_PERTURBATION = 56,
    PRIMAL_DINF_QCQO_REFORMULATE_TIME = 57,
    PRIMAL_DINF_QCQO_REFORMULATE_WORST_CHOLESKY_COLUMN_SCALING = 58,
    PRIMAL_DINF_QCQO_REFORMULATE_WORST_CHOLESKY_DIAG_SCALING = 59,
    PRIMAL_DINF_READ_DATA_TIME = 60,
    PRIMAL_DINF_REMOTE_TIME = 61,
    PRIMAL_DINF_SIM_DUAL_TIME = 62,
    PRIMAL_DINF_SIM_FEAS = 63,
    PRIMAL_DINF_SIM_OBJ = 64,
    PRIMAL_DINF_SIM_PRIMAL_TIME = 65,
    PRIMAL_DINF_SIM_TIME = 66,
    PRIMAL_DINF_SOL_BAS_DUAL_OBJ = 67,
    PRIMAL_DINF_SOL_BAS_DVIOLCON = 68,
    PRIMAL_DINF_SOL_BAS_DVIOLVAR = 69,
    PRIMAL_DINF_SOL_BAS_NRM_BARX = 70,
    PRIMAL_DINF_SOL_BAS_NRM_SLC = 71,
    PRIMAL_DINF_SOL_BAS_NRM_SLX = 72,
    PRIMAL_DINF_SOL_BAS_NRM_SUC = 73,
    PRIMAL_DINF_SOL_BAS_NRM_SUX = 74,
    PRIMAL_DINF_SOL_BAS_NRM_XC = 75,
    PRIMAL_DINF_SOL_BAS_NRM_XX = 76,
    PRIMAL_DINF_SOL_BAS_NRM_Y = 77,
    PRIMAL_DINF_SOL_BAS_PRIMAL_OBJ = 78,
    PRIMAL_DINF_SOL_BAS_PVIOLCON = 79,
    PRIMAL_DINF_SOL_BAS_PVIOLVAR = 80,
    PRIMAL_DINF_SOL_ITG_NRM_BARX = 81,
    PRIMAL_DINF_SOL_ITG_NRM_XC = 82,
    PRIMAL_DINF_SOL_ITG_NRM_XX = 83,
    PRIMAL_DINF_SOL_ITG_PRIMAL_OBJ = 84,
    PRIMAL_DINF_SOL_ITG_PVIOLACC = 85,
    PRIMAL_DINF_SOL_ITG_PVIOLBARVAR = 86,
    PRIMAL_DINF_SOL_ITG_PVIOLCON = 87,
    PRIMAL_DINF_SOL_ITG_PVIOLCONES = 88,
    PRIMAL_DINF_SOL_ITG_PVIOLDJC = 89,
    PRIMAL_DINF_SOL_ITG_PVIOLITG = 90,
    PRIMAL_DINF_SOL_ITG_PVIOLVAR = 91,
    PRIMAL_DINF_SOL_ITR_DUAL_OBJ = 92,
    PRIMAL_DINF_SOL_ITR_DVIOLACC = 93,
    PRIMAL_DINF_SOL_ITR_DVIOLBARVAR = 94,
    PRIMAL_DINF_SOL_ITR_DVIOLCON = 95,
    PRIMAL_DINF_SOL_ITR_DVIOLCONES = 96,
    PRIMAL_DINF_SOL_ITR_DVIOLVAR = 97,
    PRIMAL_DINF_SOL_ITR_NRM_BARS = 98,
    PRIMAL_DINF_SOL_ITR_NRM_BARX = 99,
    PRIMAL_DINF_SOL_ITR_NRM_SLC = 100,
    PRIMAL_DINF_SOL_ITR_NRM_SLX = 101,
    PRIMAL_DINF_SOL_ITR_NRM_SNX = 102,
    PRIMAL_DINF_SOL_ITR_NRM_SUC = 103,
    PRIMAL_DINF_SOL_ITR_NRM_SUX = 104,
    PRIMAL_DINF_SOL_ITR_NRM_XC = 105,
    PRIMAL_DINF_SOL_ITR_NRM_XX = 106,
    PRIMAL_DINF_SOL_ITR_NRM_Y = 107,
    PRIMAL_DINF_SOL_ITR_PRIMAL_OBJ = 108,
    PRIMAL_DINF_SOL_ITR_PVIOLACC = 109,
    PRIMAL_DINF_SOL_ITR_PVIOLBARVAR = 110,
    PRIMAL_DINF_SOL_ITR_PVIOLCON = 111,
    PRIMAL_DINF_SOL_ITR_PVIOLCONES = 112,
    PRIMAL_DINF_SOL_ITR_PVIOLVAR = 113,
    PRIMAL_DINF_TO_CONIC_TIME = 114,
    PRIMAL_DINF_WRITE_DATA_TIME = 115,
    PRIMAL_DINF_END = 116
} PRIMALdinfiteme;

typedef enum {
    PRIMAL_IINF_ANA_PRO_NUM_CON = 0,
    PRIMAL_IINF_ANA_PRO_NUM_CON_EQ = 1,
    PRIMAL_IINF_ANA_PRO_NUM_CON_FR = 2,
    PRIMAL_IINF_ANA_PRO_NUM_CON_LO = 3,
    PRIMAL_IINF_ANA_PRO_NUM_CON_RA = 4,
    PRIMAL_IINF_ANA_PRO_NUM_CON_UP = 5,
    PRIMAL_IINF_ANA_PRO_NUM_VAR = 6,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_BIN = 7,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_CONT = 8,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_EQ = 9,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_FR = 10,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_INT = 11,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_LO = 12,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_RA = 13,
    PRIMAL_IINF_ANA_PRO_NUM_VAR_UP = 14,
    PRIMAL_IINF_FOLDING_APPLIED = 15,
    PRIMAL_IINF_INTPNT_FACTOR_DIM_DENSE = 16,
    PRIMAL_IINF_INTPNT_ITER = 17,
    PRIMAL_IINF_INTPNT_NUM_THREADS = 18,
    PRIMAL_IINF_INTPNT_SOLVE_DUAL = 19,
    PRIMAL_IINF_MIO_ABSGAP_SATISFIED = 20,
    PRIMAL_IINF_MIO_CLIQUE_TABLE_SIZE = 21,
    PRIMAL_IINF_MIO_CONSTRUCT_SOLUTION = 22,
    PRIMAL_IINF_MIO_FINAL_NUMBIN = 23,
    PRIMAL_IINF_MIO_FINAL_NUMBINCONEVAR = 24,
    PRIMAL_IINF_MIO_FINAL_NUMCON = 25,
    PRIMAL_IINF_MIO_FINAL_NUMCONE = 26,
    PRIMAL_IINF_MIO_FINAL_NUMCONEVAR = 27,
    PRIMAL_IINF_MIO_FINAL_NUMCONT = 28,
    PRIMAL_IINF_MIO_FINAL_NUMCONTCONEVAR = 29,
    PRIMAL_IINF_MIO_FINAL_NUMDEXPCONES = 30,
    PRIMAL_IINF_MIO_FINAL_NUMDJC = 31,
    PRIMAL_IINF_MIO_FINAL_NUMDPOWCONES = 32,
    PRIMAL_IINF_MIO_FINAL_NUMINT = 33,
    PRIMAL_IINF_MIO_FINAL_NUMINTCONEVAR = 34,
    PRIMAL_IINF_MIO_FINAL_NUMPEXPCONES = 35,
    PRIMAL_IINF_MIO_FINAL_NUMPPOWCONES = 36,
    PRIMAL_IINF_MIO_FINAL_NUMQCONES = 37,
    PRIMAL_IINF_MIO_FINAL_NUMRQCONES = 38,
    PRIMAL_IINF_MIO_FINAL_NUMVAR = 39,
    PRIMAL_IINF_MIO_INITIAL_FEASIBLE_SOLUTION = 40,
    PRIMAL_IINF_MIO_NODE_DEPTH = 41,
    PRIMAL_IINF_MIO_NUM_ACTIVE_NODES = 42,
    PRIMAL_IINF_MIO_NUM_ACTIVE_ROOT_CUTS = 43,
    PRIMAL_IINF_MIO_NUM_BLOCKS_SOLVED_IN_BB = 44,
    PRIMAL_IINF_MIO_NUM_BLOCKS_SOLVED_IN_PRESOLVE = 45,
    PRIMAL_IINF_MIO_NUM_BRANCH = 46,
    PRIMAL_IINF_MIO_NUM_INT_SOLUTIONS = 47,
    PRIMAL_IINF_MIO_NUM_RELAX = 48,
    PRIMAL_IINF_MIO_NUM_REPEATED_PRESOLVE = 49,
    PRIMAL_IINF_MIO_NUM_RESTARTS = 50,
    PRIMAL_IINF_MIO_NUM_ROOT_CUT_ROUNDS = 51,
    PRIMAL_IINF_MIO_NUM_SELECTED_CLIQUE_CUTS = 52,
    PRIMAL_IINF_MIO_NUM_SELECTED_CMIR_CUTS = 53,
    PRIMAL_IINF_MIO_NUM_SELECTED_GOMORY_CUTS = 54,
    PRIMAL_IINF_MIO_NUM_SELECTED_IMPLIED_BOUND_CUTS = 55,
    PRIMAL_IINF_MIO_NUM_SELECTED_KNAPSACK_COVER_CUTS = 56,
    PRIMAL_IINF_MIO_NUM_SELECTED_LIPRO_CUTS = 57,
    PRIMAL_IINF_MIO_NUM_SEPARATED_CLIQUE_CUTS = 58,
    PRIMAL_IINF_MIO_NUM_SEPARATED_CMIR_CUTS = 59,
    PRIMAL_IINF_MIO_NUM_SEPARATED_GOMORY_CUTS = 60,
    PRIMAL_IINF_MIO_NUM_SEPARATED_IMPLIED_BOUND_CUTS = 61,
    PRIMAL_IINF_MIO_NUM_SEPARATED_KNAPSACK_COVER_CUTS = 62,
    PRIMAL_IINF_MIO_NUM_SEPARATED_LIPRO_CUTS = 63,
    PRIMAL_IINF_MIO_NUM_SOLVED_NODES = 64,
    PRIMAL_IINF_MIO_NUMBIN = 65,
    PRIMAL_IINF_MIO_NUMBINCONEVAR = 66,
    PRIMAL_IINF_MIO_NUMCON = 67,
    PRIMAL_IINF_MIO_NUMCONE = 68,
    PRIMAL_IINF_MIO_NUMCONEVAR = 69,
    PRIMAL_IINF_MIO_NUMCONT = 70,
    PRIMAL_IINF_MIO_NUMCONTCONEVAR = 71,
    PRIMAL_IINF_MIO_NUMDEXPCONES = 72,
    PRIMAL_IINF_MIO_NUMDJC = 73,
    PRIMAL_IINF_MIO_NUMDPOWCONES = 74,
    PRIMAL_IINF_MIO_NUMINT = 75,
    PRIMAL_IINF_MIO_NUMINTCONEVAR = 76,
    PRIMAL_IINF_MIO_NUMPEXPCONES = 77,
    PRIMAL_IINF_MIO_NUMPPOWCONES = 78,
    PRIMAL_IINF_MIO_NUMQCONES = 79,
    PRIMAL_IINF_MIO_NUMRQCONES = 80,
    PRIMAL_IINF_MIO_NUMVAR = 81,
    PRIMAL_IINF_MIO_OBJ_BOUND_DEFINED = 82,
    PRIMAL_IINF_MIO_PRESOLVED_NUMBIN = 83,
    PRIMAL_IINF_MIO_PRESOLVED_NUMBINCONEVAR = 84,
    PRIMAL_IINF_MIO_PRESOLVED_NUMCON = 85,
    PRIMAL_IINF_MIO_PRESOLVED_NUMCONE = 86,
    PRIMAL_IINF_MIO_PRESOLVED_NUMCONEVAR = 87,
    PRIMAL_IINF_MIO_PRESOLVED_NUMCONT = 88,
    PRIMAL_IINF_MIO_PRESOLVED_NUMCONTCONEVAR = 89,
    PRIMAL_IINF_MIO_PRESOLVED_NUMDEXPCONES = 90,
    PRIMAL_IINF_MIO_PRESOLVED_NUMDJC = 91,
    PRIMAL_IINF_MIO_PRESOLVED_NUMDPOWCONES = 92,
    PRIMAL_IINF_MIO_PRESOLVED_NUMINT = 93,
    PRIMAL_IINF_MIO_PRESOLVED_NUMINTCONEVAR = 94,
    PRIMAL_IINF_MIO_PRESOLVED_NUMPEXPCONES = 95,
    PRIMAL_IINF_MIO_PRESOLVED_NUMPPOWCONES = 96,
    PRIMAL_IINF_MIO_PRESOLVED_NUMQCONES = 97,
    PRIMAL_IINF_MIO_PRESOLVED_NUMRQCONES = 98,
    PRIMAL_IINF_MIO_PRESOLVED_NUMVAR = 99,
    PRIMAL_IINF_MIO_RELGAP_SATISFIED = 100,
    PRIMAL_IINF_MIO_TOTAL_NUM_SELECTED_CUTS = 101,
    PRIMAL_IINF_MIO_TOTAL_NUM_SEPARATED_CUTS = 102,
    PRIMAL_IINF_MIO_USER_OBJ_CUT = 103,
    PRIMAL_IINF_OPT_NUMCON = 104,
    PRIMAL_IINF_OPT_NUMVAR = 105,
    PRIMAL_IINF_OPTIMIZE_RESPONSE = 106,
    PRIMAL_IINF_PRESOLVE_NUM_PRIMAL_PERTURBATIONS = 107,
    PRIMAL_IINF_PURIFY_DUAL_SUCCESS = 108,
    PRIMAL_IINF_PURIFY_PRIMAL_SUCCESS = 109,
    PRIMAL_IINF_RD_NUMBARVAR = 110,
    PRIMAL_IINF_RD_NUMCON = 111,
    PRIMAL_IINF_RD_NUMCONE = 112,
    PRIMAL_IINF_RD_NUMINTVAR = 113,
    PRIMAL_IINF_RD_NUMQ = 114,
    PRIMAL_IINF_RD_NUMVAR = 115,
    PRIMAL_IINF_RD_PROTYPE = 116,
    PRIMAL_IINF_SIM_DUAL_DEG_ITER = 117,
    PRIMAL_IINF_SIM_DUAL_HOTSTART = 118,
    PRIMAL_IINF_SIM_DUAL_HOTSTART_LU = 119,
    PRIMAL_IINF_SIM_DUAL_INF_ITER = 120,
    PRIMAL_IINF_SIM_DUAL_ITER = 121,
    PRIMAL_IINF_SIM_NUMCON = 122,
    PRIMAL_IINF_SIM_NUMVAR = 123,
    PRIMAL_IINF_SIM_PRIMAL_DEG_ITER = 124,
    PRIMAL_IINF_SIM_PRIMAL_HOTSTART = 125,
    PRIMAL_IINF_SIM_PRIMAL_HOTSTART_LU = 126,
    PRIMAL_IINF_SIM_PRIMAL_INF_ITER = 127,
    PRIMAL_IINF_SIM_PRIMAL_ITER = 128,
    PRIMAL_IINF_SIM_SOLVE_DUAL = 129,
    PRIMAL_IINF_SOL_BAS_PROSTA = 130,
    PRIMAL_IINF_SOL_BAS_SOLSTA = 131,
    PRIMAL_IINF_SOL_ITG_PROSTA = 132,
    PRIMAL_IINF_SOL_ITG_SOLSTA = 133,
    PRIMAL_IINF_SOL_ITR_PROSTA = 134,
    PRIMAL_IINF_SOL_ITR_SOLSTA = 135,
    PRIMAL_IINF_STO_NUM_A_REALLOC = 136,
    PRIMAL_IINF_END = 137
} PRIMALiinfiteme;

typedef enum {
    PRIMAL_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_COLUMNS = 0,
    PRIMAL_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_NZ = 1,
    PRIMAL_LIINF_ANA_PRO_SCALARIZED_CONSTRAINT_MATRIX_NUM_ROWS = 2,
    PRIMAL_LIINF_BI_CLEAN_ITER = 3,
    PRIMAL_LIINF_BI_DUAL_ITER = 4,
    PRIMAL_LIINF_BI_PRIMAL_ITER = 5,
    PRIMAL_LIINF_FOLDING_BI_DUAL_ITER = 6,
    PRIMAL_LIINF_FOLDING_BI_OPTIMIZER_ITER = 7,
    PRIMAL_LIINF_FOLDING_BI_PRIMAL_ITER = 8,
    PRIMAL_LIINF_INTPNT_FACTOR_NUM_NZ = 9,
    PRIMAL_LIINF_MIO_ANZ = 10,
    PRIMAL_LIINF_MIO_FINAL_ANZ = 11,
    PRIMAL_LIINF_MIO_INTPNT_ITER = 12,
    PRIMAL_LIINF_MIO_NUM_DUAL_ILLPOSED_CER = 13,
    PRIMAL_LIINF_MIO_NUM_PRIM_ILLPOSED_CER = 14,
    PRIMAL_LIINF_MIO_PRESOLVED_ANZ = 15,
    PRIMAL_LIINF_MIO_SIMPLEX_ITER = 16,
    PRIMAL_LIINF_RD_NUMACC = 17,
    PRIMAL_LIINF_RD_NUMANZ = 18,
    PRIMAL_LIINF_RD_NUMDJC = 19,
    PRIMAL_LIINF_RD_NUMQNZ = 20,
    PRIMAL_LIINF_SIMPLEX_ITER = 21,
    PRIMAL_LIINF_END = 22
} PRIMALliinfiteme;

/* Reads the double information item `which`. */
PRIMALrescodee PRIMAL_getdouinf(PRIMALtask_t t, PRIMALdinfiteme which, PRIMALrealt *value);
/* Reads the int information item `which`. */
PRIMALrescodee PRIMAL_getintinf(PRIMALtask_t t, PRIMALiinfiteme which, int *value);
/* Reads the long (64-bit) information item `which`. */
PRIMALrescodee PRIMAL_getlintinf(PRIMALtask_t t, PRIMALliinfiteme which, PRIMALint64t *value);
/* Reads a double information item by name. */
PRIMALrescodee PRIMAL_getnadouinf(PRIMALtask_t t, const char *name, PRIMALrealt *value);
/* Reads an int information item by name. */
PRIMALrescodee PRIMAL_getnaintinf(PRIMALtask_t t, const char *name, int *value);
/* Resolves an information-item name to its index within its type. */
PRIMALrescodee PRIMAL_getinfindex(PRIMALtask_t t, PRIMALinftypee inftype, const char *name, int *index);
/* Writes the name of information item `whichinf` of the given type. */
PRIMALrescodee PRIMAL_getinfname(PRIMALtask_t t, PRIMALinftypee inftype, int whichinf, char *name);
/* Returns the number of items of one information type (its table limit). */
PRIMALrescodee PRIMAL_getinfmax(PRIMALtask_t t, PRIMALinftypee inftype, int *infmax);
#define PRIMAL_MAX_INFNAME_LEN 80
/* name as string of an information item and of a callback code (reference
 * dinfitemtostr/iinfitemtostr/liinfitemtostr/callbackcodetostr): no task. */
PRIMALrescodee PRIMAL_dinfitemtostr(PRIMALdinfiteme item, char *str);
/* Name as string of an int information item. */
PRIMALrescodee PRIMAL_iinfitemtostr(PRIMALiinfiteme item, char *str);
/* Name as string of a long (64-bit) information item. */
PRIMALrescodee PRIMAL_liinfitemtostr(PRIMALliinfiteme item, char *str);
/* Symbolic constants (reference getsymbcondim/getsymbcon/symnamtovalue/
 * iparvaltosymnam). The table is the reference 11.2.4 one, extracted from
 * its own library (1438 entries, maxlen 60). The longest name fits in
 * PRIMAL_MAX_SYMBNAME_LEN characters (including the terminator). */
#define PRIMAL_MAX_SYMBNAME_LEN 64
PRIMALrescodee PRIMAL_getsymbcondim(PRIMALenv_t env, int *num, size_t *maxlen);
/* Reads symbolic constant i: its name and (if requested) its value. */
PRIMALrescodee PRIMAL_getsymbcon(PRIMALtask_t t, int i, int sizevalue, char *name, int *value);
/* Resolves a symbolic-constant name to its value string; returns success. */
int PRIMAL_symnamtovalue(const char *name, char *value);
/* Maps an int parameter value to its symbolic-constant name. */
PRIMALrescodee PRIMAL_iparvaltosymnam(PRIMALenv_t env, int whichparam, int whichvalue,
                                      char *symbolicname);
/* Writes the problem type with MSKproblemtypee values. */
PRIMALrescodee PRIMAL_getprobtype(PRIMALtask_t t, PRIMALproblemtypee *probtype);
/* version/build/error (reference checkversion/getbuildinfo/getcodedesc/
 * getlasterror) and the A truncation threshold (get/putatruncatetol: stored,
 * not applied -- this solver does not truncate A). */
PRIMALrescodee PRIMAL_checkversion(PRIMALenv_t env, int major, int minor, int revision);
/* Writes the build state and date strings. */
PRIMALrescodee PRIMAL_getbuildinfo(char *buildstate, char *builddate);
/* byte estimate of the task memory use (reference getmemusagetask). */
PRIMALrescodee PRIMAL_getmemusagetask(PRIMALtask_t t, PRIMALint64t *meminuse, PRIMALint64t *maxmemuse);
/* Writes the symbolic name and text of a response code. */
PRIMALrescodee PRIMAL_getcodedesc(PRIMALrescodee code, char *symname, char *str);
/* Reads the last error code and message recorded on the task. */
PRIMALrescodee PRIMAL_getlasterror(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    int sizelastmsg, int *lastmsglen, char *lastmsg);
/* 64-bit variant of PRIMAL_getlasterror (sizes and length are 64-bit). */
PRIMALrescodee PRIMAL_getlasterror64(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    PRIMALint64t sizelastmsg, PRIMALint64t *lastmsglen, char *lastmsg);
/* summaries on stream (print to stdout). */
PRIMALrescodee PRIMAL_solutionsummary(PRIMALtask_t t, int whichstream);
/* Prints a summary of one solution key on the stream (stdout). */
PRIMALrescodee PRIMAL_onesolutionsummary(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
/* Prints a summary of the optimizer run on the stream (stdout). */
PRIMALrescodee PRIMAL_optimizersummary(PRIMALtask_t t, int whichstream);
/* diagnostics on stream (print to stdout). */
PRIMALrescodee PRIMAL_analyzeproblem(PRIMALtask_t t, int whichstream);
/* Prints the solution analysis of one key on the stream (stdout). */
PRIMALrescodee PRIMAL_analyzesolution(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
/* Prints the infeasibility report of one key on the stream (stdout). */
PRIMALrescodee PRIMAL_infeasibilityreport(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
/* Prints the sensitivity report on the stream (stdout). */
PRIMALrescodee PRIMAL_sensitivityreport(PRIMALtask_t t, int whichstream);
/* Reads the stored A-truncation threshold (not applied by this solver). */
PRIMALrescodee PRIMAL_getatruncatetol(PRIMALtask_t t, PRIMALrealt *tolzero);
/* Stores the A-truncation threshold (not applied by this solver). */
PRIMALrescodee PRIMAL_putatruncatetol(PRIMALtask_t t, PRIMALrealt tolzero);
/* Symbolic names (reference *tostr). The TEXT is this solver's own: the
 * reference exact form was not read, so it is neither invented nor
 * borrowed (declared deviation). The buffer must hold
 * PRIMAL_MAX_STR_LEN characters. */
#define PRIMAL_MAX_STR_LEN 1024
PRIMALrescodee PRIMAL_prostatostr(PRIMALtask_t t, PRIMALprostae prosta, char *str);
/* Name as string of a solution status. */
PRIMALrescodee PRIMAL_solstatostr(PRIMALtask_t t, PRIMALsolstae solsta, char *str);
/* Name as string of a bound key. */
PRIMALrescodee PRIMAL_bktostr(PRIMALtask_t t, PRIMALboundkeye bk, char *str);
/* Name as string of a cone type. */
PRIMALrescodee PRIMAL_conetypetostr(PRIMALtask_t t, PRIMALconetypee ct, char *str);
/* inverses of the symbolic names (reference strtoconetype/strtosk). */
PRIMALrescodee PRIMAL_strtoconetype(PRIMALtask_t t, const char *str, PRIMALconetypee *ct);
/* Name as string of a problem type. */
PRIMALrescodee PRIMAL_probtypetostr(PRIMALtask_t t, PRIMALproblemtypee pt, char *str);
/* Name as string of a response code. */
PRIMALrescodee PRIMAL_rescodetostr(PRIMALrescodee res, char *str);
/* Appends `num` variables to the task. */
PRIMALrescodee PRIMAL_appendvars(PRIMALtask_t t, int num);
/* Appends `num` constraints to the task. */
PRIMALrescodee PRIMAL_appendcons(PRIMALtask_t t, int num);
/* Returns the current number of variables. */
PRIMALrescodee PRIMAL_getnumvar(PRIMALtask_t t, int *numvar);
/* Returns the current number of constraints. */
PRIMALrescodee PRIMAL_getnumcon(PRIMALtask_t t, int *numcon);
/* Reference "preallocated" counters. Here arrays grow in place,
 * so the reserved number IS the current one (declared deviation) except
 * for cones and bars, where the capacity is real. */
PRIMALrescodee PRIMAL_getmaxnumvar(PRIMALtask_t t, int *n);
/* Reference preallocated capacity in rows (equals the current count here). */
PRIMALrescodee PRIMAL_getmaxnumcon(PRIMALtask_t t, int *n);
/* Reference preallocated capacity in cones (the real capacity here). */
PRIMALrescodee PRIMAL_getmaxnumcone(PRIMALtask_t t, int *n);
/* Reference preallocated capacity in bar variables (the real capacity here). */
PRIMALrescodee PRIMAL_getmaxnumbarvar(PRIMALtask_t t, int *n);

/* data input (column-wise linear part) */
PRIMALrescodee PRIMAL_putcj(PRIMALtask_t t, int j, PRIMALrealt cj);
/* Sets the objective constant term. */
PRIMALrescodee PRIMAL_putcfix(PRIMALtask_t t, PRIMALrealt cfix);
/* c as a vector: list (putclist), slice (putcslice) and reads (getc/getcslice).
 * The write list validates all input before touching c. */
PRIMALrescodee PRIMAL_putclist(PRIMALtask_t t, int num, const int *subj, const PRIMALrealt *val);
/* Sets c[first..last) from the given slice. */
PRIMALrescodee PRIMAL_putcslice(PRIMALtask_t t, int first, int last, const PRIMALrealt *c);
/* Reads the whole objective vector c. */
PRIMALrescodee PRIMAL_getc(PRIMALtask_t t, PRIMALrealt *c);
/* Reads c[first..last) into the given slice. */
PRIMALrescodee PRIMAL_getcslice(PRIMALtask_t t, int first, int last, PRIMALrealt *c);
/* Replaces column j of A from a sparse index/value list. */
PRIMALrescodee PRIMAL_putacol(PRIMALtask_t t, int j, int nz, const int *sub, const PRIMALrealt *val);
/* Replaces row i of A from a sparse index/value list. */
PRIMALrescodee PRIMAL_putarow(PRIMALtask_t t, int i, int nz, const int *sub, const PRIMALrealt *val);
/* Block forms (reference putarowlist/putacollist and the slices): data in
 * CSR style, `asub[ptrb[k]..ptre[k])` for the k-th row/column; the whole list
 * is validated before writing. */
PRIMALrescodee PRIMAL_putarowlist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
/* Writes several columns of A in CSR-style blocked form. */
PRIMALrescodee PRIMAL_putacollist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
/* Writes the rows of A in [first,last) from blocked CSR-style data. */
PRIMALrescodee PRIMAL_putarowslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
/* Writes the columns of A in [first,last) from blocked CSR-style data. */
PRIMALrescodee PRIMAL_putacolslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
/* a_ij = aij: replaces every stored entry of the pair (a single term
 * remains), and at 0 removes it. */
PRIMALrescodee PRIMAL_putaij(PRIMALtask_t t, int i, int j, PRIMALrealt aij);
/* list of scalar coefficients (reference putaijlist), validated before
 * writing. */
PRIMALrescodee PRIMAL_putaijlist(PRIMALtask_t t, int num, const int *subi,
                                 const int *subj, const PRIMALrealt *valij);
/* 64-bit variants (reference *64): the same readers/writers with
 * `PRIMALint64t` row pointers. */
PRIMALrescodee PRIMAL_putarowslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
/* 64-bit variant of putacolslice (64-bit row pointers). */
PRIMALrescodee PRIMAL_putacolslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
/* 64-bit variant of putarowlist (64-bit row pointers). */
PRIMALrescodee PRIMAL_putarowlist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
/* 64-bit variant of putacollist (64-bit row pointers). */
PRIMALrescodee PRIMAL_putacollist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
/* 64-bit variant of putaijlist (64-bit count). */
PRIMALrescodee PRIMAL_putaijlist64(PRIMALtask_t t, PRIMALint64t num,
    const int *subi, const int *subj, const PRIMALrealt *valij);
/* Accumulates objective quadratic terms from (qi,qj,qoval) triplets. */
PRIMALrescodee PRIMAL_putqobj(PRIMALtask_t t, int numqcnz, const int *qi, const int *qj, const PRIMALrealt *qoval);
/* loads the linear part in one call (reference inputdata/inputdata64):
 * the task must be empty; A is column-wise (aptrb[j]..aptre[j]). */
PRIMALrescodee PRIMAL_inputdata(PRIMALtask_t t, int maxnumcon, int maxnumvar,
    int numcon, int numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux);
/* 64-bit variant of inputdata (64-bit dimensions). */
PRIMALrescodee PRIMAL_inputdata64(PRIMALtask_t t, PRIMALint64t maxnumcon, PRIMALint64t maxnumvar,
    PRIMALint64t numcon, PRIMALint64t numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux);
/* q_ij = q_ji = qoij, lower triangle only (i >= j); replaces the pair. */
PRIMALrescodee PRIMAL_putqobjij(PRIMALtask_t t, int i, int j, PRIMALrealt qoij);
/* Sets the bound of variable j (key plus the two values). */
PRIMALrescodee PRIMAL_putvarbound(PRIMALtask_t t, int j, PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
/* Sets the bound of constraint i (key plus the two values). */
PRIMALrescodee PRIMAL_putconbound(PRIMALtask_t t, int i, PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
/* change ONE bound side (reference chgvarbound/chgconbound): lower!=0 ->
 * new lower = (finite ? value : -inf); else new upper = (finite ? value
 * : +inf); the bound key is recomputed. */
PRIMALrescodee PRIMAL_chgvarbound(PRIMALtask_t t, int j, int lower, int finite, PRIMALrealt value);
/* Change one bound side of a constraint; the bound key is recomputed. */
PRIMALrescodee PRIMAL_chgconbound(PRIMALtask_t t, int i, int lower, int finite, PRIMALrealt value);
/* Bound slices: [first, last), the buffer holds last-first entries. Reading
 * refuses before writing anything; writing validates the whole slice first, so a
 * refusal leaves the model untouched. */
PRIMALrescodee PRIMAL_getvarboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
/* Reads the constraint bounds in [first,last). */
PRIMALrescodee PRIMAL_getconboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
/* Writes the variable bounds in [first,last) after validating the whole slice. */
PRIMALrescodee PRIMAL_putvarboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
/* Writes the constraint bounds in [first,last) after validating the whole slice. */
PRIMALrescodee PRIMAL_putconboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
/* Bound lists (reference putvarboundlist/putconboundlist, all validated
 * before applying) and constant-bound slice (put*boundsliceconst). */
PRIMALrescodee PRIMAL_putvarboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
/* Writes the bounds of the listed constraints after validating the whole list. */
PRIMALrescodee PRIMAL_putconboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
/* Writes the same bound to the variables in [first,last). */
PRIMALrescodee PRIMAL_putvarboundsliceconst(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
/* Writes the same bound to the constraints in [first,last). */
PRIMALrescodee PRIMAL_putconboundsliceconst(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
/* quadratic constraint terms on row i:  a_i'x + 1/2 x'Q_i x in [blc_i, buc_i].
 * Same convention as putqobj: (qi,qj,qval) triplets, full coefficient for
 * cross terms (qval on (i,j), i != j, is the full x_i*x_j coefficient).
 * Convexity requirements: UP row -> Q PSD, LO row -> Q NSD; FX/RA/FR rows
 * with Q != 0 are rejected (documented deviation). */
PRIMALrescodee PRIMAL_putqconk(PRIMALtask_t t, int k, int numqcnz,
                         const int *qsubi, const int *qsubj,
                         const PRIMALrealt *qval);
/* Replaces ALL quadratic terms of ALL constraints from a row-indexed
 * triplet list (qcsubk, qcsubi, qcsubj, qcval), lower triangle only;
 * empty list zeroes. The whole list is validated before applying. */
PRIMALrescodee PRIMAL_putqcon(PRIMALtask_t t, int numqcnz,
                         const int *qcsubk, const int *qcsubi, const int *qcsubj,
                         const PRIMALrealt *qcval);
/* Number of stored quadratic terms of constraint k. */
PRIMALrescodee PRIMAL_getnumqconknz(PRIMALtask_t t, int k, int *numqcnz);
/* Reads the quadratic coefficient q_ij of constraint k. */
PRIMALrescodee PRIMAL_getqconkij(PRIMALtask_t t, int k, int i, int j, PRIMALrealt *qij);
/* Sets the objective sense (minimize/maximize). */
PRIMALrescodee PRIMAL_putobjsense(PRIMALtask_t t, PRIMALobjsensee sense);
/* Sets the type of variable j. */
PRIMALrescodee PRIMAL_putvartype(PRIMALtask_t t, int j, PRIMALvariabletypee vt);
/* Reads the type of variable j. */
PRIMALrescodee PRIMAL_getvartype(PRIMALtask_t t, int j, PRIMALvariabletypee *vt);
/* Type for a variable list (reference putvartypelist/getvartypelist):
 * the write validates indices and types of the whole list before applying. */
PRIMALrescodee PRIMAL_putvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, const PRIMALvariabletypee *vartype);
/* Reads the types of the listed variables. */
PRIMALrescodee PRIMAL_getvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, PRIMALvariabletypee *vartype);
/* Returns the number of integer variables. */
PRIMALrescodee PRIMAL_getnumintvar(PRIMALtask_t t, int *num);
/* Sets the objective constant term. */
PRIMALrescodee PRIMAL_putcfix(PRIMALtask_t t, PRIMALrealt cfix);
/* Sets an int parameter by id. */
PRIMALrescodee PRIMAL_putintparam(PRIMALtask_t t, int param, int value);
/* Sets a double parameter by id. */
PRIMALrescodee PRIMAL_putdouparam(PRIMALtask_t t, int param, PRIMALrealt value);
/* restore one parameter (or all) to the declarative-table default
 * (reference resetintparam/resetdouparam/resetparameters). */
PRIMALrescodee PRIMAL_resetintparam(PRIMALtask_t t, int param);
/* Restores one double parameter to the table default. */
PRIMALrescodee PRIMAL_resetdouparam(PRIMALtask_t t, int param);
/* Restores all parameters to the table defaults. */
PRIMALrescodee PRIMAL_resetparameters(PRIMALtask_t t);

/* data getters (needed for independent verification, e.g. KKT checks) */
PRIMALrescodee PRIMAL_getcj(PRIMALtask_t t, int j, PRIMALrealt *cj);
/* Reads a_ij as the OPERATOR (sum of the stored entries of the pair). */
PRIMALrescodee PRIMAL_getaij(PRIMALtask_t t, int i, int j, PRIMALrealt *aij);
/* Reads the objective quadratic coefficient q_ij (the pair's sum). */
PRIMALrescodee PRIMAL_getqobjij(PRIMALtask_t t, int i, int j, PRIMALrealt *qij);
/* Reads the bound of variable j. */
PRIMALrescodee PRIMAL_getvarbound(PRIMALtask_t t, int j, PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
/* Reads the bound of constraint i. */
PRIMALrescodee PRIMAL_getconbound(PRIMALtask_t t, int i, PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
/* Reads the objective sense. */
PRIMALrescodee PRIMAL_getobjsense(PRIMALtask_t t, PRIMALobjsensee *sense);
/* Reads the objective constant term. */
PRIMALrescodee PRIMAL_getcfix(PRIMALtask_t t, PRIMALrealt *cfix);
/* Reads an int parameter by id. */
PRIMALrescodee PRIMAL_getintparam(PRIMALtask_t t, int param, int *value);
/* Reads a double parameter by id. */
PRIMALrescodee PRIMAL_getdouparam(PRIMALtask_t t, int param, PRIMALrealt *value);

/* ---- model data access (reference surface) ----
 * Two DISTINCT contracts, and the line runs between the store and the operator.
 * PRIMAL_getnumanz counts the STORED ENTRIES of A (sum of the columns' nz),
 * so a coefficient written twice counts twice: storage is column-wise and no
 * dedup happens in putarow/putacol; PRIMAL_getarow/PRIMAL_getacol return that
 * store as written (both entries). PRIMAL_getaij instead answers for the
 * OPERATOR, i.e. the SUM of the entries of that (i,j): it is the coefficient
 * every route solves, and saying the first while the answer is
 * the sum would be a contradiction. Same line already drawn by PRIMAL_getqobjij
 * (sum) versus PRIMAL_getqobj + PRIMAL_getnumqobjnz (the triplet store).
 * PRIMAL_getmaxnumanz is the capacity already allocated on the same columns:
 * the invariant is numanzs <= maxnumanzs after every successful write.
 * PRIMAL_getnumqobjnz counts the Q triplets stored by putqobj (which
 * may be more than the nonzero coefficients of Q, for the same
 * reason). */
PRIMALrescodee PRIMAL_getnumanz(PRIMALtask_t t, int *numanzs);
/* Capacity already allocated on the columns of A (>= numanz). */
PRIMALrescodee PRIMAL_getmaxnumanz(PRIMALtask_t t, int *maxnumanzs);
/* Number of Q triplets stored by putqobj. */
PRIMALrescodee PRIMAL_getnumqobjnz(PRIMALtask_t t, int *numqobjnz);
/* 64-bit counter variants (reference getnumanz64/...): the same
 * numbers, widened. */
PRIMALrescodee PRIMAL_getnumanz64(PRIMALtask_t t, PRIMALint64t *numanzs);
/* 64-bit variant of getmaxnumanz. */
PRIMALrescodee PRIMAL_getmaxnumanz64(PRIMALtask_t t, PRIMALint64t *n);
/* 64-bit variant of getnumqobjnz. */
PRIMALrescodee PRIMAL_getnumqobjnz64(PRIMALtask_t t, PRIMALint64t *n);
/* 64-bit variant of getnumqconknz for constraint k. */
PRIMALrescodee PRIMAL_getnumqconknz64(PRIMALtask_t t, int k, PRIMALint64t *n);
/* Store counts of A (reference getarownumnz/getacolnumnz and their
 * slices) and A as triplets (getatrip): they count the STORED entries, like
 * getnumanz, duplicates included. `getatrip` has no outgoing count: the
 * caller sizes with getnumanz, and insufficient capacity is refused
 * without writing. */
PRIMALrescodee PRIMAL_getacolnumnz(PRIMALtask_t t, int j, int *nzj);
/* Number of stored entries of row i (duplicates counted). */
PRIMALrescodee PRIMAL_getarownumnz(PRIMALtask_t t, int i, int *nzi);
/* Number of stored entries in the columns [first,last). */
PRIMALrescodee PRIMAL_getacolslicenumnz(PRIMALtask_t t, int first, int last, int *numnz);
/* 64-bit variant of getacolslicenumnz. */
PRIMALrescodee PRIMAL_getacolslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz);
/* 64-bit variant of getarowslicenumnz. */
PRIMALrescodee PRIMAL_getarowslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz);
/* Reads the rows in [first,last) as 64-bit CSC blocks. */
PRIMALrescodee PRIMAL_getarowslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val);
/* Reads the columns in [first,last) as 64-bit CSC blocks. */
PRIMALrescodee PRIMAL_getacolslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val);
/* Number of stored entries in the rows [first,last). */
PRIMALrescodee PRIMAL_getarowslicenumnz(PRIMALtask_t t, int first, int last, int *numnz);
/* Reads all stored A entries as (row,col,val) triplets; refusal if maxnumnz is short. */
PRIMALrescodee PRIMAL_getatrip(PRIMALtask_t t, PRIMALint64t maxnumnz,
                               int *subi, int *subj, PRIMALrealt *val);
/* Reading one A row/column into maxnum-sized buffers: numret says
 * how many entries were written. */
PRIMALrescodee PRIMAL_getarow(PRIMALtask_t t, int i, int *sub, PRIMALrealt *val,
                        int maxnum, int *numret);
/* Reads the stored entries of column j into maxnum-sized buffers. */
PRIMALrescodee PRIMAL_getacol(PRIMALtask_t t, int j, int *sub, PRIMALrealt *val,
                        int maxnum, int *numret);
/* A over a row/column slice as triplets (getarowslicetrip/getacolslicetrip):
 * they count first and refuse without writing if `maxnumnz` is short. */
PRIMALrescodee PRIMAL_getarowslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val);
/* A over the columns in [first,last) as triplets. */
PRIMALrescodee PRIMAL_getacolslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val);
/* Slice variant (get*slice): the entries of row i (of column j)
 * whose index falls in [first,last), written starting at the
 * `offset` position of the sub/val buffers, which have maxnum capacity.
 * Offset is the tail of a previous slice, so maxnum-offset is the space left.
 * If the space is short the call is REFUSED (PRIMAL_RES_ERR_ARG) and the
 * user buffers stay untouched, because a partial write to
 * undo is not recoverable by the caller: it is the same rule as the
 * parameter setters (out-of-range = no effect).
 * Ordering: by increasing index (columns for a row, rows for a
 * column). An entry written twice is returned twice, because
 * no dedup happens in putarow/putacol. */
PRIMALrescodee PRIMAL_getarowslice(PRIMALtask_t t, int i, int first, int last,
                             int offset, int maxnum, int *numret,
                             int *sub, PRIMALrealt *val);
/* Same slice read as getarowslice, for column j. */
PRIMALrescodee PRIMAL_getacolslice(PRIMALtask_t t, int j, int first, int last,
                             int offset, int maxnum, int *numret,
                             int *sub, PRIMALrealt *val);
/* FULL read of the two Qs, in the same form as getarow: maxnum is the
 * buffer capacity and *numret how many triplets were written. If maxnum
 * is short the call is REFUSED and no buffer is written, *numret
 * included (the same rule as the A slices).
 * PRIMAL_getqobj reads the triplet listing AS WRITTEN, in
 * write order: it is the table PRIMAL_getnumqobjnz counts, so a
 * written 0.0 value is returned and counted (the number counts the user's
 * writes, not the operator nonzeros), and a cross term
 * (i,j) with i != j appears ONCE even though getqobjij answers the same
 * number on both halves: the store is not symmetrized, the operator is.
 * PRIMAL_getqconk reads the UPPER TRIANGLE (i <= j) in increasing order,
 * and the count comes from PRIMAL_getnumqconknz itself, not from a second
 * listing: two enumerations of the same table are two policies. A row
 * whose entries cancel (putqconk accumulates) measures zero and writes zero:
 * empty verdict, not refusal.
 * Declared deviation: the form the reference uses for these two reads
 * was NOT read (document fetch still blocked on this pass); the
 * form is our own counter's, because count and reader
 * answer for the same table and cannot diverge. */
PRIMALrescodee PRIMAL_getqobj(PRIMALtask_t t, int *qi, int *qj, PRIMALrealt *qval,
                        int maxnum, int *numret);
/* Reads the upper triangle of constraint k's Q; refusal if maxnum is short. */
PRIMALrescodee PRIMAL_getqconk(PRIMALtask_t t, int k, int *qi, int *qj, PRIMALrealt *qval,
                         int maxnum, int *numret);
/* 64-bit variant of getqobj (64-bit capacity and count). */
PRIMALrescodee PRIMAL_getqobj64(PRIMALtask_t t, int *qi, int *qj, PRIMALrealt *qval,
                        PRIMALint64t maxnum, PRIMALint64t *numret);
/* 64-bit variant of getqconk (64-bit capacity and count). */
PRIMALrescodee PRIMAL_getqconk64(PRIMALtask_t t, int k, int *qi, int *qj, PRIMALrealt *qval,
                         PRIMALint64t maxnum, PRIMALint64t *numret);

/* ---- variable and constraint names ----
 * Two independent tables, as in the reference: in an MPS a constraint and a
 * variable may be called by the same name, so the same string may
 * name one entry of one table and one of the other. Each entry owns its
 * own string; an unnamed entity reads its name as the EMPTY string (""),
 * and "" cannot be any entity's name (setting a name to "" removes
 * it).
 * Names are unique inside their own table: a name already in use is
 * REFUSED (PRIMAL_RES_ERR_ARG) without touching the previous name, because a
 * table with two indices for the same name would make getvarname a question
 * with two answers. Renaming an index with the name it already has is not a
 * conflict (idempotent).
 * getvarnameidx returns a BORROWED pointer that lives as long as the
 * task: appendvars/appendcons move the pointer table, not the
 * strings, so a `const char *` obtained before an append stays valid.
 * Missed getters answer ERR_ARG and do NOT touch the returned index.
 * Deviation: here the "not found" status is the same ERR_ARG with which
 * every getter of this API says "no" (see the Farkas rays, T85). The
 * reference would have a dedicated code for the missing name: NOT read
 * on this pass (documentation unreachable), so it is memory, not
 * source, and the choice rests on our measured convention, not on that name. */
PRIMALrescodee PRIMAL_putvarname(PRIMALtask_t t, int j, const char *name);
/* Sets the name of constraint i (independence and refusal as for variables). */
PRIMALrescodee PRIMAL_putconname(PRIMALtask_t t, int i, const char *name);
/* Borrowed name pointer of variable j; ERR_ARG if unnamed. */
PRIMALrescodee PRIMAL_getvarnameidx(PRIMALtask_t t, int j, const char **name);
/* Borrowed name pointer of constraint i; ERR_ARG if unnamed. */
PRIMALrescodee PRIMAL_getconnameidx(PRIMALtask_t t, int i, const char **name);
/* name -> index (the two inverse queries of one table) */
PRIMALrescodee PRIMAL_getvarname(PRIMALtask_t t, const char *name, int *j);
/* Resolves a constraint name to its index; ERR_ARG if absent. */
PRIMALrescodee PRIMAL_getconname(PRIMALtask_t t, const char *name, int *i);
/* Reference form for the by-name lookup: same service, other name. */
PRIMALrescodee PRIMAL_getidxvar(PRIMALtask_t t, const char *vname, int *var);
/* Reference form of the constraint by-name lookup. */
PRIMALrescodee PRIMAL_getidxcon(PRIMALtask_t t, const char *cname, int *con);
/* The whole table in one call. The user buffer has numvar (numcon)
 * entries -- read them from PRIMAL_getnumvar / PRIMAL_getnumcon, the same
 * convention as getxx -- and the write is exactly that size, because
 * there is no count to negotiate. An unnamed entity comes out as the
 * EMPTY string, identically to getvarnameidx: two reads of the same table
 * that disagree are not two formats, they are two rules.
 * Deviation: the reference delivers char** that IT copies, into fixed-length
 * caller-owned buffers; here the task's BORROWED string
 * pointers come out, which live as long as the task and must not
 * be freed. */
PRIMALrescodee PRIMAL_getallvarname(PRIMALtask_t t, const char **names);
/* Reads the whole constraint name table (numcon borrowed pointers). */
PRIMALrescodee PRIMAL_getallconname(PRIMALtask_t t, const char **names);

/* ---- bar variable names: the third table ----
 * The contract is the two scalar tables', entry by entry:
 * uniqueness INSIDE the table, "" removing the name ("" read), refusal
 * not touching the previous name, BORROWED pointer living as long as the
 * task, and getallbarname writing exactly PRIMAL_getnumbarvar entries.
 * The news is not a rule, it is the index space: a bar
 * variable is not one of the numvar scalar variables (it has its own size and
 * its own cone block), so its name lives in an INDEPENDENT namespace
 * -- the same string may name a bar and a scalar, as it already may
 * name a constraint and a variable, and a duplicate is refused only
 * inside the speaking table.
 * Declared deviation: the rule the reference uses for bar names
 * was NOT read on this pass (document fetch blocked); this is
 * the consistent extension of our rule measured in T102, not a copy. */
PRIMALrescodee PRIMAL_putbarname(PRIMALtask_t t, int j, const char *name);
/* Borrowed name pointer of bar variable j; ERR_ARG if unnamed. */
PRIMALrescodee PRIMAL_getbarnameidx(PRIMALtask_t t, int j, const char **name);
/* name -> bar index */
PRIMALrescodee PRIMAL_getbarname(PRIMALtask_t t, const char *name, int *j);
/* Reference form for the by-name lookup (same service, other name). */
PRIMALrescodee PRIMAL_getidxbarvar(PRIMALtask_t t, const char *bname, int *bar);
/* Reads the whole bar name table (numbarvar borrowed pointers). */
PRIMALrescodee PRIMAL_getallbarname(PRIMALtask_t t, const char **names);

/* ---- cone block names: the fourth table ----
 * Contract identical to the three tables measured in T102/T103/T105: uniqueness
 * INSIDE the table, "" removing the name ("" read), refusal leaving
 * the previous name alive, BORROWED pointer for the task lifetime, and
 * getallconename writing exactly PRIMAL_getnumcone entries. The
 * table capacity is the same cone_cap as the four cone arrays,
 * because two capacities to keep in step are two policies.
 * INDEPENDENT namespace, for the same reason as the bar: a cone is not a
 * scalar variable, not a constraint, not a bar. Mind the
 * confusable pair: getconname is the CONSTRAINT, getconename is the CONE, one
 * letter apart — the confusion is not closed by a comment but by
 * T106, which names constraint and cone with the SAME string in one task and
 * asserts that the two lookups answer their own indices.
 * Declared deviation: the rule the reference uses to name a cone
 * was NOT read on this pass (fetch blocked) and was not invented.
 * In particular there is no "function name" surface here with a
 * (type, index): the MSKfunctiontypee numbering is not known, and
 * giving it our numbers would be the defect T93/T94 corrected. */
PRIMALrescodee PRIMAL_putconename(PRIMALtask_t t, int k, const char *name);
/* Borrowed name pointer of cone block k; ERR_ARG if unnamed. */
PRIMALrescodee PRIMAL_getconenameidx(PRIMALtask_t t, int k, const char **name);
/* name -> cone-block index */
PRIMALrescodee PRIMAL_getconename(PRIMALtask_t t, const char *name, int *k);
/* Reference form for the by-name lookup (same service, other name). */
PRIMALrescodee PRIMAL_getidxcone(PRIMALtask_t t, const char *cname, int *cone);
/* Reads the whole cone-block name table (numcone borrowed pointers). */
PRIMALrescodee PRIMAL_getallconename(PRIMALtask_t t, const char **names);
/* reference by-name lookups (delegates with fixed assignment = 0) and
 * cone-block name length. */
PRIMALrescodee PRIMAL_getvarnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index);
/* reference constraint name->index delegate (assignment fixed to 0). */
PRIMALrescodee PRIMAL_getconnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index);
/* reference cone-block name->index delegate (assignment fixed to 0). */
PRIMALrescodee PRIMAL_getconenameindex(PRIMALtask_t t, const char *somename,
                                       int *asgn, int *index);
/* Length of the name of cone block i (0 if unnamed). */
PRIMALrescodee PRIMAL_getconenamelen(PRIMALtask_t t, int i, int *len);
/* stored Q nonzeros (objective + constraints) and c[subj[k]]; capacity
 * hints (no-op). */
PRIMALrescodee PRIMAL_getmaxnumqnz(PRIMALtask_t t, int *maxnumqnz);
/* 64-bit variant of getmaxnumqnz. */
PRIMALrescodee PRIMAL_getmaxnumqnz64(PRIMALtask_t t, PRIMALint64t *maxnumqnz);
/* Reads c[subj[k]] for the listed indices. */
PRIMALrescodee PRIMAL_getclist(PRIMALtask_t t, int num, const int *subj, PRIMALrealt *c);
/* Capacity hint for the number of variables (no-op: arrays grow in place). */
PRIMALrescodee PRIMAL_putmaxnumvar(PRIMALtask_t t, int maxnumvar);
/* Capacity hint for the number of constraints (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumcon(PRIMALtask_t t, int maxnumcon);
/* Capacity hint for the number of cones (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumcone(PRIMALtask_t t, int maxnumcone);
/* Capacity hint for the number of A nonzeros (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumanz(PRIMALtask_t t, PRIMALint64t maxnumanz);
/* Capacity hint for the number of Q nonzeros (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumqnz(PRIMALtask_t t, PRIMALint64t maxnumqnz);

/* ---- the objective name: a single slot, not a table ----
 * Same ownership rule (borrowed pointer, "" removes, NULL refused),
 * but with n = 1 the "duplicate" question does not arise: a second name REPLACES
 * the first, which is the only measurable difference from the tables.
 * Deviation: the reference copies into a caller-owned fixed-length buffer;
 * here the BORROWED pointer comes out, as for the four tables. */
PRIMALrescodee PRIMAL_putobjname(PRIMALtask_t t, const char *name);
/* Borrowed objective name pointer; ERR_ARG if unnamed. */
PRIMALrescodee PRIMAL_getobjname(PRIMALtask_t t, const char **name);
/* Task name and name lengths (reference puttaskname/gettaskname/
 * gettasknamelen, getvarnamelen/getconnamelen/getobjnamelen, getmaxnamelen).
 * Length does NOT count the terminator; an unnamed object has length 0.
 * gettaskname copies into the caller buffer, which must also hold the zero. */
PRIMALrescodee PRIMAL_puttaskname(PRIMALtask_t t, const char *name);
/* Copies the task name into the caller buffer (which holds the zero too). */
PRIMALrescodee PRIMAL_gettaskname(PRIMALtask_t t, int sizetaskname, char *taskname);
/* Length of the task name (0 if unnamed). */
PRIMALrescodee PRIMAL_gettasknamelen(PRIMALtask_t t, int *len);
/* Length of the name of variable j (0 if unnamed). */
PRIMALrescodee PRIMAL_getvarnamelen(PRIMALtask_t t, int j, int *len);
/* Length of the name of constraint i (0 if unnamed). */
PRIMALrescodee PRIMAL_getconnamelen(PRIMALtask_t t, int i, int *len);
/* Length of the objective name (0 if unnamed). */
PRIMALrescodee PRIMAL_getobjnamelen(PRIMALtask_t t, int *len);
/* Longest name length across the task's tables. */
PRIMALrescodee PRIMAL_getmaxnamelen(PRIMALtask_t t, int *maxlen);

/* conic optimization (SOCP); members are variable indices */
/* Affine expressions (AFE): f_i = sum_j F_ij x_j + g_i. The storage on
 * which the affine conic (ACC) and disjunctive (DJC) constraints build. */
PRIMALrescodee PRIMAL_appendafes(PRIMALtask_t t, PRIMALint64t num);
/* Returns the number of affine expressions in the task. */
PRIMALrescodee PRIMAL_getnumafe(PRIMALtask_t t, PRIMALint64t *numafe);
/* Sets the coefficient F_ij of affine expression i. */
PRIMALrescodee PRIMAL_putafefentry(PRIMALtask_t t, PRIMALint64t i, int j, PRIMALrealt v);
/* Replaces the whole row i of F from a sparse index/value list. */
PRIMALrescodee PRIMAL_putafefrow(PRIMALtask_t t, PRIMALint64t i, int numnz,
                                 const int *varidx, const PRIMALrealt *val);
/* Sets the constant g_i of affine expression i. */
PRIMALrescodee PRIMAL_putafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt g);
/* Reads the constant g_i of affine expression i. */
PRIMALrescodee PRIMAL_getafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt *g);
/* Number of stored nonzero entries in row i of F. */
PRIMALrescodee PRIMAL_getafefrownumnz(PRIMALtask_t t, PRIMALint64t i, int *numnz);
/* Reads row i of F as sparse indices and values. */
PRIMALrescodee PRIMAL_getafefrow(PRIMALtask_t t, PRIMALint64t i, int *numnz,
                                 int *varidx, PRIMALrealt *val);
/* block AFE surface (reference emptyafefrow/emptyafefcol,
 * putafeglist/putafegslice/getafegslice, putafefentrylist, getafeftrip,
 * getafefnumnz). A NULL buffer is not written; `getafeftrip` enumerates F as
 * (afeidx,varidx,val) in stored order and the length is the sum of
 * getafefnumnz. */
PRIMALrescodee PRIMAL_emptyafefrow(PRIMALtask_t t, PRIMALint64t afeidx);
/* Zeroes the column `varidx` of F. */
PRIMALrescodee PRIMAL_emptyafefcol(PRIMALtask_t t, int varidx);
/* Sets several g entries from an afeidx/g list. */
PRIMALrescodee PRIMAL_putafeglist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                  const PRIMALint64t *afeidx, const PRIMALrealt *g);
/* Sets g[first..last) from the given slice. */
PRIMALrescodee PRIMAL_putafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, const PRIMALrealt *slice);
/* Reads g[first..last) into the given slice. */
PRIMALrescodee PRIMAL_getafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, PRIMALrealt *g);
/* Replaces the given F entries (does not clear the rest of the rows). */
PRIMALrescodee PRIMAL_putafefentrylist(PRIMALtask_t t, PRIMALint64t numentr,
                                       const PRIMALint64t *afeidx,
                                       const int *varidx, const PRIMALrealt *val);
/* Enumerates F as (afeidx,varidx,val) triplets in stored order. */
PRIMALrescodee PRIMAL_getafeftrip(PRIMALtask_t t, PRIMALint64t *afeidx,
                                  int *varidx, PRIMALrealt *val);
/* Number of stored nonzeros in row `afeidx` of F (delegate). */
PRIMALrescodee PRIMAL_getafefnumnz(PRIMALtask_t t, PRIMALint64t afeidx, int *numnz);
/* Zeroes several rows of F. */
PRIMALrescodee PRIMAL_emptyafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                       const PRIMALint64t *afeidx);
/* Zeroes several columns of F. */
PRIMALrescodee PRIMAL_emptyafefcollist(PRIMALtask_t t, PRIMALint64t numvaridx,
                                       const int *varidx);
/* Writes the column `varidx` of F from an afeidx/value list. */
PRIMALrescodee PRIMAL_putafefcol(PRIMALtask_t t, int varidx, PRIMALint64t numnz,
                                 const PRIMALint64t *afeidx, const PRIMALrealt *val);
/* bar terms of an AFE (reference putafebarfentry and family): Fbar[i][j] is
 * a weighted combination of symmetric matrices, and <Fbar_ij, X_j> enters the
 * i-th expression. Writing (i,j) replaces the terms with the same
 * barvaridx. */
PRIMALrescodee PRIMAL_putafebarfentry(PRIMALtask_t t, PRIMALint64t afeidx, int barvaridx,
        PRIMALint64t numterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight);
/* Zeroes the bar terms of affine expression afeidx. */
PRIMALrescodee PRIMAL_emptyafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx);
/* Zeroes the bar terms of several affine expressions. */
PRIMALrescodee PRIMAL_emptyafebarfrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                          const PRIMALint64t *afeidxlist);
/* Number of bar-row entries of affine expression afeidx. */
PRIMALrescodee PRIMAL_getafebarfnumrowentries(PRIMALtask_t t, PRIMALint64t afeidx, int *numentr);
/* Entry and term counts of the bar row of affine expression afeidx. */
PRIMALrescodee PRIMAL_getafebarfrowinfo(PRIMALtask_t t, PRIMALint64t afeidx,
                                        int *numentr, PRIMALint64t *numterm);
/* Reads the bar row of affine expression afeidx (barvaridx, term indices, weights). */
PRIMALrescodee PRIMAL_getafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int *barvaridx,
        PRIMALint64t *ptrterm, PRIMALint64t *numterm, PRIMALint64t *termidx,
        PRIMALrealt *termweight);
/* Number of bar block triplets across the task's AFE bar rows. */
PRIMALrescodee PRIMAL_getafebarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip);
/* Reads the AFE bar terms as block triplets; refusal if maxnumtrip is short. */
PRIMALrescodee PRIMAL_getafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *afeidx, int *barvaridx, int *subk,
        int *subl, PRIMALrealt *valkl);
/* Sets several bar rows of F at once. */
PRIMALrescodee PRIMAL_putafebarfentrylist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *barvaridx, const PRIMALint64t *numterm,
        const PRIMALint64t *ptrterm, PRIMALint64t lenterm, const PRIMALint64t *termidx,
        const PRIMALrealt *termweight);
/* Replaces the bar row of affine expression afeidx. */
PRIMALrescodee PRIMAL_putafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int numentr,
        const int *barvaridx, const PRIMALint64t *numterm, const PRIMALint64t *ptrterm,
        PRIMALint64t lenterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight);
/* Sets Fbar from block triplets (afeidx, barvaridx, k, l, val). */
PRIMALrescodee PRIMAL_putafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t numtrip,
        const PRIMALint64t *afeidx, const int *barvaridx, const int *subk,
        const int *subl, const PRIMALrealt *valkl);
/* Number of block triplets of the Fbar implied by the ACCs. */
PRIMALrescodee PRIMAL_getaccbarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip);
/* Reads the ACC-implied Fbar as block triplets; refusal if maxnumtrip is short. */
PRIMALrescodee PRIMAL_getaccbarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *acc_afe, int *bar_var, int *blk_row,
        int *blk_col, PRIMALrealt *blk_val);
/* Conic domains (for affine conic constraints, ACC); values are
 * MSKdomaintypee's. */
typedef enum {
    PRIMAL_DOMAIN_R = 0,
    PRIMAL_DOMAIN_RZERO = 1,
    PRIMAL_DOMAIN_RPLUS = 2,
    PRIMAL_DOMAIN_RMINUS = 3,
    PRIMAL_DOMAIN_QUADRATIC_CONE = 4,
    PRIMAL_DOMAIN_RQUADRATIC_CONE = 5,
    PRIMAL_DOMAIN_PRIMAL_EXP_CONE = 6,
    PRIMAL_DOMAIN_DUAL_EXP_CONE = 7,
    PRIMAL_DOMAIN_PRIMAL_POWER_CONE = 8,
    PRIMAL_DOMAIN_DUAL_POWER_CONE = 9,
    PRIMAL_DOMAIN_PRIMAL_GEO_MEAN_CONE = 10,
    PRIMAL_DOMAIN_DUAL_GEO_MEAN_CONE = 11,
    PRIMAL_DOMAIN_SVEC_PSD_CONE = 12
} PRIMALdomaintypee;

/* Appends the linear domain R^n and returns its index. */
PRIMALrescodee PRIMAL_appendrdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends the linear domain {0}^n and returns its index. */
PRIMALrescodee PRIMAL_appendrzerodomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends the nonnegative domain R_+^n and returns its index. */
PRIMALrescodee PRIMAL_appendrplusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends the nonpositive domain R_-^n and returns its index. */
PRIMALrescodee PRIMAL_appendrminusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends a quadratic cone domain and returns its index. */
PRIMALrescodee PRIMAL_appendquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends a rotated quadratic cone domain and returns its index. */
PRIMALrescodee PRIMAL_appendrquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends a primal exponential cone domain and returns its index. */
PRIMALrescodee PRIMAL_appendprimalexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx);
/* Appends a dual exponential cone domain and returns its index. */
PRIMALrescodee PRIMAL_appenddualexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx);
/* Appends a primal power cone domain (exponent alpha) and returns its index. */
PRIMALrescodee PRIMAL_appendprimalpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx);
/* Appends a dual power cone domain (exponent alpha) and returns its index. */
PRIMALrescodee PRIMAL_appenddualpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx);
/* Appends a symmetric-vector PSD cone domain of dimension dim. */
PRIMALrescodee PRIMAL_appendsvecpsdconedomain(PRIMALtask_t t, int dim, PRIMALint64t *domidx);
/* Returns the number of conic domains. */
PRIMALrescodee PRIMAL_getnumdomain(PRIMALtask_t t, PRIMALint64t *numdomain);
/* Reads the type of the given conic domain. */
PRIMALrescodee PRIMAL_getdomaintype(PRIMALtask_t t, PRIMALint64t domidx, PRIMALdomaintypee *domtype);
/* Reads the dimension of the given conic domain. */
PRIMALrescodee PRIMAL_getdomainn(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t *n);
/* geometric-mean cones, domain names (seventh table), power-cone
 * info and the putmaxnumdomain capacity hint. */
PRIMALrescodee PRIMAL_appendprimalgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Appends a dual geometric-mean cone domain and returns its index. */
PRIMALrescodee PRIMAL_appenddualgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
/* Sets the name of the given conic domain. */
PRIMALrescodee PRIMAL_putdomainname(PRIMALtask_t t, PRIMALint64t domidx, const char *name);
/* Length of the name of the given conic domain (0 if unnamed). */
PRIMALrescodee PRIMAL_getdomainnamelen(PRIMALtask_t t, PRIMALint64t domidx, int *len);
/* Copies the name of the given conic domain into the caller buffer. */
PRIMALrescodee PRIMAL_getdomainname(PRIMALtask_t t, PRIMALint64t domidx, int sizename, char *name);
/* Reads the exponent alpha of a power-cone domain. */
PRIMALrescodee PRIMAL_getpowerdomainalpha(PRIMALtask_t t, PRIMALint64t domidx, PRIMALrealt *alpha);
/* Reads the dimension and left part of a power-cone domain. */
PRIMALrescodee PRIMAL_getpowerdomaininfo(PRIMALtask_t t, PRIMALint64t domidx,
                                         PRIMALint64t *n, PRIMALint64t *nleft);
/* Capacity hint for the number of domains (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumdomain(PRIMALtask_t t, PRIMALint64t maxnumdomain);
/* Appends a cone of type ct with the given members and parameter. */
PRIMALrescodee PRIMAL_appendcone(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt coneparam, int nummem, const int *submem);
/* cone(s) whose members are CONTIGUOUS variables j..j+nummem-1 (reference
 * appendconeseq/appendconesseq). */
PRIMALrescodee PRIMAL_appendconeseq(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt conepar,
                                    int nummem, int j);
/* Appends several contiguous-member cones in one call. */
PRIMALrescodee PRIMAL_appendconesseq(PRIMALtask_t t, int num, const PRIMALconetypee *ct,
    const PRIMALrealt *conepar, const int *nummem, const int *j);
/* removes the cones at the given indices (reference removecones). */
PRIMALrescodee PRIMAL_removecones(PRIMALtask_t t, int num, const int *subset);
/* removes the constraints at the given indices, compacting A/qcon/barA
 * (reference removecons). */
PRIMALrescodee PRIMAL_removecons(PRIMALtask_t t, int num, const int *subset);
/* removes the variables at the given indices, reshaping qcon and remapping
 * qobj and the cone members (reference removevars). */
PRIMALrescodee PRIMAL_removevars(PRIMALtask_t t, int num, const int *subset);
/* Returns the number of conic constraints (cones). */
PRIMALrescodee PRIMAL_getnumcone(PRIMALtask_t t, int *numcone);
/* Reads cone k: its type, member count and members. */
PRIMALrescodee PRIMAL_getcone(PRIMALtask_t t, int k, PRIMALconetypee *ct, int *nummem, int *submem);
/* Reads the parameter (alpha) of cone k. */
PRIMALrescodee PRIMAL_getconeparam(PRIMALtask_t t, int k, PRIMALrealt *param);
/* Returns the number of members of cone k. */
PRIMALrescodee PRIMAL_getnumconemem(PRIMALtask_t t, int k, int *nummem);
/* ---- affine conic constraints (ACC) ----
 * appendacc(domidx, numafeidx, afeidxlist, b): the vector of numafeidx
 * affine expressions afeidxlist[e] (F_e x + g_e) plus the constants b[e]
 * belongs to domain domidx (size numafeidx == getdomainn(domidx)). */
PRIMALrescodee PRIMAL_appendacc(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t numafeidx,
                                const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
/* Returns the number of affine conic constraints. */
PRIMALrescodee PRIMAL_getnumacc(PRIMALtask_t t, PRIMALint64t *numacc);
/* Reads the dimension (number of AFEs) of ACC accidx. */
PRIMALrescodee PRIMAL_getaccn(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *n);
/* Reads the domain index of ACC accidx. */
PRIMALrescodee PRIMAL_getaccdomain(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *domidx);
/* Reads the AFE index list of ACC accidx. */
PRIMALrescodee PRIMAL_getaccafeidxlist(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *afeidxlist);
/* Reads the constant vector b of ACC accidx. */
PRIMALrescodee PRIMAL_getaccb(PRIMALtask_t t, PRIMALint64t accidx, PRIMALrealt *b);
/* Block ACC surface and names (reference appendaccs/getaccs/getaccntot/
 * putaccb/putaccname/getaccname/getaccnamelen). `appendaccs` appends numaccs
 * ACCs, each of size dom_n[domidxs[i]], consuming afeidxlist/b in
 * sequence; `getaccs` is the concatenation of the per-ACC lists. */
PRIMALrescodee PRIMAL_appendaccs(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
/* Total number of AFE entries across all ACCs. */
PRIMALrescodee PRIMAL_getaccntot(PRIMALtask_t t, PRIMALint64t *n);
/* Reads all ACCs as concatenated domain/AFE lists and b. */
PRIMALrescodee PRIMAL_getaccs(PRIMALtask_t t, PRIMALint64t *domidxlist,
                              PRIMALint64t *afeidxlist, PRIMALrealt *b);
/* Rewrites the b vector of an existing ACC. */
PRIMALrescodee PRIMAL_putaccb(PRIMALtask_t t, PRIMALint64t accidx,
                              PRIMALint64t lengthb, const PRIMALrealt *b);
/* Sets the name of ACC accidx. */
PRIMALrescodee PRIMAL_putaccname(PRIMALtask_t t, PRIMALint64t accidx, const char *name);
/* Length of the name of ACC accidx (0 if unnamed). */
PRIMALrescodee PRIMAL_getaccnamelen(PRIMALtask_t t, PRIMALint64t accidx, int *len);
/* Copies the name of ACC accidx into the caller buffer. */
PRIMALrescodee PRIMAL_getaccname(PRIMALtask_t t, PRIMALint64t accidx,
                                 int sizename, char *name);
/* ACCs with contiguous AFEs (appendaccseq/appendaccsseq), activity at the
 * point (evaluateacc/evaluateaccs) and capacity hints. */
PRIMALrescodee PRIMAL_appendaccseq(PRIMALtask_t t, PRIMALint64t domidx,
                                   PRIMALint64t numafeidx, PRIMALint64t afeidxfirst,
                                   const PRIMALrealt *b);
/* Appends several ACCs over consecutive AFE ranges in one call. */
PRIMALrescodee PRIMAL_appendaccsseq(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        PRIMALint64t afeidxfirst, const PRIMALrealt *b);
/* Evaluates the activity of ACC accidx at the given solution. */
PRIMALrescodee PRIMAL_evaluateacc(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                  PRIMALrealt *activity);
/* Evaluates the activity of all ACCs at the given solution. */
PRIMALrescodee PRIMAL_evaluateaccs(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *activity);
/* Capacity hint for the number of ACCs (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumacc(PRIMALtask_t t, PRIMALint64t maxnumacc);
/* Capacity hint for the number of AFEs (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumafe(PRIMALtask_t t, PRIMALint64t maxnumafe);
/* Capacity hint for the number of DJCs (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumdjc(PRIMALtask_t t, PRIMALint64t maxnumdjc);
/* one component of an ACC's b; primal violation of an ACC set;
 * power-domain sequences. */
PRIMALrescodee PRIMAL_putaccbj(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t j, PRIMALrealt bj);
/* ACC duals (reference getaccdoty/getaccdotys/putaccdoty): the
 * multipliers of the rows the ACC produced, in our `y`
 * convention (the reference `doty` convention was not read). */
PRIMALrescodee PRIMAL_getaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 PRIMALrealt *doty);
/* Reads the doty vectors of all ACCs at once. */
PRIMALrescodee PRIMAL_getaccdotys(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *doty);
/* Writes the doty vector of ACC accidx. */
PRIMALrescodee PRIMAL_putaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 const PRIMALrealt *doty);
/* Primal violation of the listed ACCs at the given solution. */
PRIMALrescodee PRIMAL_getpviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol);
/* Dual violation of the listed ACCs at the given solution. */
PRIMALrescodee PRIMAL_getdviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol);
/* Appends several primal power-cone domains in one call. */
PRIMALrescodee PRIMAL_appendprimalpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist);
/* Appends several dual power-cone domains in one call. */
PRIMALrescodee PRIMAL_appenddualpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist);
/* the F and g implied by the AFE order inside the ACCs. */
PRIMALrescodee PRIMAL_getaccfnumnz(PRIMALtask_t t, PRIMALint64t *accfnnz);
/* Reads the g vector implied by the ACCs. */
PRIMALrescodee PRIMAL_getaccgvector(PRIMALtask_t t, PRIMALrealt *g);
/* Reads the F implied by the ACCs as (row,col,val) triplets. */
PRIMALrescodee PRIMAL_getaccftrip(PRIMALtask_t t, PRIMALint64t *frow,
                                  int *fcol, PRIMALrealt *fval);
/* ---- disjunctive constraints (DJC, reference style) ----
 * A DJC is the OR of numterm clauses; clause i is the conjunction of
 * termsizelist[i] domains applied to affine expressions. domidxlist concatenates
 * the domains of all clauses (length sum termsizelist); afeidxlist concatenates
 * the expressions, one per domain component (length = sum of the
 * domain sizes). b, optional (NULL = all zero), is the constant
 * SUBTRACTED from each expression: expression k is F_k x + g_k - b_k.
 * appenddjcs pre-allocates num empty slots; putdjc fills slot djcidx.
 * The model extends at once (one selection binary per clause and big-M
 * rows, like every MIP of this solver), so an already-written djcidx is not
 * rewritable (ERR_ARG). Declared deviation: only LINEAR domains
 * (R/RZERO/RPLUS/RMINUS); a conic domain in a DJC is ERR_ARG. */
PRIMALrescodee PRIMAL_appenddjcs(PRIMALtask_t t, PRIMALint64t num);
/* Fills the description of DJC djcidx (OR of clauses over affine expressions). */
PRIMALrescodee PRIMAL_putdjc(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t numdomidx, const PRIMALint64t *domidxlist,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist,
        const PRIMALrealt *b, PRIMALint64t numterms,
        const PRIMALint64t *termsizelist);
/* idxlast-idxfirst consecutive DJCs; termsindjc[i] = number of terms of DJC
 * idxfirst+i; the rest is the concatenation of the putdjc descriptions. */
PRIMALrescodee PRIMAL_putdjcslice(PRIMALtask_t t, PRIMALint64t idxfirst,
        PRIMALint64t idxlast, PRIMALint64t numdomidx,
        const PRIMALint64t *domidxlist, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
        PRIMALint64t numterms, const PRIMALint64t *termsizelist,
        const PRIMALint64t *termsindjc);
/* Returns the number of disjunctive constraints. */
PRIMALrescodee PRIMAL_getnumdjc(PRIMALtask_t t, PRIMALint64t *num);
/* Number of domain entries in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumdomain(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
/* Number of AFE entries in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumafe(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
/* Number of clauses (terms) in DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcnumterm(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
/* Reads the domain index list of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcdomainidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *domidxlist);
/* Reads the AFE index list of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcafeidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *afeidxlist);
/* Reads the b vector of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjcb(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALrealt *b);
/* Reads the clause size list of DJC djcidx. */
PRIMALrescodee PRIMAL_getdjctermsizelist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *termsizelist);
/* Total number of domain entries across all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumdomaintot(PRIMALtask_t t, PRIMALint64t *n);
/* Total number of AFE entries across all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumafetot(PRIMALtask_t t, PRIMALint64t *n);
/* Total number of clauses across all DJCs. */
PRIMALrescodee PRIMAL_getdjcnumtermtot(PRIMALtask_t t, PRIMALint64t *n);
/* bulk read of ALL the DJCs (reference getdjcs): the lists are the
 * concatenation of the per-DJC ones, `numterms` has one entry per DJC. Each
 * buffer has the length given by the getdjcnum*tot; a NULL buffer is skipped. */
PRIMALrescodee PRIMAL_getdjcs(PRIMALtask_t t, PRIMALint64t *domidxlist,
        PRIMALint64t *afeidxlist, PRIMALrealt *b, PRIMALint64t *termsizelist,
        PRIMALint64t *numterms);
/* primal violation of a set of DJCs (reference getpvioldjc): for each
 * djcidxlist[k] writes viol[k] = min_i(max_j viol(T_ij)) read on the published
 * point and on the current model. */
PRIMALrescodee PRIMAL_getpvioldjc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numdjcidx, const PRIMALint64t *djcidxlist, PRIMALrealt *viol);
/* DJC names (reference putdjcname/getdjcname/getdjcnamelen): buffer of
 * sizename bytes, which must also hold the terminator; a refusal does not write. */
PRIMALrescodee PRIMAL_putdjcname(PRIMALtask_t t, PRIMALint64t djcidx, const char *name);
/* Length of the name of DJC djcidx (0 if unnamed). */
PRIMALrescodee PRIMAL_getdjcnamelen(PRIMALtask_t t, PRIMALint64t djcidx, int *len);
/* Copies the name of DJC djcidx into the caller buffer. */
PRIMALrescodee PRIMAL_getdjcname(PRIMALtask_t t, PRIMALint64t djcidx, int sizename, char *name);

/* SDP (semi-definite): bar variables X_j >= 0 (dim x dim symmetric matrices)
 * and linear terms as inner products <A^k, X_j> with sparse
 * symmetric matrices from the "matrix store" (appendsparsesymmat). */
PRIMALrescodee PRIMAL_appendsparsesymmat(PRIMALtask_t t, int dim, int nnz,
                                   const int *subi, const int *subj,
                                   const PRIMALrealt *val, int *idx);
/* several matrices in one call (reference appendsparsesymmatlist): dims[k] and
 * nz[k] give the shape, subi/subj/valij are concatenated, idx[k] returns the
 * ids. The whole list is validated before appending. */
PRIMALrescodee PRIMAL_appendsparsesymmatlist(PRIMALtask_t t, int num, const int *dims,
    const PRIMALint64t *nz, const int *subi, const int *subj, const PRIMALrealt *valij,
    PRIMALint64t *idx);
/* Reads the sparse symmetric matrix stored at idx. */
PRIMALrescodee PRIMAL_getsparsesymmat(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxlen,
    int *subi, int *subj, PRIMALrealt *valij);
/* Appends `num` bar variables with the given dimensions. */
PRIMALrescodee PRIMAL_appendbarvars(PRIMALtask_t t, int num, const int *dim);
/* removes the bar variables at the given indices, remapping the terms (reference
 * removebarvars). */
PRIMALrescodee PRIMAL_removebarvars(PRIMALtask_t t, int num, const int *subset);
/* constraint i: adds to (i) the scalar terms  sum_k val_k <A^{sub_k}, X_j> */
PRIMALrescodee PRIMAL_putbaraij(PRIMALtask_t t, int i, int j, int num,
                           const int *sub, const PRIMALrealt *val);
/* block write of bar A (more efficient API for large SDPs):
 * constraint i sees sum_k val_k <A^{blk_sub_k}, X_{blk_j_k}> for k=0..num-1 */
PRIMALrescodee PRIMAL_putbarablockij(PRIMALtask_t t, int i, int j, int num,
                               const int *blk_sub, const PRIMALrealt *blk_val);
/* read of the bar A terms of the pair (i,j): list (symidx, coef).
 * Contract of the accessors that fill user buffers (`T102`/`T104`):
 * maxnum is the capacity of symidx/val and *num the number of TERMS of that
 * pair. If the space is short the call is REFUSED (ERR_ARG) without having
 * touched any buffer and without having written *num: a truncated list that
 * answers OK is indistinguishable from a complete list, and an <A,X> built
 * on the prefix is a different model. symidx and val both NULL is the door
 * of the count only (this pair has no getnum... for (i,j)) and is never
 * refused, because it writes nothing; an (i,j) with no terms answers 0
 * with OK -- empty verdict, not refusal. */
PRIMALrescodee PRIMAL_getbaraidxij(PRIMALtask_t t, int i, int j, int maxnum,
                             int *num, int *symidx, PRIMALrealt *val);
/* read of the bar C terms of variable j: list (symidx, coef), with the
 * SAME capacity and impossible-silent-refusal contract as above. */
PRIMALrescodee PRIMAL_getbarcidxj(PRIMALtask_t t, int j, int maxnum,
                             int *num, int *symidx, PRIMALrealt *val);
/* Sparsity and per-block info of A-bar/C-bar (reference getbarasparsity/
 * getbaraidxinfo/getbaraidx and the C variants). `idx` names a block: here
 * `idx = i*numbarvar + j` for A-bar and `idx = j` for C-bar (this solver's
 * convention: the reference's vectorization was not read). */
PRIMALrescodee PRIMAL_getbarasparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxij);
/* Number of stored entries in A-bar block idx. */
PRIMALrescodee PRIMAL_getbaraidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num);
/* Reads A-bar block idx: its (i,j) and the term list. */
PRIMALrescodee PRIMAL_getbaraidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *i, int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights);
/* Sparsity pattern of C-bar over bar variables. */
PRIMALrescodee PRIMAL_getbarcsparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxj);
/* Number of stored entries in C-bar block idx. */
PRIMALrescodee PRIMAL_getbarcidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num);
/* Reads C-bar block idx: its variable j and the term list. */
PRIMALrescodee PRIMAL_getbarcidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights);
/* Block-triplet form of A-bar and C-bar (reference
 * getbarablocktriplet/getbarcblocktriplet): one row for every stored entry
 * of the lower triangle of every block. A: (i, j, k, l, val), C: (j, k, l, val);
 * the counters give the exact number of such entries. Insufficient capacity with
 * buffers provided = refusal without writing. */
PRIMALrescodee PRIMAL_getnumbarablocktriplets(PRIMALtask_t t, PRIMALint64t *num);
/* Number of block triplets of C-bar. */
PRIMALrescodee PRIMAL_getnumbarcblocktriplets(PRIMALtask_t t, PRIMALint64t *num);
/* Reads A-bar as (i,j,k,l,val) block triplets; refusal if maxnum is short. */
PRIMALrescodee PRIMAL_getbarablocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subi, int *subj, int *subk, int *subl, PRIMALrealt *valijkl);
/* Reads C-bar as (j,k,l,val) block triplets; refusal if maxnum is short. */
PRIMALrescodee PRIMAL_getbarcblocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subj, int *subk, int *subl, PRIMALrealt *valjkl);
/* objective: adds  sum_k val_k <A^{sub_k}, X_j> */
PRIMALrescodee PRIMAL_putbarcj(PRIMALtask_t t, int j, int num,
                         const int *sub, const PRIMALrealt *val);
/* block bar writes (reference putbarablocktriplet/putbarcblocktriplet/
 * putbaraijlist): they add terms to the store. A-bar by entries
 * (con,bar,k,l,val); C-bar (bar,k,l,val); the list by (i,j). */
PRIMALrescodee PRIMAL_putbarablocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const int *subk, const int *subl,
        const PRIMALrealt *valijkl);
/* Adds C-bar block triplets (bar,k,l,val) to the store. */
PRIMALrescodee PRIMAL_putbarcblocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subj, const int *subk, const int *subl, const PRIMALrealt *valjkl);
/* Adds lists of matrices to the (i,j) entries of A-bar. */
PRIMALrescodee PRIMAL_putbaraijlist(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const PRIMALint64t *alphaptrb,
        const PRIMALint64t *alphaptre, const PRIMALint64t *matidx,
        const PRIMALrealt *weights);
/* Returns the number of bar variables. */
PRIMALrescodee PRIMAL_getnumbarvar(PRIMALtask_t t, int *num);
/* solution of the bar variable j (dim_j x dim_j matrix, row-major) */
PRIMALrescodee PRIMAL_getbarxj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *xj);
/* Bar dual block: writes dim_j*dim_j doubles in row-major order. */
PRIMALrescodee PRIMAL_getbarsj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *sj);
/* reference bar surface: counters (getnumbaranz/getnumbarcnz), names
 * (putbarvarname/getbarvarname/getbarvarnameindex/getbarvarnamelen), slices of
 * barx/barsj (getbarxslice/getbarsslice, concatenated dense d*d blocks),
 * warm start (putbarxj/putbarsj) and the capacity hint
 * putmaxnumbarvar. */
PRIMALrescodee PRIMAL_getnumbaranz(PRIMALtask_t t, PRIMALint64t *nz);
/* Number of stored C-bar terms. */
PRIMALrescodee PRIMAL_getnumbarcnz(PRIMALtask_t t, PRIMALint64t *nz);
/* Sets the name of bar variable j (reference form). */
PRIMALrescodee PRIMAL_putbarvarname(PRIMALtask_t t, int j, const char *name);
/* Copies the name of bar variable i into the caller buffer. */
PRIMALrescodee PRIMAL_getbarvarname(PRIMALtask_t t, int i, int sizename, char *name);
/* reference bar name->index delegate (assignment fixed to 0). */
PRIMALrescodee PRIMAL_getbarvarnameindex(PRIMALtask_t t, const char *somename,
                                         int *asgn, int *index);
/* Length of the name of bar variable i (0 if unnamed). */
PRIMALrescodee PRIMAL_getbarvarnamelen(PRIMALtask_t t, int i, int *len);
/* Reads the bar solution over bar variables [first,last). */
PRIMALrescodee PRIMAL_getbarxslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barxslice);
/* Reads the bar dual over bar variables [first,last). */
PRIMALrescodee PRIMAL_getbarsslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barsslice);
/* Reads dim_j*dim_j row-major doubles for the primal bar warm start. */
PRIMALrescodee PRIMAL_putbarxj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barxj);
/* Reads dim_j*dim_j row-major doubles for the dual bar warm start. */
PRIMALrescodee PRIMAL_putbarsj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barsj);
/* Capacity hint for the number of bar variables (no-op here). */
PRIMALrescodee PRIMAL_putmaxnumbarvar(PRIMALtask_t t, int maxnumbarvar);

/* internal getters for CBF I/O (bar dimensions, barC/barA terms, matrix store) */
PRIMALrescodee PRIMAL_getbarsize(PRIMALtask_t t, int j, int *dim);
/* Dimension of bar variable j (reference getdimbarvarj). */
PRIMALrescodee PRIMAL_getdimbarvarj(PRIMALtask_t t, int j, int *dimbarvarj);
/* Packed length d(d+1)/2 of bar variable j. */
PRIMALrescodee PRIMAL_getlenbarvarj(PRIMALtask_t t, int j, PRIMALint64t *lenbarvarj);
/* Number of stored C-bar terms. */
PRIMALrescodee PRIMAL_getnumbarcterm(PRIMALtask_t t, int *num);
/* Reads stored C-bar term k (bar variable, symmetric id, coefficient). */
PRIMALrescodee PRIMAL_getbarcitem(PRIMALtask_t t, int k, int *jbar, int *msym, PRIMALrealt *coef);
/* Number of stored A-bar terms. */
PRIMALrescodee PRIMAL_getnumbaraterm(PRIMALtask_t t, int *num);
/* Reads stored A-bar term k (constraint, bar variable, symmetric id, coefficient). */
PRIMALrescodee PRIMAL_getbaraitem(PRIMALtask_t t, int k, int *con, int *jbar, int *msym, PRIMALrealt *coef);
/* Number of symmetric matrices in the matrix store. */
PRIMALrescodee PRIMAL_getnumsymmat(PRIMALtask_t t, int *num);
/* Reads the dimension and nonzero count of stored symmetric matrix m. */
PRIMALrescodee PRIMAL_getsymmatinfo(PRIMALtask_t t, int m, int *dim, int *nnz);
/* Reads stored entry e of symmetric matrix m as (i,j,val). */
PRIMALrescodee PRIMAL_getsymmatentry(PRIMALtask_t t, int m, int e, int *i, int *j, PRIMALrealt *val);

/* optimize + results */
PRIMALrescodee PRIMAL_optimize(PRIMALtask_t t);
/* ---- basis (solvebasis) ---- */
/* status keys for variables/rows (PRIMAL PRIMALstakey): basic, superbasic,
 * at lower, at upper. PRIMAL_SK_UNDEF for elements never set. */
typedef enum {
    PRIMAL_SK_UNDEF = 0,
    PRIMAL_SK_BAS = 1,
    PRIMAL_SK_SUPBAS = 2,
    PRIMAL_SK_LOW = 3,
    PRIMAL_SK_UPR = 4
} PRIMALstakeye;
/* sets/reads the basis status (for rows: BAS/supbas are equivalent to a
 * basic slack). Vectors of length numcon (skc) / numvar (skx). */
PRIMALrescodee PRIMAL_putskc(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc);
/* Sets the variable status keys of the basis. */
PRIMALrescodee PRIMAL_putskx(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skx);
/* Reads the row status keys of the basis. */
PRIMALrescodee PRIMAL_getskc(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skc);
/* Reads the variable status keys of the basis. */
PRIMALrescodee PRIMAL_getskx(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skx);
/* Name as string of a basis status key. */
PRIMALrescodee PRIMAL_sktostr(PRIMALtask_t t, PRIMALstakeye sk, char *str);
/* Parses a basis status key from its name. */
PRIMALrescodee PRIMAL_strtosk(PRIMALtask_t t, const char *str, PRIMALstakeye *sk);
/* evaluates the current basis (skc+skx): primal solution (basic solution) and
 * duals from the basis system, verifying primal/dual feasibility with the
 * public getters; if the basis is not primal feasible (incompatible rows)
 * returns ERR_INFEASIBLE; if not dual-feasible it optimizes from scratch
 * (documented deviation: the clone's simplex does not warm start from
 * arbitrary bases, the basis serves as a specification of the BASIC SOLUTION). */
PRIMALrescodee PRIMAL_solvebasis(PRIMALtask_t t);
/* basis write/read in MPS BAS format (XLOWER/XUPPER/XBASIC for
 * variables, XBASIC for rows); names c%d / x%d consistent with writedata */
PRIMALrescodee PRIMAL_writebasis(PRIMALtask_t t, const char *filename);
/* Reads a basis in MPS BAS format from the file. */
PRIMALrescodee PRIMAL_readbasis(PRIMALtask_t t, const char *filename);
/* Solution imports are complete, validated, transactional images. Imported
 * vectors are unverified: statuses UNKNOWN and objective
 * getters unavailable until solve. Malformed imports preserve the old result.
 * Solution I/O (reference writesolution/readsolution, writebsolution/
 * readbsolution, writebsolutionhandle). Declared deviation: the reference's
 * FORMAT was not read; here there is a text format and a binary format
 * of our own that round-trip. `*file` are the same calls. */
typedef void (*PRIMALhwritefunc)(void *handle, const char *data, int len);
typedef int (*PRIMALhreadfunc)(void *handle, char *buffer, int *len);
PRIMALrescodee PRIMAL_writesolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename);
/* Reads a solution key from the text file. */
PRIMALrescodee PRIMAL_readsolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename);
/* Writes all solution keys to the file. */
PRIMALrescodee PRIMAL_writesolutionfile(PRIMALtask_t t, const char *filename);
/* Reads all solution keys from the file. */
PRIMALrescodee PRIMAL_readsolutionfile(PRIMALtask_t t, const char *filename);
/* Writes the native binary solution; nonzero compression is rejected. */
PRIMALrescodee PRIMAL_writebsolution(PRIMALtask_t t, const char *filename, int compress);
/* Reads the solution from the binary form. */
PRIMALrescodee PRIMAL_readbsolution(PRIMALtask_t t, const char *filename, int compress);
/* Writes the binary solution through a user write callback. */
PRIMALrescodee PRIMAL_writebsolutionhandle(PRIMALtask_t t, PRIMALhwritefunc func,
                                           void *handle, int compress);
/* JSON form (JSOL): a flat object of our own. */
PRIMALrescodee PRIMAL_writejsonsol(PRIMALtask_t t, const char *filename);
/* Reads a solution from a JSON file. */
PRIMALrescodee PRIMAL_readjsonsol(PRIMALtask_t t, const char *filename);
/* Reads a solution from a JSON string. */
PRIMALrescodee PRIMAL_readjsonstring(PRIMALtask_t t, const char *data);
/* ---- sensitivity (LP, post-optimal) ----
 * Range of cost c_j for which the current SOLUTION (x*, duals included)
 * stays optimal (linear problems already solved only). Returns
 * PRIMAL_RES_ERR_ARG if not applicable. */
PRIMALrescodee PRIMAL_costsensitivity(PRIMALtask_t t, int j,
                                 PRIMALrealt *lcost, PRIMALrealt *ucost);
/* Range of the RHS bound of row i (active side) for which the current DUALS
 * stay optimal: the row stays at the active bound with the same basis. */
PRIMALrescodee PRIMAL_rhssensitivity(PRIMALtask_t t, int i,
                                PRIMALrealt *lrange, PRIMALrealt *urange);
/* progress callback: called with a user info string after each major
 * iteration / solution update (PRIMAL PRIMAL_progresscb equivalent).
 * Return value ignored. Signature matches PRIMALcallbackfunc. */
typedef void (*PRIMALprogresscb)(void *handle, const char *info);
/* Sets the progress callback and its handle. */
PRIMALrescodee PRIMAL_setprogresscb(PRIMALtask_t t, PRIMALprogresscb cb, void *handle);
/* ---- general callbacks (reference putcallbackfunc/getcallbackfunc/
 * putresponsefunc) ----
 * The general callback receives an event code (the numbers are those of
 * MSKcallbackcodee) and three detail vectors; this solver emits the
 * BEGIN/END events of OPTIMIZER/READ/WRITE and, for the route that answers, those of
 * SIMPLEX/INTPNT/MIO/CONIC with the vectors at NULL (detail not
 * populated, declared deviation). The response callback is invoked when
 * a solve ends with a code different from PRIMAL_RES_OK.
 * Declared deviation: the per-iteration codes (MSK_CALLBACK_IM_* and the
 * MSK_CALLBACK_INTPNT/CONIC/PRIMAL_SIMPLEX "middle" ones) are not emitted. */
typedef enum {
    PRIMAL_CALLBACK_BEGIN_CONIC     = 1,
    PRIMAL_CALLBACK_BEGIN_INTPNT    = 15,
    PRIMAL_CALLBACK_BEGIN_MIO       = 17,
    PRIMAL_CALLBACK_BEGIN_OPTIMIZER = 19,
    PRIMAL_CALLBACK_BEGIN_READ      = 28,
    PRIMAL_CALLBACK_BEGIN_SIMPLEX   = 30,
    PRIMAL_CALLBACK_BEGIN_WRITE     = 33,
    PRIMAL_CALLBACK_END_CONIC       = 38,
    PRIMAL_CALLBACK_END_INTPNT      = 52,
    PRIMAL_CALLBACK_END_MIO         = 54,
    PRIMAL_CALLBACK_END_OPTIMIZER   = 56,
    PRIMAL_CALLBACK_END_READ        = 65,
    PRIMAL_CALLBACK_END_SIMPLEX     = 67,
    PRIMAL_CALLBACK_END_WRITE       = 71,
    PRIMAL_CALLBACK_READ_OPF        = 95,
    PRIMAL_CALLBACK_READ_OPF_SECTION = 96,
    PRIMAL_CALLBACK_WRITE_OPF       = 107,
    /* per-iteration / solution-update (the reference's "middle" ones) */
    PRIMAL_CALLBACK_CONIC           = 34,
    PRIMAL_CALLBACK_PRIMAL_SIMPLEX  = 93,
    PRIMAL_CALLBACK_INTPNT          = 90,
    /* internal sub-steps */
    PRIMAL_CALLBACK_IM_LU           = 79,
    PRIMAL_CALLBACK_IM_ORDER        = 84
} PRIMALcallbackcodee;
/* Name as string of a callback event code. */
PRIMALrescodee PRIMAL_callbackcodetostr(PRIMALcallbackcodee code, char *str);

typedef void (*PRIMALcallbackcb)(PRIMALtask_t task, void *handle,
    PRIMALcallbackcodee code, const PRIMALrealt *info,
    const PRIMALint32t *intinfo, const PRIMALint64t *lliinfo);
typedef void (*PRIMALresponsecb)(PRIMALtask_t task, void *handle, PRIMALrescodee res);

/* Sets the general event callback. */
PRIMALrescodee PRIMAL_putcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb cb, void *handle);
/* Reads the general event callback and its handle. */
PRIMALrescodee PRIMAL_getcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb *cb, void **handle);
/* Sets the response callback invoked on a non-OK solve code. */
PRIMALrescodee PRIMAL_putresponsefunc(PRIMALtask_t t, PRIMALresponsecb cb, void *handle);
/* name for the info strings of the progress callback (max 63 chars,
 * truncated like PRIMAL_setinfoconnname) */
PRIMALrescodee PRIMAL_setinfoconnname(PRIMALtask_t t, const char *connname);
/* Reads the primal variable vector of the solution. */
PRIMALrescodee PRIMAL_getxx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xx);
/* Reads the dual variable vector of the solution. */
PRIMALrescodee PRIMAL_gety(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *y);
/* warm start: provides a starting point (used by the interior-point route;
 * ignored by simplex/MIP/cones). whichsol: PRIMAL_SOL_ITR or PRIMAL_SOL_BAS. */
PRIMALrescodee PRIMAL_putxx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *xx);
/* Provides a dual starting point for the interior-point route. */
PRIMALrescodee PRIMAL_puty(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *y);
/* Reads the lower-constraint slack vector of the solution. */
PRIMALrescodee PRIMAL_getslc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *slc);
/* Reads the upper-constraint slack vector of the solution. */
PRIMALrescodee PRIMAL_getsuc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *suc);
/* Reads the lower-variable slack vector of the solution. */
PRIMALrescodee PRIMAL_getslx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *slx);
/* Reads the upper-variable slack vector of the solution. */
PRIMALrescodee PRIMAL_getsux(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *sux);
/* Reads the primal objective; unavailable after model edits until re-solved. */
PRIMALrescodee PRIMAL_getprimalobj(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pobj);
/* Reads the dual objective value. */
PRIMALrescodee PRIMAL_getdualobj(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *dobj);
/* Reads the solution status of the given key. */
PRIMALrescodee PRIMAL_getsolsta(PRIMALtask_t t, PRIMALsolt which, PRIMALsolstae *solsta);
/* Reads the problem status of the given key. */
PRIMALrescodee PRIMAL_getprosta(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *prosta);
/* `getsolution`: reads the complete solution in one call (reference
 * MSK_getsolution). Every pointer is optional (NULL = skip). `skn` (status
 * keys of the cones) is SK_UNDEF for every cone and `snx` (conic dual per variable)
 * is 0: declared deviations (no basis for a conic block; the dual of a
 * cone lives inside the block). Without a published point the point buffers
 * answer ERR_ARG, like the individual getters; the states are read all the same. */
PRIMALrescodee PRIMAL_getsolution(PRIMALtask_t t, PRIMALsolt which,
    PRIMALprostae *problemsta, PRIMALsolstae *solutionsta,
    PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
    PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y,
    PRIMALrealt *slc, PRIMALrealt *suc, PRIMALrealt *slx, PRIMALrealt *sux,
    PRIMALrealt *snx);
/* Reads the cone status keys (SK_UNDEF for every cone here). */
PRIMALrescodee PRIMAL_getskn(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skn);
/* Reads the per-variable conic dual vector (0 here: a declared deviation). */
PRIMALrescodee PRIMAL_getsnx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *snx);
/* ---- Farkas certificates (infeasibility rays, vectors) ----
 * PRIMAL_getdualray fills y (numcon entries): a certificate that the primal has
 * no feasible point, measured as   sum_i y_i b_i(active) > 0  and  A'y <= 0
 * on every nonnegative variable (equality rows; see the ranged form below).
 * PRIMAL_getprimalray fills rho (numvar entries): a recession direction of the
 * feasible set along which the objective is unbounded, measured as
 *   (A rho)_i <= 0 if row i has a finite upper bound, >= 0 if it has a finite
 *   lower bound, = 0 if both (ranged or fixed);  rho_j >= 0 if x_j has a finite
 *   lower bound, <= 0 if it has a finite upper bound, = 0 if both (bounded) or
 *   free;  and s * c'rho < 0 (s = +1 min, -1 max).
 * Both are scaled to max |.| = 1. A vector is handed out only when it measured
 * as one of the two alternatives in the form the solver actually solved; until
 * then the getters answer PRIMAL_RES_ERR_ARG, exactly as a solution getter does
 * for a problem with no solution. The same holds for a dual ray whose support
 * needs a variable-bound row: it has no one-entry-per-constraint image. */
PRIMALrescodee PRIMAL_getdualray(PRIMALtask_t t, PRIMALrealt *y);
/* Fills rho with a measured primal recession (unboundedness) ray. */
PRIMALrescodee PRIMAL_getprimalray(PRIMALtask_t t, PRIMALrealt *rho);
/* ---- solution quality: max primal/dual violations of the solution ----
 * (measured on the problem data with the getters; -1 if not applicable) */
PRIMALrescodee PRIMAL_getprimalinfeas(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pinf);
/* Max dual violation of the solution (-1 if not applicable). */
PRIMALrescodee PRIMAL_getdualinfeas(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *dinf);
/* ---- solution information: per-index violations and summary ----
 * Reference family (MOSEK 11.2.4): `getpviolcon`/`getpviolvar`/
 * `getpviolbarvar`/`getpviolcones` write into `viol` the primal violation
 * of the indices listed in `sub` (which is a vector of indices, not an interval);
 * an index out of domain, `viol`/`sub` null or no solution are
 * `PRIMAL_RES_ERR_ARG` **without writing anything**. `PRIMAL_getsolutioninfo` reports
 * the maxima of the family plus `pobj`/`dobj` (accepts NULL for the fields not
 * requested). The primal half is exact; the dual one (dviol*) is in our
 * convention of the `y`/`slc`/`sux` getters (README «Dual conventions»), with the
 * cone members left unmeasured — see the comment in `primal.c`. */
PRIMALrescodee PRIMAL_getpviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
/* Primal violation of the listed variables. */
PRIMALrescodee PRIMAL_getpviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
/* Primal violation of the listed bar variables. */
PRIMALrescodee PRIMAL_getpviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol);
/* Primal violation of the listed cones. */
PRIMALrescodee PRIMAL_getpviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol);
/* The dual half of the same family. Same form of `sub`/`viol` and same
 * refusal contract; the values are in our getter convention
 * (README «Dual conventions», sign mirrored with respect to the reference) and a cone
 * member stays unmeasured in `dviolvar`. */
PRIMALrescodee PRIMAL_getdviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
/* Dual violation of the listed variables. */
PRIMALrescodee PRIMAL_getdviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
/* Dual violation of the listed bar variables. */
PRIMALrescodee PRIMAL_getdviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol);
/* Dual violation of the listed cones. */
PRIMALrescodee PRIMAL_getdviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol);
/* Reports the solution's maxima plus pobj/dobj (NULL skips a field). */
PRIMALrescodee PRIMAL_getsolutioninfo(PRIMALtask_t t, PRIMALsolt which,
    PRIMALrealt *pobj, PRIMALrealt *pviolcon, PRIMALrealt *pviolvar,
    PRIMALrealt *pviolbarvar, PRIMALrealt *pviolcone, PRIMALrealt *pviolitg,
    PRIMALrealt *dobj, PRIMALrealt *dviolcon, PRIMALrealt *dviolvar,
    PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone);
/* ---- feasibility repair (elastic): min sum(s^- + s^+) s.t.
 * lo - s^- <= Ax <= up + s^+, lx - s^- <= x <= ux + s^+ ----
 * Brings the repaired solution (x) back into the task. */
PRIMALrescodee PRIMAL_feasrepair(PRIMALtask_t t);
/* Reads a solution item (`part`) over [first,last). */
PRIMALrescodee PRIMAL_getsolutionslice(PRIMALtask_t t, PRIMALsolt which, int part,
                                 int first, int last, PRIMALrealt *values);
/* Slices of the solution vectors (reference: getxxslice, getyslice,
 * getslcslice, getsucslice, getslxslice, getsuxslice, getskxslice, getskcslice)
 * and reduced cost `(s_l^x)_j - (s_u^x)_j` (getreducedcosts). Form [first,last),
 * `last-first` entries; a refusal does not write. */
PRIMALrescodee PRIMAL_getxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *xx);
/* Reads the dual vector over [first,last). */
PRIMALrescodee PRIMAL_getyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *y);
/* Reads the lower-constraint slacks over [first,last). */
PRIMALrescodee PRIMAL_getslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slc);
/* Reads the upper-constraint slacks over [first,last). */
PRIMALrescodee PRIMAL_getsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *suc);
/* Reads the lower-variable slacks over [first,last). */
PRIMALrescodee PRIMAL_getslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slx);
/* Reads the upper-variable slacks over [first,last). */
PRIMALrescodee PRIMAL_getsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *sux);
/* Reads the reduced costs over [first,last) (== -(slx+sux)). */
PRIMALrescodee PRIMAL_getreducedcosts(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *redcosts);
/* Reads the variable basis keys over [first,last). */
PRIMALrescodee PRIMAL_getskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALstakeye *skx);
/* Reads the row basis keys over [first,last). */
PRIMALrescodee PRIMAL_getskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALstakeye *skc);
/* Writes the variable basis keys over [first,last). */
PRIMALrescodee PRIMAL_putskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skx);
/* Writes the row basis keys over [first,last). */
PRIMALrescodee PRIMAL_putskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skc);
/* 2-norms of the primal solution: ||x^c||, ||x||, ||X_bar||_F. */
PRIMALrescodee PRIMAL_getprimalsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmxc, PRIMALrealt *nrmxx, PRIMALrealt *nrmbarx);
/* 2-norms of the dual solution: ||y||, ||slc||, ||suc||, ||slx||, ||sux||, ||snx||, ||Z_bar||_F. */
PRIMALrescodee PRIMAL_getdualsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmy, PRIMALrealt *nrmslc, PRIMALrealt *nrmsuc,
        PRIMALrealt *nrmslx, PRIMALrealt *nrmsux, PRIMALrealt *nrmsnx,
        PRIMALrealt *nrmbars);
/* `x^c`: the value of the constraint variables (reference getxc/getxcslice):
 * the first member of the row, read from `row_activity` (scalar | quadratic |
 * bar) -- the same read as `getpviolcon`. */
PRIMALrescodee PRIMAL_getxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc);
/* Reads the row-activity vector x^c over [first,last). */
PRIMALrescodee PRIMAL_getxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *xc);
/* setters of x^c and s_n^x (reference putxc/putxcslice/putsnx/putsnxslice/
 * getsnxslice). s_n^x is not computed by any route: it is storage. */
PRIMALrescodee PRIMAL_putxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc);
/* Writes the row-activity vector x^c over [first,last). */
PRIMALrescodee PRIMAL_putxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xc);
/* Writes the s_n^x storage vector. */
PRIMALrescodee PRIMAL_putsnx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *snx);
/* Writes the s_n^x storage vector over [first,last). */
PRIMALrescodee PRIMAL_putsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *snx);
/* Reads the s_n^x storage vector over [first,last). */
PRIMALrescodee PRIMAL_getsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  PRIMALrealt *snx);
/* setters of the solution vectors (reference putslc/putsuc/putslx/putsux,
 * putxxslice/putyslice/putslxslice/putsuxslice): they write into the point buffer
 * (exists after the first opt_prepare); a slice is [first, last) with
 * last-first entries; a refusal does not write. */
PRIMALrescodee PRIMAL_putslc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slc);
/* Writes the upper-constraint slack vector. */
PRIMALrescodee PRIMAL_putsuc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *suc);
/* Writes the lower-variable slack vector. */
PRIMALrescodee PRIMAL_putslx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slx);
/* Writes the upper-variable slack vector. */
PRIMALrescodee PRIMAL_putsux(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *sux);
/* Writes the primal vector over [first,last). */
PRIMALrescodee PRIMAL_putxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xx);
/* Writes the dual vector over [first,last). */
PRIMALrescodee PRIMAL_putyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                const PRIMALrealt *y);
/* Writes the lower-variable slacks over [first,last). */
PRIMALrescodee PRIMAL_putslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slx);
/* Writes the upper-variable slacks over [first,last). */
PRIMALrescodee PRIMAL_putsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *sux);
/* Writes the lower-constraint slacks over [first,last). */
PRIMALrescodee PRIMAL_putslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slc);
/* Writes the upper-constraint slacks over [first,last). */
PRIMALrescodee PRIMAL_putsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *suc);
/* Writes the same bound to the listed variables. */
PRIMALrescodee PRIMAL_putvarboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkx, PRIMALrealt blx, PRIMALrealt bux);
/* Writes the same bound to the listed constraints. */
PRIMALrescodee PRIMAL_putconboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkc, PRIMALrealt blc, PRIMALrealt buc);
/* write the problem in a readable text form (debugging) */
PRIMALrescodee PRIMAL_writedata(PRIMALtask_t t, const char *filename);
/* Reads a problem from a file in a format known by extension/content. */
PRIMALrescodee PRIMAL_readdata(PRIMALtask_t t, const char *filename);
/* reference forms around readdata/writedata (the declared format does not
 * change what is read: the reader recognizes the file by content) and
 * `getapiecenumnz`, the nonzeros of A in a rectangular piece. */
PRIMALrescodee PRIMAL_readdataautoformat(PRIMALtask_t t, const char *filename);
/* Reads a file in the declared data format and compression (gzip/zstd refused). */
PRIMALrescodee PRIMAL_readdataformat(PRIMALtask_t t, const char *filename,
                                     PRIMALdataformate format,
                                     PRIMALcompresstypee compress);
/* Reads a task from a file (form of readdata). */
PRIMALrescodee PRIMAL_readtask(PRIMALtask_t t, const char *filename);
/* Writes a task to a file (form of writedata). */
PRIMALrescodee PRIMAL_writetask(PRIMALtask_t t, const char *filename);
/* Number of A nonzeros in the rectangular piece [firsti,lasti) x [firstj,lastj). */
PRIMALrescodee PRIMAL_getapiecenumnz(PRIMALtask_t t, int firsti, int lasti,
                                     int firstj, int lastj, int *numnz);

/* logging callback: void (*)(void *handle, const char *msg) */
typedef void (*PRIMALlogcb)(void *handle, const char *msg);
/* Sets the logging callback and its handle. */
PRIMALrescodee PRIMAL_setlogcb(PRIMALtask_t t, PRIMALlogcb logcb, void *loghandle);
/* Links a stream callback to one task stream. */
PRIMALrescodee PRIMAL_linkfunctotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                     void *handle, PRIMALstreamfunc func);
/* file stream, env stream, echo (reference linkfileto*stream,
 * linkfunctoenvstream, unlinkfuncfrom*stream, echo*). The env stream is
 * inherited by tasks created afterwards. */
PRIMALrescodee PRIMAL_linkfiletotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                           const char *filename, int append);
/* Unlinks a stream callback from one task stream. */
PRIMALrescodee PRIMAL_unlinkfuncfromtaskstream(PRIMALtask_t t, PRIMALstreamtypee which);
/* Formats a message into the task stream (printf-style). */
PRIMALrescodee PRIMAL_echotask(PRIMALtask_t t, PRIMALstreamtypee which, const char *format, ...);
/* Links a file to one environment stream. */
PRIMALrescodee PRIMAL_linkfiletoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                          const char *filename, int append);
/* Links a callback to one environment stream. */
PRIMALrescodee PRIMAL_linkfunctoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                          void *handle, PRIMALstreamfunc func);
/* Unlinks a stream callback from one environment stream. */
PRIMALrescodee PRIMAL_unlinkfuncfromenvstream(PRIMALenv_t env, PRIMALstreamtypee which);
/* Formats a message into the environment stream (printf-style). */
PRIMALrescodee PRIMAL_echoenv(PRIMALenv_t env, PRIMALstreamtypee which, const char *format, ...);
/* Prints the introductory banner to the environment stream. */
PRIMALrescodee PRIMAL_echointro(PRIMALenv_t env, int longver);

/* ---- reference parity: long parameters, name generators, diagnostics,
 * optimize*, repair/sensitivity, toconic ---- */
/* Reads a long (64-bit) integer parameter by id. */
PRIMALrescodee PRIMAL_getlintparam(PRIMALtask_t t, int param, PRIMALint64t *parvalue);
/* Sets a long (64-bit) integer parameter by id. */
PRIMALrescodee PRIMAL_putlintparam(PRIMALtask_t t, int param, PRIMALint64t parvalue);
/* Sets a parameter by name and string value. */
PRIMALrescodee PRIMAL_putparam(PRIMALtask_t t, const char *parname, const char *parvalue);
/* Writes the parameter settings to a file. */
PRIMALrescodee PRIMAL_writeparamfile(PRIMALtask_t t, const char *filename);
/* Reads parameter settings from a file. */
PRIMALrescodee PRIMAL_readparamfile(PRIMALtask_t t, const char *filename);
/* Generates variable names from a format and axis description. */
PRIMALrescodee PRIMAL_generatevarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Generates constraint names from a format and axis description. */
PRIMALrescodee PRIMAL_generateconnames(PRIMALtask_t t, int num, const int *subi, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Generates cone-block names from a format and axis description. */
PRIMALrescodee PRIMAL_generateconenames(PRIMALtask_t t, int num, const int *subk, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Generates bar-variable names from a format and axis description. */
PRIMALrescodee PRIMAL_generatebarvarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Generates ACC names from a format and axis description. */
PRIMALrescodee PRIMAL_generateaccnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Generates DJC names from a format and axis description. */
PRIMALrescodee PRIMAL_generatedjcnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
/* Reads cone k's type, parameter and member count. */
PRIMALrescodee PRIMAL_getconeinfo(PRIMALtask_t t, int k, PRIMALconetypee *ct,
                                  PRIMALrealt *conepar, int *nummem);
/* Prints the parameter settings to stdout. */
PRIMALrescodee PRIMAL_printparam(PRIMALtask_t t);
/* Prints a read summary on the stream (stdout). */
PRIMALrescodee PRIMAL_readsummary(PRIMALtask_t t, int whichstream);
/* Optimizes and returns the termination code. */
PRIMALrescodee PRIMAL_optimizetrm(PRIMALtask_t t, PRIMALrescodee *trmcode);
/* Optimizes several tasks (optionally in parallel) and returns codes. */
PRIMALrescodee PRIMAL_optimizebatch(PRIMALenv_t env, int israce, PRIMALrealt maxtime,
        int numthreads, PRIMALint64t numtask, const PRIMALtask_t *task,
        PRIMALrescodee *trmcode, PRIMALrescodee *rcode);
/* Elastic primal repair with the given row/column weights. */
PRIMALrescodee PRIMAL_primalrepair(PRIMALtask_t t, const PRIMALrealt *wlc,
        const PRIMALrealt *wuc, const PRIMALrealt *wlx, const PRIMALrealt *wux);
/* Dual sensitivity ranges for the listed variables. */
PRIMALrescodee PRIMAL_dualsensitivity(PRIMALtask_t t, int numj, const int *subj,
        PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej);
/* Primal sensitivity ranges for the listed rows and variables. */
PRIMALrescodee PRIMAL_primalsensitivity(PRIMALtask_t t, int numi, const int *subi,
        const int *marki, int numj, const int *subj, const int *markj,
        PRIMALrealt *leftpricei, PRIMALrealt *rightpricei, PRIMALrealt *leftrangei,
        PRIMALrealt *rightrangei, PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej);
/* Rewrites the model in conic form. */
PRIMALrescodee PRIMAL_toconic(PRIMALtask_t t);
/* put-style ACC, AFE/cones/barA per row */
PRIMALrescodee PRIMAL_putacc(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t domidx,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
/* Writes the descriptions of several ACCs at once. */
PRIMALrescodee PRIMAL_putacclist(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *accidxs, const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
/* Writes several AFE rows from blocked index/value data. */
PRIMALrescodee PRIMAL_putafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *numnzrow, const PRIMALint64t *ptrrow,
        PRIMALint64t lenidxval, const int *varidx, const PRIMALrealt *val);
/* Rewrites cone k with the given type, parameter and members. */
PRIMALrescodee PRIMAL_putcone(PRIMALtask_t t, int k, PRIMALconetypee ct, PRIMALrealt conepar,
                              int nummem, const int *submem);
/* Writes several A-bar rows from blocked matrix lists. */
PRIMALrescodee PRIMAL_putbararowlist(PRIMALtask_t t, int num, const int *subi,
        const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *subj,
        const PRIMALint64t *nummat, const PRIMALint64t *matidx, const PRIMALrealt *weights);
/* "new"-style solution API and per-index setters */
PRIMALrescodee PRIMAL_solutiondef(PRIMALtask_t t, PRIMALsolt which, int *isdef);
/* Sets the solution entry of constraint i (key, x, sl, su). */
PRIMALrescodee PRIMAL_putconsolutioni(PRIMALtask_t t, int i, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su);
/* Sets the dual solution entry of constraint i. */
PRIMALrescodee PRIMAL_putsolutionyi(PRIMALtask_t t, int i, PRIMALsolt which, PRIMALrealt y);
/* Sets the solution entry of variable j (key, x, sl, su, sn). */
PRIMALrescodee PRIMAL_putvarsolutionj(PRIMALtask_t t, int j, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su, PRIMALrealt sn);
/* Reads the full solution including the ACC doty vector. */
PRIMALrescodee PRIMAL_getsolutionnew(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *problemsta,
        PRIMALsolstae *solutionsta, PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
        PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y, PRIMALrealt *slc, PRIMALrealt *suc,
        PRIMALrealt *slx, PRIMALrealt *sux, PRIMALrealt *snx, PRIMALrealt *doty);
/* Writes the full solution including the ACC doty vector. */
PRIMALrescodee PRIMAL_putsolutionnew(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx, const PRIMALrealt *doty);
/* Writes the full solution (without doty). */
PRIMALrescodee PRIMAL_putsolution(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx);
/* Reports the solution maxima including the ACC/DJC fields. */
PRIMALrescodee PRIMAL_getsolutioninfonew(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pobj,
        PRIMALrealt *pviolcon, PRIMALrealt *pviolvar, PRIMALrealt *pviolbarvar,
        PRIMALrealt *pviolcone, PRIMALrealt *pviolacc, PRIMALrealt *pvioldjc,
        PRIMALrealt *pviolitg, PRIMALrealt *dobj, PRIMALrealt *dviolcon,
        PRIMALrealt *dviolvar, PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone,
        PRIMALrealt *dviolacc);
/* string/handle I/O, basis solve, sparse Cholesky, clone/dual/subproblem */
PRIMALrescodee PRIMAL_readlpstring(PRIMALtask_t t, const char *data);
/* Reads an OPF problem from a string (no parser: ERR_ARG). */
PRIMALrescodee PRIMAL_readopfstring(PRIMALtask_t t, const char *data);
/* Reads a PTF problem from a string (no parser: ERR_ARG). */
PRIMALrescodee PRIMAL_readptfstring(PRIMALtask_t t, const char *data);
/* Reads task data through a user read callback. */
PRIMALrescodee PRIMAL_readdatahandle(PRIMALtask_t t, PRIMALhreadfunc hread, void *h,
                                     int format, int compress, const char *path);
/* Writes task data through a user write callback. */
PRIMALrescodee PRIMAL_writedatahandle(PRIMALtask_t t, PRIMALhwritefunc func, void *handle,
                                      int format, int compress);
/* Initializes a basis solve; writes the basis variable indices. */
PRIMALrescodee PRIMAL_initbasissolve(PRIMALtask_t t, int *basis);
/* Solves against the initialized basis (forward or transposed). */
PRIMALrescodee PRIMAL_solvewithbasis(PRIMALtask_t t, int transp, int numnz, int *sub,
                                     PRIMALrealt *val, int *numnzout);
/* Condition estimates of the basis and its inverse. */
PRIMALrescodee PRIMAL_basiscond(PRIMALtask_t t, PRIMALrealt *nrmbasis, PRIMALrealt *nrminvbasis);
/* Sparse Cholesky of a symmetric matrix with the given ordering. */
PRIMALrescodee PRIMAL_computesparsecholesky(PRIMALenv_t env, int numthreads, int ordermethod,
        PRIMALrealt tolsingular, int n, const int *anzc, const PRIMALint64t *aptrc,
        const int *asubc, const PRIMALrealt *avalc, int **perm, PRIMALrealt **diag,
        int **lnzc, PRIMALint64t **lptrc, PRIMALint64t *lensubnval, int **lsubc,
        PRIMALrealt **lvalc);
/* Deep-copies the task into clonedtask. */
PRIMALrescodee PRIMAL_clonetask(PRIMALtask_t t, PRIMALtask_t *clonedtask);
/* Builds the dual problem into dualtask (restricted form). */
PRIMALrescodee PRIMAL_getdualproblem(PRIMALtask_t t, PRIMALtask_t *dualtask);
/* Builds an infeasible subproblem of the given solution into inftask. */
PRIMALrescodee PRIMAL_getinfeasiblesubproblem(PRIMALtask_t t, PRIMALsolt which,
                                              PRIMALtask_t *inftask);

/* ---- linear algebra (reference, "Linear algebra" group) ----
 * Dense matrices are COLUMN-major (the reference convention). `uplo` and
 * `transpose` have the values of MSKuploe/MSKtransposee (LO 0/UP 1, NO 0/YES 1). */
typedef enum { PRIMAL_TRANSPOSE_NO = 0, PRIMAL_TRANSPOSE_YES = 1 } PRIMALtransposee;
typedef enum { PRIMAL_UPLO_LO = 0, PRIMAL_UPLO_UP = 1 } PRIMALUploe;

/* Dot product x'y. */
PRIMALrescodee PRIMAL_dot(PRIMALenv_t env, int n, const PRIMALrealt *x,
                          const PRIMALrealt *y, PRIMALrealt *xty);
/* y += alpha*x. */
PRIMALrescodee PRIMAL_axpy(PRIMALenv_t env, int n, PRIMALrealt alpha,
                           const PRIMALrealt *x, PRIMALrealt *y);
/* y = alpha*op(a)*x + beta*y. */
PRIMALrescodee PRIMAL_gemv(PRIMALenv_t env, PRIMALtransposee transa, int m, int n,
                           PRIMALrealt alpha, const PRIMALrealt *a, const PRIMALrealt *x,
                           PRIMALrealt beta, PRIMALrealt *y);
/* c = alpha*op(a)*op(b) + beta*c. */
PRIMALrescodee PRIMAL_gemm(PRIMALenv_t env, PRIMALtransposee transa, PRIMALtransposee transb,
                           int m, int n, int k, PRIMALrealt alpha, const PRIMALrealt *a,
                           const PRIMALrealt *b, PRIMALrealt beta, PRIMALrealt *c);
/* c = alpha*a*op(a) + beta*c (symmetric rank-k update). */
PRIMALrescodee PRIMAL_syrk(PRIMALenv_t env, PRIMALUploe uplo, PRIMALtransposee trans, int n,
                           int k, PRIMALrealt alpha, const PRIMALrealt *a, PRIMALrealt beta,
                           PRIMALrealt *c);
/* Cholesky factorization of a dense symmetric positive definite matrix. */
PRIMALrescodee PRIMAL_potrf(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a);
/* Eigenvalues of a dense symmetric matrix. */
PRIMALrescodee PRIMAL_syeig(PRIMALenv_t env, PRIMALUploe uplo, int n, const PRIMALrealt *a,
                            PRIMALrealt *w);
/* Eigenvalues (and eigenvectors) of a dense symmetric matrix. */
PRIMALrescodee PRIMAL_syevd(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a,
                            PRIMALrealt *w);
/* Triangular solve with a sparse lower-triangular Cholesky factor. */
PRIMALrescodee PRIMAL_sparsetriangularsolvedense(PRIMALenv_t env, PRIMALtransposee transposed,
        int n, const int *lnzc, const PRIMALint64t *lptrc, PRIMALint64t lensubnval,
        const int *lsubc, const PRIMALrealt *lvalc, PRIMALrealt *b);

/* ---- UTF-8 <-> wide-char conversion (reference utf8towchar/wchartoutf8) ----
 * PRIMALwchart is wchar_t (UTF-32 on this platform).  `len` = output units
 * written, `conv` = input units consumed.  Declared deviation: the reference's
 * exact len/conv convention was not read (not invented). */
typedef wchar_t PRIMALwchart;
/* Converts a UTF-8 string to wide characters. */
PRIMALrescodee PRIMAL_utf8towchar(size_t outputlen, size_t *len, size_t *conv,
                                  PRIMALwchart *output, const char *input);
/* Converts a wide-character string to UTF-8. */
PRIMALrescodee PRIMAL_wchartoutf8(size_t outputlen, size_t *len, size_t *conv,
                                  char *output, const PRIMALwchart *input);

#ifdef __cplusplus
}
#endif

#endif /* PRIMAL_H */
