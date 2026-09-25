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
 * CONVENTIONE DUALI (documentata, verificata dai test):
 *   Le soluzioni duali riportate (y, slc, suc, slx, sux) soddisfano,
 *   per il problema ORIGINALE come scritto (qualsiasi senso):
 *       c + Qx + A'y + z = 0          con z = slx + sux
 *   dove (in forma min-normalizzata, cioe' moltiplicando per s=+1 min / -1 max):
 *       y_i > 0  <=> riga i al bound SUPERIORE attivo
 *       y_i < 0  <=> riga i al bound INFERIORE attivo
 *       z_j > 0  <=> variabile j al bound superiore attivo
 *       z_j < 0  <=> variabile j al bound inferiore attivo
 *   slc/suc e slx/sux sono la scomposizione per segno di y e z:
 *       slc = min(y,0), suc = max(y,0), slx = min(z,0), sux = max(z,0).
 *   Multipli di riga/variabile non attivi sono 0 (complementarita').
 *   L'obiettivo duale vale: dobj = -sum_i y_i*b_i(attivo) - sum_j z_j*xb_j(attivo)
 *   in forma min (forte dualita' |pobj - s*dobj_min| ~ 0, verificata dai test).
 *
 * STATO DELLA SOLUZIONE (PRIMAL_getsolsta):
 *   PRIMAL_SOL_STA_OPTIMAL, PRIMAL_SOL_STA_PRIM_INFEAS_CER, PRIMAL_SOL_STA_DUAL_INFEAS_CER,
 *   PRIMAL_SOL_STA_INTEGER_OPTIMAL, PRIMAL_SOL_STA_UNKNOWN.
 * PRIMAL_optimize restituisce PRIMAL_RES_OK con il verdetto in solsta; per
 * comoperatorsita' con la vecchia bozza, infeasibilita'/illimitatezza
 * restituiscono anche PRIMAL_RES_ERR_INFEASIBLE / PRIMAL_RES_ERR_UNBOUNDED.
 * Un membro *_CER NOMINA un vettore di Farkas: viene pubblicato solo dove
 * PRIMAL_getdualray / PRIMAL_getprimalray rispondono (il percorso LP/QP, e solo
 * se il raggio misura nella forma risolta). Dove il verdetto c'e' ma il vettore
 * no — il percorso a tagli tangenti, l'albero e branch-and-bound, una base non
 * ammissibile — solsta resta UNKNOWN e il verdetto si legge in PRIMAL_getprosta.
 * I numeri grezzi e l'accoppiamento prosta+solsta sono fissati da T93 e T94.
 *
 * Deviazioni documentate dal PRIMAL reale:
 *  - una sola soluzione interna, servita sia per PRIMAL_SOL_ITR che PRIMAL_SOL_BAS;
 *  - PRIMAL_OPTIMIZER_DUAL_SIMPLEX e PRIMAL_SIMPLEX usano entrambi il simplesso
 *    primario a due fasi; i QP usano sempre il punto interno (INTPNT).
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
PRIMALrescodee PRIMAL_appendsos2(PRIMALtask_t t, int num, const int *submem, const PRIMALrealt *weight);
PRIMALrescodee PRIMAL_getnumsos(PRIMALtask_t t, int *numsos);
PRIMALrescodee PRIMAL_getsos(PRIMALtask_t t, int k, int *sostype, int *num,
                       int *submem, PRIMALrealt *weight);

/* conic types */
typedef enum {
    PRIMAL_CT_QUAD = 0,   /* (t, x1..xk): t >= sqrt(sum xi^2)    */
    PRIMAL_CT_RQUAD = 1,  /* rotated: 2*x1*x2 >= sum_{i>=3} xi^2 */
    PRIMAL_CT_PEXP = 2,   /* (x1,x2,x3): x1 >= x2*exp(x3/x2), x2 >= 0 (param ignorato) */
    PRIMAL_CT_DEXP = 3,   /* duale: x1 <= x2*exp(x3/x2), x2 <= 0       */
    PRIMAL_CT_PPOW = 4,   /* (x1,x2,x3): x1^a*x2^(1-a) >= |x3|, a=param in (0,1) */
    /* 5 and 6 are the reference's DPOW (dual power cone) and ZERO: not
     * implemented here, and deliberately left empty rather than reused, so a
     * caller passing one of those numbers is refused instead of answered with
     * the wrong cone. RPOW is this solver's own cone and has no reference
     * number, so it sits past the reference's list. */
    PRIMAL_CT_RPOW = 7    /* ruotato: 2*x1^(2a)*x2^(2(1-a)) >= x3^2; a=1/2 -> RQUAD */
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
#define PRIMAL_IPAR_NUM_THREADS               15   /* thread per le parti parallele (default 1) */
#define PRIMAL_IPAR_INTPNT_MAX_NUM_COR        16   /* correttori IPM: -1/1 = 1 (default), >=2 higher-order */
#define PRIMAL_IPAR_PRESOLVE_LEVEL            17   /* 0 off, 1 empty/singleton (default), 2 +duplicati */
#define PRIMAL_IPAR_CONCURRENT_TIME           18   /* optimizer concorrente: 0 tie-break (default), 1 piu' veloce */
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
/* numero di parametri di un tipo nella tabella (1 dou, 2 int, 3 str). */
PRIMALrescodee PRIMAL_getnumparam(PRIMALtask_t t, int partype, int *numparam);
/* nomi dei parametri: getparamname/whichparam/getparammax, le ricerche per nome
 * (isdouparname/isintparname/isstrparname) e la lettura per nome
 * (getnaintparam/getnadouparam/getnastrparam). Il nome e' quello del nostro
 * enum; non esiste un parametro stringa. */
PRIMALrescodee PRIMAL_getparamname(PRIMALtask_t t, int partype, int param, char *parname);
PRIMALrescodee PRIMAL_getparammax(PRIMALtask_t t, int partype, int *parammax);
PRIMALrescodee PRIMAL_whichparam(PRIMALtask_t t, const char *parname, int *partype, int *param);
PRIMALrescodee PRIMAL_isdouparname(PRIMALtask_t t, const char *parname, int *param);
PRIMALrescodee PRIMAL_isintparname(PRIMALtask_t t, const char *parname, int *param);
PRIMALrescodee PRIMAL_isstrparname(PRIMALtask_t t, const char *parname, int *param);
PRIMALrescodee PRIMAL_getnaintparam(PRIMALtask_t t, const char *paramname, int *parvalue);
PRIMALrescodee PRIMAL_getnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt *parvalue);
PRIMALrescodee PRIMAL_getnastrparam(PRIMALtask_t t, const char *paramname,
                                    int sizeparamname, int *len, char *parvalue);
/* setter per nome di int/double e famiglia stringa (nessun parametro stringa:
 * get/put/resetstrparam* rispondono ERR_ARG). */
PRIMALrescodee PRIMAL_putnaintparam(PRIMALtask_t t, const char *paramname, int parvalue);
PRIMALrescodee PRIMAL_putnadouparam(PRIMALtask_t t, const char *paramname, PRIMALrealt parvalue);
PRIMALrescodee PRIMAL_putnastrparam(PRIMALtask_t t, const char *paramname, const char *parvalue);
PRIMALrescodee PRIMAL_getstrparam(PRIMALtask_t t, int param, int maxlen, int *len, char *parvalue);
PRIMALrescodee PRIMAL_getstrparamlen(PRIMALtask_t t, int param, int *len);
PRIMALrescodee PRIMAL_putstrparam(PRIMALtask_t t, int param, const char *parvalue);
PRIMALrescodee PRIMAL_resetstrparam(PRIMALtask_t t, int param);

/* stream callbacks (PRIMAL compatible) */
typedef enum { PRIMAL_STREAM_LOG = 0 } PRIMALstreamtypee;
typedef void (*PRIMALstreamfunc)(void *handle, const char *msg);
/* tipo di nome per analyzenames (riferimento MSKnametypee) */
typedef enum {
    PRIMAL_NAME_TYPE_GEN = 0,
    PRIMAL_NAME_TYPE_MPS = 1,
    PRIMAL_NAME_TYPE_LP  = 2
} PRIMALnametypee;
PRIMALrescodee PRIMAL_analyzenames(PRIMALtask_t t, PRIMALstreamtypee whichstream,
                                   PRIMALnametypee nametype);
/* varianti "al" (allocate-and-return) dei parametri stringa (riferimento
 * getstrparamal/getnastrparamal). Questo solver non ha parametri stringa:
 * rispondono ERR_ARG senza allocare. */
PRIMALrescodee PRIMAL_getstrparamal(PRIMALtask_t t, int param, int numaddchr, char **value);
PRIMALrescodee PRIMAL_getnastrparamal(PRIMALtask_t t, const char *paramname, int numaddchr,
                                      char **value);

/* environment/task lifecycle */
/* callback di uscita su errore fatale (riferimento MSKexitfunc). */
typedef void (*PRIMALexitfunc)(void *handle, const char *msg);
PRIMALrescodee PRIMAL_putexitfunc(PRIMALenv_t env, PRIMALexitfunc exitfunc, void *handle);
PRIMALrescodee PRIMAL_makeenv(PRIMALenv_t *env, void *usercb);
PRIMALrescodee PRIMAL_maketask(PRIMALenv_t env, int maxcon, int maxvar, PRIMALtask_t *task);
/* gestione task del riferimento: makeemptytask (nessuna dimensione dichiarata),
 * getenv, commitchanges/resizetask/updatesolutioninfo (no-op), deletesolution
 * (toglie il punto e il verdetto). */
PRIMALrescodee PRIMAL_makeemptytask(PRIMALenv_t env, PRIMALtask_t *task);
PRIMALrescodee PRIMAL_getenv(PRIMALtask_t t, PRIMALenv_t *env);
PRIMALrescodee PRIMAL_commitchanges(PRIMALtask_t t);
PRIMALrescodee PRIMAL_resizetask(PRIMALtask_t t, int maxnumcon, int maxnumvar,
                                 int maxnumcone, PRIMALint64t maxnumanz,
                                 PRIMALint64t maxnumqnz);
PRIMALrescodee PRIMAL_updatesolutioninfo(PRIMALtask_t t, PRIMALsolt which);
PRIMALrescodee PRIMAL_deletesolution(PRIMALtask_t t, PRIMALsolt which);
PRIMALrescodee PRIMAL_deletetask(PRIMALtask_t *task);
PRIMALrescodee PRIMAL_deleteenv(PRIMALenv_t *env);

/* Environment/task memory helpers (reference MSK_callocenv/MSK_freeenv and their
 * task variants). This solver keeps no internal memory pool, so the debug
 * variants (callocdbg/freedbg) behave like the plain calloc/free and the memory
 * checks (checkmem) always answer OK: there is no pool to overrun or to inspect
 * (declared deviation). globalenvinitialize/finalize have no global state to set
 * up here. freeenv/freetask free a buffer allocated by these helpers, they are
 * NOT the environment/task destructors (those are deleteenv/deletetask). */
void *PRIMAL_callocenv(PRIMALenv_t env, size_t number, size_t size);
void *PRIMAL_callocdbgenv(PRIMALenv_t env, size_t number, size_t size,
                          const char *file, unsigned line);
void PRIMAL_freeenv(PRIMALenv_t env, void *buffer);
void PRIMAL_freedbgenv(PRIMALenv_t env, void *buffer, const char *file, unsigned line);
void *PRIMAL_calloctask(PRIMALtask_t task, size_t number, size_t size);
void *PRIMAL_callocdbgtask(PRIMALtask_t task, size_t number, size_t size,
                           const char *file, unsigned line);
void PRIMAL_freetask(PRIMALtask_t task, void *buffer);
void PRIMAL_freedbgtask(PRIMALtask_t task, void *buffer, const char *file, unsigned line);
PRIMALrescodee PRIMAL_globalenvinitialize(PRIMALint64t maxnumalloc, const char *dbgfile);
PRIMALrescodee PRIMAL_globalenvfinalize(void);
PRIMALrescodee PRIMAL_checkmemenv(PRIMALenv_t env, const char *file, int line);
PRIMALrescodee PRIMAL_checkmemtask(PRIMALtask_t task, const char *file, int line);
/* utilità informative (riferimento getversion/isinfinity/getresponseclass):
 * getversion riporta la versione di QUESTO solver; isinfinity usa la nostra
 * infinità (IEEE, `INF == INFINITY`); getresponseclass mappa un codice nella
 * classe del riferimento (OK 0, WRN 1, TRM 2, ERR 3, UNK 4). */
PRIMALrescodee PRIMAL_getversion(int *major, int *minor, int *revision);
int PRIMAL_isinfinity(PRIMALrealt value);
PRIMALrescodee PRIMAL_getresponseclass(PRIMALrescodee res, int *responseclass);
/* Tipo di problema, coi valori di MSKproblemtypee. Regola documentata nel .c:
 * QC+coni -> MIXED, QC -> QCQO, obiettivo quadratico -> QO, coni/barre -> CONIC,
 * altrimenti LO (le variabili intere non entrano nella classe). */
typedef enum {
    PRIMAL_PROBTYPE_LO    = 0,
    PRIMAL_PROBTYPE_QO    = 1,
    PRIMAL_PROBTYPE_QCQO  = 2,
    PRIMAL_PROBTYPE_CONIC = 3,
    PRIMAL_PROBTYPE_MIXED = 4
} PRIMALproblemtypee;
/* information items (riferimento MSKdinfiteme/MSKiinfiteme/MSKliinfiteme e
 * MSKinftypee). Gli indici sono quelli del riferimento 11.2.4 (letti da
 * constants.html il 2026-09-19); END e' il limite della tabella, non un item. */
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

PRIMALrescodee PRIMAL_getdouinf(PRIMALtask_t t, PRIMALdinfiteme which, PRIMALrealt *value);
PRIMALrescodee PRIMAL_getintinf(PRIMALtask_t t, PRIMALiinfiteme which, int *value);
PRIMALrescodee PRIMAL_getlintinf(PRIMALtask_t t, PRIMALliinfiteme which, PRIMALint64t *value);
PRIMALrescodee PRIMAL_getnadouinf(PRIMALtask_t t, const char *name, PRIMALrealt *value);
PRIMALrescodee PRIMAL_getnaintinf(PRIMALtask_t t, const char *name, int *value);
PRIMALrescodee PRIMAL_getinfindex(PRIMALtask_t t, PRIMALinftypee inftype, const char *name, int *index);
PRIMALrescodee PRIMAL_getinfname(PRIMALtask_t t, PRIMALinftypee inftype, int whichinf, char *name);
PRIMALrescodee PRIMAL_getinfmax(PRIMALtask_t t, PRIMALinftypee inftype, int *infmax);
#define PRIMAL_MAX_INFNAME_LEN 80
/* nome come stringa di un information item e di un callback code (riferimento
 * dinfitemtostr/iinfitemtostr/liinfitemtostr/callbackcodetostr): nessun task. */
PRIMALrescodee PRIMAL_dinfitemtostr(PRIMALdinfiteme item, char *str);
PRIMALrescodee PRIMAL_iinfitemtostr(PRIMALiinfiteme item, char *str);
PRIMALrescodee PRIMAL_liinfitemtostr(PRIMALliinfiteme item, char *str);
/* Symbolic constants (riferimento getsymbcondim/getsymbcon/symnamtovalue/
 * iparvaltosymnam). La tabella e' quella del riferimento 11.2.4, estratta dalla
 * sua stessa libreria (1438 voci, maxlen 60). Il nome piu' lungo sta in
 * PRIMAL_MAX_SYMBNAME_LEN caratteri (incluso il terminatore). */
#define PRIMAL_MAX_SYMBNAME_LEN 64
PRIMALrescodee PRIMAL_getsymbcondim(PRIMALenv_t env, int *num, size_t *maxlen);
PRIMALrescodee PRIMAL_getsymbcon(PRIMALtask_t t, int i, int sizevalue, char *name, int *value);
int PRIMAL_symnamtovalue(const char *name, char *value);
PRIMALrescodee PRIMAL_iparvaltosymnam(PRIMALenv_t env, int whichparam, int whichvalue,
                                      char *symbolicname);
PRIMALrescodee PRIMAL_getprobtype(PRIMALtask_t t, PRIMALproblemtypee *probtype);
/* versione/build/errore (riferimento checkversion/getbuildinfo/getcodedesc/
 * getlasterror) e la soglia di troncamento di A (get/putatruncatetol: memorizzata,
 * non applicata -- questo solver non tronca A). */
PRIMALrescodee PRIMAL_checkversion(PRIMALenv_t env, int major, int minor, int revision);
PRIMALrescodee PRIMAL_getbuildinfo(char *buildstate, char *builddate);
/* stima in byte dell'uso di memoria del task (riferimento getmemusagetask). */
PRIMALrescodee PRIMAL_getmemusagetask(PRIMALtask_t t, PRIMALint64t *meminuse, PRIMALint64t *maxmemuse);
PRIMALrescodee PRIMAL_getcodedesc(PRIMALrescodee code, char *symname, char *str);
PRIMALrescodee PRIMAL_getlasterror(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    int sizelastmsg, int *lastmsglen, char *lastmsg);
PRIMALrescodee PRIMAL_getlasterror64(PRIMALtask_t t, PRIMALrescodee *lastrescode,
    PRIMALint64t sizelastmsg, PRIMALint64t *lastmsglen, char *lastmsg);
/* riassunti su stream (stampano su stdout). */
PRIMALrescodee PRIMAL_solutionsummary(PRIMALtask_t t, int whichstream);
PRIMALrescodee PRIMAL_onesolutionsummary(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
PRIMALrescodee PRIMAL_optimizersummary(PRIMALtask_t t, int whichstream);
/* diagnostica su stream (stampano su stdout). */
PRIMALrescodee PRIMAL_analyzeproblem(PRIMALtask_t t, int whichstream);
PRIMALrescodee PRIMAL_analyzesolution(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
PRIMALrescodee PRIMAL_infeasibilityreport(PRIMALtask_t t, int whichstream, PRIMALsolt whichsol);
PRIMALrescodee PRIMAL_sensitivityreport(PRIMALtask_t t, int whichstream);
PRIMALrescodee PRIMAL_getatruncatetol(PRIMALtask_t t, PRIMALrealt *tolzero);
PRIMALrescodee PRIMAL_putatruncatetol(PRIMALtask_t t, PRIMALrealt tolzero);
/* Nomi simbolici (riferimento *tostr). Il TESTO e' quello di questo solver: la
 * forma esatta del riferimento non e' stata letta, quindi non e' inventata ma
 * nemmeno presa in prestito (deviazione dichiarata). Il buffer deve reggere
 * PRIMAL_MAX_STR_LEN caratteri. */
#define PRIMAL_MAX_STR_LEN 1024
PRIMALrescodee PRIMAL_prostatostr(PRIMALtask_t t, PRIMALprostae prosta, char *str);
PRIMALrescodee PRIMAL_solstatostr(PRIMALtask_t t, PRIMALsolstae solsta, char *str);
PRIMALrescodee PRIMAL_bktostr(PRIMALtask_t t, PRIMALboundkeye bk, char *str);
PRIMALrescodee PRIMAL_conetypetostr(PRIMALtask_t t, PRIMALconetypee ct, char *str);
/* inversi dei nomi simbolici (riferimento strtoconetype/strtosk). */
PRIMALrescodee PRIMAL_strtoconetype(PRIMALtask_t t, const char *str, PRIMALconetypee *ct);
PRIMALrescodee PRIMAL_probtypetostr(PRIMALtask_t t, PRIMALproblemtypee pt, char *str);
PRIMALrescodee PRIMAL_rescodetostr(PRIMALrescodee res, char *str);
PRIMALrescodee PRIMAL_appendvars(PRIMALtask_t t, int num);
PRIMALrescodee PRIMAL_appendcons(PRIMALtask_t t, int num);
PRIMALrescodee PRIMAL_getnumvar(PRIMALtask_t t, int *numvar);
PRIMALrescodee PRIMAL_getnumcon(PRIMALtask_t t, int *numcon);
/* Contatori "preallocati" del riferimento. Qui gli array crescono sul posto,
 * quindi il numero riservato E' quello corrente (deviazione dichiarata) tranne
 * per coni e barre, dove la capacita' e' reale. */
PRIMALrescodee PRIMAL_getmaxnumvar(PRIMALtask_t t, int *n);
PRIMALrescodee PRIMAL_getmaxnumcon(PRIMALtask_t t, int *n);
PRIMALrescodee PRIMAL_getmaxnumcone(PRIMALtask_t t, int *n);
PRIMALrescodee PRIMAL_getmaxnumbarvar(PRIMALtask_t t, int *n);

/* data input (column-wise linear part) */
PRIMALrescodee PRIMAL_putcj(PRIMALtask_t t, int j, PRIMALrealt cj);
PRIMALrescodee PRIMAL_putcfix(PRIMALtask_t t, PRIMALrealt cfix);
/* c come vettore: lista (putclist), fetta (putcslice) e letture (getc/getcslice).
 * La lista di scrittura valida tutto l'input prima di toccare c. */
PRIMALrescodee PRIMAL_putclist(PRIMALtask_t t, int num, const int *subj, const PRIMALrealt *val);
PRIMALrescodee PRIMAL_putcslice(PRIMALtask_t t, int first, int last, const PRIMALrealt *c);
PRIMALrescodee PRIMAL_getc(PRIMALtask_t t, PRIMALrealt *c);
PRIMALrescodee PRIMAL_getcslice(PRIMALtask_t t, int first, int last, PRIMALrealt *c);
PRIMALrescodee PRIMAL_putacol(PRIMALtask_t t, int j, int nz, const int *sub, const PRIMALrealt *val);
PRIMALrescodee PRIMAL_putarow(PRIMALtask_t t, int i, int nz, const int *sub, const PRIMALrealt *val);
/* Forme in blocco (riferimento putarowlist/putacollist e le slice): dati in
 * stile CSR, `asub[ptrb[k]..ptre[k])` per la k-esima riga/colonna; l'intera lista
 * e' validata prima di scrivere. */
PRIMALrescodee PRIMAL_putarowlist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putacollist(PRIMALtask_t t, int num, const int *sub,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putarowslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putacolslice(PRIMALtask_t t, int first, int last,
    const int *ptrb, const int *ptre, const int *asub, const PRIMALrealt *aval);
/* a_ij = aij: rimpiazza ogni entrata memorizzata della coppia (un solo termine
 * resta), e a 0 la rimuove. */
PRIMALrescodee PRIMAL_putaij(PRIMALtask_t t, int i, int j, PRIMALrealt aij);
/* lista di coefficienti scalari (riferimento putaijlist), validata prima di
 * scrivere. */
PRIMALrescodee PRIMAL_putaijlist(PRIMALtask_t t, int num, const int *subi,
                                 const int *subj, const PRIMALrealt *valij);
/* varianti a 64 bit (riferimento *64): gli stessi lettori/scrittori con i
 * puntatori di riga `PRIMALint64t`. */
PRIMALrescodee PRIMAL_putarowslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putacolslice64(PRIMALtask_t t, int first, int last,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putarowlist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putacollist64(PRIMALtask_t t, int num, const int *sub,
    const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *asub,
    const PRIMALrealt *aval);
PRIMALrescodee PRIMAL_putaijlist64(PRIMALtask_t t, PRIMALint64t num,
    const int *subi, const int *subj, const PRIMALrealt *valij);
PRIMALrescodee PRIMAL_putqobj(PRIMALtask_t t, int numqcnz, const int *qi, const int *qj, const PRIMALrealt *qoval);
/* carica la parte lineare in una chiamata (riferimento inputdata/inputdata64):
 * il task deve essere vuoto; A e' per colonna (aptrb[j]..aptre[j]). */
PRIMALrescodee PRIMAL_inputdata(PRIMALtask_t t, int maxnumcon, int maxnumvar,
    int numcon, int numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux);
PRIMALrescodee PRIMAL_inputdata64(PRIMALtask_t t, PRIMALint64t maxnumcon, PRIMALint64t maxnumvar,
    PRIMALint64t numcon, PRIMALint64t numvar, const PRIMALrealt *c, PRIMALrealt cfix,
    const int *aptrb, const int *aptre, const int *asub, const PRIMALrealt *aval,
    const PRIMALboundkeye *bkc, const PRIMALrealt *blc, const PRIMALrealt *buc,
    const PRIMALboundkeye *bkx, const PRIMALrealt *blx, const PRIMALrealt *bux);
/* q_ij = q_ji = qoij, solo triangolo inferiore (i >= j); rimpiazza la coppia. */
PRIMALrescodee PRIMAL_putqobjij(PRIMALtask_t t, int i, int j, PRIMALrealt qoij);
PRIMALrescodee PRIMAL_putvarbound(PRIMALtask_t t, int j, PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
PRIMALrescodee PRIMAL_putconbound(PRIMALtask_t t, int i, PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
/* cambiano UN lato del bound (riferimento chgvarbound/chgconbound): lower!=0 ->
 * new lower = (finite ? value : -inf); altrimenti new upper = (finite ? value
 * : +inf); il bound key viene ricalcolato. */
PRIMALrescodee PRIMAL_chgvarbound(PRIMALtask_t t, int j, int lower, int finite, PRIMALrealt value);
PRIMALrescodee PRIMAL_chgconbound(PRIMALtask_t t, int i, int lower, int finite, PRIMALrealt value);
/* Bound slices: [first, last), the buffer holds last-first entries. Reading
 * refuses before writing anything; writing validates the whole slice first, so a
 * refusal leaves the model untouched. */
PRIMALrescodee PRIMAL_getvarboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
PRIMALrescodee PRIMAL_getconboundslice(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
PRIMALrescodee PRIMAL_putvarboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
PRIMALrescodee PRIMAL_putconboundslice(PRIMALtask_t t, int first, int last,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
/* Liste di bound (riferimento putvarboundlist/putconboundlist, validate tutte
 * prima di applicare) e fetta a bound costante (put*boundsliceconst). */
PRIMALrescodee PRIMAL_putvarboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
PRIMALrescodee PRIMAL_putconboundlist(PRIMALtask_t t, int num, const int *sub,
    const PRIMALboundkeye *bk, const PRIMALrealt *bl, const PRIMALrealt *bu);
PRIMALrescodee PRIMAL_putvarboundsliceconst(PRIMALtask_t t, int first, int last,
    PRIMALboundkeye bk, PRIMALrealt bl, PRIMALrealt bu);
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
/* Rimpiazza TUTTI i termini quadratici di TUTTI i vincoli da una lista di
 * triplette con indice di riga (qcsubk, qcsubi, qcsubj, qcval), solo triangolo
 * inferiore; lista vuota azzera. L'intera lista e' validata prima di applicare. */
PRIMALrescodee PRIMAL_putqcon(PRIMALtask_t t, int numqcnz,
                         const int *qcsubk, const int *qcsubi, const int *qcsubj,
                         const PRIMALrealt *qcval);
PRIMALrescodee PRIMAL_getnumqconknz(PRIMALtask_t t, int k, int *numqcnz);
PRIMALrescodee PRIMAL_getqconkij(PRIMALtask_t t, int k, int i, int j, PRIMALrealt *qij);
PRIMALrescodee PRIMAL_putobjsense(PRIMALtask_t t, PRIMALobjsensee sense);
PRIMALrescodee PRIMAL_putvartype(PRIMALtask_t t, int j, PRIMALvariabletypee vt);
PRIMALrescodee PRIMAL_getvartype(PRIMALtask_t t, int j, PRIMALvariabletypee *vt);
/* Tipo per una lista di variabili (riferimento putvartypelist/getvartypelist):
 * la scrittura valida indici e tipi dell'intera lista prima di applicare. */
PRIMALrescodee PRIMAL_putvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, const PRIMALvariabletypee *vartype);
PRIMALrescodee PRIMAL_getvartypelist(PRIMALtask_t t, int num,
                                     const int *subj, PRIMALvariabletypee *vartype);
PRIMALrescodee PRIMAL_getnumintvar(PRIMALtask_t t, int *num);
PRIMALrescodee PRIMAL_putcfix(PRIMALtask_t t, PRIMALrealt cfix);
PRIMALrescodee PRIMAL_putintparam(PRIMALtask_t t, int param, int value);
PRIMALrescodee PRIMAL_putdouparam(PRIMALtask_t t, int param, PRIMALrealt value);
/* riportano un parametro (o tutti) al default della tabella dichiarativa
 * (riferimento resetintparam/resetdouparam/resetparameters). */
PRIMALrescodee PRIMAL_resetintparam(PRIMALtask_t t, int param);
PRIMALrescodee PRIMAL_resetdouparam(PRIMALtask_t t, int param);
PRIMALrescodee PRIMAL_resetparameters(PRIMALtask_t t);

/* data getters (needed for independent verification, e.g. KKT checks) */
PRIMALrescodee PRIMAL_getcj(PRIMALtask_t t, int j, PRIMALrealt *cj);
PRIMALrescodee PRIMAL_getaij(PRIMALtask_t t, int i, int j, PRIMALrealt *aij);
PRIMALrescodee PRIMAL_getqobjij(PRIMALtask_t t, int i, int j, PRIMALrealt *qij);
PRIMALrescodee PRIMAL_getvarbound(PRIMALtask_t t, int j, PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
PRIMALrescodee PRIMAL_getconbound(PRIMALtask_t t, int i, PRIMALboundkeye *bk, PRIMALrealt *bl, PRIMALrealt *bu);
PRIMALrescodee PRIMAL_getobjsense(PRIMALtask_t t, PRIMALobjsensee *sense);
PRIMALrescodee PRIMAL_getcfix(PRIMALtask_t t, PRIMALrealt *cfix);
PRIMALrescodee PRIMAL_getintparam(PRIMALtask_t t, int param, int *value);
PRIMALrescodee PRIMAL_getdouparam(PRIMALtask_t t, int param, PRIMALrealt *value);

/* ---- accesso ai dati del modello (superficie del riferimento) ----
 * Due contratti DISTINTI, e la linea sta fra il negozio e l'operatore.
 * PRIMAL_getnumanz conta le ENTRATE DI A MEMORIZZATE (somma delle nz delle
 * colonne), percio' un coefficiente scritto due volte conta due volte: la
 * memorizzazione e' colonnare e nessuna deduplica avviene in putarow/putacol;
 * PRIMAL_getarow/PRIMAL_getacol restituiscono quel negozio per come e' scritto
 * (entrambe le entrate). PRIMAL_getaij risponde invece dell'OPERATORE, cioe'
 * della SOMMA delle entrate di quell'(i,j): e' il coefficiente che ogni strada
 * risolve, e sarebbe una contraddizione dire il primo mentre la risposta e'
 * della somma. Stessa linea gia' tracciata da PRIMAL_getqobjij (somma) contro
 * PRIMAL_getqobj + PRIMAL_getnumqobjnz (il negozio di triplette).
 * PRIMAL_getmaxnumanz e' la capacita' gia' allocata sulle stesse colonne:
 * l'invariante e' numanzs <= maxnumanzs dopo ogni scrittura riuscita.
 * PRIMAL_getnumqobjnz conta le triplette di Q memorizzate da putqobj (che
 * possono essere di piu' dei coefficienti non nulli di Q, per lo stesso
 * motivo). */
PRIMALrescodee PRIMAL_getnumanz(PRIMALtask_t t, int *numanzs);
PRIMALrescodee PRIMAL_getmaxnumanz(PRIMALtask_t t, int *maxnumanzs);
PRIMALrescodee PRIMAL_getnumqobjnz(PRIMALtask_t t, int *numqobjnz);
/* varianti a 64 bit dei contatori (riferimento getnumanz64/...): gli stessi
 * numeri, allargati. */
PRIMALrescodee PRIMAL_getnumanz64(PRIMALtask_t t, PRIMALint64t *numanzs);
PRIMALrescodee PRIMAL_getmaxnumanz64(PRIMALtask_t t, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getnumqobjnz64(PRIMALtask_t t, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getnumqconknz64(PRIMALtask_t t, int k, PRIMALint64t *n);
/* Conteggi del negozio di A (riferimento getarownumnz/getacolnumnz e le loro
 * fette) e A in triplette (getatrip): contano le ENTRATE memorizzate, come
 * getnumanz, duplicati inclusi. `getatrip` non ha un conteggio in uscita: chi
 * chiama dimensiona con getnumanz, e una capienza insufficiente e' rifiutata
 * senza scrivere. */
PRIMALrescodee PRIMAL_getacolnumnz(PRIMALtask_t t, int j, int *nzj);
PRIMALrescodee PRIMAL_getarownumnz(PRIMALtask_t t, int i, int *nzi);
PRIMALrescodee PRIMAL_getacolslicenumnz(PRIMALtask_t t, int first, int last, int *numnz);
PRIMALrescodee PRIMAL_getacolslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz);
PRIMALrescodee PRIMAL_getarowslicenumnz64(PRIMALtask_t t, int first, int last, PRIMALint64t *numnz);
PRIMALrescodee PRIMAL_getarowslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val);
PRIMALrescodee PRIMAL_getacolslice64(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, PRIMALint64t *ptrb, PRIMALint64t *ptre,
        int *sub, PRIMALrealt *val);
PRIMALrescodee PRIMAL_getarowslicenumnz(PRIMALtask_t t, int first, int last, int *numnz);
PRIMALrescodee PRIMAL_getatrip(PRIMALtask_t t, PRIMALint64t maxnumnz,
                               int *subi, int *subj, PRIMALrealt *val);
/* Lettura di una riga/colonna di A in buffer di dimensione maxnum: numret dice
 * quante entrate sono state scritte. */
PRIMALrescodee PRIMAL_getarow(PRIMALtask_t t, int i, int *sub, PRIMALrealt *val,
                        int maxnum, int *numret);
PRIMALrescodee PRIMAL_getacol(PRIMALtask_t t, int j, int *sub, PRIMALrealt *val,
                        int maxnum, int *numret);
/* A su una fetta di righe/colonne in triplette (getarowslicetrip/getacolslicetrip):
 * contano prima e rifiutano senza scrivere se `maxnumnz` non basta. */
PRIMALrescodee PRIMAL_getarowslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val);
PRIMALrescodee PRIMAL_getacolslicetrip(PRIMALtask_t t, int first, int last,
        PRIMALint64t maxnumnz, int *subi, int *subj, PRIMALrealt *val);
/* Variante a fetta (get*slice): sono le entrate della riga i (della colonna j)
 * il cui indice cade in [first,last), scritte a partire dalla posizione
 * `offset` dei buffer sub/val, che hanno capienza maxnum. Offset e' la coda di
 * una fetta precedente, percio' maxnum-offset e' lo spazio rimasto.
 * Se lo spazio non basta la chiamata e' RIFIUTATA (PRIMAL_RES_ERR_ARG) e i
 * buffer dell'utente restano intonsi, perche' una scrittura parziale da
 * disfare non e' recuperabile da chi chiama: e' la stessa regola dei setter
 * dei parametri (fuori-range = niente effetto).
 * Ordinamento: per indice crescente (colonne per una riga, righe per una
 * colonna). Un'entrata scritta due volte e' restituita due volte, perche'
 * nessuna deduplica avviene in putarow/putacol. */
PRIMALrescodee PRIMAL_getarowslice(PRIMALtask_t t, int i, int first, int last,
                             int offset, int maxnum, int *numret,
                             int *sub, PRIMALrealt *val);
PRIMALrescodee PRIMAL_getacolslice(PRIMALtask_t t, int j, int first, int last,
                             int offset, int maxnum, int *numret,
                             int *sub, PRIMALrealt *val);
/* Lettura INTERA delle due Q, sulla stessa forma di getarow: maxnum e' la
 * capienza dei buffer e *numret quante triplette sono state scritte. Se maxnum
 * non basta la chiamata e' RIFIUTATA e nessun buffer viene scritto, *numret
 * compreso (la stessa regola delle slice di A).
 * PRIMAL_getqobj legge il listato di triplette COSI' COME E' STATO SCRITTO, in
 * ordine di scrittura: e' la tabella che PRIMAL_getnumqobjnz conta, quindi un
 * valore 0.0 scritto e' restituito e contato (il numero conta le scritture
 * dell'utente, non i non nulli dell'operatore), e un termine incrociato
 * (i,j) con i != j compare UNA volta sola anche se getqobjij risponde lo stesso
 * numero su entrambe le meta': il negozio non e' simmetrizzato, l'operatore si'.
 * PRIMAL_getqconk legge il TRIANGOLO SUPERIORE (i <= j) in ordine crescente,
 * e il conteggio viene da PRIMAL_getnumqconknz stesso, non da un secondo
 * elenco: due enumerazioni della stessa tabella sono due politiche. Una riga
 * le cui entrate si annullano (putqconk accumula) misura zero e scrive zero:
 * verdetto vuoto, non rifiuto.
 * Deviazione dichiarata: la forma che il riferimento usa per queste due letture
 * NON e' stata letta (fetch dei documenti bloccato anche in questo giro); la
 * forma e' quella del nostro stesso contatore, perche' conteggio e lettore
 * rispondono della stessa tabella e non possono divergere. */
PRIMALrescodee PRIMAL_getqobj(PRIMALtask_t t, int *qi, int *qj, PRIMALrealt *qval,
                        int maxnum, int *numret);
PRIMALrescodee PRIMAL_getqconk(PRIMALtask_t t, int k, int *qi, int *qj, PRIMALrealt *qval,
                         int maxnum, int *numret);
PRIMALrescodee PRIMAL_getqobj64(PRIMALtask_t t, int *qi, int *qj, PRIMALrealt *qval,
                        PRIMALint64t maxnum, PRIMALint64t *numret);
PRIMALrescodee PRIMAL_getqconk64(PRIMALtask_t t, int k, int *qi, int *qj, PRIMALrealt *qval,
                         PRIMALint64t maxnum, PRIMALint64t *numret);

/* ---- nomi di variabili e vincoli ----
 * Due tabelle indipendenti, come nel riferimento: in un MPS un vincolo e una
 * variabile possono chiamarsi allo stesso modo, quindi la stessa stringa puo'
 * nominare uno dell'una e uno dell'altra tabella. Ogni entrata possiede la
 * propria stringa; un ente senza nome legge il nome come stringa VUOTA (""),
 * e "" non puo' essere il nome di nessun ente (mettere "" a un nome serve a
 * toglierlo).
 * I nomi sono univoci dentro la propria tabella: un nome gia' in uso e'
 * RIFIUTATO (PRIMAL_RES_ERR_ARG) senza toccare il nome precedente, perche' una
 * tabella con due indici per lo stesso nome renderebbe getvarname una domanda
 * con due risposte. Rinominare un indice con il nome che ha gia' non e' un
 * conflitto (idempotente).
 * getvarnameidx restituisce un puntatore PRESTATO che vive finche' vive il
 * task: appendvars/appendcons muovono la tabella dei puntatori, non le
 * stringhe, quindi un `const char *` ottenuto prima di un append resta valido.
 * I getter non trovati rispondono ERR_ARG e NON toccano l'indice restituito.
 * Deviazione: qui lo statuto di "non trovato" e' lo stesso ERR_ARG con cui
 * ogni getter di questo API dice "no" (vedi i raggi di Farkas, T85). Il
 * riferimento avrebbe un codice dedicato per il nome inesistente: NON letto
 * in questo giro (documentazione irraggiungibile), quindi e' memoria e non
 * fonte, e la scelta sta sulla convenzione nostra misurata, non su quel nome. */
PRIMALrescodee PRIMAL_putvarname(PRIMALtask_t t, int j, const char *name);
PRIMALrescodee PRIMAL_putconname(PRIMALtask_t t, int i, const char *name);
PRIMALrescodee PRIMAL_getvarnameidx(PRIMALtask_t t, int j, const char **name);
PRIMALrescodee PRIMAL_getconnameidx(PRIMALtask_t t, int i, const char **name);
/* nome -> indice (le due domande inverse di una stessa tabella) */
PRIMALrescodee PRIMAL_getvarname(PRIMALtask_t t, const char *name, int *j);
PRIMALrescodee PRIMAL_getconname(PRIMALtask_t t, const char *name, int *i);
/* Forma del riferimento per la ricerca per nome: stesso servizio, altro nome. */
PRIMALrescodee PRIMAL_getidxvar(PRIMALtask_t t, const char *vname, int *var);
PRIMALrescodee PRIMAL_getidxcon(PRIMALtask_t t, const char *cname, int *con);
/* L'intera tabella in una chiamata. Il buffer dell'utente ha numvar (numcon)
 * elementi -- li legge da PRIMAL_getnumvar / PRIMAL_getnumcon, la stessa
 * convenzione di getxx -- e la scrittura e' esattamente quella misura, perche'
 * non c'e' un conteggio da negoziare. Un ente senza nome esce come la stringa
 * VUOTA, identicamente a getvarnameidx: due letture della stessa tabella che
 * non concordano non sono due formati, sono due regole.
 * Deviazione: il riferimento consegna char** dove LUI copia, in buffer di
 * lunghezza fissa di proprieta' di chi chiama; qui escono i puntatori
 * PRESTATI delle stringhe del task, che vivono finche' vive il task e non
 * vanno liberati. */
PRIMALrescodee PRIMAL_getallvarname(PRIMALtask_t t, const char **names);
PRIMALrescodee PRIMAL_getallconname(PRIMALtask_t t, const char **names);

/* ---- nomi delle variabili di barra: la terza tabella ----
 * Il contratto e' quello delle due tabelle scalari, elemento per elemento:
 * univocita' DENTRO la tabella, "" che toglie il nome (lettura ""), rifiuto che
 * non tocca il nome precedente, puntatore PRESTATO che vive finche' vive il
 * task, e getallbarname che scrive esattamente PRIMAL_getnumbarvar elementi.
 * La novita' non e' una regola, e' lo spazio degli indici: una variabile di
 * barra non e' una delle numvar variabili scalari (ha dimensione propria e il
 * proprio blocco di cono), quindi il suo nome vive in un namespace INDIPENDENTE
 * -- la stessa stringa puo' nominare una barra e una scalare, come gia' puo'
 * nominare un vincolo e una variabile, e un duplicato e' rifiutato solo
 * dentro la tabella di chi parla.
 * Deviazione dichiarata: la regola che il riferimento usa per i nomi di barra
 * NON e' stata letta in questo giro (fetch dei documenti bloccato); questa e'
 * l'estensione coerente della regola nostra misurata in T102, non una copia. */
PRIMALrescodee PRIMAL_putbarname(PRIMALtask_t t, int j, const char *name);
PRIMALrescodee PRIMAL_getbarnameidx(PRIMALtask_t t, int j, const char **name);
/* nome -> indice della barra */
PRIMALrescodee PRIMAL_getbarname(PRIMALtask_t t, const char *name, int *j);
/* Forma del riferimento per la ricerca per nome (stesso servizio, altro nome). */
PRIMALrescodee PRIMAL_getidxbarvar(PRIMALtask_t t, const char *bname, int *bar);
PRIMALrescodee PRIMAL_getallbarname(PRIMALtask_t t, const char **names);

/* ---- nomi dei blocchi di cono: la quarta tabella ----
 * Contratto identico alle tre tabelle misurate in T102/T103/T105: univocita'
 * DENTRO la tabella, "" che toglie il nome (lettura ""), rifiuto che lascia
 * vivo il nome precedente, puntatore PRESTATO per la vita del task, e
 * getallconename che scrive esattamente PRIMAL_getnumcone elementi. La
 * capacita' della tabella e' la stessa cone_cap dei quattro array dei coni,
 * perche' due capacita' da tenere in pari sono due politiche.
 * Namespace INDIPENDENTE, per lo stesso motivo della barra: un cono non e' una
 * variabile scalare, non e' un vincolo, non e' una barra. Attenzione alla
 * coppia confondibile: getconname e' il VINCOLO, getconename e' il CONO, una
 * lettera di differenza — la confusione non e' chiusa da un commento ma da
 * T106, che nomina vincolo e cono con la STESSA stringa nello stesso task e
 * asserisce che le due ricerche rispondono i propri indici.
 * Deviazione dichiarata: la regola che il riferimento usa per nominare un cono
 * NON e' stata letta in questo giro (fetch bloccato) e non e' stata inventata.
 * In particolare qui non esiste una superficie "nome di funzione" con un
 * (tipo, indice): la numerazione di MSKfunctiontypee non e' conosciuta, e
 * darle numeri nostri sarebbe il difetto che T93/T94 hanno corretto. */
PRIMALrescodee PRIMAL_putconename(PRIMALtask_t t, int k, const char *name);
PRIMALrescodee PRIMAL_getconenameidx(PRIMALtask_t t, int k, const char **name);
/* nome -> indice del blocco di cono */
PRIMALrescodee PRIMAL_getconename(PRIMALtask_t t, const char *name, int *k);
/* Forma del riferimento per la ricerca per nome (stesso servizio, altro nome). */
PRIMALrescodee PRIMAL_getidxcone(PRIMALtask_t t, const char *cname, int *cone);
PRIMALrescodee PRIMAL_getallconename(PRIMALtask_t t, const char **names);
/* ricerche per nome del riferimento (deleghe con l'assegnazione fissa = 0) e
 * lunghezza del nome di un blocco conico. */
PRIMALrescodee PRIMAL_getvarnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index);
PRIMALrescodee PRIMAL_getconnameindex(PRIMALtask_t t, const char *somename,
                                      int *asgn, int *index);
PRIMALrescodee PRIMAL_getconenameindex(PRIMALtask_t t, const char *somename,
                                       int *asgn, int *index);
PRIMALrescodee PRIMAL_getconenamelen(PRIMALtask_t t, int i, int *len);
/* nonnulli di Q memorizzati (obiettivo + vincoli) e c[subj[k]]; suggerimenti di
 * capacita' (no-op). */
PRIMALrescodee PRIMAL_getmaxnumqnz(PRIMALtask_t t, int *maxnumqnz);
PRIMALrescodee PRIMAL_getmaxnumqnz64(PRIMALtask_t t, PRIMALint64t *maxnumqnz);
PRIMALrescodee PRIMAL_getclist(PRIMALtask_t t, int num, const int *subj, PRIMALrealt *c);
PRIMALrescodee PRIMAL_putmaxnumvar(PRIMALtask_t t, int maxnumvar);
PRIMALrescodee PRIMAL_putmaxnumcon(PRIMALtask_t t, int maxnumcon);
PRIMALrescodee PRIMAL_putmaxnumcone(PRIMALtask_t t, int maxnumcone);
PRIMALrescodee PRIMAL_putmaxnumanz(PRIMALtask_t t, PRIMALint64t maxnumanz);
PRIMALrescodee PRIMAL_putmaxnumqnz(PRIMALtask_t t, PRIMALint64t maxnumqnz);

/* ---- il nome dell'obiettivo: un solo posto, non una tabella ----
 * Stessa regola di proprieta' (puntatore prestato, "" toglie, NULL rifiutato),
 * ma con n = 1 la domanda "dupplicato" non si pone: un secondo nome SOSTITUISCE
 * il primo, che e' l'unica differenza misurabile dalle tabelle.
 * Deviazione: il riferimento copia in un buffer di lunghezza fissa di proprieta'
 * di chi chiama; qui esce il puntatore PRESTATO, come per le quattro tabelle. */
PRIMALrescodee PRIMAL_putobjname(PRIMALtask_t t, const char *name);
PRIMALrescodee PRIMAL_getobjname(PRIMALtask_t t, const char **name);
/* Nome del task e lunghezze dei nomi (riferimento puttaskname/gettaskname/
 * gettasknamelen, getvarnamelen/getconnamelen/getobjnamelen, getmaxnamelen).
 * La lunghezza NON conta il terminatore; un oggetto senza nome ha lunghezza 0.
 * gettaskname copia nel buffer di chi chiama, che deve contenere anche lo zero. */
PRIMALrescodee PRIMAL_puttaskname(PRIMALtask_t t, const char *name);
PRIMALrescodee PRIMAL_gettaskname(PRIMALtask_t t, int sizetaskname, char *taskname);
PRIMALrescodee PRIMAL_gettasknamelen(PRIMALtask_t t, int *len);
PRIMALrescodee PRIMAL_getvarnamelen(PRIMALtask_t t, int j, int *len);
PRIMALrescodee PRIMAL_getconnamelen(PRIMALtask_t t, int i, int *len);
PRIMALrescodee PRIMAL_getobjnamelen(PRIMALtask_t t, int *len);
PRIMALrescodee PRIMAL_getmaxnamelen(PRIMALtask_t t, int *maxlen);

/* conic optimization (SOCP); members are variable indices */
/* Espressioni affini (AFE): f_i = sum_j F_ij x_j + g_i. Lo storage su cui
 * costruiscono i vincoli conici affini (ACC) e disgiuntivi (DJC). */
PRIMALrescodee PRIMAL_appendafes(PRIMALtask_t t, PRIMALint64t num);
PRIMALrescodee PRIMAL_getnumafe(PRIMALtask_t t, PRIMALint64t *numafe);
PRIMALrescodee PRIMAL_putafefentry(PRIMALtask_t t, PRIMALint64t i, int j, PRIMALrealt v);
PRIMALrescodee PRIMAL_putafefrow(PRIMALtask_t t, PRIMALint64t i, int numnz,
                                 const int *varidx, const PRIMALrealt *val);
PRIMALrescodee PRIMAL_putafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt g);
PRIMALrescodee PRIMAL_getafeg(PRIMALtask_t t, PRIMALint64t i, PRIMALrealt *g);
PRIMALrescodee PRIMAL_getafefrownumnz(PRIMALtask_t t, PRIMALint64t i, int *numnz);
PRIMALrescodee PRIMAL_getafefrow(PRIMALtask_t t, PRIMALint64t i, int *numnz,
                                 int *varidx, PRIMALrealt *val);
/* superficie in blocco degli AFE (riferimento emptyafefrow/emptyafefcol,
 * putafeglist/putafegslice/getafegslice, putafefentrylist, getafeftrip,
 * getafefnumnz). Un buffer NULL non viene scritto; `getafeftrip` enumera F in
 * (afeidx,varidx,val) nell'ordine memorizzato e la lunghezza e' la somma dei
 * getafefnumnz. */
PRIMALrescodee PRIMAL_emptyafefrow(PRIMALtask_t t, PRIMALint64t afeidx);
PRIMALrescodee PRIMAL_emptyafefcol(PRIMALtask_t t, int varidx);
PRIMALrescodee PRIMAL_putafeglist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                  const PRIMALint64t *afeidx, const PRIMALrealt *g);
PRIMALrescodee PRIMAL_putafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, const PRIMALrealt *slice);
PRIMALrescodee PRIMAL_getafegslice(PRIMALtask_t t, PRIMALint64t first,
                                   PRIMALint64t last, PRIMALrealt *g);
PRIMALrescodee PRIMAL_putafefentrylist(PRIMALtask_t t, PRIMALint64t numentr,
                                       const PRIMALint64t *afeidx,
                                       const int *varidx, const PRIMALrealt *val);
PRIMALrescodee PRIMAL_getafeftrip(PRIMALtask_t t, PRIMALint64t *afeidx,
                                  int *varidx, PRIMALrealt *val);
PRIMALrescodee PRIMAL_getafefnumnz(PRIMALtask_t t, PRIMALint64t afeidx, int *numnz);
PRIMALrescodee PRIMAL_emptyafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                       const PRIMALint64t *afeidx);
PRIMALrescodee PRIMAL_emptyafefcollist(PRIMALtask_t t, PRIMALint64t numvaridx,
                                       const int *varidx);
PRIMALrescodee PRIMAL_putafefcol(PRIMALtask_t t, int varidx, PRIMALint64t numnz,
                                 const PRIMALint64t *afeidx, const PRIMALrealt *val);
/* termini bar di un AFE (riferimento putafebarfentry e famiglia): Fbar[i][j] e'
 * una combinazione pesata di matrici simmetriche, e <Fbar_ij, X_j> entra nella
 * i-esima espressione. La scrittura di (i,j) rimpiazza i termini con lo stesso
 * barvaridx. */
PRIMALrescodee PRIMAL_putafebarfentry(PRIMALtask_t t, PRIMALint64t afeidx, int barvaridx,
        PRIMALint64t numterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight);
PRIMALrescodee PRIMAL_emptyafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx);
PRIMALrescodee PRIMAL_emptyafebarfrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
                                          const PRIMALint64t *afeidxlist);
PRIMALrescodee PRIMAL_getafebarfnumrowentries(PRIMALtask_t t, PRIMALint64t afeidx, int *numentr);
PRIMALrescodee PRIMAL_getafebarfrowinfo(PRIMALtask_t t, PRIMALint64t afeidx,
                                        int *numentr, PRIMALint64t *numterm);
PRIMALrescodee PRIMAL_getafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int *barvaridx,
        PRIMALint64t *ptrterm, PRIMALint64t *numterm, PRIMALint64t *termidx,
        PRIMALrealt *termweight);
PRIMALrescodee PRIMAL_getafebarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip);
PRIMALrescodee PRIMAL_getafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *afeidx, int *barvaridx, int *subk,
        int *subl, PRIMALrealt *valkl);
PRIMALrescodee PRIMAL_putafebarfentrylist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *barvaridx, const PRIMALint64t *numterm,
        const PRIMALint64t *ptrterm, PRIMALint64t lenterm, const PRIMALint64t *termidx,
        const PRIMALrealt *termweight);
PRIMALrescodee PRIMAL_putafebarfrow(PRIMALtask_t t, PRIMALint64t afeidx, int numentr,
        const int *barvaridx, const PRIMALint64t *numterm, const PRIMALint64t *ptrterm,
        PRIMALint64t lenterm, const PRIMALint64t *termidx, const PRIMALrealt *termweight);
PRIMALrescodee PRIMAL_putafebarfblocktriplet(PRIMALtask_t t, PRIMALint64t numtrip,
        const PRIMALint64t *afeidx, const int *barvaridx, const int *subk,
        const int *subl, const PRIMALrealt *valkl);
PRIMALrescodee PRIMAL_getaccbarfnumblocktriplets(PRIMALtask_t t, PRIMALint64t *numtrip);
PRIMALrescodee PRIMAL_getaccbarfblocktriplet(PRIMALtask_t t, PRIMALint64t maxnumtrip,
        PRIMALint64t *numtrip, PRIMALint64t *acc_afe, int *bar_var, int *blk_row,
        int *blk_col, PRIMALrealt *blk_val);
/* Domini conici (per i vincoli conici affini, ACC); i valori sono quelli di
 * MSKdomaintypee. */
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

PRIMALrescodee PRIMAL_appendrdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendrzerodomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendrplusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendrminusdomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendrquadraticconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendprimalexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appenddualexpconedomain(PRIMALtask_t t, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendprimalpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appenddualpowerconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALrealt alpha, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appendsvecpsdconedomain(PRIMALtask_t t, int dim, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_getnumdomain(PRIMALtask_t t, PRIMALint64t *numdomain);
PRIMALrescodee PRIMAL_getdomaintype(PRIMALtask_t t, PRIMALint64t domidx, PRIMALdomaintypee *domtype);
PRIMALrescodee PRIMAL_getdomainn(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t *n);
/* coni di media geometrica, nomi dei domini (settima tabella), info sul cono di
 * potenza e il suggerimento di capacita' putmaxnumdomain. */
PRIMALrescodee PRIMAL_appendprimalgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_appenddualgeomeanconedomain(PRIMALtask_t t, PRIMALint64t n, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_putdomainname(PRIMALtask_t t, PRIMALint64t domidx, const char *name);
PRIMALrescodee PRIMAL_getdomainnamelen(PRIMALtask_t t, PRIMALint64t domidx, int *len);
PRIMALrescodee PRIMAL_getdomainname(PRIMALtask_t t, PRIMALint64t domidx, int sizename, char *name);
PRIMALrescodee PRIMAL_getpowerdomainalpha(PRIMALtask_t t, PRIMALint64t domidx, PRIMALrealt *alpha);
PRIMALrescodee PRIMAL_getpowerdomaininfo(PRIMALtask_t t, PRIMALint64t domidx,
                                         PRIMALint64t *n, PRIMALint64t *nleft);
PRIMALrescodee PRIMAL_putmaxnumdomain(PRIMALtask_t t, PRIMALint64t maxnumdomain);
PRIMALrescodee PRIMAL_appendcone(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt coneparam, int nummem, const int *submem);
/* cono/i i cui membri sono variabili CONTIGUE j..j+nummem-1 (riferimento
 * appendconeseq/appendconesseq). */
PRIMALrescodee PRIMAL_appendconeseq(PRIMALtask_t t, PRIMALconetypee ct, PRIMALrealt conepar,
                                    int nummem, int j);
PRIMALrescodee PRIMAL_appendconesseq(PRIMALtask_t t, int num, const PRIMALconetypee *ct,
    const PRIMALrealt *conepar, const int *nummem, const int *j);
/* rimuove i coni agli indici dati (riferimento removecones). */
PRIMALrescodee PRIMAL_removecones(PRIMALtask_t t, int num, const int *subset);
/* rimuove i vincoli agli indici dati, compattando A/qcon/barA (riferimento
 * removecons). */
PRIMALrescodee PRIMAL_removecons(PRIMALtask_t t, int num, const int *subset);
/* rimuove le variabili agli indici dati, rimodellando qcon e rimappando qobj e i
 * membri di cono (riferimento removevars). */
PRIMALrescodee PRIMAL_removevars(PRIMALtask_t t, int num, const int *subset);
PRIMALrescodee PRIMAL_getnumcone(PRIMALtask_t t, int *numcone);
PRIMALrescodee PRIMAL_getcone(PRIMALtask_t t, int k, PRIMALconetypee *ct, int *nummem, int *submem);
PRIMALrescodee PRIMAL_getconeparam(PRIMALtask_t t, int k, PRIMALrealt *param);
PRIMALrescodee PRIMAL_getnumconemem(PRIMALtask_t t, int k, int *nummem);
/* ---- vincoli conici affini (ACC) ----
 * appendacc(domidx, numafeidx, afeidxlist, b): il vettore delle numafeidx
 * espressioni affini afeidxlist[e] (F_e x + g_e) piu' le costanti b[e]
 * appartiene al dominio domidx (dimensione numafeidx == getdomainn(domidx)). */
PRIMALrescodee PRIMAL_appendacc(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t numafeidx,
                                const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_getnumacc(PRIMALtask_t t, PRIMALint64t *numacc);
PRIMALrescodee PRIMAL_getaccn(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getaccdomain(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *domidx);
PRIMALrescodee PRIMAL_getaccafeidxlist(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t *afeidxlist);
PRIMALrescodee PRIMAL_getaccb(PRIMALtask_t t, PRIMALint64t accidx, PRIMALrealt *b);
/* superficie ACC in blocco e nomi (riferimento appendaccs/getaccs/getaccntot/
 * putaccb/putaccname/getaccname/getaccnamelen). `appendaccs` appende numaccs
 * ACC, ciascuno di dimensione dom_n[domidxs[i]], consumando afeidxlist/b in
 * sequenza; `getaccs` e' la concatenazione delle liste per-ACC. */
PRIMALrescodee PRIMAL_appendaccs(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_getaccntot(PRIMALtask_t t, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getaccs(PRIMALtask_t t, PRIMALint64t *domidxlist,
                              PRIMALint64t *afeidxlist, PRIMALrealt *b);
PRIMALrescodee PRIMAL_putaccb(PRIMALtask_t t, PRIMALint64t accidx,
                              PRIMALint64t lengthb, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_putaccname(PRIMALtask_t t, PRIMALint64t accidx, const char *name);
PRIMALrescodee PRIMAL_getaccnamelen(PRIMALtask_t t, PRIMALint64t accidx, int *len);
PRIMALrescodee PRIMAL_getaccname(PRIMALtask_t t, PRIMALint64t accidx,
                                 int sizename, char *name);
/* ACC con AFE contigui (appendaccseq/appendaccsseq), attivita' al punto
 * (evaluateacc/evaluateaccs) e suggerimenti di capacita'. */
PRIMALrescodee PRIMAL_appendaccseq(PRIMALtask_t t, PRIMALint64t domidx,
                                   PRIMALint64t numafeidx, PRIMALint64t afeidxfirst,
                                   const PRIMALrealt *b);
PRIMALrescodee PRIMAL_appendaccsseq(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        PRIMALint64t afeidxfirst, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_evaluateacc(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                  PRIMALrealt *activity);
PRIMALrescodee PRIMAL_evaluateaccs(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *activity);
PRIMALrescodee PRIMAL_putmaxnumacc(PRIMALtask_t t, PRIMALint64t maxnumacc);
PRIMALrescodee PRIMAL_putmaxnumafe(PRIMALtask_t t, PRIMALint64t maxnumafe);
PRIMALrescodee PRIMAL_putmaxnumdjc(PRIMALtask_t t, PRIMALint64t maxnumdjc);
/* un componente di b di un ACC; violazione primale di un insieme di ACC;
 * sequenze di domini di potenza. */
PRIMALrescodee PRIMAL_putaccbj(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t j, PRIMALrealt bj);
/* i duali di un ACC (riferimento getaccdoty/getaccdotys/putaccdoty): i
 * moltiplicatori delle righe che l'ACC ha prodotto, nella convenzione dei nostri
 * `y` (la convenzione del riferimento per `doty` non e' stata letta). */
PRIMALrescodee PRIMAL_getaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 PRIMALrealt *doty);
PRIMALrescodee PRIMAL_getaccdotys(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *doty);
PRIMALrescodee PRIMAL_putaccdoty(PRIMALtask_t t, PRIMALsolt which, PRIMALint64t accidx,
                                 const PRIMALrealt *doty);
PRIMALrescodee PRIMAL_getpviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getdviolacc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numaccidx, const PRIMALint64t *accidxlist, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_appendprimalpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist);
PRIMALrescodee PRIMAL_appenddualpowerconedomainseq(PRIMALtask_t t, PRIMALint64t num,
        const PRIMALint64t *n, const PRIMALint64t *nleft, const PRIMALrealt *alpha,
        PRIMALint64t *domidxlist);
/* la F e la g implicite nell'ordine degli AFE dentro gli ACC. */
PRIMALrescodee PRIMAL_getaccfnumnz(PRIMALtask_t t, PRIMALint64t *accfnnz);
PRIMALrescodee PRIMAL_getaccgvector(PRIMALtask_t t, PRIMALrealt *g);
PRIMALrescodee PRIMAL_getaccftrip(PRIMALtask_t t, PRIMALint64t *frow,
                                  int *fcol, PRIMALrealt *fval);
/* ---- disjunctive constraints (DJC, stile riferimento) ----
 * Un DJC e' l'OR di numterm clausole; la clausola i e' la congiunzione di
 * termsizelist[i] domini applicati a espressioni affini. domidxlist concatena i
 * domini di tutte le clausole (lunghezza sum termsizelist); afeidxlist concatena
 * le espressioni, una per componente di dominio (lunghezza = somma delle
 * dimensioni dei domini). b, opzionale (NULL = tutti zero), e' la costante
 * SOTTRATTA a ogni espressione: l'espressione k e' F_k x + g_k - b_k.
 * appenddjcs pre-alloca num slot vuoti; putdjc riempie lo slot djcidx.
 * Il modello si estende subito (una binaria di selezione per clausola e righe
 * big-M, come ogni MIP di questo solver), quindi un djcidx gia' scritto non e'
 * riscrivibile (ERR_ARG). Deviazione dichiarata: solo domini LINEARI
 * (R/RZERO/RPLUS/RMINUS); un dominio conico in un DJC e' ERR_ARG. */
PRIMALrescodee PRIMAL_appenddjcs(PRIMALtask_t t, PRIMALint64t num);
PRIMALrescodee PRIMAL_putdjc(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t numdomidx, const PRIMALint64t *domidxlist,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist,
        const PRIMALrealt *b, PRIMALint64t numterms,
        const PRIMALint64t *termsizelist);
/* idxlast-idxfirst DJC consecutivi; termsindjc[i] = numero di termini della DJC
 * idxfirst+i; il resto e' la concatenazione delle descrizioni di putdjc. */
PRIMALrescodee PRIMAL_putdjcslice(PRIMALtask_t t, PRIMALint64t idxfirst,
        PRIMALint64t idxlast, PRIMALint64t numdomidx,
        const PRIMALint64t *domidxlist, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
        PRIMALint64t numterms, const PRIMALint64t *termsizelist,
        const PRIMALint64t *termsindjc);
PRIMALrescodee PRIMAL_getnumdjc(PRIMALtask_t t, PRIMALint64t *num);
PRIMALrescodee PRIMAL_getdjcnumdomain(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getdjcnumafe(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getdjcnumterm(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getdjcdomainidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *domidxlist);
PRIMALrescodee PRIMAL_getdjcafeidxlist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *afeidxlist);
PRIMALrescodee PRIMAL_getdjcb(PRIMALtask_t t, PRIMALint64t djcidx, PRIMALrealt *b);
PRIMALrescodee PRIMAL_getdjctermsizelist(PRIMALtask_t t, PRIMALint64t djcidx,
        PRIMALint64t *termsizelist);
PRIMALrescodee PRIMAL_getdjcnumdomaintot(PRIMALtask_t t, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getdjcnumafetot(PRIMALtask_t t, PRIMALint64t *n);
PRIMALrescodee PRIMAL_getdjcnumtermtot(PRIMALtask_t t, PRIMALint64t *n);
/* lettura in blocco di TUTTI i DJC (riferimento getdjcs): le liste sono la
 * concatenazione di quelle per-DJC, `numterms` ha una entrata per DJC. Ogni
 * buffer ha la lunghezza data dai getdjcnum*tot; un buffer NULL viene saltato. */
PRIMALrescodee PRIMAL_getdjcs(PRIMALtask_t t, PRIMALint64t *domidxlist,
        PRIMALint64t *afeidxlist, PRIMALrealt *b, PRIMALint64t *termsizelist,
        PRIMALint64t *numterms);
/* violazione primale di un insieme di DJC (riferimento getpvioldjc): per ogni
 * djcidxlist[k] scrive viol[k] = min_i(max_j viol(T_ij)) letto sul punto
 * pubblicato e sul modello corrente. */
PRIMALrescodee PRIMAL_getpvioldjc(PRIMALtask_t t, PRIMALsolt which,
        PRIMALint64t numdjcidx, const PRIMALint64t *djcidxlist, PRIMALrealt *viol);
/* nomi dei DJC (putdjcname/getdjcname/getdjcnamelen del riferimento): buffer di
 * sizename byte, che deve contenere anche il terminatore; un rifiuto non scrive. */
PRIMALrescodee PRIMAL_putdjcname(PRIMALtask_t t, PRIMALint64t djcidx, const char *name);
PRIMALrescodee PRIMAL_getdjcnamelen(PRIMALtask_t t, PRIMALint64t djcidx, int *len);
PRIMALrescodee PRIMAL_getdjcname(PRIMALtask_t t, PRIMALint64t djcidx, int sizename, char *name);

/* SDP (semi-definite): variabili bar X_j >= 0 (matrici simmetriche dim x dim)
 * e termini lineari in forma di prodotto interno <A^k, X_j> con matrici
 * simmetriche sparse dallo "matrix store" (appendsparsesymmat). */
PRIMALrescodee PRIMAL_appendsparsesymmat(PRIMALtask_t t, int dim, int nnz,
                                   const int *subi, const int *subj,
                                   const PRIMALrealt *val, int *idx);
/* piu' matrici in una chiamata (riferimento appendsparsesymmatlist): dims[k] e
 * nz[k] danno la forma, subi/subj/valij sono concatenati, idx[k] restituisce gli
 * id. L'intera lista e' validata prima di appendere. */
PRIMALrescodee PRIMAL_appendsparsesymmatlist(PRIMALtask_t t, int num, const int *dims,
    const PRIMALint64t *nz, const int *subi, const int *subj, const PRIMALrealt *valij,
    PRIMALint64t *idx);
PRIMALrescodee PRIMAL_getsparsesymmat(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxlen,
    int *subi, int *subj, PRIMALrealt *valij);
PRIMALrescodee PRIMAL_appendbarvars(PRIMALtask_t t, int num, const int *dim);
/* rimuove le variabili bar agli indici dati, rimappando i termini (riferimento
 * removebarvars). */
PRIMALrescodee PRIMAL_removebarvars(PRIMALtask_t t, int num, const int *subset);
/* vincolo i: aggiunge a (i) i termini scalari  sum_k val_k <A^{sub_k}, X_j> */
PRIMALrescodee PRIMAL_putbaraij(PRIMALtask_t t, int i, int j, int num,
                           const int *sub, const PRIMALrealt *val);
/* scrittura per blocchi di bar A (API piu' efficiente per SDP grandi):
 * il vincolo i vede sum_k val_k <A^{blk_sub_k}, X_{blk_j_k}> per k=0..num-1 */
PRIMALrescodee PRIMAL_putbarablockij(PRIMALtask_t t, int i, int j, int num,
                               const int *blk_sub, const PRIMALrealt *blk_val);
/* lettura dei termini bar A della coppia (i,j): elenco (symidx, coef).
 * Contratto degli accessori che riempiono buffer dell'utente (`T102`/`T104`):
 * maxnum e' la capienza di symidx/val e *num il numero di TERMINI di quella
 * coppia. Se lo spazio non basta la chiamata e' RIFIUTATA (ERR_ARG) senza aver
 * toccato nessun buffer e senza aver scritto *num: una lista troncata che
 * risponde OK e' indistinguibile da una lista completa, e un <A,X> costruito
 * sul prefisso e' un altro modello. Simmetrici e val entrambi NULL e' la porta
 * del solo conteggio (questa coppia non ha un getnum... per (i,j)) e non viene
 * mai rifiutata, perche' non scrive nulla; una (i,j) senza termini risponde 0
 * con OK -- verdetto vuoto, non rifiuto. */
PRIMALrescodee PRIMAL_getbaraidxij(PRIMALtask_t t, int i, int j, int maxnum,
                             int *num, int *symidx, PRIMALrealt *val);
/* lettura dei termini bar C della variabile j: elenco (symidx, coef), con lo
 * STESSO contratto di capienza e di rifiuto silenzioso-impossibile di sopra. */
PRIMALrescodee PRIMAL_getbarcidxj(PRIMALtask_t t, int j, int maxnum,
                             int *num, int *symidx, PRIMALrealt *val);
/* Sparsita' e info per blocco di A-bar/C-bar (riferimento getbarasparsity/
 * getbaraidxinfo/getbaraidx e le varianti C). `idx` nomina un blocco: qui
 * `idx = i*numbarvar + j` per A-bar e `idx = j` per C-bar (convenzione di questo
 * solver: la vettorizzazione del riferimento non e' stata letta). */
PRIMALrescodee PRIMAL_getbarasparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxij);
PRIMALrescodee PRIMAL_getbaraidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num);
PRIMALrescodee PRIMAL_getbaraidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *i, int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights);
PRIMALrescodee PRIMAL_getbarcsparsity(PRIMALtask_t t, PRIMALint64t maxnumnz,
                                      PRIMALint64t *numnz, PRIMALint64t *idxj);
PRIMALrescodee PRIMAL_getbarcidxinfo(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t *num);
PRIMALrescodee PRIMAL_getbarcidx(PRIMALtask_t t, PRIMALint64t idx, PRIMALint64t maxnum,
    int *j, PRIMALint64t *num, PRIMALint64t *sub, PRIMALrealt *weights);
/* Forma a triplette di blocco di A-bar e C-bar (riferimento
 * getbarablocktriplet/getbarcblocktriplet): una riga per ogni entrata memorizzata
 * del triangolo inferiore di ogni blocco. A: (i, j, k, l, val), C: (j, k, l, val);
 * i contatori danno il numero esatto di tali entrate. Capienza insufficiente con
 * buffer forniti = rifiuto senza scrivere. */
PRIMALrescodee PRIMAL_getnumbarablocktriplets(PRIMALtask_t t, PRIMALint64t *num);
PRIMALrescodee PRIMAL_getnumbarcblocktriplets(PRIMALtask_t t, PRIMALint64t *num);
PRIMALrescodee PRIMAL_getbarablocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subi, int *subj, int *subk, int *subl, PRIMALrealt *valijkl);
PRIMALrescodee PRIMAL_getbarcblocktriplet(PRIMALtask_t t, PRIMALint64t maxnum, PRIMALint64t *num,
    int *subj, int *subk, int *subl, PRIMALrealt *valjkl);
/* obiettivo: aggiunge  sum_k val_k <A^{sub_k}, X_j> */
PRIMALrescodee PRIMAL_putbarcj(PRIMALtask_t t, int j, int num,
                         const int *sub, const PRIMALrealt *val);
/* scritture bar in blocco (riferimento putbarablocktriplet/putbarcblocktriplet/
 * putbaraijlist): aggiungono termini al negozio. A-bar per entrate
 * (con,bar,k,l,val); C-bar (bar,k,l,val); la lista per (i,j). */
PRIMALrescodee PRIMAL_putbarablocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const int *subk, const int *subl,
        const PRIMALrealt *valijkl);
PRIMALrescodee PRIMAL_putbarcblocktriplet(PRIMALtask_t t, PRIMALint64t num,
        const int *subj, const int *subk, const int *subl, const PRIMALrealt *valjkl);
PRIMALrescodee PRIMAL_putbaraijlist(PRIMALtask_t t, PRIMALint64t num,
        const int *subi, const int *subj, const PRIMALint64t *alphaptrb,
        const PRIMALint64t *alphaptre, const PRIMALint64t *matidx,
        const PRIMALrealt *weights);
PRIMALrescodee PRIMAL_getnumbarvar(PRIMALtask_t t, int *num);
/* soluzione della variabile bar j (matrice dim_j x dim_j, row-major) */
PRIMALrescodee PRIMAL_getbarxj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *xj);
/* duale bar approssimato Z_j = C_j - sum_i y_i A^i (PSD per costruzione) */
PRIMALrescodee PRIMAL_getbarsj(PRIMALtask_t t, PRIMALsolt which, int j, PRIMALrealt *sj);
/* superficie bar del riferimento: contatori (getnumbaranz/getnumbarcnz), nomi
 * (putbarvarname/getbarvarname/getbarvarnameindex/getbarvarnamelen), fette di
 * barx/barsj (getbarxslice/getbarsslice, blocchi densi d*d concatenati),
 * warm start (putbarxj/putbarsj) e il suggerimento di capacita'
 * putmaxnumbarvar. */
PRIMALrescodee PRIMAL_getnumbaranz(PRIMALtask_t t, PRIMALint64t *nz);
PRIMALrescodee PRIMAL_getnumbarcnz(PRIMALtask_t t, PRIMALint64t *nz);
PRIMALrescodee PRIMAL_putbarvarname(PRIMALtask_t t, int j, const char *name);
PRIMALrescodee PRIMAL_getbarvarname(PRIMALtask_t t, int i, int sizename, char *name);
PRIMALrescodee PRIMAL_getbarvarnameindex(PRIMALtask_t t, const char *somename,
                                         int *asgn, int *index);
PRIMALrescodee PRIMAL_getbarvarnamelen(PRIMALtask_t t, int i, int *len);
PRIMALrescodee PRIMAL_getbarxslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barxslice);
PRIMALrescodee PRIMAL_getbarsslice(PRIMALtask_t t, PRIMALsolt which, int first,
                                   int last, PRIMALint64t slicesize, PRIMALrealt *barsslice);
PRIMALrescodee PRIMAL_putbarxj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barxj);
PRIMALrescodee PRIMAL_putbarsj(PRIMALtask_t t, PRIMALsolt which, int j, const PRIMALrealt *barsj);
PRIMALrescodee PRIMAL_putmaxnumbarvar(PRIMALtask_t t, int maxnumbarvar);

/* getter interni per I/O CBF (dimensioni bar, termini barC/barA, matrix store) */
PRIMALrescodee PRIMAL_getbarsize(PRIMALtask_t t, int j, int *dim);
PRIMALrescodee PRIMAL_getdimbarvarj(PRIMALtask_t t, int j, int *dimbarvarj);
PRIMALrescodee PRIMAL_getlenbarvarj(PRIMALtask_t t, int j, PRIMALint64t *lenbarvarj);
PRIMALrescodee PRIMAL_getnumbarcterm(PRIMALtask_t t, int *num);
PRIMALrescodee PRIMAL_getbarcitem(PRIMALtask_t t, int k, int *jbar, int *msym, PRIMALrealt *coef);
PRIMALrescodee PRIMAL_getnumbaraterm(PRIMALtask_t t, int *num);
PRIMALrescodee PRIMAL_getbaraitem(PRIMALtask_t t, int k, int *con, int *jbar, int *msym, PRIMALrealt *coef);
PRIMALrescodee PRIMAL_getnumsymmat(PRIMALtask_t t, int *num);
PRIMALrescodee PRIMAL_getsymmatinfo(PRIMALtask_t t, int m, int *dim, int *nnz);
PRIMALrescodee PRIMAL_getsymmatentry(PRIMALtask_t t, int m, int e, int *i, int *j, PRIMALrealt *val);

/* optimize + results */
PRIMALrescodee PRIMAL_optimize(PRIMALtask_t t);
/* ---- basis (solvebasis) ---- */
/* status keys per variabili/righe (PRIMAL PRIMALstakey): base, superbasic,
 * at lower, at upper. PRIMAL_SK_UNDEF per elementi mai impostati. */
typedef enum {
    PRIMAL_SK_UNDEF = 0,
    PRIMAL_SK_BAS = 1,
    PRIMAL_SK_SUPBAS = 2,
    PRIMAL_SK_LOW = 3,
    PRIMAL_SK_UPR = 4
} PRIMALstakeye;
/* imposta/legge lo status della base (per righe: BAS/supbas equivalgono a
 * slack di base). Vettori lunghi numcon (skc) / numvar (skx). */
PRIMALrescodee PRIMAL_putskc(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc);
PRIMALrescodee PRIMAL_putskx(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skx);
PRIMALrescodee PRIMAL_getskc(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skc);
PRIMALrescodee PRIMAL_getskx(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skx);
PRIMALrescodee PRIMAL_sktostr(PRIMALtask_t t, PRIMALstakeye sk, char *str);
PRIMALrescodee PRIMAL_strtosk(PRIMALtask_t t, const char *str, PRIMALstakeye *sk);
/* valuta la base corrente (skc+skx): soluzione primal (basic solution) e
 * duali dal sistema della base, verificando primal/duale feasibility con i
 * getter pubblici; se la base non e' primal feasible (righe incompatibili)
 * ritorna ERR_INFEASIBLE; se non duale-feasible ottimizza da zero
 * (deviazione documentata: il simplesso del clone non fa warm start da
 * base arbitrarie, la base serve come specifica della SOLUZIONE basic). */
PRIMALrescodee PRIMAL_solvebasis(PRIMALtask_t t);
/* scrittura/lettura base in formato MPS BAS (XLOWER/XUPPER/XBASIC per
 * variabili, XBASIC per righe); nomi c%d / x%d coerenti con writedata */
PRIMALrescodee PRIMAL_writebasis(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readbasis(PRIMALtask_t t, const char *filename);
/* solution I/O (riferimento writesolution/readsolution, writebsolution/
 * readbsolution, writebsolutionhandle). Deviazione dichiarata: il FORMATO del
 * riferimento non e' stato letto; qui c'e' un formato testuale e uno binario
 * propri che fanno round-trip. `*file` sono le stesse chiamate. */
typedef void (*PRIMALhwritefunc)(void *handle, const char *data, int len);
typedef int (*PRIMALhreadfunc)(void *handle, char *buffer, int *len);
PRIMALrescodee PRIMAL_writesolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename);
PRIMALrescodee PRIMAL_readsolution(PRIMALtask_t t, PRIMALsolt whichsol, const char *filename);
PRIMALrescodee PRIMAL_writesolutionfile(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readsolutionfile(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_writebsolution(PRIMALtask_t t, const char *filename, int compress);
PRIMALrescodee PRIMAL_readbsolution(PRIMALtask_t t, const char *filename, int compress);
PRIMALrescodee PRIMAL_writebsolutionhandle(PRIMALtask_t t, PRIMALhwritefunc func,
                                           void *handle, int compress);
/* forma JSON (JSOL): un oggetto piatto proprio. */
PRIMALrescodee PRIMAL_writejsonsol(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readjsonsol(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readjsonstring(PRIMALtask_t t, const char *data);
/* ---- sensitivity (LP, post-ottimo) ----
 * Range del costo c_j per cui la SOLUZIONE corrente (x*, duali inclusi)
 * resta ottimale (solo problemi lineari gia' risolti). Ritorna
 * PRIMAL_RES_ERR_ARG se non applicabile. */
PRIMALrescodee PRIMAL_costsensitivity(PRIMALtask_t t, int j,
                                 PRIMALrealt *lcost, PRIMALrealt *ucost);
/* Range del bound RHS della riga i (lato attivo) per cui i DUALI correnti
 * restano ottimali: la riga resta al bound attivo con la stessa base. */
PRIMALrescodee PRIMAL_rhssensitivity(PRIMALtask_t t, int i,
                                PRIMALrealt *lrange, PRIMALrealt *urange);
/* progress callback: called with a user info string after each major
 * iteration / solution update (PRIMAL PRIMAL_progresscb equivalent).
 * Return value ignored. Signature matches PRIMALcallbackfunc. */
typedef void (*PRIMALprogresscb)(void *handle, const char *info);
PRIMALrescodee PRIMAL_setprogresscb(PRIMALtask_t t, PRIMALprogresscb cb, void *handle);
/* ---- callback generali (riferimento putcallbackfunc/getcallbackfunc/
 * putresponsefunc) ----
 * Il callback generale riceve un codice di evento (i numeri sono quelli di
 * MSKcallbackcodee) e tre vettori di dettaglio; questo solver emette gli eventi
 * BEGIN/END di OPTIMIZER/READ/WRITE e, per la rotta che risponde, quelli di
 * SIMPLEX/INTPNT/MIO/CONIC con i vettori a NULL (dettaglio non
 * popolato, deviazione dichiarata). Il callback di risposta e' invocato quando
 * un solve termina con un codice diverso da PRIMAL_RES_OK.
 * Deviazione dichiarata: i codici per-iterazione (MSK_CALLBACK_IM_* e i
 * MSK_CALLBACK_INTPNT/CONIC/PRIMAL_SIMPLEX "di mezzo") non sono emessi. */
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
    /* per-iterazione / solution-update (i "middle" del riferimento) */
    PRIMAL_CALLBACK_CONIC           = 34,
    PRIMAL_CALLBACK_PRIMAL_SIMPLEX  = 93,
    PRIMAL_CALLBACK_INTPNT          = 90,
    /* sub-step interni */
    PRIMAL_CALLBACK_IM_LU           = 79,
    PRIMAL_CALLBACK_IM_ORDER        = 84
} PRIMALcallbackcodee;
PRIMALrescodee PRIMAL_callbackcodetostr(PRIMALcallbackcodee code, char *str);

typedef void (*PRIMALcallbackcb)(PRIMALtask_t task, void *handle,
    PRIMALcallbackcodee code, const PRIMALrealt *info,
    const PRIMALint32t *intinfo, const PRIMALint64t *lliinfo);
typedef void (*PRIMALresponsecb)(PRIMALtask_t task, void *handle, PRIMALrescodee res);

PRIMALrescodee PRIMAL_putcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb cb, void *handle);
PRIMALrescodee PRIMAL_getcallbackfunc(PRIMALtask_t t, PRIMALcallbackcb *cb, void **handle);
PRIMALrescodee PRIMAL_putresponsefunc(PRIMALtask_t t, PRIMALresponsecb cb, void *handle);
/* name for the info strings of the progress callback (max 63 chars,
 * truncated like PRIMAL_setinfoconnname) */
PRIMALrescodee PRIMAL_setinfoconnname(PRIMALtask_t t, const char *connname);
PRIMALrescodee PRIMAL_getxx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xx);
PRIMALrescodee PRIMAL_gety(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *y);
/* warm start: fornisce un punto di partenza (usato dal percorso punto interno;
 * ignorato da simplex/MIP/coni). whichsol: PRIMAL_SOL_ITR o PRIMAL_SOL_BAS. */
PRIMALrescodee PRIMAL_putxx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *xx);
PRIMALrescodee PRIMAL_puty(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *y);
PRIMALrescodee PRIMAL_getslc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *slc);
PRIMALrescodee PRIMAL_getsuc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *suc);
PRIMALrescodee PRIMAL_getslx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *slx);
PRIMALrescodee PRIMAL_getsux(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *sux);
PRIMALrescodee PRIMAL_getprimalobj(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pobj);
PRIMALrescodee PRIMAL_getdualobj(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *dobj);
PRIMALrescodee PRIMAL_getsolsta(PRIMALtask_t t, PRIMALsolt which, PRIMALsolstae *solsta);
PRIMALrescodee PRIMAL_getprosta(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *prosta);
/* `getsolution`: legge in una chiamata la soluzione completa (riferimento
 * MSK_getsolution). Ogni puntatore e' opzionale (NULL = salta). `skn` (chiavi di
 * stato dei coni) e' SK_UNDEF per ogni cono e `snx` (duale conico per variabile)
 * e' 0: deviazioni dichiarate (nessuna base per un blocco conico; il duale di un
 * cono vive dentro il blocco). Senza punto pubblicato i buffer del punto
 * rispondono ERR_ARG, come i getter singoli; gli stati si leggono lo stesso. */
PRIMALrescodee PRIMAL_getsolution(PRIMALtask_t t, PRIMALsolt which,
    PRIMALprostae *problemsta, PRIMALsolstae *solutionsta,
    PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
    PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y,
    PRIMALrealt *slc, PRIMALrealt *suc, PRIMALrealt *slx, PRIMALrealt *sux,
    PRIMALrealt *snx);
PRIMALrescodee PRIMAL_getskn(PRIMALtask_t t, PRIMALsolt which, PRIMALstakeye *skn);
PRIMALrescodee PRIMAL_getsnx(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *snx);
/* ---- Farkas certificates (raggi di infeasibilita', vettori) ----
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
PRIMALrescodee PRIMAL_getprimalray(PRIMALtask_t t, PRIMALrealt *rho);
/* ---- solution quality: max violazioni primal/dual della soluzione ----
 * (misurate sui dati del problema con i getter; -1 se non applicabile) */
PRIMALrescodee PRIMAL_getprimalinfeas(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pinf);
PRIMALrescodee PRIMAL_getdualinfeas(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *dinf);
/* ---- solution information: violazioni per indice e riepilogo ----
 * Famiglia del riferimento (MOSEK 11.2.4): `getpviolcon`/`getpviolvar`/
 * `getpviolbarvar`/`getpviolcones` scrivono in `viol` la violazione primale
 * degli indici elencati in `sub` (che e' un vettore di indici, non un intervallo);
 * un indice fuori dominio, `viol`/`sub` null o nessuna soluzione sono
 * `PRIMAL_RES_ERR_ARG` **senza scrivere nulla**. `PRIMAL_getsolutioninfo` riporta
 * i massimi della famiglia piu' `pobj`/`dobj` (accetta NULL per i campi non
 * richiesti). La meta' primale e' esatta; la duale (dviol*) e' nella nostra
 * convenzione dei getter `y`/`slc`/`sux` (README «Dual conventions»), con i
 * membri di cono lasciati non misurati — v. il commento in `primal.c`. */
PRIMALrescodee PRIMAL_getpviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getpviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getpviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getpviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol);
/* La meta' duale della stessa famiglia. Stessa forma di `sub`/`viol` e stesso
 * contratto di rifiuto; i valori sono nella nostra convenzione dei getter
 * (README «Dual conventions», segno speculare al riferimento) e un membro di cono
 * resta non misurato in `dviolvar`. */
PRIMALrescodee PRIMAL_getdviolcon(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getdviolvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                  const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getdviolbarvar(PRIMALtask_t t, PRIMALsolt which, int num,
                                     const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getdviolcones(PRIMALtask_t t, PRIMALsolt which, int num,
                                    const int *sub, PRIMALrealt *viol);
PRIMALrescodee PRIMAL_getsolutioninfo(PRIMALtask_t t, PRIMALsolt which,
    PRIMALrealt *pobj, PRIMALrealt *pviolcon, PRIMALrealt *pviolvar,
    PRIMALrealt *pviolbarvar, PRIMALrealt *pviolcone, PRIMALrealt *pviolitg,
    PRIMALrealt *dobj, PRIMALrealt *dviolcon, PRIMALrealt *dviolvar,
    PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone);
/* ---- feasibility repair (elastic): min sum(s^- + s^+) s.t.
 * lo - s^- <= Ax <= up + s^+, lx - s^- <= x <= ux + s^+ ----
 * Riporta la soluzione riparata (x) nel task. */
PRIMALrescodee PRIMAL_feasrepair(PRIMALtask_t t);
PRIMALrescodee PRIMAL_getsolutionslice(PRIMALtask_t t, PRIMALsolt which, int part,
                                 int first, int last, PRIMALrealt *values);
/* Fette dei vettori di soluzione (riferimento: getxxslice, getyslice,
 * getslcslice, getsucslice, getslxslice, getsuxslice, getskxslice, getskcslice)
 * e costo ridotto `(s_l^x)_j - (s_u^x)_j` (getreducedcosts). Forma [first,last),
 * `last-first` entrate; un rifiuto non scrive. */
PRIMALrescodee PRIMAL_getxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *xx);
PRIMALrescodee PRIMAL_getyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *y);
PRIMALrescodee PRIMAL_getslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slc);
PRIMALrescodee PRIMAL_getsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *suc);
PRIMALrescodee PRIMAL_getslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *slx);
PRIMALrescodee PRIMAL_getsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *sux);
PRIMALrescodee PRIMAL_getreducedcosts(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *redcosts);
PRIMALrescodee PRIMAL_getskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALstakeye *skx);
PRIMALrescodee PRIMAL_getskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALstakeye *skc);
PRIMALrescodee PRIMAL_putskxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skx);
PRIMALrescodee PRIMAL_putskcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALstakeye *skc);
/* norme 2 della soluzione primale: ||x^c||, ||x||, ||X_bar||_F. */
PRIMALrescodee PRIMAL_getprimalsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmxc, PRIMALrealt *nrmxx, PRIMALrealt *nrmbarx);
PRIMALrescodee PRIMAL_getdualsolutionnorms(PRIMALtask_t t, PRIMALsolt which,
        PRIMALrealt *nrmy, PRIMALrealt *nrmslc, PRIMALrealt *nrmsuc,
        PRIMALrealt *nrmslx, PRIMALrealt *nrmsux, PRIMALrealt *nrmsnx,
        PRIMALrealt *nrmbars);
/* `x^c`: il valore delle variabili di vincolo (riferimento getxc/getxcslice):
 * il primo membro della riga, letto da `row_activity` (scalare | quadratico |
 * barra) -- la stessa lettura di `getpviolcon`. */
PRIMALrescodee PRIMAL_getxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc);
PRIMALrescodee PRIMAL_getxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last, PRIMALrealt *xc);
/* setter di x^c e s_n^x (riferimento putxc/putxcslice/putsnx/putsnxslice/
 * getsnxslice). s_n^x non e' calcolato da nessun percorso: e' storage. */
PRIMALrescodee PRIMAL_putxc(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *xc);
PRIMALrescodee PRIMAL_putxcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xc);
PRIMALrescodee PRIMAL_putsnx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *snx);
PRIMALrescodee PRIMAL_putsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *snx);
PRIMALrescodee PRIMAL_getsnxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  PRIMALrealt *snx);
/* setter dei vettori di soluzione (riferimento putslc/putsuc/putslx/putsux,
 * putxxslice/putyslice/putslxslice/putsuxslice): scrivono nel buffer del punto
 * (esiste dopo il primo opt_prepare); una fetta e' [first, last) con
 * last-first entrate; un rifiuto non scrive. */
PRIMALrescodee PRIMAL_putslc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slc);
PRIMALrescodee PRIMAL_putsuc(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *suc);
PRIMALrescodee PRIMAL_putslx(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *slx);
PRIMALrescodee PRIMAL_putsux(PRIMALtask_t t, PRIMALsolt which, const PRIMALrealt *sux);
PRIMALrescodee PRIMAL_putxxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                 const PRIMALrealt *xx);
PRIMALrescodee PRIMAL_putyslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                const PRIMALrealt *y);
PRIMALrescodee PRIMAL_putslxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slx);
PRIMALrescodee PRIMAL_putsuxslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *sux);
PRIMALrescodee PRIMAL_putslcslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *slc);
PRIMALrescodee PRIMAL_putsucslice(PRIMALtask_t t, PRIMALsolt which, int first, int last,
                                  const PRIMALrealt *suc);
PRIMALrescodee PRIMAL_putvarboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkx, PRIMALrealt blx, PRIMALrealt bux);
PRIMALrescodee PRIMAL_putconboundlistconst(PRIMALtask_t t, int num, const int *sub,
        PRIMALboundkeye bkc, PRIMALrealt blc, PRIMALrealt buc);
/* write the problem in a readable text form (debugging) */
PRIMALrescodee PRIMAL_writedata(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readdata(PRIMALtask_t t, const char *filename);
/* forme del riferimento attorno a readdata/writedata (il formato dichiarato non
 * cambia cio' che si legge: il lettore riconosce il file dal contenuto) e
 * `getapiecenumnz`, i nonnulli di A in un pezzo rettangolare. */
PRIMALrescodee PRIMAL_readdataautoformat(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readdataformat(PRIMALtask_t t, const char *filename,
                                     PRIMALdataformate format,
                                     PRIMALcompresstypee compress);
PRIMALrescodee PRIMAL_readtask(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_writetask(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_getapiecenumnz(PRIMALtask_t t, int firsti, int lasti,
                                     int firstj, int lastj, int *numnz);

/* logging callback: void (*)(void *handle, const char *msg) */
typedef void (*PRIMALlogcb)(void *handle, const char *msg);
PRIMALrescodee PRIMAL_setlogcb(PRIMALtask_t t, PRIMALlogcb logcb, void *loghandle);
PRIMALrescodee PRIMAL_linkfunctotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                     void *handle, PRIMALstreamfunc func);
/* stream su file, stream di env, echo (riferimento linkfileto*stream,
 * linkfunctoenvstream, unlinkfuncfrom*stream, echo*). Lo stream di env e'
 * ereditato dai task creati dopo. */
PRIMALrescodee PRIMAL_linkfiletotaskstream(PRIMALtask_t t, PRIMALstreamtypee which,
                                           const char *filename, int append);
PRIMALrescodee PRIMAL_unlinkfuncfromtaskstream(PRIMALtask_t t, PRIMALstreamtypee which);
PRIMALrescodee PRIMAL_echotask(PRIMALtask_t t, PRIMALstreamtypee which, const char *format, ...);
PRIMALrescodee PRIMAL_linkfiletoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                          const char *filename, int append);
PRIMALrescodee PRIMAL_linkfunctoenvstream(PRIMALenv_t env, PRIMALstreamtypee which,
                                          void *handle, PRIMALstreamfunc func);
PRIMALrescodee PRIMAL_unlinkfuncfromenvstream(PRIMALenv_t env, PRIMALstreamtypee which);
PRIMALrescodee PRIMAL_echoenv(PRIMALenv_t env, PRIMALstreamtypee which, const char *format, ...);
PRIMALrescodee PRIMAL_echointro(PRIMALenv_t env, int longver);

/* ---- parita' del riferimento: parametri long, generatori di nomi, diagnostica,
 * optimize*, repair/sensitivity, toconic ---- */
PRIMALrescodee PRIMAL_getlintparam(PRIMALtask_t t, int param, PRIMALint64t *parvalue);
PRIMALrescodee PRIMAL_putlintparam(PRIMALtask_t t, int param, PRIMALint64t parvalue);
PRIMALrescodee PRIMAL_putparam(PRIMALtask_t t, const char *parname, const char *parvalue);
PRIMALrescodee PRIMAL_writeparamfile(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_readparamfile(PRIMALtask_t t, const char *filename);
PRIMALrescodee PRIMAL_generatevarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_generateconnames(PRIMALtask_t t, int num, const int *subi, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_generateconenames(PRIMALtask_t t, int num, const int *subk, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_generatebarvarnames(PRIMALtask_t t, int num, const int *subj, const char *fmt,
        int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_generateaccnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_generatedjcnames(PRIMALtask_t t, PRIMALint64t num, const PRIMALint64t *sub,
        const char *fmt, int ndims, const int *dims, const PRIMALint64t *sp, int numnamedaxis,
        const int *namedaxisidxs, PRIMALint64t numnames, const char **names);
PRIMALrescodee PRIMAL_getconeinfo(PRIMALtask_t t, int k, PRIMALconetypee *ct,
                                  PRIMALrealt *conepar, int *nummem);
PRIMALrescodee PRIMAL_printparam(PRIMALtask_t t);
PRIMALrescodee PRIMAL_readsummary(PRIMALtask_t t, int whichstream);
PRIMALrescodee PRIMAL_optimizetrm(PRIMALtask_t t, PRIMALrescodee *trmcode);
PRIMALrescodee PRIMAL_optimizebatch(PRIMALenv_t env, int israce, PRIMALrealt maxtime,
        int numthreads, PRIMALint64t numtask, const PRIMALtask_t *task,
        PRIMALrescodee *trmcode, PRIMALrescodee *rcode);
PRIMALrescodee PRIMAL_primalrepair(PRIMALtask_t t, const PRIMALrealt *wlc,
        const PRIMALrealt *wuc, const PRIMALrealt *wlx, const PRIMALrealt *wux);
PRIMALrescodee PRIMAL_dualsensitivity(PRIMALtask_t t, int numj, const int *subj,
        PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej);
PRIMALrescodee PRIMAL_primalsensitivity(PRIMALtask_t t, int numi, const int *subi,
        const int *marki, int numj, const int *subj, const int *markj,
        PRIMALrealt *leftpricei, PRIMALrealt *rightpricei, PRIMALrealt *leftrangei,
        PRIMALrealt *rightrangei, PRIMALrealt *leftpricej, PRIMALrealt *rightpricej,
        PRIMALrealt *leftrangej, PRIMALrealt *rightrangej);
PRIMALrescodee PRIMAL_toconic(PRIMALtask_t t);
/* ACC in stile put, AFE/coni/barA per riga */
PRIMALrescodee PRIMAL_putacc(PRIMALtask_t t, PRIMALint64t accidx, PRIMALint64t domidx,
        PRIMALint64t numafeidx, const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_putacclist(PRIMALtask_t t, PRIMALint64t numaccs,
        const PRIMALint64t *accidxs, const PRIMALint64t *domidxs, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidxlist, const PRIMALrealt *b);
PRIMALrescodee PRIMAL_putafefrowlist(PRIMALtask_t t, PRIMALint64t numafeidx,
        const PRIMALint64t *afeidx, const int *numnzrow, const PRIMALint64t *ptrrow,
        PRIMALint64t lenidxval, const int *varidx, const PRIMALrealt *val);
PRIMALrescodee PRIMAL_putcone(PRIMALtask_t t, int k, PRIMALconetypee ct, PRIMALrealt conepar,
                              int nummem, const int *submem);
PRIMALrescodee PRIMAL_putbararowlist(PRIMALtask_t t, int num, const int *subi,
        const PRIMALint64t *ptrb, const PRIMALint64t *ptre, const int *subj,
        const PRIMALint64t *nummat, const PRIMALint64t *matidx, const PRIMALrealt *weights);
/* API di soluzione in stile "new" e setter per indice */
PRIMALrescodee PRIMAL_solutiondef(PRIMALtask_t t, PRIMALsolt which, int *isdef);
PRIMALrescodee PRIMAL_putconsolutioni(PRIMALtask_t t, int i, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su);
PRIMALrescodee PRIMAL_putsolutionyi(PRIMALtask_t t, int i, PRIMALsolt which, PRIMALrealt y);
PRIMALrescodee PRIMAL_putvarsolutionj(PRIMALtask_t t, int j, PRIMALsolt which,
        PRIMALstakeye sk, PRIMALrealt x, PRIMALrealt sl, PRIMALrealt su, PRIMALrealt sn);
PRIMALrescodee PRIMAL_getsolutionnew(PRIMALtask_t t, PRIMALsolt which, PRIMALprostae *problemsta,
        PRIMALsolstae *solutionsta, PRIMALstakeye *skc, PRIMALstakeye *skx, PRIMALstakeye *skn,
        PRIMALrealt *xc, PRIMALrealt *xx, PRIMALrealt *y, PRIMALrealt *slc, PRIMALrealt *suc,
        PRIMALrealt *slx, PRIMALrealt *sux, PRIMALrealt *snx, PRIMALrealt *doty);
PRIMALrescodee PRIMAL_putsolutionnew(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx, const PRIMALrealt *doty);
PRIMALrescodee PRIMAL_putsolution(PRIMALtask_t t, PRIMALsolt which, const PRIMALstakeye *skc,
        const PRIMALstakeye *skx, const PRIMALstakeye *skn, const PRIMALrealt *xc,
        const PRIMALrealt *xx, const PRIMALrealt *y, const PRIMALrealt *slc,
        const PRIMALrealt *suc, const PRIMALrealt *slx, const PRIMALrealt *sux,
        const PRIMALrealt *snx);
PRIMALrescodee PRIMAL_getsolutioninfonew(PRIMALtask_t t, PRIMALsolt which, PRIMALrealt *pobj,
        PRIMALrealt *pviolcon, PRIMALrealt *pviolvar, PRIMALrealt *pviolbarvar,
        PRIMALrealt *pviolcone, PRIMALrealt *pviolacc, PRIMALrealt *pvioldjc,
        PRIMALrealt *pviolitg, PRIMALrealt *dobj, PRIMALrealt *dviolcon,
        PRIMALrealt *dviolvar, PRIMALrealt *dviolbarvar, PRIMALrealt *dviolcone,
        PRIMALrealt *dviolacc);
/* I/O a stringa/handle, basis solve, Cholesky sparsa, clone/duale/subproblem */
PRIMALrescodee PRIMAL_readlpstring(PRIMALtask_t t, const char *data);
PRIMALrescodee PRIMAL_readopfstring(PRIMALtask_t t, const char *data);
PRIMALrescodee PRIMAL_readptfstring(PRIMALtask_t t, const char *data);
PRIMALrescodee PRIMAL_readdatahandle(PRIMALtask_t t, PRIMALhreadfunc hread, void *h,
                                     int format, int compress, const char *path);
PRIMALrescodee PRIMAL_writedatahandle(PRIMALtask_t t, PRIMALhwritefunc func, void *handle,
                                      int format, int compress);
PRIMALrescodee PRIMAL_initbasissolve(PRIMALtask_t t, int *basis);
PRIMALrescodee PRIMAL_solvewithbasis(PRIMALtask_t t, int transp, int numnz, int *sub,
                                     PRIMALrealt *val, int *numnzout);
PRIMALrescodee PRIMAL_basiscond(PRIMALtask_t t, PRIMALrealt *nrmbasis, PRIMALrealt *nrminvbasis);
PRIMALrescodee PRIMAL_computesparsecholesky(PRIMALenv_t env, int numthreads, int ordermethod,
        PRIMALrealt tolsingular, int n, const int *anzc, const PRIMALint64t *aptrc,
        const int *asubc, const PRIMALrealt *avalc, int **perm, PRIMALrealt **diag,
        int **lnzc, PRIMALint64t **lptrc, PRIMALint64t *lensubnval, int **lsubc,
        PRIMALrealt **lvalc);
PRIMALrescodee PRIMAL_clonetask(PRIMALtask_t t, PRIMALtask_t *clonedtask);
PRIMALrescodee PRIMAL_getdualproblem(PRIMALtask_t t, PRIMALtask_t *dualtask);
PRIMALrescodee PRIMAL_getinfeasiblesubproblem(PRIMALtask_t t, PRIMALsolt which,
                                              PRIMALtask_t *inftask);

/* ---- algebra lineare (riferimento, gruppo "Linear algebra") ----
 * Le matrici dense sono COLONNA-major (la convenzione del riferimento). `uplo` e
 * `transpose` hanno i valori di MSKuploe/MSKtransposee (LO 0/UP 1, NO 0/YES 1). */
typedef enum { PRIMAL_TRANSPOSE_NO = 0, PRIMAL_TRANSPOSE_YES = 1 } PRIMALtransposee;
typedef enum { PRIMAL_UPLO_LO = 0, PRIMAL_UPLO_UP = 1 } PRIMALUploe;

PRIMALrescodee PRIMAL_dot(PRIMALenv_t env, int n, const PRIMALrealt *x,
                          const PRIMALrealt *y, PRIMALrealt *xty);
PRIMALrescodee PRIMAL_axpy(PRIMALenv_t env, int n, PRIMALrealt alpha,
                           const PRIMALrealt *x, PRIMALrealt *y);
PRIMALrescodee PRIMAL_gemv(PRIMALenv_t env, PRIMALtransposee transa, int m, int n,
                           PRIMALrealt alpha, const PRIMALrealt *a, const PRIMALrealt *x,
                           PRIMALrealt beta, PRIMALrealt *y);
PRIMALrescodee PRIMAL_gemm(PRIMALenv_t env, PRIMALtransposee transa, PRIMALtransposee transb,
                           int m, int n, int k, PRIMALrealt alpha, const PRIMALrealt *a,
                           const PRIMALrealt *b, PRIMALrealt beta, PRIMALrealt *c);
PRIMALrescodee PRIMAL_syrk(PRIMALenv_t env, PRIMALUploe uplo, PRIMALtransposee trans, int n,
                           int k, PRIMALrealt alpha, const PRIMALrealt *a, PRIMALrealt beta,
                           PRIMALrealt *c);
PRIMALrescodee PRIMAL_potrf(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a);
PRIMALrescodee PRIMAL_syeig(PRIMALenv_t env, PRIMALUploe uplo, int n, const PRIMALrealt *a,
                            PRIMALrealt *w);
PRIMALrescodee PRIMAL_syevd(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a,
                            PRIMALrealt *w);
PRIMALrescodee PRIMAL_sparsetriangularsolvedense(PRIMALenv_t env, PRIMALtransposee transposed,
        int n, const int *lnzc, const PRIMALint64t *lptrc, PRIMALint64t lensubnval,
        const int *lsubc, const PRIMALrealt *lvalc, PRIMALrealt *b);

/* ---- UTF-8 <-> wide-char conversion (reference utf8towchar/wchartoutf8) ----
 * PRIMALwchart is wchar_t (UTF-32 on this platform).  `len` = output units
 * written, `conv` = input units consumed.  Declared deviation: the reference's
 * exact len/conv convention was not read (not invented). */
typedef wchar_t PRIMALwchart;
PRIMALrescodee PRIMAL_utf8towchar(size_t outputlen, size_t *len, size_t *conv,
                                  PRIMALwchart *output, const char *input);
PRIMALrescodee PRIMAL_wchartoutf8(size_t outputlen, size_t *len, size_t *conv,
                                  char *output, const PRIMALwchart *input);

#ifdef __cplusplus
}
#endif

#endif /* PRIMAL_H */
