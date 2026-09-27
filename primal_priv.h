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
/* primal_priv.h - internal sharing for the split primal_* translation units.
 *
 * Holds what one translation unit cannot hold alone: the task struct,
 * small shared types, cross-unit macros, and declarations of helpers
 * used from more than one unit (verbatim moves from primal.c).
 * NOT installed and NOT part of the public API: build-only.
 */
#ifndef PRIMAL_PRIV_H
#define PRIMAL_PRIV_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stddef.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include "primal.h"
#include "linalg.h"
#include "stdform.h"
#include "simplex.h"
#include "ipm.h"
#include "sdp.h"
#include "expcone.h"
#include "mpsio.h"
#include "scaling.h"
#include "socp.h"
#include "presolve.h"

#define INF INFINITY
/* ---------------- column storage ---------------- */
typedef struct {
    int nz, cap;
    int *sub;
    double *val;
} Col;

/* A cone incidence, rather than a variable: shared members have distinct duals. */
typedef struct { int cone, pos, var; double value; } SocDual;

struct PRIMAL_env_s {
    int dummy;
    PRIMALstreamfunc streamfunc; void *streamhandle; FILE *streamfile;
    PRIMALexitfunc exitfunc; void *exithandle;
};

struct PRIMAL_task_s {
    PRIMALenv_t env;
    int numcon, numvar;

    Col *cols;          /* per-variable column of A */
    double *c;          /* linear objective */
    double cfix;
    /* quadratic objective: stored NATIVE SPARSE as an accumulated triplet list
     * (qt_*). The dense n x n form (qobj) and are materialized lazily only by
     * the consumers that need them (conic/QCQP eigendecomposition, getqobjij);
     * the LP/QP hot path uses sparse matvecs (task_xQx / task_Qx) and builds the
     * scaled Qi directly from the triplets, so a large sparse Q never forces a
     * dense n^2 task allocation. */
    int *qt_i, *qt_j; double *qt_v; int qt_n, qt_cap;
    double *qobj;       /* lazy dense cache (numvar x numvar symmetric), or NULL */
    int has_qobj;

    PRIMALboundkeye *bkx; double *blx, *bux;
    PRIMALboundkeye *bkc; double *blc, *buc;
    PRIMALvariabletypee *vartype;   /* MIP: per-variable integrality */
    /* names: two independent tables (a constraint and a variable may share a
     * name, as an MPS allows), sized exactly numvar / numcon like the bound
     * arrays. Each entry OWNS its string; NULL means unnamed. */
    char **varname, **conname;

    /* quadratic constraint terms: per-row dense symmetric Q_i (numvar x numvar)
     * or NULL; has_qcon counts the rows with a nonzero Q */
    double **qcon;   /* [numcon][numvar*numvar], NULL when unused */
    int has_qcon;

    /* affine expressions (AFE): f_i = sum_j F_ij x_j + g_i, stored per row sparse */
    int numafe, afecap;
    int *afe_nz, *afe_cap;
    int **afe_sub; double **afe_val;
    double *afeg;
    /* bar terms of an AFE (reference putafebarfentry): <Fbar_ij, X_j> enters
     * the i-th expression. A term is (barvaridx, symidx, weight); the
     * write of (i,j) replaces the terms with the same barvaridx. */
    int *afe_barnz, *afe_barcap;
    int **afe_baridx, **afe_barsym;
    double **afe_barcoef;

    /* conic domains (for affine conic constraints, ACC) */
    int numdomain, domcap;
    int *dom_type;
    PRIMALint64t *dom_n;
    double *dom_param;
    char **domname;            /* the seventh name table, on domcap */

    /* affine conic constraints (ACC): F_acc x + b_acc in domain acc_dom.
     * acc_afe[k][e] is the AFE index of the e-th component, acc_b[k][e]
     * the added constant; the dimension is acc_nafe[k] == dom_n[acc_dom[k]]. */
    int numacc, acccap;
    PRIMALint64t *acc_dom;
    PRIMALint64t *acc_nafe;
    PRIMALint64t **acc_afe;
    double **acc_b;
    PRIMALint64t *acc_varbase; /* first auxiliary variable, -1 for linear domains */
    PRIMALint64t *acc_rowbase; /* the first row the ACC produced (for doty) */
    char **accname;            /* the sixth name table, on acccap */

    /* disjunctive constraints (DJC, reference style): OR of numterm clauses,
     * each a conjunction of domains on affine expressions. The metadata
     * are the exact description of putdjc; the extended model (selection
     * binaries + big-M rows) is reserved at write time and refreshed from
     * current AFEs and bounds before optimization. There is a single capacity for all the arrays
     * and for the fifth name table. */
    int numdjc, djccap;
    int *djc_rowbase, *djc_varbase, *djc_nrow;
    PRIMALint64t *djc_ndom;        /* for each DJC: |domidxlist| */
    PRIMALint64t *djc_nafe;        /* for each DJC: |afeidxlist| */
    PRIMALint64t *djc_numterm;     /* for each DJC: number of clauses */
    PRIMALint64t **djc_dom;        /* for each DJC: list of domains */
    PRIMALint64t **djc_afe;        /* for each DJC: list of the AFEs */
    double **djc_b;                /* for each DJC: vector b (or NULL) */
    PRIMALint64t **djc_termsize;   /* for each DJC: size of each clause */
    char **djcname;                /* the fifth name table, on djccap */

    /* SOS constraints (MIP): type 1/2, members, weights */
    int numsos, soscap;
    int *sos_type, *sos_n, **sos_mem;
    double **sos_w;

    /* conic constraints (SOCP): members are variable indices */
    int numcones, cone_cap;
    int *cone_type, *cone_nmem, **cone_mem;
    SocDual *soc_dual;
    int nsoc_dual;
    double *cone_param;   /* alpha for PPOW/RPOW */
    char **conename;      /* the fourth name table, on cone_cap */

    /* SDP: bar variables and matrix store */
    int numbarvar, barcap;
    int *barDim;             /* dimension of each bar variable */
    char **barname;          /* the third name table, on barcap */
    int nsym, symcap;
    int *sym_dim, *sym_nnz, *sym_cap;
    int **sym_subi, **sym_subj;
    double **sym_val;
    int nbarA, capbarA, nbarC, capbarC;
    int *barA_con, *barA_bar, *barA_sym; double *barA_coef;
    int *barC_bar, *barC_sym; double *barC_coef;
    double **barx;           /* bar solution (per variable, dim*dim) */
    double **barsj;          /* approximate bar dual */

    PRIMALobjsensee sense;
    char *objname;           /* the objective: a single name, a one-slot table */
    char *taskname;          /* the task name (reference puttaskname) */
    int optimizer;
    int presolve;          /* LP presolve: 0 off, 1 on (default) */
    int presolve_level;    /* presolve depth: 0 off, 1 basic, 2 aggressive */

    double tol_gap, tol_pfeas, tol_dfeas, tol_near_rel;
    /* Per-class interior-point tolerance sets (reference: INTPNT_CO_TOL_* for the
     * conic route, INTPNT_QO_TOL_* for the quadratic route; the plain set above is
     * the LP route). Same defaults, distinct storage so a route reads its own. */
    double tol_co_pfeas, tol_co_dfeas, tol_co_gap;
    double tol_qo_pfeas, tol_qo_dfeas, tol_qo_gap;
    /* Wall-clock cap (seconds; <0 = no limit) and the absolute clock() deadline
     * it produces for the current solve (set by PRIMAL_optimize, read by the
     * IPM loops via ipm_set_deadline and by the B&B loop directly). */
    double optimizer_max_time;
    double opt_deadline;
    /* MIP-phase cap (reference MSK_DPAR_MIO_MAX_TIME) and the effective B&B
     * deadline (the tighter of it and opt_deadline). */
    double mio_max_time;
    double mip_deadline;
    /* Objective cuts (min-space; -DBL_MAX/+DBL_MAX = none) -- the PRIMAL_DPAR_*_OBJ_CUT. */
    double lower_obj_cut;
    double upper_obj_cut;
    /* PSD tolerance (reference MSK_DPAR_SEMIDEFINITE_TOL_APPROX, default 1e-10):
     * used as the relative factor of the encoder's convexity threshold and of the
     * witness PSD-cone membership check. */
    double semi_tol_approx;
    int max_iter_intpnt, max_iter_simplex, intpnt_max_cor;
    int log;
    int scaling;                       /* LP/QP row/column equilibration on/off */
    double atruncatetol;               /* A truncation threshold (reference
                                        * get/putatruncatetol): stored, not
                                        * applied (this solver does not truncate A) */
    int mip_max_nodes;                 /* branch & bound node cap */
    int num_threads;                   /* threads for probing / strong branching */
    int concurrent_time;               /* concurrent optimizer: 0 tie-break, 1 fastest */
    double mip_tol_abs_gap, mip_tol_rel_gap;   /* B&B pruning gap */
    double mip_tol_inther;             /* integrality threshold */
    double mip_tol_feas;               /* what an incumbent must measure in */

    PRIMALlogcb logcb; void *loghandle;
    FILE *logfile;                              /* stream to file (linkfiletotaskstream) */
    PRIMALcallbackcb cbfn; void *cbhandle;      /* general callback (events) */
    PRIMALresponsecb respfn; void *resphandle;  /* response callback (errors) */

    /* progress callback (per-iteration / solution-update info strings) */
    PRIMALprogresscb progcb; void *proghandle;
    char infoname[64];

    /* solution */
    int has_sol;
    PRIMALsolstae solsta;
    /* Problem status is derived from solsta (prosta_of), exactly as the
     * reference pairs the two. This field carries the one outcome the pairing
     * does not cover: a mixed-integer problem found infeasible, where the
     * reference reports PRIM_INFEAS with no solution status at all. */
    PRIMALprostae prosta;
    PRIMALrescodee last_rc;
    double opt_time;   /* CPU seconds of the last optimize (getdouinf) */
    double *x, *y, *slc, *suc, *slx, *sux;
    double *snx;          /* s_n^x: conic multipliers per variable (storage) */
    double *xc;           /* x^c: row activity, if set by hand */
    int has_xc;
    double pobj, dobj;
    double mip_bound; /* Global numerical search bound, original objective sense. */
    int mip_bound_defined, mip_result;
    int result_stale;
    int derived_dirty, derived_syncing;
    /* basis (solvebasis): status keys for rows/variables */
    PRIMALstakeye *skc, *skx;
    int skccap, skxcap;   /* capacity of skc (numcon) and skx (numvar) */
    /* Farkas certificates, in the user's space, published only when measured
     * (has_dray: infeasible, b'y > 0 and A'y <= 0; has_pray: unbounded) */
    double *pray, *dray;
    int has_pray, has_dray;
    /* basis solve (initbasissolve/solvewithbasis): dense LU factorization of B */
    void *basis_lu;
    int *basis_vec; int basis_n;
    /* warm start (PRIMAL_putxx/PRIMAL_puty) */
    double *warm_x, *warm_y; int has_warm;
    int warmxcap, warmycap;   /* capacity of warm_x (numvar) and warm_y (numcon) */
    /* One capacity per lazy table. These five are allocated at the numvar /
     * numcon of the FIRST write and are indexed afterwards by the CURRENT one,
     * so a later PRIMAL_appendvars/appendcons leaves them short: the writer
     * overruns the block and the reader walks off it. Growing them is what the
     * model arrays (c, cols, bkc, ...) already do; the capacity is the one
     * number that says whether a table still has the model's shape. */
    int qcon_cap;             /* length of the t->qcon row-pointer array */
};
typedef struct {
    int id;
    int kind;
    size_t off;
    double dflt, lo, hi;
    const char *name;   /* the symbolic name (reference getparamname/whichparam) */
} PrimalParam;
/* row count of PRIMAL_PARAMS: an extern int (not a sizeof macro) because
 * the array has incomplete type in every unit but primal_core.c. The single
 * definition below the table keeps value and table in one place. */
extern const int primal_nparam_count;
#define PRIMAL_NPARAM (primal_nparam_count)
#define SDP_MAXROUND 200
#define SDP_BIGM 1e6
typedef struct {
    double *lx, *ux;
    double bound;   /* lower bound of the parent: the best-bound key */
} MipNode;
typedef struct {
    PRIMALtask_t t; int s, nvar, ncon;
    const double *lx, *ux, *lc, *uc;
    int *bins; int start, end;
    signed char *fix;
} ProbeJob;
typedef struct {
    PRIMALtask_t trelax; int s, nvar;
    const double *lx, *ux, *lc, *uc, *x;
    const int *cand; int start, end;
    double *score;
} SBJob;
typedef struct {
    const int *Aptr, *Arow; const double *Aval;
    const int *Qptr, *Qrow; const double *Qval;
    int m, n; const double *b, *c; PRIMALtask_t t;
    double *xt, *ystd, *zst; const double *x0, *y0;
    int method; double *dray, *pray; int status; double elapsed;
} ConcJob;
typedef struct { int n; int *idx; int *bkc; double *blc, *buc; } SavedRows;
typedef struct { PRIMALtask_t t; int s; PRIMALrescodee rc; } MipKidJob;
enum { P_INT = 0, P_DOU = 1, P_DOUI = 2 };
enum { STD_OPT = 0, STD_INFEASIBLE, STD_UNBOUNDED, STD_STALLED, STD_MEMORY };

/* the model store: defined once in primal_core.c */
extern const PrimalParam PRIMAL_PARAMS[];
extern const char *const cone_kind_name[];
extern const int cone_kind_count;
extern const char *const dinf_names[];
extern const char *const iinf_names[];
extern const char *const liinf_names[];

/* shared helpers: each defined once in the unit noted, used from more
 * than one unit (verbatim moves from primal.c, `static` dropped) */
/* Inner product of the bar objective matrix C with the symmetric matrix X. */
double barC_dot(const PRIMALtask_t t, const double *X);
/* Verdict for a point sitting on the solver's own SDP_BIGM cap: measures a
 * recession direction of the model and publishes it when it exists. */
PRIMALrescodee bar_cap_verdict(PRIMALtask_t t, int s, double **symPq,
                                      int nb, const int *barOff, const int *barPq,
                                      const double *z);
/* Copies the bar store (symmetric registry, bar variables, barA/barC) into dst. */
PRIMALrescodee bar_copy(PRIMALtask_t src, PRIMALtask_t dst);
/* Dual violation of bar block j: max(0, -lambda_min(barsj[j])). */
double bar_dual_viol(const PRIMALtask_t t, int j);
/* Smallest eigenvalue of the d x d symmetric matrix A; pmax gets the largest. */
double bar_min_eig(int d, const double *A, double *pmax);
/* Packed lower-triangle position of the (p,q) entry of a d x d symmetric block. */
int bar_pack(int d, int p, int q);
/* Total stored size of the bar variables (sum of dim_j*dim_j). */
int bar_tot(const PRIMALtask_t t);
/* Expands a bound key plus bl/bu into the explicit (lo,up) range. */
void bound_range(PRIMALboundkeye bk, double bl, double bu, double *lo, double *up);
/* Derives implied bounds from the scalar rows; returns 1 when the box empties. */
int bound_tighten(PRIMALtask_t t, int nvar, int ncon,
                         double *lx, double *ux, const double *lc, const double *uc,
                         int *lo_row, double *lo_coef, int *up_row, double *up_coef);
/* Builds the CSC form of A into freshly allocated ptr/sub/val arrays. */
int build_csc(PRIMALtask_t t, int **ptr_out, int **sub_out, double **val_out);
/* Fires the general callback with the given event code. */
void cb_fire(PRIMALtask_t t, PRIMALcallbackcodee code);
/* Signed slack of vector v in the dual cone K* of type ct. */
double cone_dual_signed_slack(int ct, double a, const double *v, int nk);
/* Worst dual-cone violation measured over the task's cones; returns it. */
double cone_dual_worst(PRIMALtask_t t, int s, int *nmeas, int verb);
/* Signed slack of vector v in the cone of type ct. */
double cone_signed_slack(int ct, double a, const double *v, int nk);
/* Temporarily removes redundant rows before the conic route, recording them in sv. */
void conic_presolve_apply(PRIMALtask_t t, SavedRows *sv);
/* Restores the rows removed by conic_presolve_apply. */
void conic_presolve_restore(PRIMALtask_t t, SavedRows *sv);
/* Materializes and returns the dense symmetric objective Q, or NULL. */
double *ensure_dense_qobj(PRIMALtask_t t);
/* Allocates the task's model arrays; returns a resource code. */
PRIMALrescodee ensure_size(PRIMALtask_t t);
/* Effective iteration cap: v <= 0 means no limit (INT_MAX). */
int iter_cap(int v);
/* Begins an iteration/solution-update callback scope for the task. */
void iter_cb_begin(PRIMALtask_t t);
/* Ends the iteration callback scope opened by iter_cb_begin. */
void iter_cb_end(void);
/* Grows a lazily-sized table to at least need elements, updating *cap. */
void *lazy_grow(void *tbl, int *cap, int need, size_t esz);
/* Builds CG/cover cuts into a clone tc of the task; returns the number built. */
int mip_build_cuts(PRIMALtask_t t, int s,
                           const double *lc0, const double *uc0,
                           PRIMALtask_t *tc_out, int *nconf_out);
/* True when the cut model has all-integer data, so a pure Gomory cut is valid. */
int mip_gomory_applicable(PRIMALtask_t tc);
/* Adds one Gomory cut from the final simplex tableau; returns the number added. */
int mip_gomory_round(PRIMALtask_t tc, int s);
/* Worker for a parallel branch-and-bound child node (pthread entry point). */
void *mip_kid_run(void *arg);
/* True when point w satisfies the node bounds and rows within ftol/itol. */
int mip_point_measures(PRIMALtask_t t, const double *lx, const double *ux,
                              const double *lc, const double *uc, const double *w,
                              double ftol, double itol);
/* Solves the LP/QP relaxation of a node over the given box; xout gets the point. */
int mip_relax(PRIMALtask_t t, int s, const double *lx, const double *ux,
                     const double *lc, const double *uc,
                     double *xout, double *pmin);
/* Solves the conic/SDP relaxation of a node; barX_out gets the bar block. */
int mip_relax_conic(PRIMALtask_t t, int s, PRIMALenv_t env2,
                           const double *lx, const double *ux,
                           const double *lc, const double *uc,
                           double *xout, double *pmin, double *barX_out);
/* Builds and measures an infeasibility/unboundedness witness of the model. */
int model_lp_witness(PRIMALtask_t t, int s, double **symPq, int mode,
                            const char *tag, double *rayout, double *dualout);
/* Invalidates the published point and verdict after the model dimensions grow. */
void model_resized(PRIMALtask_t t);
/* Finds the index of name in the table; ERR_ARG if it is absent. */
PRIMALrescodee name_find(const char **names, int n, const char *name, int *idx);
/* Length in bytes of a name (0 for NULL or empty). */
int name_len_of(const char *s);
/* Sets entry idx of the name table, enforcing uniqueness inside the table. */
PRIMALrescodee name_put(char **names, int n, int idx, const char *name);
/* Allocates and zeroes the per-solve buffers before optimization. */
PRIMALrescodee opt_prepare(PRIMALtask_t t);
/* Dispatcher core: runs the applicable optimization routes in order. */
PRIMALrescodee opt_routes(PRIMALtask_t t);
/* Conic route (SOC/RQUAD/exp-power/bar) with the tangent-cut fallback. */
PRIMALrescodee optimize_conic(PRIMALtask_t t, int s);
/* Branch-and-bound route for models with integer variables. */
PRIMALrescodee optimize_mip(PRIMALtask_t t, int s);
/* Quadratic-objective/constraint route via the QCQP encoder. */
PRIMALrescodee optimize_quad(PRIMALtask_t t, int s);
/* Tangent-cuts (outer approximation) route for SDP models. */
PRIMALrescodee optimize_sdp(PRIMALtask_t t, int s);
/* Native primal-dual interior-point route for conic/SDP models. */
PRIMALrescodee optimize_sdp_ipm(PRIMALtask_t t, int s);
/* Copies the parameter fields from the task src into the task dst. */
void param_copy(struct PRIMAL_task_s *dst, struct PRIMAL_task_s *src);
/* Applies the declarative-table defaults to a task. */
void param_defaults(struct PRIMAL_task_s *t);
/* Looks up the parameter table row for (kind, id), or NULL if unknown. */
const PrimalParam *param_find(int kind, int id);
/* Returns a pointer to the task field named by table row d. */
void *param_slot(const PrimalParam *d, struct PRIMAL_task_s *t);
/* Worker for parallel binary-variable probing (pthread entry point). */
void *probe_worker(void *arg);
void mip_run_jobs(int n, size_t stride, void *jobs, void *(*worker)(void *));
/* Value of quadratic row i at w (linear plus quadratic terms). */
double quad_row_value(const PRIMALtask_t t, int i, const double *w);
/* Picks the better of two candidate ray vectors (own vs the last iterate). */
const double *ray_candidate(const double *own, const double *iter, int n);
/* Publishes a measured Farkas ray and its status; returns 1 on success. */
int ray_publish(PRIMALtask_t t, const StdForm *sf,
                       const double *rs, const double *ds,
                       double *yray, double *xray, int status);
/* Removes rows implied by the bounds; returns the number removed. */
int row_redundant(PRIMALtask_t t, int nvar, int ncon,
                         const double *lx, const double *ux, double *lc, double *uc);
/* Worker for parallel strong branching (pthread entry point). */
void *sb_worker(void *arg);
/* Returns the objective Q values scaled by s and by the dual ds. */
double *scaled_qvals(PRIMALtask_t t, double s, const double *ds);
/* Rebuilds one component of an SOC block's dual from the raw dual value. */
double soc_dual_component(const PRIMALtask_t t, int k, int i, double raw);
/* True when which is one of the declared solution keys. */
int sol_key_ok(PRIMALsolt which);
/* Runs simplex and IPM concurrently on one standard form; returns the winner. */
int solve_std_conc(const int *Aptr, const int *Arow, const double *Aval,
                          const int *Qptr, const int *Qrow, const double *Qval,
                          int m, int n, const double *b, const double *c, PRIMALtask_t t,
                          double *xt, double *ystd, double *zst,
                          const double *x0, const double *y0, double *dray, double *pray);
/* Solves one standard form on the route chosen by std_route_method. */
int solve_std_routed(const int *Aptr, const int *Arow, const double *Aval,
                            const int *Qptr, const int *Qrow, const double *Qval,
                            int m, int n, const double *b, const double *c,
                            PRIMALtask_t t, double *xt, double *ystd, double *zst,
                            const double *x0, const double *y0, int method,
                            double *dray, double *pray);
/* Chooses the solver route for a standard form from its class and size. */
int std_route_method(int hasQ, int m, int n, PRIMALtask_t t);
/* Computes Q x for the task's quadratic objective into out. */
void task_Qx(PRIMALtask_t t, const double *x, double *out);
/* Computes x' Q x for the task's quadratic objective. */
double task_xQx(PRIMALtask_t t, const double *x);
/* Writes one message to the task's log stream. */
void tlog(PRIMALtask_t t, const char *msg);
/* Writes one message to the task's progress stream. */
void tprog(PRIMALtask_t t, const char *msg);

int exp_member(int ct, int i);

double exp_factor(int ct, int i);

void model_changed(PRIMALtask_t t);

double row_activity(PRIMALtask_t t, int i, const double *x);

int sos_violation(PRIMALtask_t t, int k, const double *x, double tol, double *pivot);

PRIMALrescodee acc_store(PRIMALtask_t t, PRIMALint64t domidx, PRIMALint64t numafeidx,
                        const PRIMALint64t *afeidxlist, const PRIMALrealt *b,
                        PRIMALint64t rowbase, PRIMALint64t varbase);
PRIMALrescodee acc_sync(PRIMALtask_t t);
PRIMALrescodee derived_sync(PRIMALtask_t t);
PRIMALrescodee djc_sync(PRIMALtask_t t);
double djc_violation(PRIMALtask_t t, int d, const double *x);
PRIMALrescodee djc_copy(PRIMALtask_t s, PRIMALtask_t d);

#endif /* PRIMAL_PRIV_H */
