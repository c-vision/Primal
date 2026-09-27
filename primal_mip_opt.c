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
/* primal_mip_opt.c - optimize_mip (branch & bound).
 * Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Jobs are prepared by the caller. A failed creation executes the unchanged
 * job synchronously; only threads actually created may be joined. */
void mip_run_jobs(int n, size_t stride, void *jobs, void *(*worker)(void *)) {
    pthread_t threads[64]; int made[64] = {0};
    for (int k = 0; k < n; k++) {
        void *job = (char *)jobs+(size_t)k*stride;
        made[k] = pthread_create(&threads[k],NULL,worker,job) == 0;
        if (!made[k]) worker(job);
    }
    for (int k = 0; k < n; k++) if (made[k]) pthread_join(threads[k],NULL);
}

/* Run branch and bound over the node relaxations. Builds cut copies,
 * tightens bounds, probes binaries, then searches the tree until the
 * node cap or an optimal incumbent. Returns the result code. */
PRIMALrescodee optimize_mip(PRIMALtask_t t, int s) {
    t->mip_result = 1; t->mip_bound_defined = 0;
    int nvar = t->numvar, ncon = t->numcon;

    /* conic/quadratic/SDP node relaxations need a shadow env */
    int use_conic = (t->numcones > 0) || (t->has_qcon > 0) ||
                    (t->has_qobj && t->numcones > 0) || (t->numbarvar > 0);
    PRIMALenv_t senv = NULL;
    if (use_conic) {
        PRIMALrescodee rce = PRIMAL_makeenv(&senv, NULL);
        if (rce != PRIMAL_RES_OK) return rce;
    }
    /* Chvatal-Gomory cuts enter a copy used only by the relaxations; the
     * user model (numcon, A, bounds) never sees them. The copy has
     * `ncon+ncuts` rows, so the B&B row boxes (`lc`/`uc`) are allocated to
     * the same length and the cut entries are constant.
     * Pairwise-probing conflict cuts (via (a)) are computed BEFORE the clone
     * and travel with the others: they need the task row bounds, computed
     * here (lc0/uc0 over ncon), because the B&B lx/ux/lc/uc do not exist yet. */
    PRIMALtask_t tc = NULL;
    double *lc0 = NULL, *uc0 = NULL;
    if (getenv("GMB_MIP_CONFLICT") && !getenv("GMB_NO_MIP_CUTS") && ncon > 0) {
        lc0 = (double *)malloc((size_t)ncon * sizeof(double));
        uc0 = (double *)malloc((size_t)ncon * sizeof(double));
        if (lc0 && uc0) {
            for (int i = 0; i < ncon; i++)
                bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc0[i], &uc0[i]);
        } else { free(lc0); free(uc0); lc0 = NULL; uc0 = NULL; }
    }
    int cfcuts = 0;
    int cgcuts = mip_build_cuts(t, s, lc0, uc0, &tc, &cfcuts);   /* TOTAL, conflicts included */
    free(lc0); free(uc0);
    int gocuts = 0;
    if (!getenv("GMB_NO_MIP_CUTS")) {
        /* Gomory cuts from the tableau: a clone is needed (even with no CG
         * cuts) to leave the user model untouched. */
        if (!tc && mip_gomory_applicable(t)) {
            if (PRIMAL_clonetask(t, &tc) != PRIMAL_RES_OK) tc = NULL;
        }
        if (tc) gocuts = mip_gomory_round(tc, s);
    }
    int ncuts = cgcuts + gocuts;
    int nrelax = ncon + ncuts;
    PRIMALtask_t trelax = tc ? tc : t;
    if (cgcuts - cfcuts > 0) { char cb[64]; snprintf(cb, sizeof cb, "MIP: %d Chvatal-Gomory cuts\n", cgcuts - cfcuts); tlog(t, cb); }
    if (gocuts > 0) { char cb[64]; snprintf(cb, sizeof cb, "MIP: %d Gomory cuts\n", gocuts); tlog(t, cb); }
    if (cfcuts > 0) { char cb[64]; snprintf(cb, sizeof cb, "MIP: %d conflict cuts\n", cfcuts); tlog(t, cb); }
#define MIP_RELAX(tv, lxx, uxx, xo, pm, bx) \
    (use_conic ? mip_relax_conic(tv, s, senv, lxx, uxx, lc, uc, xo, pm, bx) \
               : mip_relax(tv, s, lxx, uxx, lc, uc, xo, pm))

    /* min-form bounds */
    double *lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc = (double *)malloc((size_t)(nrelax > 0 ? nrelax : 1) * sizeof(double));
    double *uc = (double *)malloc((size_t)(nrelax > 0 ? nrelax : 1) * sizeof(double));
    if (!lx || !ux || !lc || !uc) {
        free(lx); free(ux); free(lc); free(uc); PRIMAL_deletetask(&tc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int j = 0; j < nvar; j++) {
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
        /* A semi variable's domain is {0} u [l,u]. Solving the LP relaxation
         * over [l,u] alone DROPS the deactivation, so its optimum is not a lower
         * bound for the node and pruning on it was unsound: min x over
         * {0}u[2,5] answered 2. Relaxing the root box to [0,u] puts the
         * deactivation back in the relaxation (a valid superset); the existing
         * gap-branch (0<x<l -> x=0 or x>=l) then removes the forbidden band, and
         * every child box -- x=0, or lx=l on the active side -- is a valid
         * relaxation of its own subregion. */
        int vt = t->vartype[j];
        if ((vt == PRIMAL_VAR_TYPE_SEMI_CONT || vt == PRIMAL_VAR_TYPE_SEMI_INT) &&
            lx[j] > 0.0)
            lx[j] = 0.0;
    }
    for (int i = 0; i < ncon; i++) bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc[i], &uc[i]);
    if (!getenv("GMB_NO_BOUND_TIGHTEN")) {
        int nbt = bound_tighten(t, nvar, ncon, lx, ux, lc, uc, NULL, NULL, NULL, NULL);
        if (nbt > 0 && getenv("GMB_DBG")) {
            char cb[64];
            snprintf(cb, sizeof cb, "MIP: %d tightened bounds\n", nbt);
            tlog(t, cb);
        }
    }
    for (int k = 0; k < ncuts; k++) {
        PRIMALboundkeye bk = PRIMAL_BK_FR; double lo = 0.0, up = 0.0;
        PRIMAL_getconbound(tc, ncon + k, &bk, &lo, &up);
        lc[ncon + k] = lo; uc[ncon + k] = up;
    }

    /* Discard "toxic" cuts: the root relaxation WITH cuts must not worsen
     * the bound (higher pmin in min form) relative to WITHOUT. A cut row
     * slicing off the continuous-relaxation optimum can stall the solver on
     * a suboptimal optimum (reproducible bug: a binary knapsack with the
     * cover x0+x1<=1 gave pmin=-8.5 instead of -9). Kill-switch
     * GMB_NO_MIP_CUT_CHECK=1. */
    if (tc && ncuts > 0 && !use_conic && t->num_threads <= 1 &&
        !getenv("GMB_NO_MIP_CUT_CHECK")) {
        double *xo = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
        double pw = 0.0, pn = 0.0;
        int sw = xo ? MIP_RELAX(tc, lx, ux, xo, &pw, NULL) : 3;
        int sn = xo ? MIP_RELAX(t, lx, ux, xo, &pn, NULL) : 3;
        if (xo && sw == 0 && sn == 0 && pw > pn + 1e-9) {
            PRIMAL_deletetask(&tc); tc = NULL;
            cgcuts = 0; gocuts = 0; ncuts = 0; nrelax = ncon;
            trelax = t;
            tlog(t, "MIP: cuts discarded (node relaxation worsened)\n");
        }
        free(xo);
    }

    /* probing: for every binary variable (limited to the first 20) try
     * x_j = 0 and x_j = 1; if one side is infeasible, the variable is fixed. The
     * MIP publishes no duals, so it is sound. */
    if (!getenv("GMB_NO_MIP_PROBING")) {
        int bins[64], nbin = 0;
        for (int j = 0; j < nvar && nbin < 20; j++)
            if (t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN) bins[nbin++] = j;
        if (nbin > 0) {
            signed char fix[64];
            int nth = t->num_threads > 1 ? t->num_threads : 1;
            if (nth > nbin) nth = nbin;
            if (nth > 1) {
                ProbeJob jobs[64]; int njob = 0;
                int chunk = (nbin + nth - 1) / nth;
                for (int k = 0; k < nth; k++) {
                    int st = k * chunk, en = st + chunk;
                    if (en > nbin) en = nbin;
                    if (st >= en) continue;
                    jobs[k].t = t; jobs[k].s = s; jobs[k].nvar = nvar; jobs[k].ncon = ncon;
                    jobs[k].lx = lx; jobs[k].ux = ux; jobs[k].lc = lc; jobs[k].uc = uc;
                    jobs[k].bins = bins; jobs[k].start = st; jobs[k].end = en; jobs[k].fix = fix;
                    njob++;
                }
                mip_run_jobs(njob,sizeof(ProbeJob),jobs,probe_worker);
            } else {
                ProbeJob jb;
                jb.t = t; jb.s = s; jb.nvar = nvar; jb.ncon = ncon;
                jb.lx = lx; jb.ux = ux; jb.lc = lc; jb.uc = uc;
                jb.bins = bins; jb.start = 0; jb.end = nbin; jb.fix = fix;
                probe_worker(&jb);
            }
            int nfix = 0;
            for (int k = 0; k < nbin; k++) {
                int v = bins[k];
                if (fix[k] == 1) { lx[v] = ux[v] = 1.0; nfix++; }
                else if (fix[k] == 0) { lx[v] = ux[v] = 0.0; nfix++; }
            }
            if (nfix > 0 && getenv("GMB_DBG")) {
                char cb[64];
                snprintf(cb, sizeof cb, "MIP: %d probed fixings\n", nfix);
                tlog(t, cb);
            }
        }
    }

    /* node stack (DFS) */
    int cap = 64, sp = 0;
    MipNode *stk = (MipNode *)malloc((size_t)cap * sizeof(MipNode));
    double *bestx = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    if (!stk || !bestx) {
        free(stk); free(bestx); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    int nbartot = bar_tot(t);
    double *bestX = (nbartot > 0) ? (double *)calloc((size_t)nbartot, sizeof(double)) : NULL;
    double *barXbuf = (nbartot > 0) ? (double *)malloc((size_t)nbartot * sizeof(double)) : NULL;
    if (nbartot > 0 && (!bestX || !barXbuf)) {
        free(bestX); free(barXbuf);
        free(stk); free(bestx); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    stk[sp].lx = (double *)malloc((size_t)nvar * sizeof(double));
    stk[sp].ux = (double *)malloc((size_t)nvar * sizeof(double));
    if (!stk[sp].lx || !stk[sp].ux) {
        free(stk[sp].lx); free(stk[sp].ux); free(stk); free(bestx);
        free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    memcpy(stk[sp].lx, lx, (size_t)nvar * sizeof(double));
    memcpy(stk[sp].ux, ux, (size_t)nvar * sizeof(double));
    stk[sp].bound = -INF;
    sp++;

    double best = INF;
    const double itol = t->mip_tol_inther;   /* PRIMAL_DPAR_MIP_TOL_INTHER */
    const double ftol = t->mip_tol_feas;     /* PRIMAL_DPAR_MIP_TOL_FEAS */
    double *xs = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *flx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *flux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!xs || !flx || !flux) {
        for (int q = 0; q < sp; q++) { free(stk[q].lx); free(stk[q].ux); }
        free(xs); free(flx); free(flux);
        free(stk); free(bestx); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    long nodes = 0;
    int root_status = 0;   /* remember root relaxation status */

    /* initial incumbent from PRIMAL_putxx (mioinitsol): accepted only if it
     * measures in the model, by the same predicate a branch-and-bound incumbent
     * has to pass. */
    if (t->has_warm && t->warm_x) {
        double *w = t->warm_x;
        int ok = mip_point_measures(t, lx, ux, lc, uc, w, t->mip_tol_feas, itol);
        if (ok) {
            double p = t->cfix;
            for (int j = 0; j < nvar; j++) p += t->c[j] * w[j];
            if (t->has_qobj) p += 0.5 * task_xQx(t, w);
            best = s * p;   /* min-form objective */
            memcpy(bestx, w, (size_t)nvar * sizeof(double));
            tlog(t, "MIP initial solution accepted\n");
        } else {
            tlog(t, "MIP initial solution rejected\n");
        }
    }
    t->has_warm = 0;

    /* Parallel B&B (num_threads > 1): solve the root, branch on the most
     * fractional and solve the two children in two threads (clone of the task
     * with the child's bound). If the root is integral, continue sequentially. */
    if (t->num_threads > 1) {
        double *xr = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
        int bjr = -1; double vv = 0.0, root_bound = -INF;
        if (xr) {
            double pminr = 0.0;
            int str = MIP_RELAX(trelax, lx, ux, xr, &pminr, NULL);
            double bfr = -1.0;
            if (str == 0) root_bound = pminr;
            if (str == 0)
                for (int j = 0; j < nvar; j++) {
                    if (t->vartype[j] != PRIMAL_VAR_TYPE_INT &&
                        t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) continue;
                    double fl = floor(xr[j] + 1e-9), fr = xr[j] - fl;
                    if (fr < itol || (1.0 - fr) < itol) continue;
                    if (fr > 0.5) fr = 1.0 - fr;
                    if (fr > bfr) { bfr = fr; bjr = j; }
                }
            if (bjr >= 0) vv = xr[bjr];
            free(xr);
        }
        if (bjr >= 0) {
            PRIMALtask_t kids[2] = {NULL, NULL};
            for (int side = 0; side < 2; side++) {
                if (PRIMAL_clonetask(t, &kids[side]) != PRIMAL_RES_OK) { kids[side] = NULL; continue; }
                kids[side]->num_threads = 1;
                kids[side]->opt_deadline = t->opt_deadline;
                kids[side]->mip_deadline = t->mip_deadline;
                if (opt_prepare(kids[side]) != PRIMAL_RES_OK) { PRIMAL_deletetask(&kids[side]); continue; }
                if (kids[side]->mip_max_nodes > 1) kids[side]->mip_max_nodes /= 2;
                kids[side]->bkx[bjr] = PRIMAL_BK_RA;
                if (side == 0) kids[side]->bux[bjr] = floor(vv);
                else kids[side]->blx[bjr] = floor(vv) + 1.0;
            }
            pthread_t th[2]; MipKidJob kj[2]; int made[2] = {0, 0};
            for (int side = 0; side < 2; side++) {
                kj[side].t = kids[side]; kj[side].s = s; kj[side].rc = PRIMAL_RES_ERR_ARG;
                if (!kids[side]) continue;
                if (pthread_create(&th[side], NULL, mip_kid_run, &kj[side]) == 0) made[side] = 1;
                else kj[side].rc = optimize_mip(kids[side], s);
            }
            for (int side = 0; side < 2; side++) if (made[side]) pthread_join(th[side], NULL);
            int bside = -1; double bkey = best;
            for (int side = 0; side < 2; side++) {
                if (!kids[side] || !kids[side]->has_sol) continue;
                double key = s * kids[side]->pobj;
                if (key < bkey) { bkey = key; bside = side; }
            }
            PRIMALrescodee prc;
            int complete = 1;
            double bound = INF;
            for (int side = 0; side < 2; side++) {
                if (!kids[side]) { complete = 0; bound = fmin(bound, root_bound); continue; }
                if (kj[side].rc == PRIMAL_RES_ERR_INFEASIBLE) continue;
                if (kj[side].rc != PRIMAL_RES_OK) complete = 0;
                double child_bound = kids[side]->mip_bound_defined
                    ? fmax(root_bound, s * kids[side]->mip_bound) : root_bound;
                bound = fmin(bound, child_bound);
            }
            t->mip_bound_defined = isfinite(bound);
            t->mip_bound = s * bound;
            if (bside >= 0 || best < INF) {
                memcpy(t->x, bside >= 0 ? kids[bside]->x : bestx, (size_t)nvar * sizeof(double));
                for (int j = 0; bside >= 0 && j < t->numbarvar; j++) {
                    int d = t->barDim[j];
                    memcpy(t->barx[j], kids[bside]->barx[j], (size_t)d * d * sizeof(double));
                }
                t->pobj = bside >= 0 ? kids[bside]->pobj : s * best;
                t->solsta = complete ? PRIMAL_SOL_STA_INTEGER_OPTIMAL : PRIMAL_SOL_STA_PRIM_FEAS;
                t->prosta = PRIMAL_PRO_STA_PRIM_FEAS;
                t->has_sol = 1; t->dobj = t->mip_bound_defined ? t->mip_bound : NAN;
                prc = complete ? PRIMAL_RES_OK : PRIMAL_RES_TRM_MAX_ITER;
            } else {
                int inf = 1;
                for (int side = 0; side < 2; side++)
                    if (!kids[side] || kj[side].rc != PRIMAL_RES_ERR_INFEASIBLE) inf = 0;
                t->solsta = PRIMAL_SOL_STA_UNKNOWN;
                t->prosta = inf ? PRIMAL_PRO_STA_PRIM_INFEAS : PRIMAL_PRO_STA_UNKNOWN;
                prc = inf ? PRIMAL_RES_ERR_INFEASIBLE : PRIMAL_RES_TRM_MAX_ITER;
            }
            for (int side = 0; side < 2; side++) if (kids[side]) PRIMAL_deletetask(&kids[side]);
            free(stk[0].lx); free(stk[0].ux); free(stk); free(bestx); free(bestX); free(barXbuf);
            free(xs); free(flx); free(flux); free(lx); free(ux); free(lc); free(uc);
            PRIMAL_deletetask(&tc);
            if (senv) PRIMAL_deleteenv(&senv);
            return prc;
        }
    }

    int deadline_hit = 0, incomplete = 0;
    double open_bound = INF, closed_bound = INF, active_bound = INF;
    while (sp > 0 && nodes < t->mip_max_nodes) {
        if (t->mip_deadline >= 0.0 && (double)clock() >= t->mip_deadline) { deadline_hit = 1; break; }
        /* best-bound: expand the node with the lowest bound (the parent leaves
         * it in `bound`), not the deepest. Among equal bounds the most recent
         * wins, which is the previous DFS order. */
        int besti = sp - 1;
        for (int q = sp - 2; q >= 0; q--) if (stk[q].bound < stk[besti].bound) besti = q;
        MipNode nd = stk[besti];
        stk[besti] = stk[--sp];
        active_bound = nd.bound;
        double *x = (double *)malloc((size_t)nvar * sizeof(double));
        double pmin;
        if (!x) { incomplete = 1; free(nd.lx); free(nd.ux); break; }
        int st = MIP_RELAX(trelax, nd.lx, nd.ux, x, &pmin, NULL);
        nodes++;
        if (nodes == 1) root_status = st;
        if (st != 0) {   /* infeasible (or unbounded child): prune */
            if (st != 1) { incomplete = 1; open_bound = fmin(open_bound, nd.bound); }
            active_bound = INF;
            if (st == 3 && getenv("GMB_DBG"))
                fprintf(stderr, "  [mip] node=%ld relaxation gave no answer\n", nodes);
            free(x); free(nd.lx); free(nd.ux);
            continue;
        }
        active_bound = pmin;
        /* Round the root solution, pin integers and solve the restricted model
         * for an incumbent. Its objective is not the original node's bound. */
        if (nodes == 1 && best == INF) {
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                          (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol));
                xs[j] = iv ? floor(x[j] + 0.5) : x[j];
            }
            memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
            memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                if (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                    (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol))
                    flx[j] = flux[j] = xs[j];
            }
            double pfb = 0.0;
            int rst = MIP_RELAX(trelax, flx, flux, xs, &pfb, NULL);
            if (rst == 0 && mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs, ftol, itol)) {
                double p = t->cfix;
                for (int j = 0; j < nvar; j++) p += t->c[j] * xs[j];
                if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                if (s * p < best) {
                    best = s * p;
                    memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                }
            }
        }
        /* Diving (fractional diving, gated GMB_MIP_DIVING, default off): from
         * the node point, fix in turn the most fractional integer to its
         * rounding (clamped to the node bounds) and re-solve the reduced
         * relaxation. If a feasible point that improves the incumbent emerges,
         * accept it. PRIMAL heuristic: it updates only the incumbent, so by
         * construction it cannot change the optimum (only the path).
         * DEDICATED buffers (dlx/dux/dx): the four failed versions shared
         * flx/flux/xs with RINS/RENS/pump; here nothing is shared, so the side
         * effect of T84 C5 can no longer come from there.
         * Like the other primal heuristics it is ON by default with kill-switch
         * GMB_NO_MIP_DIVING=1 (the family has RINS/RENS/FPUMP/LS in the same
         * style): the suite is green with the default on and the 15 MIP samples
         * with stdout identical to off (measured 2026-09-25). */
        if (!getenv("GMB_NO_MIP_DIVING") && (nodes == 1 || (nodes % 8) == 3)) {
            double *dlx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
            double *dux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
            double *dx  = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
            if (dlx && dux && dx) {
                memcpy(dlx, nd.lx, (size_t)nvar * sizeof(double));
                memcpy(dux, nd.ux, (size_t)nvar * sizeof(double));
                memcpy(dx, x, (size_t)nvar * sizeof(double));
                int dives = nvar < 10 ? nvar : 10;
                for (int d = 0; d < dives; d++) {
                    int bj = -1; double bfrac = 0.0;
                    for (int j = 0; j < nvar; j++) {
                        int vt = t->vartype[j];
                        int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                                  (vt == PRIMAL_VAR_TYPE_SEMI_INT && dx[j] > ftol));
                        if (!iv) continue;
                        double fr = fabs(dx[j] - floor(dx[j] + 0.5));
                        if (fr > 1e-4 && fr > bfrac) { bfrac = fr; bj = j; }
                    }
                    if (bj < 0) break;   /* nothing more fractional: the point is integral */
                    double r = floor(dx[bj] + 0.5);
                    if (r < dlx[bj]) r = dlx[bj];
                    if (r > dux[bj]) r = dux[bj];
                    r = floor(r + 0.5);   /* it must stay an INTEGER after the clamp */
                    if (r < dlx[bj] || r > dux[bj]) break;   /* the nearest integer is outside the node */
                    double oj = dlx[bj], okk = dux[bj];
                    if (r == oj && r == okk) break;   /* the jump does not tighten the box */
                    dlx[bj] = dux[bj] = r;
                    double pdv = 0.0;
                    int rst = MIP_RELAX(trelax, dlx, dux, dx, &pdv, NULL);
                    if (rst != 0) break;   /* the jump made the node empty: stop */
                    /* The published incumbent must be REALLY integral: INTHER
                     * (t->mip_tol_inther) declares the RELAXATION integral to
                     * decide its branch, but here a point is delivered, so
                     * integrality is judged with the same tight tolerance as
                     * feasibility (T84 C4: with itol=0.6 a fractional vertex
                     * must not come out as INTEGER_OPTIMAL). */
                    if (mip_point_measures(t, nd.lx, nd.ux, lc, uc, dx, ftol, ftol)) {
                        double p = t->cfix;
                        for (int j = 0; j < nvar; j++) p += t->c[j] * dx[j];
                        if (t->has_qobj) p += 0.5 * task_xQx(t, dx);
                        if (s * p < best - 1e-9) {
                            best = s * p;
                            memcpy(bestx, dx, (size_t)nvar * sizeof(double));
                            tlog(t, "MIP: diving incumbent\n");
                        }
                        break;
                    }
                }
            }
            free(dlx); free(dux); free(dx);
        }
        /* RINS (Relaxation Induced Neighborhood Search): with an incumbent in
         * hand, fix the integers on which the relaxation and the incumbent AGREE
         * and solve the reduced relaxation -- a feasible point of the node.
         * Standard primal heuristic (MOSEK has RINS/RENS/feasibility pump);
         * here it is the second, after the root rounding. Kill-switch
         * GMB_NO_MIP_RINS=1. */
        if (!getenv("GMB_NO_MIP_RINS") && best < INF && (nodes == 1 || (nodes % 8) == 0)) {
            memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
            memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
            int agree = 0;
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                          (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol));
                double r = floor(bestx[j] + 0.5);
                if (iv && fabs(x[j] - r) < 1e-6) { flx[j] = flux[j] = r; agree++; }
            }
            if (agree > 0) {
                double prb = 0.0;
                int rst = MIP_RELAX(trelax, flx, flux, xs, &prb, NULL);
                if (rst == 0 && mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs, ftol, itol)) {
                    double p = t->cfix;
                    for (int j = 0; j < nvar; j++) p += t->c[j] * xs[j];
                    if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                    if (s * p < best) {
                        best = s * p;
                        memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                    }
                }
            }
        }
        /* RENS (Relaxation Enforced Neighborhood Search): fix the ALREADY
         * near-integral integers in the relaxation (|x_j - round| < 1e-4) and
         * solve the reduced one. No incumbent needed: it uses the node point.
         * Kill-switch GMB_NO_MIP_RENS. */
        if (!getenv("GMB_NO_MIP_RENS") && (nodes == 1 || (nodes % 8) == 4)) {
            memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
            memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
            int fixed = 0;
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                          (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol));
                double r = floor(x[j] + 0.5);
                if (iv && fabs(x[j] - r) < 1e-4) { flx[j] = flux[j] = r; fixed++; }
            }
            if (fixed > 0) {
                double prb = 0.0;
                int rst = MIP_RELAX(trelax, flx, flux, xs, &prb, NULL);
                if (rst == 0 && mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs, ftol, itol)) {
                    double p = t->cfix;
                    for (int j = 0; j < nvar; j++) p += t->c[j] * xs[j];
                    if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                    if (s * p < best) {
                        best = s * p;
                        memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                    }
                }
            }
        }
        /* Feasibility pump (one-shot version): round ALL the integers of the
         * node point, fix them and solve the relaxation on the continuous ones
         * only. If the resulting point is feasible for the MIP, update the
         * incumbent. Unlike RENS (which fixes only the near-integers) the pump
         * fixes every integer, so the relaxation is a pure LP on the continuous
         * variables. Kill-switch GMB_NO_MIP_FPUMP=1. */
        if (!getenv("GMB_NO_MIP_FPUMP") && (nodes == 1 || (nodes % 8) == 2)) {
            memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
            memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
            int fixed = 0;
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                          (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol));
                if (!iv) continue;
                double r = floor(x[j] + 0.5);
                if (vt == PRIMAL_VAR_TYPE_INT_BIN) { if (r < 0.0) r = 0.0; if (r > 1.0) r = 1.0; }
                if (r < nd.lx[j]) r = nd.lx[j];
                if (r > nd.ux[j]) r = nd.ux[j];
                if (fabs(r - floor(r + 0.5)) > 1e-9) continue;   /* non-integral clamp: skip */
                flx[j] = flux[j] = r; fixed++;
            }
            if (fixed > 0) {
                double prb = 0.0;
                int rst = MIP_RELAX(trelax, flx, flux, xs, &prb, NULL);
                if (rst == 0 && mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs, ftol, itol)) {
                    double p = t->cfix;
                    for (int j = 0; j < nvar; j++) p += t->c[j] * xs[j];
                    if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                    if (s * p < best) {
                        best = s * p;
                        memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                    }
                }
            }
        }
        /* Local search: from the best incumbent try the flip of every binary
         * variable; if the point stays feasible and improves, accept it. Only
         * feasible improvements, so sound. Kill-switch
         * GMB_NO_MIP_LOCALSEARCH=1. */
        if (!getenv("GMB_NO_MIP_LOCALSEARCH") && best < INF && (nodes == 1 || (nodes % 8) == 6)) {
            for (int round = 0; round < 4; round++) {
                int improved = 0;
                for (int j = 0; j < nvar; j++) {
                    if (t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) continue;
                    memcpy(xs, bestx, (size_t)nvar * sizeof(double));
                    xs[j] = 1.0 - xs[j];
                    if (!mip_point_measures(t, lx, ux, lc, uc, xs, ftol, itol)) continue;
                    double p = t->cfix;
                    for (int k = 0; k < nvar; k++) p += t->c[k] * xs[k];
                    if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                    if (s * p < best - 1e-9) {
                        best = s * p;
                        memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                        improved = 1;
                    }
                }
                if (!improved) break;
            }
        }
        double gap = t->mip_tol_abs_gap;
        if (t->mip_tol_rel_gap > 0.0 && best < INF) {
            double rg = t->mip_tol_rel_gap * (1.0 + fabs(best));
            if (rg > gap) gap = rg;
        }
        if (best < INF && pmin >= best - gap) {
            closed_bound = fmin(closed_bound, pmin); active_bound = INF;
            free(x); free(nd.lx); free(nd.ux); continue; }

        /* branching priority:
         * 1. semi-continuous/integer violated (0 < x < l): two children
         *    { x = 0 (ux=0), x >= l (lx=l) }
         * 2. SOS violation: partition the member set by weight order
         * 3. most fractional integer variable (as before) */
        int bj = -1; double bfrac = -1.0;
        int bsemi = -1;               /* semi variable to branch on */
        int bsos = -1;                /* SOS constraint to branch on */
        /* semi-continuous/semi-integer check: x_j must be 0 or >= l_j */
        for (int j = 0; j < nvar && bsemi < 0; j++) {
            if (t->vartype[j] != PRIMAL_VAR_TYPE_SEMI_CONT &&
                t->vartype[j] != PRIMAL_VAR_TYPE_SEMI_INT) continue;
            double l = t->blx[j];
            if (x[j] > 1e-7 && x[j] < l - 1e-7) bsemi = j;
        }
        double sos_pivot = 0.0;
        int invalid_sos = 0;
        for (int k = 0; k < t->numsos && bsemi < 0 && bsos < 0; k++) {
            int violation = sos_violation(t, k, x, ftol, &sos_pivot);
            if (violation < 0) { invalid_sos = 1; break; }
            if (violation) bsos = k;
        }
        if (invalid_sos) {
            incomplete = 1; free(x); free(nd.lx); free(nd.ux); break;
        }
        /* integer check: collect the fractional candidates (for strong
         * branching) and keep the most fractional as default */
        {
            int cand[64]; double candf[64]; int ncand = 0;
            for (int j = 0; j < nvar; j++) {
                if (t->vartype[j] != PRIMAL_VAR_TYPE_INT && t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN)
                    continue;
                double fl = floor(x[j] + 1e-9);   /* snap near-integers */
                double fr = x[j] - fl;
                if (fr < itol || (1.0 - fr) < itol) continue;
                if (fr > 0.5) fr = 1.0 - fr;
                if (fr > bfrac) { bfrac = fr; bj = j; }
                if (ncand < 64) { cand[ncand] = j; candf[ncand] = fr; ncand++; }
            }
            /* strong branching: on the first K candidates by fraction solve
             * the two children and choose the highest lowest bound. Limited to
             * the first nodes, because it costs 2K LPs per node. */
            if (ncand > 1 && nodes < 200) {
                int K = ncand < 3 ? ncand : 3;
                double bestscore = -INF; int bestj = -1;
                for (int a2 = 0; a2 < K; a2++) {   /* selection of the K largest */
                    int mi = a2;
                    for (int b2 = a2 + 1; b2 < ncand; b2++) if (candf[b2] > candf[mi]) mi = b2;
                    int tj = cand[a2]; cand[a2] = cand[mi]; cand[mi] = tj;
                    double tf = candf[a2]; candf[a2] = candf[mi]; candf[mi] = tf;
                }
                {
                    double scores[64];
                    int nth = t->num_threads > 1 ? t->num_threads : 1;
                    if (nth > K) nth = K;
                    if (nth > 1) {
                        SBJob jobs[64]; int njob = 0;
                        int chunk = (K + nth - 1) / nth;
                        for (int k = 0; k < nth; k++) {
                            int st = k * chunk, en = st + chunk;
                            if (en > K) en = K;
                            if (st >= en) continue;
                            jobs[k].trelax = trelax; jobs[k].s = s; jobs[k].nvar = nvar;
                            jobs[k].lx = nd.lx; jobs[k].ux = nd.ux;
                            jobs[k].lc = lc; jobs[k].uc = uc; jobs[k].x = x;
                            jobs[k].cand = cand; jobs[k].start = st; jobs[k].end = en;
                            jobs[k].score = scores;
                            njob++;
                        }
                        mip_run_jobs(njob,sizeof(SBJob),jobs,sb_worker);
                    } else {
                        SBJob jb;
                        jb.trelax = trelax; jb.s = s; jb.nvar = nvar;
                        jb.lx = nd.lx; jb.ux = nd.ux; jb.lc = lc; jb.uc = uc; jb.x = x;
                        jb.cand = cand; jb.start = 0; jb.end = K; jb.score = scores;
                        sb_worker(&jb);
                    }
                    for (int a2 = 0; a2 < K; a2++)
                        if (scores[a2] > bestscore) { bestscore = scores[a2]; bestj = cand[a2]; }
                }
                if (bestj >= 0) bj = bestj;
            }
        }
        /* semi-integer: also needs integrality when x >= l */
        if (bj < 0 && bsemi < 0) {
            for (int j = 0; j < nvar; j++) {
                if (t->vartype[j] != PRIMAL_VAR_TYPE_SEMI_INT) continue;
                if (x[j] <= 1e-7) continue;              /* x = 0: fine */
                double fl = floor(x[j] + 1e-9);
                double fr = x[j] - fl;
                if (fr >= itol && (1.0 - fr) >= itol) { bfrac = fr > 0.5 ? 1.0 - fr : fr; bj = j; }
            }
        }

        if (bsemi >= 0) {
            /* branch on semi-continuous/integer variable: x = 0 or x >= l */
            int j = bsemi;
            double l = t->blx[j];
            free(x);
            if (sp + 2 > cap) {
                cap *= 2;
                MipNode *ns = (MipNode *)realloc(stk, (size_t)cap * sizeof(MipNode));
                if (!ns) { incomplete = 1; free(nd.lx); free(nd.ux); break; }
                stk = ns;
            }
            int ok = 1;
            for (int side = 0; side < 2; side++) {
                stk[sp].lx = (double *)malloc((size_t)nvar * sizeof(double));
                stk[sp].ux = (double *)malloc((size_t)nvar * sizeof(double));
                if (!stk[sp].lx || !stk[sp].ux) ok = 0;
                else {
                    memcpy(stk[sp].lx, nd.lx, (size_t)nvar * sizeof(double));
                    memcpy(stk[sp].ux, nd.ux, (size_t)nvar * sizeof(double));
                    if (side == 0) stk[sp].lx[j] = 0.0, stk[sp].ux[j] = 0.0;  /* x = 0 */
                    else {
                        if (stk[sp].lx[j] < l) stk[sp].lx[j] = l;            /* x >= l */
                        if (stk[sp].ux[j] < l) { ok = 0; free(stk[sp].lx); free(stk[sp].ux); }
                    }
                    if (ok) { stk[sp].bound = pmin; sp++; }
                }
                if (!ok) { incomplete = 1; break; }
            }
            free(nd.lx); free(nd.ux);
            if (!ok) { incomplete = 1; break; }
            continue;
        }

        if (bsos >= 0) {
            /* SOS1 partitions the ordered set; SOS2 children overlap at the
             * pivot. Intersect with x_j=0, including the lower bound: merely
             * setting an upper bound to zero permits negative SOS violations. */
            free(x);
            if (sp + 2 > cap) {
                cap *= 2;
                MipNode *ns = (MipNode *)realloc(stk, (size_t)cap * sizeof(MipNode));
                if (!ns) { incomplete = 1; free(nd.lx); free(nd.ux); break; }
                stk = ns;
            }
            int ok = 1;
            for (int side = 0; side < 2; side++) {
                double *cl = (double *)malloc((size_t)nvar * sizeof(double));
                double *cu = (double *)malloc((size_t)nvar * sizeof(double));
                if (!cl || !cu) { free(cl); free(cu); ok = 0; break; }
                memcpy(cl, nd.lx, (size_t)nvar * sizeof(double));
                memcpy(cu, nd.ux, (size_t)nvar * sizeof(double));
                int feasible = 1;
                for (int q = 0; q < t->sos_n[bsos]; q++) {
                    int j = t->sos_mem[bsos][q];
                    double w = t->sos_w[bsos][q];
                    int zero = side == 0
                        ? (t->sos_type[bsos] == 1 ? w <= sos_pivot : w < sos_pivot)
                        : w > sos_pivot;
                    if (zero) {
                        cl[j] = fmax(cl[j], 0.0); cu[j] = fmin(cu[j], 0.0);
                        if (cl[j] > cu[j]) feasible = 0;
                    }
                }
                if (feasible) {
                    stk[sp].lx = cl; stk[sp].ux = cu; stk[sp].bound = pmin; sp++;
                } else { free(cl); free(cu); }
            }
            free(nd.lx); free(nd.ux);
            if (!ok) { incomplete = 1; break; }
            active_bound = INF;
            continue;
        }

        if (bj < 0) {   /* the relaxation is integral within itol: a CANDIDATE,
                         * not an incumbent. What goes on record is the rounded
                         * point, and only if the model accepts it. */
            int moved = -1; double mvd = 0.0;
            for (int j = 0; j < nvar; j++) {
                int vt = t->vartype[j];
                int iv = (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                          (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol));
                xs[j] = iv ? floor(x[j] + 0.5) : x[j];
                double d = fabs(x[j] - xs[j]);
                if (iv && d > mvd) { mvd = d; moved = j; }
            }
            int accepted;
            if (t->numbarvar > 0) {
                /* SDP MIP: `mip_point_measures` does not see the bars, so it
                 * cannot accept the point of a model with bars. The integers are
                 * pinned to the rounded value and the SDP is solved again:
                 * (xs, X) is feasible for the node, hence for the model, and
                 * its X goes into `bestX` together with xs. */
                memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
                memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
                for (int j = 0; j < nvar; j++) {
                    int vt = t->vartype[j];
                    if (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                        (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol))
                        flx[j] = flux[j] = xs[j];
                }
                double pfb = 0.0;
                int rst = MIP_RELAX(trelax, flx, flux, xs, &pfb, barXbuf);
                accepted = (rst == 0);
            } else {
                accepted = mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs, ftol, itol);
                if (!accepted) {
                    /* Rounding moved an integer, and every row it lives in moved with
                     * it -- for a big-M disjunction that shift is the difference
                     * between the two sides. The repair is the standard one: pin the
                     * integers at their rounded values and let the CONTINUOUS
                     * variables absorb the shift. That is another relaxation of THIS
                     * node, so nothing is lost; if it has no feasible point, the node
                     * is refined by branching instead. */
                    memcpy(flx, nd.lx, (size_t)nvar * sizeof(double));
                    memcpy(flux, nd.ux, (size_t)nvar * sizeof(double));
                    for (int j = 0; j < nvar; j++) {
                        int vt = t->vartype[j];
                        if (vt == PRIMAL_VAR_TYPE_INT || vt == PRIMAL_VAR_TYPE_INT_BIN ||
                            (vt == PRIMAL_VAR_TYPE_SEMI_INT && x[j] > ftol))
                            flx[j] = flux[j] = xs[j];
                    }
                    double pfb = 0.0;
                    int rst = MIP_RELAX(trelax, flx, flux, xs, &pfb, NULL);
                    if (getenv("GMB_DBG"))
                        fprintf(stderr, "  [mip] node=%ld candidate rejected:"
                                " j=%d lp=%.12g -> %.12g (|d|=%.3g), pinned relaxation st=%d\n",
                                nodes, moved, moved >= 0 ? x[moved] : 0.0,
                                moved >= 0 ? xs[moved] : 0.0, mvd, rst);
                    if (rst == 0)
                        accepted = mip_point_measures(t, nd.lx, nd.ux, lc, uc, xs,
                                                      ftol, itol);
                }
            }
            if (accepted) {
                double p = t->cfix;
                for (int j = 0; j < nvar; j++) p += t->c[j] * xs[j];
                if (t->has_qobj) p += 0.5 * task_xQx(t, xs);
                if (t->numbarvar > 0) p += barC_dot(t, barXbuf);
                if (s * p < best) {
                    best = s * p;   /* min-form; p is the objective as written */
                memcpy(bestx, xs, (size_t)nvar * sizeof(double));
                if (t->numbarvar > 0)
                    memcpy(bestX, barXbuf, (size_t)nbartot * sizeof(double));
                }
                double close_gap = fmax(t->mip_tol_abs_gap,
                    t->mip_tol_rel_gap * (1.0 + fabs(best)));
                /* Only the ORIGINAL relaxation bound can close this node. */
                if (pmin >= best - close_gap) {
                    closed_bound = fmin(closed_bound, pmin); active_bound = INF;
                    free(x); free(nd.lx); free(nd.ux);
                continue;
            }
            }
            /* Still nothing to publish. If the snap moved some integer off its
             * LP value, that variable is a legitimate branch: floor/ceil of the
             * LP value keeps every whole number on one side or the other, so
             * refining loses no integer point. If nothing moved, the node's own
             * LP answer is what disagrees with the model, and the node is
             * dropped rather than published. */
            bj = moved;
            if (bj < 0) {
                incomplete = 1; open_bound = fmin(open_bound, pmin);
                tlog(t, "MIP leaf rejected: the integral point does not measure\n");
                if (getenv("GMB_DBG"))
                    fprintf(stderr, "  [mip] node=%ld leaf rejected: nothing moved"
                            " and the point does not measure\n", nodes);
                free(x); free(nd.lx); free(nd.ux);
                continue;
            }
            bfrac = mvd;
        }
        /* children: x_bj <= floor(v), x_bj >= floor(v)+1 of the LP value.
         * The floor is taken on v itself, NOT on v+itol: a value that is
         * already integral within itol would floor to its own upper bound, the
         * left child would BE the parent, and the node would regenerate
         * forever (measured on the big-M disjunction of T67 at INHER=1e-5:
         * 100000 nodes, none of them a refinement). Rounding down loses no
         * integer point either way: every whole number is on one side or the
         * other. Each cut is intersected with the node's own box, and a side
         * that is empty or leaves the box untouched is not a refinement, so it
         * is not created. */
        double v = x[bj];
        double vlo = floor(v);
        double vup = vlo + 1.0;
        int semi_bj = (t->vartype[bj] == PRIMAL_VAR_TYPE_SEMI_INT);
        double lsemi_bj = t->blx[bj];
        free(x);
        if (sp + 2 > cap) {
            cap *= 2;
            MipNode *ns = (MipNode *)realloc(stk, (size_t)cap * sizeof(MipNode));
            if (!ns) { incomplete = 1; free(nd.lx); free(nd.ux); break; }
            stk = ns;
        }
        int ok = 1, refined = 0;
        const double box_l = nd.lx[bj], box_u = nd.ux[bj];
        for (int side = 0; side < 2; side++) {
            double cl = nd.lx[bj], cu = nd.ux[bj];
            int replace = 0;   /* the deactivation is not an intersection */
            if (side == 0) {
                if (semi_bj && vlo < lsemi_bj - 1e-9) {
                    /* x = 0: the only alternative to [l, u] when no whole number
                     * fits in [l, floor]. The node's box holds l in its lower
                     * bound, so this side REPLACES the interval with {0}. */
                    cl = 0.0; cu = 0.0; replace = 1;
                } else cu = vlo;                        /* x <= floor */
            } else {
                if (semi_bj && vup < lsemi_bj - 1e-9) cl = lsemi_bj;
                else cl = vup;                          /* x >= ceil  */
            }
            if (!replace) {
                if (cl < nd.lx[bj]) cl = nd.lx[bj];
                if (cu > nd.ux[bj]) cu = nd.ux[bj];
            }
            if (cl > cu || (cl == nd.lx[bj] && cu == nd.ux[bj])) continue;
            stk[sp].lx = (double *)malloc((size_t)nvar * sizeof(double));
            stk[sp].ux = (double *)malloc((size_t)nvar * sizeof(double));
            if (!stk[sp].lx || !stk[sp].ux) { free(stk[sp].lx); free(stk[sp].ux); ok = 0; break; }
            memcpy(stk[sp].lx, nd.lx, (size_t)nvar * sizeof(double));
            memcpy(stk[sp].ux, nd.ux, (size_t)nvar * sizeof(double));
            stk[sp].lx[bj] = cl;
            stk[sp].ux[bj] = cu;
            stk[sp].bound = pmin;
            sp++;
            refined = 1;
        }
        free(nd.lx); free(nd.ux);
        if (!ok) { incomplete = 1; break; }   /* alloc failure: unwind (best so far kept) */
        if (!refined) {
            incomplete = 1; open_bound = fmin(open_bound, pmin);
            /* The relaxation answer sits on a box that no branch can cut: the
             * node's own LP is what disagrees with the model, and there is
             * nothing left to search inside it. */
            tlog(t, "MIP leaf rejected: no branch refines the node\n");
            if (getenv("GMB_DBG"))
                fprintf(stderr, "  [mip] node=%ld leaf rejected, no refinement"
                        " (j=%d lp=%g box=[%g,%g])\n", nodes, bj, v, box_l, box_u);
            continue;
        }
    }

    /* Retain bounds before releasing the frontier. Closed leaves also matter
     * when a nonzero gap tolerance permits early pruning. */
    double global_bound = fmin(best, fmin(closed_bound, fmin(open_bound, active_bound)));
    for (int q = 0; q < sp; q++) global_bound = fmin(global_bound, stk[q].bound);
    t->mip_bound_defined = isfinite(global_bound);
    t->mip_bound = s * global_bound;
    /* free remaining stack */
    for (int q = 0; q < sp; q++) { free(stk[q].lx); free(stk[q].ux); }
    free(stk);
    free(xs); free(flx); free(flux);
    free(lx); free(ux); free(lc); free(uc);
    PRIMAL_deletetask(&tc);

    PRIMALrescodee rc;
    /* Leaving the loop with work on the stack means the node cap (or an
     * allocation failure) stopped the search, not the bounds. An incumbent
     * found that way is a feasible point, not a proof: publishing it as
     * INTEGER_OPTIMAL asserts optimality the tree never established, so the
     * reference's pairing for that outcome (Table 7.3: PRIM_FEAS + PRIM_FEAS,
     * "integer feasible point") is what goes out, with a termination code. */
    if (root_status == 1)      rc = PRIMAL_RES_ERR_INFEASIBLE;
    else if (incomplete || (sp > 0 && nodes < t->mip_max_nodes && !deadline_hit)) rc = PRIMAL_RES_TRM_MAX_ITER;
    else if (deadline_hit)     rc = PRIMAL_RES_TRM_MAX_ITER;   /* time cap */
    else if (nodes >= t->mip_max_nodes && sp > 0)  rc = PRIMAL_RES_TRM_MAX_ITER;
    else if (nodes >= t->mip_max_nodes && sp == 0 && best == INF) rc = PRIMAL_RES_TRM_MAX_ITER;
    else if (best < INF)       rc = PRIMAL_RES_OK;
    else if (root_status == 3) rc = PRIMAL_RES_TRM_MAX_ITER;
    else if (nodes >= t->mip_max_nodes) rc = PRIMAL_RES_TRM_MAX_ITER;
    else                       rc = PRIMAL_RES_ERR_INFEASIBLE;  /* exhausted w/o incumbent */

    if (rc == PRIMAL_RES_OK) {
        t->has_sol = 1;
        memcpy(t->x, bestx, (size_t)nvar * sizeof(double));
        if (t->numbarvar > 0 && bestX) {
            int off = 0;
            for (int j = 0; j < t->numbarvar; j++) {
                int d = t->barDim[j];
                memcpy(t->barx[j], bestX + off, (size_t)d * d * sizeof(double));
                off += d * d;
            }
        }
        double po = t->cfix;
        for (int j = 0; j < nvar; j++) po += t->c[j] * t->x[j];
        if (t->has_qobj) po += 0.5 * task_xQx(t, t->x);
        if (t->numbarvar > 0 && bestX) po += barC_dot(t, bestX);
        t->pobj = po;
        t->dobj = t->mip_bound_defined ? t->mip_bound : NAN;      /* MIP: no duals; dobj = pobj */
        t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
        tlog(t, "integer optimal solution found\n");
    } else if (rc == PRIMAL_RES_ERR_INFEASIBLE) {
        /* Table 7.3: an infeasible integer problem is PRIM_INFEAS with
         * solsta UNKNOWN. A certificate status is not published here because
         * branch-and-bound has no Farkas vector to go with it, and this
         * solver does not declare a certificate without the vector that
         * measures (PRIMAL_getdualray answers ERR_ARG). */
        t->solsta = PRIMAL_SOL_STA_UNKNOWN;
        t->prosta = PRIMAL_PRO_STA_PRIM_INFEAS;
        tlog(t, "MIP infeasible\n");
    } else if (rc == PRIMAL_RES_ERR_UNBOUNDED) {
        /* Table 7.3 pairs an unbounded integer problem with DUAL_INFEAS, but
         * the certificate member is not published: the relaxation's ray is not
         * lifted into the integer model and no vector goes to the user. */
        t->solsta = PRIMAL_SOL_STA_UNKNOWN;
        t->prosta = PRIMAL_PRO_STA_DUAL_INFEAS;
        tlog(t, "MIP unbounded relaxation\n");
    } else {
        /* max nodes: keep the incumbent if there is one. With none there is no
         * point to publish at all, and the all-zero buffer opt_prepare left is
         * not one -- it made getxx answer OK with x = 0 and getprimalinfeas
         * answer that nothing is violated, for a point no node ever proposed. */
        if (best < INF) {
            t->has_sol = 1;
            memcpy(t->x, bestx, (size_t)nvar * sizeof(double));
            if (t->numbarvar > 0 && bestX) {
                int off = 0;
                for (int j = 0; j < t->numbarvar; j++) {
                    int d = t->barDim[j];
                    memcpy(t->barx[j], bestX + off, (size_t)d * d * sizeof(double));
                    off += d * d;
                }
            }
            t->pobj = 0.0;
            for (int j = 0; j < nvar; j++) t->pobj += t->c[j] * t->x[j];
            t->pobj += t->cfix;
            if (t->has_qobj) t->pobj += 0.5 * task_xQx(t, t->x);
            if (t->numbarvar > 0 && bestX) t->pobj += barC_dot(t, bestX);
        }
        t->dobj = t->mip_bound_defined ? t->mip_bound : NAN;
        /* Table 7.3: an integer-feasible point that is not proven optimal is
         * PRIM_FEAS, not UNKNOWN -- the incumbent is worth publishing. With no
         * incumbent there is no conclusion, and the derived problem status is
         * UNKNOWN as the table pairs it. */
        t->solsta = (best < INF) ? PRIMAL_SOL_STA_PRIM_FEAS : PRIMAL_SOL_STA_UNKNOWN;
        tlog(t, "MIP node limit\n");
    }
    free(bestx); free(bestX); free(barXbuf);
#undef MIP_RELAX
    if (senv) PRIMAL_deleteenv(&senv);
    return rc;
}

/* =====================================================================
 * Conic path (SOCP): assemble   min c'x  s.t.  Ex = d,  Gx + h in K
 * from the task.  K = product of R_+ (bounds + linear rows) and SOC
 * (user QUAD cones; RQUAD via aux variables U,V,W_k,T,R with equalities
 * U=u, V=v, W_k=w_k, T=(u+v)/sqrt2, R=(u-v)/sqrt2).
 * Conic stationarity:  c + E'y - G'lam = 0,  dual obj = -(d'y + h'lam).
 * Dual mapping back (empirically validated against the LP path):
 *   ROWLO: ymin_i = -lam   ROWUP: ymin_i = +lam
 *   VARLO: zmin_j = -lam   VARUP: zmin_j = +lam
 *   FX var x_j = b:  zmin_j = +y_eq   (E row = e_j)
 *   RQUAD aux eq (U-u=0): zmin_u += -y_eq ... etc.
 * ===================================================================== */
