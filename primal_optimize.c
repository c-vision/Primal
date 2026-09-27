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
/* primal_optimize.c - opt_routes and the LP/QP solve leg.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Route the task to the engine selected by its model shape (MIP, SDP, conic,
 * QP or LP/QP), after validating ranged bounds and preparing the solve
 * buffers. Returns the engine's result code. */
PRIMALrescodee opt_routes(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    int nvar = t->numvar, ncon = t->numcon;
    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;

    PRIMALrescodee rc = opt_prepare(t);
    if (rc != PRIMAL_RES_OK) return rc;

    /* validate bounds */
    for (int j = 0; j < nvar; j++)
        if (t->bkx[j] == PRIMAL_BK_RA && t->blx[j] > t->bux[j] + 1e-12 * (1.0 + fabs(t->blx[j])))
            return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < ncon; i++)
        if (t->bkc[i] == PRIMAL_BK_RA && t->blc[i] > t->buc[i] + 1e-12 * (1.0 + fabs(t->blc[i])))
            return PRIMAL_RES_ERR_ARG;

    /* MIP path: branch & bound over LP relaxations (integer variables,
     * semi-continuous/semi-integer, SOS constraints) */
    {
        int nint = 0;
        for (int j = 0; j < nvar; j++)
            if (t->vartype[j] != PRIMAL_VAR_TYPE_CONT) nint++;
        if (nint > 0 || t->numsos > 0)
            {
                cb_fire(t, PRIMAL_CALLBACK_BEGIN_MIO);
                PRIMALrescodee rm = optimize_mip(t, s);
                cb_fire(t, PRIMAL_CALLBACK_END_MIO);
                return rm;
            }
    }

    /* SDP path: primal-dual interior point on the PSD blocks and the SOC
     * blocks from QUAD/RQUAD cones; falls back to the tangent-cut outer
     * approximation (bars) / conic path when the conversion does not handle
     * the case (a quadratic objective). */
    if (t->numbarvar > 0) {
        /* A quadratic row or objective is not part of either bar conversion:
         * the RQUAD encoder carries the PSD blocks into its own shadow task,
         * while both routes below would answer on a model without them. */
        if (t->has_qcon > 0 || t->has_qobj)
            return optimize_quad(t, s);
        /* GMB_NO_SDP_IPM forces the tangent-cut route, like GMB_NO_EXP_IPM does for
         * the exp/power cones. Development-only, and it is the only way to put a
         * model of one's choosing through the cut loop: without it the native IPM
         * answers every SDP in the sample set at round 0, and the cut loop's own
         * stopping rule (8 stalled rounds, 0.9x) cannot be measured on anything
         * but the one sample that lands there by itself. */
        PRIMALrescodee r = getenv("GMB_NO_SDP_IPM") ? PRIMAL_RES_TRM_MAX_ITER
                                                  : optimize_sdp_ipm(t, s);
        if (r == PRIMAL_RES_OK || r == PRIMAL_RES_ERR_INFEASIBLE || r == PRIMAL_RES_ERR_UNBOUNDED)
            return r;
        int nat_ok = (t->has_sol && t->solsta == PRIMAL_SOL_STA_OPTIMAL);
        if (t->numcones == 0) {
            PRIMALrescodee rc2 = optimize_sdp(t, s);  /* fallback: tangent-cut outer approximation */
            if (rc2 == PRIMAL_RES_OK || !nat_ok) return rc2;
            t->solsta = PRIMAL_SOL_STA_OPTIMAL;
            return PRIMAL_RES_OK;
        }
        /* combined bar + cones: fall through to the conic path */
    }

    /* quadratic rows / QP objective with cones: exact RQUAD encoding into
     * a conic shadow task; pure conic problems go straight to socp.c */
    if (t->has_qcon > 0 || (t->has_qobj && t->numcones > 0))
        return optimize_quad(t, s);
    /* A PURE QP goes through the dense route by default (fast). The RQUAD
     * encoder gives an exact dual (pobj == dobj, y = the KKT multipliers) but
     * it is 40-180x slower: measured on the qp_* benchmark, n=50 0.0016 s dense
     * vs 0.061 s conic, n=200 0.26 s vs >20 s. The dense route's dual postsolve
     * is inaccurate only for ranged/upper bounds (a known open item, counted as
     * a fuzz warning), so the encoder is opt-in: GMB_QP_CONIC forces it. A
     * wall-clock cap or an objective cut also keeps the dense route (the conic
     * route does not read the ipm.c deadline / cut globals). */
    if (t->has_qobj && t->numvar <= 400 && getenv("GMB_QP_CONIC")) {
        int capped = (t->optimizer_max_time >= 0.0) ||
                     (t->lower_obj_cut > -0.5 * DBL_MAX) || (t->upper_obj_cut < 0.5 * DBL_MAX);
        if (!capped) {
            PRIMALrescodee rq = optimize_quad(t, s);
            if (rq == PRIMAL_RES_OK) return rq;
        }
    }

    /* conic path: SOCP via socp.c (bars handled as extended variables +
     * PSD tangent cuts when both bars and cones are present).  Exp/power
     * cones are native barrier blocks of the unified conic IPM in sdp.c and
     * are tried there first; the tangent-cut outer approximation below stays
     * the fallback (and GMB_NO_EXP_IPM forces it, for the parity test). */
    if (t->numcones > 0) {
        int nexpp = 0;
        for (int k = 0; k < t->numcones; k++)
            if (t->cone_type[k] != PRIMAL_CT_QUAD && t->cone_type[k] != PRIMAL_CT_RQUAD) nexpp = 1;
        int nat_ok = 0;
        if (nexpp && !getenv("GMB_NO_EXP_IPM")) {
            PRIMALrescodee r = optimize_sdp_ipm(t, s);
            if (r == PRIMAL_RES_OK) return r;
            /* The native route may have saved a near-optimal candidate
             * (T101/risk_parity): a published point but not a solved model.
             * The cuts stay the first choice; if they too give no answer this
             * separate path delivers it. */
            nat_ok = (t->has_sol && t->solsta == PRIMAL_SOL_STA_OPTIMAL);
            /* ERR_ARG is not a failed solve: the conversion refused the model
             * before sdp_ipm ever ran, so no [route] line was printed and the
             * trace would otherwise say nothing about who answers.  A quadratic
             * objective or row cannot reach this point either (both branches
             * above route those shapes to the encoder), so what is left here is
             * a cone type neither engine represents. */
            if (r == PRIMAL_RES_ERR_ARG && getenv("GMB_DBG")) fprintf(stderr,
                "  [route] native conic IPM refused the model"
                " (unknown cone type), cuts answer\n");
        }
        SavedRows sv;
        if (t->presolve && t->presolve_level >= 1) conic_presolve_apply(t, &sv);
        else { sv.n = 0; sv.idx = NULL; sv.bkc = NULL; sv.blc = sv.buc = NULL; }
        PRIMALrescodee rcon = optimize_conic(t, s);
        conic_presolve_restore(t, &sv);
        if (rcon == PRIMAL_RES_OK) return rcon;
        if (nat_ok) {
            /* the cuts give no answer: deliver the native route's near-optimal
             * candidate, already published (has_sol). */
            t->solsta = PRIMAL_SOL_STA_OPTIMAL;
            return PRIMAL_RES_OK;
        }
        return rcon;
    }

    /* min-form internal data */
    double *ci = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *sqv = NULL;            /* min-form scaled Q triplet VALUES (sparse) */
    int hasQ = t->has_qobj;
    double *lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    if (!ci || !lx || !ux || !lc || !uc) {
        free(ci); free(sqv); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int j = 0; j < nvar; j++) ci[j] = s * t->c[j];
    for (int j = 0; j < nvar; j++) bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    for (int i = 0; i < ncon; i++) bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc[i], &uc[i]);
    if (!getenv("GMB_NO_PRESOLVE_ROWS")) {
        int nrr = row_redundant(t, nvar, ncon, lx, ux, lc, uc);
        if (nrr > 0 && getenv("GMB_DBG")) {
            char cb[64];
            snprintf(cb, sizeof cb, "LP: %d redundant rows\n", nrr);
            tlog(t, cb);
        }
    }
    /* bound tightening LP/QP: the bounds are tightened and, for each tightened
     * bound, the row and coefficient that produced it are recorded -- they are
     * needed by the dual postsolve below. */
    int *bt_lorow = (int *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(int));
    int *bt_uprow = (int *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(int));
    double *bt_locoef = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    double *bt_upcoef = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    if (!bt_lorow || !bt_uprow || !bt_locoef || !bt_upcoef) {
        free(bt_lorow); free(bt_uprow); free(bt_locoef); free(bt_upcoef);
        free(ci); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    if (!getenv("GMB_NO_BOUND_TIGHTEN")) {
        int nbt = bound_tighten(t, nvar, ncon, lx, ux, lc, uc,
                                bt_lorow, bt_locoef, bt_uprow, bt_upcoef);
        if (nbt > 0 && getenv("GMB_DBG")) {
            char cb[64];
            snprintf(cb, sizeof cb, "LP: %d tightened bounds\n", nbt);
            tlog(t, cb);
        }
    }
    /* copy the tightened bounds BEFORE the scaling: scale_equilibrate modifies
     * lx/ux in place, and the postsolve must compare the point with the
     * unscaled tightened bounds. */
    double *bt_lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *bt_ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!bt_lx || !bt_ux) {
        free(bt_lorow); free(bt_uprow); free(bt_locoef); free(bt_upcoef);
        free(bt_lx); free(bt_ux);
        free(ci); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }
    memcpy(bt_lx, lx, (size_t)nvar * sizeof(double));
    memcpy(bt_ux, ux, (size_t)nvar * sizeof(double));
    int *ptr = NULL, *sub = NULL;
    double *aval = NULL;
    if (!build_csc(t, &ptr, &sub, &aval)) {
        free(ci); free(lx); free(ux); free(lc); free(uc);
        return PRIMAL_RES_ERR_ALLOC;
    }

    /* row/column equilibration (powers of 2, exact), PRIMAL_IPAR_SCALING=0
     * turns it off. Q is NOT scaled here (it is sparse): it is scaled below
     * with ds, as D·Q·D on the triplets. */
    double *rs = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *ds = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!rs || !ds) {
        free(rs); free(ds); free(ci); free(lx); free(ux); free(lc); free(uc);
        free(ptr); free(sub); free(aval);
        return PRIMAL_RES_ERR_ALLOC;
    }
    if (t->scaling) {
        scale_equilibrate(nvar, ncon, ptr, sub, aval, lc, uc, lx, ux, ci, NULL, rs, ds);
    } else {
        for (int i = 0; i < ncon; i++) rs[i] = 1.0;
        for (int j = 0; j < nvar; j++) ds[j] = 1.0;
    }
    if (hasQ) {
        sqv = scaled_qvals(t, s, ds);   /* min-form scaled Q values (s·ds_i·ds_j·v) */
        if (!sqv) {
            free(rs); free(ds); free(ci); free(lx); free(ux); free(lc); free(uc);
            free(ptr); free(sub); free(aval);
            return PRIMAL_RES_ERR_ALLOC;
        }
    }

    StdForm *sf = stdform_build(nvar, ncon, ci, t->qt_i, t->qt_j, sqv,
                                hasQ ? t->qt_n : 0, lx, ux, lc, uc, ptr, sub, aval);
    free(ci); free(lx); free(ux); free(lc); free(uc);
    if (!sf) {
        free(ptr); free(sub); free(aval); free(sqv);
        return PRIMAL_RES_ERR_ARG;
    }

    int status = 0;

    double *xt   = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    double *ystd = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
    double *zst  = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    if (!xt || !ystd || !zst) {
        free(xt); free(ystd); free(zst); free(rs); free(ds);
        stdform_free(sf); free(ptr); free(sub); free(aval); free(sqv);
        return PRIMAL_RES_ERR_ALLOC;
    }

    /* ---- LP presolve: reduces the standard-form problem (LP only) ----
     * The reductions are safe and the postsolve recovers exact primal+duals;
     * the ambiguous cases are left to the solver (no false infeasible/unbounded).
     * The warm start is skipped when the presolve re-addresses rows/columns. */
    Presolve *pre = NULL;
    const int *As_ptr = sf->Aptr, *As_row = sf->Arow;
    const double *As_val = sf->Aval, *bsolve = sf->b, *csolve = sf->c;
    const int *Qs_ptr = sf->Qptr, *Qs_row = sf->Qrow;
    const double *Qs_val = sf->Qval;
    int msolve = sf->m, nsolve = sf->n;
    if (t->presolve && t->presolve_level >= 1 && !hasQ) {
        Presolve *pp = NULL;
        if (lp_presolve(sf->Aptr, sf->Arow, sf->Aval, sf->b, sf->c, sf->m, sf->n, 1e-9,
                        t->presolve_level, &pp) == 0 &&
            pp && presolve_changed(pp)) {
            pre = pp;
            presolve_reduced(pre, &As_ptr, &As_row, &As_val, &bsolve, &csolve, &msolve, &nsolve);
            Qs_ptr = NULL; Qs_row = NULL; Qs_val = NULL;   /* presolve is LP only */
        } else if (pp) {
            presolve_free(pp);
        }
    }
    int method = std_route_method(hasQ, msolve, nsolve, t);   /* 0 simplex,1 IPM,2/3 sparse */

    /* warm start: map the user (x,y) into the std space for the IPM */
    double *x0 = NULL, *y0 = NULL;
    if (t->has_warm && method != 0 && !pre) {
        x0 = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
        y0 = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
        if (x0 && y0) {
            int bad = 0;
            for (int j = 0; j < nvar; j++)
                if (t->warm_x[j] != t->warm_x[j]) bad = 1;
            if (!bad) {
                for (int j = 0; j < nvar; j++) {
                    const SfrVar *v = &sf->vars[j];
                    if (v->fixed) continue;
                    double xs = t->warm_x[j] * ds[j] - v->shift;
                    if (v->ncols == 1) {
                        x0[v->col[0]] = v->tau[0] * xs;
                        if (x0[v->col[0]] < 1e-8) x0[v->col[0]] = 1e-8;
                    } else {
                        x0[v->col[0]] = xs > 1e-8 ? xs : 1e-8;
                        x0[v->col[1]] = -xs > 1e-8 ? -xs : 1e-8;
                    }
                }
                for (int r = 0; r < sf->m; r++) {
                    if (sf->rows[r].kind == SFRK_VARUB) continue;
                    int i = sf->rows[r].orig;
                    if (i < 0 || i >= ncon) continue;
                    /* count the std rows of constraint i (ranged -> 2) */
                    int nr = 0;
                    for (int r2 = 0; r2 < sf->m; r2++)
                        if (sf->rows[r2].kind != SFRK_VARUB && sf->rows[r2].orig == i) nr++;
                    /* y_rep (original sense) -> ymin min-form -> ystd */
                    y0[r] = -sf->rows[r].sigma * (s * t->warm_y[i] / rs[i]) / (double)nr;
                }
            } else { free(x0); free(y0); x0 = y0 = NULL; }
        } else { free(x0); free(y0); x0 = y0 = NULL; }
    }

    /* Ray capture: own/pray are what the tableau simplex writes in the space it
     * solved (reduced when presolve ran); yray/xray are the same candidates in
     * the full standard form, which is where they get measured. */
    double *own_y = (double *)calloc((size_t)(msolve > 0 ? msolve : 1), sizeof(double));
    double *own_x = (double *)calloc((size_t)(nsolve > 0 ? nsolve : 1), sizeof(double));
    double *yray  = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
    double *xray  = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));

    if (pre) {
        double *xred = (double *)calloc((size_t)(nsolve > 0 ? nsolve : 1), sizeof(double));
        double *yred = (double *)calloc((size_t)(msolve > 0 ? msolve : 1), sizeof(double));
        double *zred = (double *)calloc((size_t)(nsolve > 0 ? nsolve : 1), sizeof(double));
        if (msolve == 0 && nsolve == 0) {
            status = STD_OPT;                               /* all fixed: optimal */
            presolve_postsolve(pre, NULL, NULL, xt, ystd);
        } else if (nsolve == 0) {
            status = STD_INFEASIBLE;     /* rows 0=b_i!=0 left over: infeasible */
            for (int r = 0; r < msolve && own_y; r++)
                if (bsolve[r] != 0.0) { own_y[r] = bsolve[r] > 0.0 ? 1.0 : -1.0; break; }
        } else if (msolve == 0) {
            status = STD_UNBOUNDED;      /* free min c'x with c_j<0: unbounded */
            for (int j = 0; j < nsolve && own_x; j++)
                if (csolve[j] < 0.0) { own_x[j] = 1.0; break; }
        } else if (xred && yred && zred) {
            status = solve_std_routed(As_ptr, As_row, As_val, Qs_ptr, Qs_row, Qs_val,
                                     msolve, nsolve, bsolve, csolve, t,
                                     xred, yred, zred, NULL, NULL, method, own_y, own_x);
            if (status == STD_OPT) presolve_postsolve(pre, xred, yred, xt, ystd);
        } else {
            status = STD_MEMORY;                            /* out of memory */
        }
        if (status != STD_OPT && status != STD_MEMORY) {
            /* undo the reductions linearly, then let the measurement decide */
            const double *cy = ray_candidate(own_y, yred, msolve);
            const double *cx = ray_candidate(own_x, xred, nsolve);
            presolve_postsolve_dir(pre, cx, cy, xray, yray);
        }
        free(xred); free(yred); free(zred);
        presolve_free(pre);
    } else {
        if (t->num_threads > 1 && (method == 0 || method == 1))
            status = solve_std_conc(sf->Aptr, sf->Arow, sf->Aval, sf->Qptr, sf->Qrow, sf->Qval,
                                    sf->m, sf->n, sf->b, sf->c, t,
                                    xt, ystd, zst, x0, y0, own_y, own_x);
        else
            status = solve_std_routed(sf->Aptr, sf->Arow, sf->Aval, sf->Qptr, sf->Qrow, sf->Qval,
                                      sf->m, sf->n, sf->b, sf->c, t,
                                      xt, ystd, zst, x0, y0, method, own_y, own_x);
        if (status != STD_OPT && status != STD_MEMORY) {
            const double *cy = ray_candidate(own_y, ystd, sf->m);
            const double *cx = ray_candidate(own_x, xt, sf->n);
            if (cy) memcpy(yray, cy, (size_t)sf->m * sizeof(double));
            if (cx) memcpy(xray, cx, (size_t)sf->n * sizeof(double));
        }
    }
    free(own_y); free(own_x);
    free(x0); free(y0);
    t->has_warm = 0;
    {
        char pb[96];
        snprintf(pb, sizeof pb, "%s relaxation solved",
                 (method == 2 || method == 3) ? "sparse interior point" :
                 method == 1 ? "interior point" : "simplex");
        tprog(t, pb);
        cb_fire(t, (method == 0 || method == 4) ? PRIMAL_CALLBACK_PRIMAL_SIMPLEX
                                                : PRIMAL_CALLBACK_INTPNT);
    }

    /* primal solution: map + column descaling (x_orig = D x') */
    stdform_map_x(sf, xt, t->x);
    for (int j = 0; j < nvar; j++) t->x[j] *= ds[j];


    status = ray_publish(t, sf, rs, ds, yray, xray, status);
    free(yray); free(xray);

    if (status != STD_OPT) {
        /* Not optimal: report the status the measurement leaves behind; the
         * duals of a problem with no optimum are not meaningful. No point is
         * published -- has_sol stays down, so getxx/getprimalobj/the infeasibility
         * getters refuse instead of answering with the all-zero buffer that
         * opt_prepare left (which used to sit next to a getprimalinfeas saying
         * that very point violates the model). A certificate that measured is
         * still published, through has_dray/has_pray, and rc carries the
         * verdict either way. */
        t->solsta = (status == STD_INFEASIBLE && t->has_dray) ? PRIMAL_SOL_STA_PRIM_INFEAS_CER
                  : (status == STD_UNBOUNDED && t->has_pray) ? PRIMAL_SOL_STA_DUAL_INFEAS_CER
                  : PRIMAL_SOL_STA_UNKNOWN;
        for (int j = 0; j < nvar; j++) t->x[j] = 0.0;
        tlog(t, status == STD_INFEASIBLE ? (t->has_dray ? "primal infeasible\n"
                                                        : "primal infeasible (ray not certified)\n")
              : status == STD_UNBOUNDED ? (t->has_pray ? "dual infeasible (unbounded)\n"
                                                       : "dual infeasible (ray not certified)\n")
              : status == STD_MEMORY ? "out of memory\n" : "iteration limit\n");
        free(xt); free(ystd); free(zst); free(rs); free(ds);
        stdform_free(sf); free(ptr); free(sub); free(aval); free(sqv);
        free(bt_lorow); free(bt_uprow); free(bt_locoef); free(bt_upcoef); free(bt_lx); free(bt_ux);
        if (status == STD_INFEASIBLE) return PRIMAL_RES_ERR_INFEASIBLE;
        if (status == STD_UNBOUNDED) return PRIMAL_RES_ERR_UNBOUNDED;
        if (status == STD_MEMORY) return PRIMAL_RES_ERR_ALLOC;
        return PRIMAL_RES_TRM_MAX_ITER;
    }

    /* ---------- duals (min form) ---------- */
    double *ymin = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    double *zmin = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    double *Qxv = t->has_qobj ? (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double)) : NULL;
    if (!ymin || !zmin || (t->has_qobj && !Qxv)) {
        free(ymin); free(zmin); free(Qxv); free(rs); free(ds);
        free(xt); free(ystd); free(zst);
        stdform_free(sf); free(ptr); free(sub); free(aval); free(sqv);
        free(bt_lorow); free(bt_uprow); free(bt_locoef); free(bt_upcoef); free(bt_lx); free(bt_ux);
        return PRIMAL_RES_ERR_ALLOC;
    }
    /* Qx (sparse, original Q) — reused for zmin, pobj and dobj */
    double xQx = 0.0;
    if (Qxv) { task_Qx(t, t->x, Qxv); for (int j = 0; j < nvar; j++) xQx += t->x[j] * Qxv[j]; }
    stdform_map_y(sf, ystd, ymin);
    for (int i = 0; i < ncon; i++) ymin[i] *= rs[i];   /* y_orig = R y' */
    /* z_min = -(ci + s*Q x + A_orig' ymin), with sparse original Q and unscaled x */
    for (int j = 0; j < nvar; j++) {
        double qv = Qxv ? s * Qxv[j] : 0.0;
        double av = 0.0;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) av += c->val[k] * ymin[c->sub[k]];
        zmin[j] = -(s * t->c[j] + qv + av);
    }
    /* dual postsolve of the bound tightening: if a tightened bound is ACTIVE,
     * the bound multiplier goes to the row that implied it. `zmin` is
     * recomputed from `ymin`, so the KKT of the original model holds. */
    {
        int changed = 0;
        for (int j = 0; j < nvar; j++) {
            int i = -1; double aij = 0.0;
            if (bt_uprow[j] >= 0 && t->x[j] >= bt_ux[j] - 1e-7 * (1.0 + fabs(bt_ux[j]))) {
                i = bt_uprow[j]; aij = bt_upcoef[j];
            } else if (bt_lorow[j] >= 0 && t->x[j] <= bt_lx[j] + 1e-7 * (1.0 + fabs(bt_lx[j]))) {
                i = bt_lorow[j]; aij = bt_locoef[j];
            }
            if (i >= 0 && aij != 0.0) {
                double qv = Qxv ? s * Qxv[j] : 0.0;
                double av = 0.0;
                const Col *c = &t->cols[j];
                for (int k = 0; k < c->nz; k++) av += c->val[k] * ymin[c->sub[k]];
                double zm = -(s * t->c[j] + qv + av);
                ymin[i] += zm / aij;
                changed = 1;
            }
        }
        if (changed)
            for (int j = 0; j < nvar; j++) {
                double qv = Qxv ? s * Qxv[j] : 0.0;
                double av = 0.0;
                const Col *c = &t->cols[j];
                for (int k = 0; k < c->nz; k++) av += c->val[k] * ymin[c->sub[k]];
                zmin[j] = -(s * t->c[j] + qv + av);
            }
    }
    free(bt_lorow); free(bt_uprow); free(bt_locoef); free(bt_upcoef); free(bt_lx); free(bt_ux);

    /* reported duals: y = s*ymin, z = s*zmin; sign-split into slc/suc/slx/sux */
    for (int i = 0; i < ncon; i++) {
        double yy = s * ymin[i];
        t->y[i]   = yy;
        t->slc[i] = yy < 0.0 ? yy : 0.0;
        t->suc[i] = yy > 0.0 ? yy : 0.0;
    }
    for (int j = 0; j < nvar; j++) {
        double zz = s * zmin[j];
        t->slx[j] = zz < 0.0 ? zz : 0.0;
        t->sux[j] = zz > 0.0 ? zz : 0.0;
    }

    /* ---------- objective values ---------- */
    /* primal: original objective (as written) */
    double po = t->cfix;
    for (int j = 0; j < nvar; j++) po += t->c[j] * t->x[j];
    if (t->has_qobj) po += 0.5 * xQx;   /* xQx = x'Qx (original Q, sparse) */
    t->pobj = po;

    /* dual (min form): dobj_min = -(sum ymin_i * b_i^act + sum zmin_j * xb_j^act)
     * - 1/2 x'(sQ) x ; reported in the original sense via s. */
    double dob = 0.0;
    if (t->has_qobj) dob -= 0.5 * s * xQx;
    for (int i = 0; i < ncon; i++) {
        double yy = ymin[i];
        /* a multiplier on an infinite bound is 0 at optimality; skip the term so
         * numerical noise (|yy| ~ 1e-9) on a free/one-sided bound cannot make
         * dobj = -/+inf */
        if (yy > 1e-9)            { if (isfinite(t->buc[i])) dob -= yy * t->buc[i]; }
        else if (yy < -1e-9)      { if (isfinite(t->blc[i])) dob -= yy * t->blc[i]; }
    }
    for (int j = 0; j < nvar; j++) {
        double zz = zmin[j];
        if (zz > 1e-9)            { if (isfinite(t->bux[j])) dob -= zz * t->bux[j]; }
        else if (zz < -1e-9)      { if (isfinite(t->blx[j])) dob -= zz * t->blx[j]; }
    }
    /* cfix is in pobj (above) and in neither sum: without it the published dual
     * is the dual of the model minus its constant. Added unscaled because it is
     * a term of the objective as written, not of the min-normalized one. */
    t->dobj = s * dob + t->cfix;

    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_OPTIMAL;
    tlog(t, "optimal solution found\n");

    free(ymin); free(zmin); free(Qxv); free(rs); free(ds);
    free(xt); free(ystd); free(zst);
    stdform_free(sf);
    free(ptr); free(sub); free(aval); free(sqv);
    return PRIMAL_RES_OK;
}

/* =====================================================================
 * Reference linear algebra (the "Linear algebra" group)
 * =====================================================================
 * Dense matrices are column-major, as in the reference. `dot`/`axpy` are the
 * two vector helpers; `gemv`/`gemm`/`syrk` the products; `potrf` the in-place
 * Cholesky; `syeig`/`syevd` the eigenvalues (via the Jacobi rotations of
 * `linalg.c`); `sparsetriangularsolvedense` the substitution on sparse L (or
 * L') in column format. */

/* Compute the dot product xty = x'y over n entries. */
PRIMALrescodee PRIMAL_dot(PRIMALenv_t env, int n, const PRIMALrealt *x,
                          const PRIMALrealt *y, PRIMALrealt *xty) {
    (void)env;
    if (!x || !y || !xty || n < 0) return PRIMAL_RES_ERR_ARG;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    *xty = s;
    return PRIMAL_RES_OK;
}

/* Compute y += alpha*x over n entries. */
PRIMALrescodee PRIMAL_axpy(PRIMALenv_t env, int n, PRIMALrealt alpha,
                           const PRIMALrealt *x, PRIMALrealt *y) {
    (void)env;
    if (!x || !y || n < 0) return PRIMAL_RES_ERR_ARG;
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
    return PRIMAL_RES_OK;
}

/* Compute y = alpha*op(A)*x + beta*y for a column-major A, m x n (NoT) or
 * n x m (Transposed); y has m entries (NoT) or n entries (Transposed). */
PRIMALrescodee PRIMAL_gemv(PRIMALenv_t env, PRIMALtransposee transa, int m, int n,
                           PRIMALrealt alpha, const PRIMALrealt *a, const PRIMALrealt *x,
                           PRIMALrealt beta, PRIMALrealt *y) {
    (void)env;
    if (!a || !x || !y || m < 0 || n < 0) return PRIMAL_RES_ERR_ARG;
    if (transa == PRIMAL_TRANSPOSE_NO) {
        for (int i = 0; i < m; i++) {           /* y (m) = alpha*A*x + beta*y */
            double s = 0.0;
            for (int j = 0; j < n; j++) s += a[i + (size_t)j * m] * x[j];
            y[i] = alpha * s + beta * y[i];
        }
    } else {
        for (int j = 0; j < n; j++) {           /* y (n) = alpha*A'*x + beta*y */
            double s = 0.0;
            for (int i = 0; i < m; i++) s += a[i + (size_t)j * m] * x[i];
            y[j] = alpha * s + beta * y[j];
        }
    }
    return PRIMAL_RES_OK;
}

/* Compute C = alpha*op(A)*op(B) + beta*C for column-major A, B, C; op(A) is
 * m x k and op(B) is k x n. */
PRIMALrescodee PRIMAL_gemm(PRIMALenv_t env, PRIMALtransposee transa, PRIMALtransposee transb,
                           int m, int n, int k, PRIMALrealt alpha, const PRIMALrealt *a,
                           const PRIMALrealt *b, PRIMALrealt beta, PRIMALrealt *c) {
    (void)env;
    if (!a || !b || !c || m < 0 || n < 0 || k < 0) return PRIMAL_RES_ERR_ARG;
    /* op(A) is m x k, op(B) is k x n, C is m x n. A (NoT) is m x k, A (T) is k x m;
     * B (NoT) is k x n, B (T) is n x k. All column-major. */
    for (int l = 0; l < n; l++)
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int j = 0; j < k; j++) {
                double av = (transa == PRIMAL_TRANSPOSE_NO) ? a[i + (size_t)j * m]
                                                            : a[j + (size_t)i * k];
                double bv = (transb == PRIMAL_TRANSPOSE_NO) ? b[j + (size_t)l * k]
                                                            : b[l + (size_t)j * n];
                s += av * bv;
            }
            c[i + (size_t)l * m] = alpha * s + beta * c[i + (size_t)l * m];
        }
    return PRIMAL_RES_OK;
}

/* Compute the `uplo` triangle of C = alpha*A*A' + beta*C (NoT) or
 * alpha*A'*A + beta*C (Transposed) for column-major A, C. */
PRIMALrescodee PRIMAL_syrk(PRIMALenv_t env, PRIMALUploe uplo, PRIMALtransposee trans, int n,
                           int k, PRIMALrealt alpha, const PRIMALrealt *a, PRIMALrealt beta,
                           PRIMALrealt *c) {
    (void)env;
    if (!a || !c || n < 0 || k < 0) return PRIMAL_RES_ERR_ARG;
    /* trans=NoT: A is n x k, C = alpha*A*A' + beta*C.
     * trans=Yes: A is k x n, C = alpha*A'*A + beta*C. Only the `uplo` triangle. */
    for (int j = 0; j < n; j++)
        for (int i = (uplo == PRIMAL_UPLO_LO ? j : 0);
             i < (uplo == PRIMAL_UPLO_LO ? n : j + 1); i++) {
            double s = 0.0;
            for (int l = 0; l < k; l++) {
                double ai = (trans == PRIMAL_TRANSPOSE_NO) ? a[i + (size_t)l * n]
                                                           : a[l + (size_t)i * k];
                double aj = (trans == PRIMAL_TRANSPOSE_NO) ? a[j + (size_t)l * n]
                                                           : a[l + (size_t)j * k];
                s += ai * aj;
            }
            c[i + (size_t)j * n] = alpha * s + beta * c[i + (size_t)j * n];
        }
    return PRIMAL_RES_OK;
}

/* In-place Cholesky factorization of the `uplo` triangle of the symmetric
 * n x n matrix a; ERR_ARG if a pivot is not positive. */
PRIMALrescodee PRIMAL_potrf(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a) {
    (void)env;
    if (!a || n < 0) return PRIMAL_RES_ERR_ARG;
    if (uplo == PRIMAL_UPLO_LO) {
        /* A = L L', L lower-triangular, written into the lower triangle */
        for (int j = 0; j < n; j++) {
            double s = a[j + (size_t)j * n];
            for (int l = 0; l < j; l++) s -= a[j + (size_t)l * n] * a[j + (size_t)l * n];
            if (!(s > 0.0)) return PRIMAL_RES_ERR_ARG;
            a[j + (size_t)j * n] = sqrt(s);
            for (int i = j + 1; i < n; i++) {
                double t = a[i + (size_t)j * n];
                for (int l = 0; l < j; l++) t -= a[i + (size_t)l * n] * a[j + (size_t)l * n];
                a[i + (size_t)j * n] = t / a[j + (size_t)j * n];
            }
        }
    } else {
        /* A = U' U, U upper-triangular, written into the upper triangle */
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++) {
                double s = a[i + (size_t)j * n];
                for (int l = 0; l < i; l++) s -= a[l + (size_t)i * n] * a[l + (size_t)j * n];
                if (i == j) {
                    if (!(s > 0.0)) return PRIMAL_RES_ERR_ARG;
                    a[i + (size_t)j * n] = sqrt(s);
                } else {
                    a[i + (size_t)j * n] = s / a[i + (size_t)i * n];
                }
            }
    }
    return PRIMAL_RES_OK;
}

/* Copy the symmetric matrix (from the `uplo` triangle) into row-major for the
 * Jacobi kernel. */
static double *sym_to_rowmajor(PRIMALUploe uplo, int n, const PRIMALrealt *a) {
    double *r = (double *)malloc((size_t)n * n * sizeof(double));
    if (!r) return NULL;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int lower = (i >= j);
            if ((uplo == PRIMAL_UPLO_LO && lower) || (uplo == PRIMAL_UPLO_UP && !lower))
                r[i * n + j] = a[i + (size_t)j * n];
            else
                r[i * n + j] = a[j + (size_t)i * n];
        }
    return r;
}

/* Compute the eigenvalues of the symmetric n x n matrix a (read from the
 * `uplo` triangle) into w, via the Jacobi rotations. */
PRIMALrescodee PRIMAL_syeig(PRIMALenv_t env, PRIMALUploe uplo, int n, const PRIMALrealt *a,
                            PRIMALrealt *w) {
    (void)env;
    if (!a || !w || n < 0) return PRIMAL_RES_ERR_ARG;
    if (n == 0) return PRIMAL_RES_OK;
    double *r = sym_to_rowmajor(uplo, n, a);
    double *evec = (double *)malloc((size_t)n * n * sizeof(double));
    if (!r || !evec) { free(r); free(evec); return PRIMAL_RES_ERR_ALLOC; }
    dmat_eig_jacobi(n, r, w, evec);
    free(r); free(evec);
    return PRIMAL_RES_OK;
}

/* Compute the eigenvalues and eigenvectors of the symmetric n x n matrix a:
 * w receives the eigenvalues, a the eigenvector matrix (columns). */
PRIMALrescodee PRIMAL_syevd(PRIMALenv_t env, PRIMALUploe uplo, int n, PRIMALrealt *a,
                            PRIMALrealt *w) {
    (void)env;
    if (!a || !w || n < 0) return PRIMAL_RES_ERR_ARG;
    if (n == 0) return PRIMAL_RES_OK;
    double *r = sym_to_rowmajor(uplo, n, a);
    double *evec = (double *)malloc((size_t)n * n * sizeof(double));
    if (!r || !evec) { free(r); free(evec); return PRIMAL_RES_ERR_ALLOC; }
    dmat_eig_jacobi(n, r, w, evec);   /* evec row-major, column k = eigenvector k */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) a[i + (size_t)j * n] = evec[j * n + i];
    free(r); free(evec);
    return PRIMAL_RES_OK;
}

/* Solve L x = b (or L' x = b when transposed) in place for a sparse
 * lower-triangular L in column format (lnzc/lptrc/lsubc/lvalc); ERR_ARG if a
 * column has no diagonal entry. */
PRIMALrescodee PRIMAL_sparsetriangularsolvedense(PRIMALenv_t env, PRIMALtransposee transposed,
        int n, const int *lnzc, const PRIMALint64t *lptrc, PRIMALint64t lensubnval,
        const int *lsubc, const PRIMALrealt *lvalc, PRIMALrealt *b) {
    (void)env; (void)lensubnval;
    if (!lnzc || !lptrc || !lsubc || !lvalc || !b || n < 0) return PRIMAL_RES_ERR_ARG;
    /* diagonal of each column: the entry with lsubc == j */
    double *diag = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!diag) return PRIMAL_RES_ERR_ALLOC;
    for (int j = 0; j < n; j++) {
        diag[j] = 0.0;
        for (PRIMALint64t k = lptrc[j]; k < lptrc[j] + lnzc[j]; k++)
            if (lsubc[k] == j) { diag[j] = lvalc[k]; break; }
        if (diag[j] == 0.0) { free(diag); return PRIMAL_RES_ERR_ARG; }
    }
    if (transposed == PRIMAL_TRANSPOSE_NO) {
        for (int j = 0; j < n; j++) {          /* L x = b, forward */
            b[j] /= diag[j];
            for (PRIMALint64t k = lptrc[j]; k < lptrc[j] + lnzc[j]; k++)
                if (lsubc[k] > j) b[lsubc[k]] -= lvalc[k] * b[j];
        }
    } else {
        for (int j = n - 1; j >= 0; j--) {     /* L' x = b, backward */
            for (PRIMALint64t k = lptrc[j]; k < lptrc[j] + lnzc[j]; k++)
                if (lsubc[k] > j) b[j] -= lvalc[k] * b[lsubc[k]];
            b[j] /= diag[j];
        }
    }
    free(diag);
    return PRIMAL_RES_OK;
}

/* =====================================================================
 * Reference parity surface (long parameters, name generators, diagnostics,
 * optimize*, repair/sensitivity)
 * ===================================================================== */

/* getlintparam/putlintparam: the reference's "long int" parameters are our int
 * parameters widened to 64 bit (a value outside int is ERR_ARG). */
