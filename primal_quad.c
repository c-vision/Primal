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
/* primal_quad.c - opt_prepare, bar_copy, quad_encode_task, optimize_quad.
 * Shares primal_priv.h. Modified 2026-09-27 for numerical/result contracts.
 */
#include "primal_priv.h"

/* Clears the previous solve's point, statuses and objective before a new run;
 * also (re)allocates the solution vectors the engines write into. */
PRIMALrescodee opt_prepare(PRIMALtask_t t) {
    if (t) t->result_stale = 0;
    int nvar = t->numvar, ncon = t->numcon;
    t->has_sol = 0;
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    t->prosta = PRIMAL_PRO_STA_UNKNOWN;
    t->pobj = 0.0;
    t->dobj = 0.0;
    t->mip_result = 0; t->mip_bound_defined = 0;
    free(t->x); free(t->y); free(t->slc); free(t->suc); free(t->slx); free(t->sux);
    free(t->soc_dual); t->soc_dual = NULL; t->nsoc_dual = 0;
    free(t->snx); free(t->xc);
    free(t->pray); free(t->dray);
    t->has_pray = 0; t->has_dray = 0;
    t->has_xc = 0;
    t->x   = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    t->y   = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    t->slc = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    t->suc = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    t->slx = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    t->sux = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    t->snx = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    t->xc  = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    t->pray = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    t->dray = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    if (!t->x || !t->y || !t->slc || !t->suc || !t->slx || !t->sux ||
        !t->snx || !t->xc || !t->pray || !t->dray)
        return PRIMAL_RES_ERR_ALLOC;
    return ensure_size(t);
}

/* Copy the bar store of src into dst: the matrix registry, the bar variables
 * and every bar A / bar C term.  Public getters only, and the symmat ids are
 * the ones dst hands back rather than the ones src used, so dst's registry may
 * be ordered differently.  Both term stores are append-only, so one call per
 * term is the same list the caller built. */
PRIMALrescodee bar_copy(PRIMALtask_t src, PRIMALtask_t dst) {
    int nsym = 0, nb = 0, nba = 0, NBC = 0;
    PRIMALrescodee rc;
    PRIMAL_getnumsymmat(src, &nsym);
    PRIMAL_getnumbarvar(src, &nb);
    if (nb == 0) return PRIMAL_RES_OK;

    int *symmap = (int *)malloc((size_t)(nsym > 0 ? nsym : 1) * sizeof(int));
    int *dims = (int *)malloc((size_t)nb * sizeof(int));
    if (!symmap || !dims) { free(symmap); free(dims); return PRIMAL_RES_ERR_ALLOC; }
    for (int m = 0; m < nsym; m++) {
        int d = 0, nz = 0, nid = -1;
        rc = PRIMAL_getsymmatinfo(src, m, &d, &nz);
        if (rc != PRIMAL_RES_OK) { free(symmap); free(dims); return rc; }
        int *si = (int *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int));
        int *sj = (int *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int));
        double *sv = (double *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(double));
        if (!si || !sj || !sv) {
            free(si); free(sj); free(sv); free(symmap); free(dims);
            return PRIMAL_RES_ERR_ALLOC;
        }
        for (int e = 0; e < nz; e++) {
            int ii = 0, jj = 0; double vv = 0.0;
            PRIMAL_getsymmatentry(src, m, e, &ii, &jj, &vv);
            si[e] = ii; sj[e] = jj; sv[e] = vv;
        }
        rc = PRIMAL_appendsparsesymmat(dst, d, nz, si, sj, sv, &nid);
        free(si); free(sj); free(sv);
        if (rc != PRIMAL_RES_OK) { free(symmap); free(dims); return rc; }
        symmap[m] = nid;
    }
    for (int j = 0; j < nb; j++) PRIMAL_getbarsize(src, j, &dims[j]);
    rc = PRIMAL_appendbarvars(dst, nb, dims);
    free(dims);
    if (rc != PRIMAL_RES_OK) { free(symmap); return rc; }

    PRIMAL_getnumbaraterm(src, &nba);
    for (int k = 0; k < nba; k++) {
        int i = 0, j = 0, m = 0; double cf = 0.0;
        PRIMAL_getbaraitem(src, k, &i, &j, &m, &cf);
        if (i < 0 || i >= dst->numcon || j < 0 || j >= dst->numbarvar) continue;
        rc = PRIMAL_putbaraij(dst, i, j, 1, &symmap[m], &cf);
        if (rc != PRIMAL_RES_OK) { free(symmap); return rc; }
    }
    PRIMAL_getnumbarcterm(src, &NBC);
    for (int k = 0; k < NBC; k++) {
        int j = 0, m = 0; double cf = 0.0;
        PRIMAL_getbarcitem(src, k, &j, &m, &cf);
        if (j < 0 || j >= dst->numbarvar) continue;
        rc = PRIMAL_putbarcj(dst, j, 1, &symmap[m], &cf);
        if (rc != PRIMAL_RES_OK) { free(symmap); return rc; }
    }
    free(symmap);
    return PRIMAL_RES_OK;
}

/* =====================================================================
 * Quadratic encoding: QCQP -> conic shadow task.
 * Every row with Q_i != 0 (and the objective Q when cones are present)
 * is rewritten with an exact RQUAD encoding via eigen-decomposition:
 *   Q = sum_r lam_r v_r v_r'  =>  1/2 x'Qx = 1/2 sum_r z_r^2,
 *   z_r = sqrt(lam_r) v_r'x   (aux equality row, lam_r > 0)
 * cone RQUAD(u, one, z_1..z_R) => u >= 1/2 sum z_r^2 = 1/2 x'Qx.
 * - UP row  (a'x <= uc): add +u to the row and u >= 1/2 x'Qx with Q PSD
 * - LO row  (a'x >= lc): add -u to the row and u >= 1/2 x'(-Q)x with -Q PSD
 * - objective Q (conic path): obj 1/2 x'Qx becomes +w in the objective with
 *   w >= 1/2 x'Qx (Q PSD; NSD => maximize ... not supported, deviation).
 * The shadow task is solved by optimize_conic and the solution (x and the
 * row duals of the ORIGINAL rows) is copied back.
 * Domain: the cone represents the convex side only, so an entity with an
 * eigenvalue on the wrong side beyond 1e-9*max(1,max|lam|) makes the whole
 * model come back PRIMAL_RES_ERR_ARG: the alternative is answering, with
 * PRIMAL_RES_OK, on a model whose feasible set (or objective) is not the one
 * the user wrote.  Eigenvalues below that threshold stay rank deficiency.
 * ===================================================================== */
static PRIMALrescodee quad_encode_task(PRIMALtask_t t, PRIMALtask_t *shadow_out) {
    int nvar = t->numvar, ncon = t->numcon;
    PRIMALrescodee rc;

    /* count aux: eigenrows per quadratic row + objective */
    int nquad_row = 0;
    /* A quadratic objective is encoded (rather than dropped) whenever the task
     * also has cones or bar variables: those are the shapes the LP/QP route
     * cannot take, so the RQUAD epigraph is the only faithful representation. */
    int nquad_obj = t->has_qobj ? 1 : 0;
    for (int i = 0; i < ncon; i++) {
        if (!t->qcon || !t->qcon[i]) continue;
        int nz = 0;
        for (int e = 0; e < nvar * nvar && !nz; e++) if (t->qcon[i][e] != 0.0) nz = 1;
        if (nz && t->bkc[i] != PRIMAL_BK_UP && t->bkc[i] != PRIMAL_BK_LO &&
                  t->bkc[i] != PRIMAL_BK_FR) return PRIMAL_RES_ERR_ARG;
        if (nz) nquad_row++;
    }
    if (nquad_row == 0 && nquad_obj == 0) return PRIMAL_RES_ERR_ARG;

    /* which rows keep their identity: ALL original rows are kept;
     * quadratic rows get u added to their linear part and become
     * linear-only; each contributes 1 new aux var u + R z-vars +
     * (R equalities z_r = sqrt(lam) v_r'x). We need one "one" var
     * shared by all cones (fixed 1). */
    /* total aux computation requires eigenvalues first; do rows one by one
     * into a fresh task assembled incrementally via putarow on extended rows
     * is complex; simpler: build a full shadow matrix description here. */

    /* ---- pass 1: eigen-decompose each quadratic row and the objective ---- */
    int maxdim = nvar > 0 ? nvar : 1;
    double *eval = (double *)malloc((size_t)maxdim * sizeof(double));
    double *evec = (double *)malloc((size_t)maxdim * (size_t)maxdim * sizeof(double));
    double *Q = (double *)malloc((size_t)maxdim * (size_t)maxdim * sizeof(double));
    if (!eval || !evec || !Q) { free(eval); free(evec); free(Q); return PRIMAL_RES_ERR_ALLOC; }

    /* structure of the shadow problem (built in one pass below):
     * vars: [0..nvar) original | aux block
     * aux layout per quadratic entity: u (1) + z_1..z_R (R) ... then a
     * shared "one" variable at the very end.
     * rows: original ncon rows (with linear part extended) + aux equalities
     *       z_r - sqrt(lam) v_r'x = 0
     * cones: one RQUAD per entity over (u, one, z_1..z_R). */

    /* first pass: copy Q, eig, count usable eigenvalues
     * objective: MIN needs Q PSD (lam>0, +u in obj); MAX needs Q NSD
     * (uses -Q, lam<0 of Q, -u in obj). Rows: UP -> PSD, LO -> NSD. */
    int *R = (int *)calloc((size_t)(ncon + 1), sizeof(int));   /* per row + obj */
    int obj_need = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;
    int bad_curv = 0;
    int bad_i = -1;
    double bad_lam = 0.0;
    for (int i = 0; i <= ncon; i++) {   /* i == ncon => objective */
        if (i < ncon && t->bkc[i] == PRIMAL_BK_FR) continue;
        const double *Qi = (i == ncon) ? (t->has_qobj ? ensure_dense_qobj(t) : NULL)
                                       : (t->qcon ? t->qcon[i] : NULL);
        if (!Qi) continue;
        int nz = 0;
        for (int e = 0; e < nvar * nvar && !nz; e++) if (Qi[e] != 0.0) nz = 1;
        if (!nz) continue;
        memcpy(Q, Qi, (size_t)nvar * (size_t)nvar * sizeof(double));
        dmat_eig_jacobi(nvar, Q, eval, evec);
        int need = (i == ncon) ? obj_need
                               : ((t->bkc[i] == PRIMAL_BK_UP) ? 1 : -1);
        /* Curvature on the wrong side of the required sign is not a rounding
         * artefact: the RQUAD cone can only represent the convex part, so
         * encoding this entity would answer on a model with a bigger feasible
         * set (or a flat objective) and still report PRIMAL_RES_OK.  The
         * threshold is relative to the entity's own spectrum so a numerically
         * flat direction stays rank deficiency, not a refusal. */
        double lmax = 0.0;
        for (int r = 0; r < nvar; r++) if (fabs(eval[r]) > lmax) lmax = fabs(eval[r]);
        double bad_thr = t->semi_tol_approx * (lmax > 1.0 ? lmax : 1.0);
        int cnt = 0;
        for (int r = 0; r < nvar; r++) {
            if (need == 1 && eval[r] > 1e-10) cnt++;
            if (need == -1 && eval[r] < -1e-10) cnt++;
            if (need == 1 && eval[r] < -bad_thr && !bad_curv) { bad_curv = 1; bad_i = i; bad_lam = eval[r]; }
            if (need == -1 && eval[r] > bad_thr && !bad_curv) { bad_curv = 1; bad_i = i; bad_lam = eval[r]; }
        }
        if (cnt == 0) {   /* Q PSD/NSD with tiny eigenvalues: rank 0, nothing to do */
            continue;
        }
        R[i] = cnt;
    }
    if (bad_curv) {
        if (bad_curv && getenv("GMB_DBG")) {
            if (bad_i == ncon) fprintf(stderr,
                "  [route] model refused: the quadratic objective has eigenvalue %.3g on the wrong side (domain not convex)\n", bad_lam);
            else fprintf(stderr,
                "  [route] model refused: quadratic row %d has eigenvalue %.3g on the wrong side (domain not convex)\n", bad_i, bad_lam);
        }
        free(R); free(eval); free(evec); free(Q);
        /* bad_curv: the model is not convex on the side the encoding needs.
         * nent == 0: every quadratic part is ~0. */
        return PRIMAL_RES_ERR_ARG;
    }

    /* ---- build shadow task ---- */
    PRIMALenv_t env2 = NULL;
    PRIMALtask_t sh = NULL;
    /* env reuse: create a private env (cheap struct) */
    rc = PRIMAL_makeenv(&env2, NULL);
    if (rc != PRIMAL_RES_OK) { free(R); free(eval); free(evec); free(Q); return rc; }
    /* layout: nvar originals + per-entity [u + R z] + one "one" var */
    int naux = 0;
    for (int i = 0; i <= ncon; i++) naux += R[i] > 0 ? 1 + R[i] : 0;
    int ntot_var = nvar + naux + 1;
    int nrow_eq = 0;
    for (int i = 0; i <= ncon; i++) nrow_eq += R[i];
    rc = PRIMAL_maketask(env2, 0, 0, &sh);
    if (rc != PRIMAL_RES_OK) {
        PRIMAL_deleteenv(&env2); free(R); free(eval); free(evec); free(Q); return rc;
    }
    PRIMAL_appendvars(sh, ntot_var);
    PRIMAL_appendcons(sh, ncon + nrow_eq);
    PRIMAL_putobjsense(sh, t->sense);
    PRIMAL_putcfix(sh, t->cfix);
    for (int j = 0; j < nvar; j++) {
        PRIMAL_putvarbound(sh, j, t->bkx[j], t->blx[j], t->bux[j]);
        PRIMAL_putcj(sh, j, t->c[j]);
    }
    /* "one" variable: last */
    int one = ntot_var - 1;
    PRIMAL_putvarbound(sh, one, PRIMAL_BK_FX, 1.0, 1.0);

    int auxv = nvar;      /* cursor over aux vars */
    int row2 = ncon;      /* cursor over new equality rows */
    for (int i = 0; i <= ncon; i++) {
        if (R[i] <= 0) continue;
        const double *Qi = (i == ncon) ? ensure_dense_qobj(t) : t->qcon[i];
        memcpy(Q, Qi, (size_t)nvar * (size_t)nvar * sizeof(double));
        dmat_eig_jacobi(nvar, Q, eval, evec);
        int isobj = (i == ncon);
        /* objective: MIN -> Q PSD side (+u in obj); MAX -> Q NSD side
         * (uses -Q: u >= 1/2 x'(-Q)x, obj -= u). Rows: UP -> PSD, LO -> NSD. */
        int psd_side = isobj ? (t->sense != PRIMAL_OPTIMIZE_MAXIMIZE)
                             : (t->bkc[i] == PRIMAL_BK_UP);
        int u = auxv; auxv++;
        PRIMAL_putvarbound(sh, u, PRIMAL_BK_LO, 0.0, INFINITY);
        int zbase = auxv;
        auxv += R[i];
        for (int r = 0, done = 0; r < nvar && done < R[i]; r++) {
            int keep = psd_side ? (eval[r] > 1e-10) : (eval[r] < -1e-10);
            if (!keep) continue;
            double sq = sqrt(psd_side ? eval[r] : -eval[r]);
            /* equality: z - sqrt(lam) * v_r'x = 0 on row row2 */
            int z = zbase + done;
            PRIMAL_putvarbound(sh, z, PRIMAL_BK_FR, -INFINITY, INFINITY);
            /* dmat_eig_jacobi returns eigenvector k in COLUMN k of a
             * row-major buffer: component j of eigenvector r is
             * evec[j*nvar+r]. Read by row, every lambda paired with the wrong
             * direction and the quadratic row with the cross term was answered
             * for another quadratic form. */
            int nzc = 0;
            for (int j = 0; j < nvar; j++) if (fabs(evec[j * nvar + r]) > 1e-14) nzc++;
            if (nzc == 0) nzc = 1;
            int *sub = (int *)malloc((size_t)(nzc + 1) * sizeof(int));
            double *val = (double *)malloc((size_t)(nzc + 1) * sizeof(double));
            if (!sub || !val) { free(sub); free(val); rc = PRIMAL_RES_ERR_ALLOC; goto sh_fail; }
            int w = 0;
            for (int j = 0; j < nvar; j++) {
                if (fabs(evec[j * nvar + r]) > 1e-14) {
                    sub[w] = j; val[w] = -sq * evec[j * nvar + r]; w++;
                }
            }
            sub[w] = z; val[w] = 1.0; w++;
            PRIMAL_putarow(sh, row2, w, sub, val);
            PRIMAL_putconbound(sh, row2, PRIMAL_BK_FX, 0.0, 0.0);
            free(sub); free(val);
            row2++; done++;
        }
        /* cone RQUAD(u, one, z_1..z_R): u >= 1/2 sum z^2 => sgn * 1/2 x'Qx */
        {
            int cm[256];
            int nmem = 2 + R[i];
            if (nmem > 256) { rc = PRIMAL_RES_ERR_ARG; goto sh_fail; }
            cm[0] = u; cm[1] = one;
            for (int r = 0; R[i] > 0 && r < R[i]; r++) cm[2 + r] = zbase + r;
            PRIMAL_appendcone(sh, PRIMAL_CT_RQUAD, 0.0, nmem, cm);
        }
        /* objective handling: 1/2 x'Qx term via u:
         * MIN + Q PSD:  obj += u;   MAX + Q NSD: obj -= u
         * (u >= 1/2 x'Qx in the first case, u >= 1/2 x'(-Q)x in the second,
         *  so the objective bounds the quadratic term from the correct side) */
        if (isobj) {
            double co = psd_side ? +1.0 : -1.0;
            PRIMAL_putcj(sh, u, co);
        }
    }

    /* ---- row fixup: original rows (linear part + quadratic sgn*u term) ---- */
    {
        /* map entity -> u var: rebuilt by scanning the same order */
        int auxv2 = nvar;
        int *uof = (int *)malloc((size_t)(ncon + 1) * sizeof(int));
        int *R2 = (int *)calloc((size_t)(ncon + 1), sizeof(int));
        if (!uof || !R2) { free(uof); free(R2); rc = PRIMAL_RES_ERR_ALLOC; goto sh_fail; }
        for (int i = 0; i <= ncon; i++) R2[i] = R[i];
        for (int i = 0; i <= ncon; i++) {
            if (R2[i] <= 0) { uof[i] = -1; continue; }
            uof[i] = auxv2;
            auxv2 += 1 + R2[i];
        }
        for (int i = 0; i < ncon; i++) {
            /* re-put original row i with the u term added */
            int nu = uof[i];
            int nzmax = 1;
            for (int j = 0; j < nvar; j++) nzmax += t->cols[j].nz + 1;
            int *sub = (int *)malloc((size_t)nzmax * sizeof(int));
            double *val = (double *)malloc((size_t)nzmax * sizeof(double));
            if (!sub || !val) { free(sub); free(val); free(uof); free(R2); rc = PRIMAL_RES_ERR_ALLOC; goto sh_fail; }
            int w = 0;
            for (int j = 0; j < nvar; j++) {
                const Col *c = &t->cols[j];
                for (int k = 0; k < c->nz; k++)
                    if (c->sub[k] == i) { sub[w] = j; val[w] = c->val[k]; w++; }
            }
            if (nu >= 0) {
                double sgn = (t->bkc[i] == PRIMAL_BK_UP) ? +1.0 : -1.0;
                sub[w] = nu; val[w] = sgn; w++;
            }
            PRIMAL_putarow(sh, i, w, sub, val);
            PRIMAL_putconbound(sh, i, t->bkc[i], t->blc[i], t->buc[i]);
            free(sub); free(val);
        }
        free(uof); free(R2);
    }

    /* carry the model's other structures over: the PSD blocks (rows and bar
     * variables, both of which exist in sh by now) and the user cones */
    rc = bar_copy(t, sh);
    if (rc != PRIMAL_RES_OK) goto sh_fail;
    for (int k = 0; k < t->numcones; k++)
        PRIMAL_appendcone(sh, (PRIMALconetypee)t->cone_type[k], t->cone_param[k],
                       t->cone_nmem[k], t->cone_mem[k]);

    free(R); free(eval); free(evec); free(Q);
    *shadow_out = sh;
    return PRIMAL_RES_OK;
sh_fail:
    free(R); free(eval); free(evec); free(Q);
    if (sh) PRIMAL_deletetask(&sh);
    PRIMAL_deleteenv(&env2);
    return rc;
}

/* solve t through the quadratic encoding; maps the shadow solution back */
PRIMALrescodee optimize_quad(PRIMALtask_t t, int s) {
    PRIMALtask_t sh = NULL;
    PRIMALrescodee rc = quad_encode_task(t, &sh);
    if (rc != PRIMAL_RES_OK) return rc;
    PRIMALenv_t shenv = sh->env;
    /* copy progress callback + params */
    sh->progcb = t->progcb; sh->proghandle = t->proghandle;
    memcpy(sh->infoname, t->infoname, sizeof sh->infoname);
    param_copy(sh, t);
    if (t->has_qobj && t->numcones == 0 && t->numbarvar == 0) {   /* pure QP: its QO triple governs (T173) */
        sh->tol_co_pfeas = t->tol_qo_pfeas;
        sh->tol_co_dfeas = t->tol_qo_dfeas;
        sh->tol_co_gap   = t->tol_qo_gap;
    }
    sh->logcb = t->logcb; sh->loghandle = t->loghandle;

    PRIMALrescodee rcp = opt_prepare(sh);
    if (rcp != PRIMAL_RES_OK) { PRIMAL_deletetask(&sh); return rcp; }
    PRIMALrescodee rcs;
    int nexpp = 0;
    for (int k = 0; k < sh->numcones; k++)
        if (sh->cone_type[k] != PRIMAL_CT_QUAD && sh->cone_type[k] != PRIMAL_CT_RQUAD)
            nexpp = 1;
    if (sh->numbarvar > 0 || (nexpp && !getenv("GMB_NO_EXP_IPM"))) {
        /* The encoded shadow has no quadratic part left, which is the shape the
         * unified conic IPM solves natively; the extended-variable route is its
         * outer-approximation fallback.  This is the same ladder PRIMAL_optimize
         * runs on the model itself: bars AND exp/power cones are native blocks
         * there, and the encoder copies the user's cones into the shadow — so a
         * bar-free shadow holding a PPOW block is still a model the native route
         * exists for.  Asking it of the cuts alone was one decision written
         * twice, with two policies. */
        rcs = optimize_sdp_ipm(sh, s);
        if (rcs != PRIMAL_RES_OK && rcs != PRIMAL_RES_ERR_INFEASIBLE &&
            rcs != PRIMAL_RES_ERR_UNBOUNDED) rcs = optimize_conic(sh, s);
    } else {
        rcs = optimize_conic(sh, s);
    }
    /* map solution back: x, and y of the ORIGINAL rows only (aux equalities
     * and cones are an internal encoding) */
    if (rcs == PRIMAL_RES_OK && sh->has_sol) {
        int nvar = t->numvar;
        memcpy(t->x, sh->x, (size_t)t->numvar * sizeof(double));
        memcpy(t->y, sh->y, (size_t)t->numcon * sizeof(double));
        /* The row duals in the LP/QP convention (same split as the dense
         * route): y >= 0 on an upper side, y <= 0 on a lower side.  The conic
         * route leaves slc/suc unset, so getdviolcon read |y| as a violation. */
        for (int i = 0; i < t->numcon; i++) {
            double yy = t->y[i];
            t->slc[i] = yy < 0.0 ? yy : 0.0;
            t->suc[i] = yy > 0.0 ? yy : 0.0;
        }
        for (int j = 0; j < t->numbarvar; j++) {
            int d = t->barDim[j];
            memcpy(t->barx[j], sh->barx[j], (size_t)d * (size_t)d * sizeof(double));
            memcpy(t->barsj[j], sh->barsj[j], (size_t)d * (size_t)d * sizeof(double));
        }
        /* z (var duals) from KKT on the ORIGINAL quadratic problem:
         * c + Qx + A'y + z = 0 with Q = sum of row quadratics? No: z is the
         * var-bound multiplier; recompute z = -(c + A'y) for the linear
         * part; the quadratic row terms contribute to ROW duals only. */
        double *qxv = NULL;
        if (t->has_qobj && t->qt_n > 0) {
            qxv = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
            if (qxv) task_Qx(t, t->x, qxv);
        }
        for (int j = 0; j < nvar; j++) {
            double av = 0.0;
            const Col *c = &t->cols[j];
            for (int k = 0; k < c->nz; k++) av += c->val[k] * t->y[c->sub[k]];
            for (int i = 0; i < t->numcon; i++) if (t->qcon && t->qcon[i]) {
                double qg = 0.0;
                for (int k = 0; k < nvar; k++) qg += t->qcon[i][j*nvar+k] * t->x[k];
                av += t->y[i] * qg;
            }
            double zz = -(t->c[j] + (qxv ? qxv[j] : 0.0) + av);
            t->slx[j] = zz < 0.0 ? zz : 0.0;
            t->sux[j] = zz > 0.0 ? zz : 0.0;
        }
        free(qxv);
        /* pobj: original objective (linear + 1/2 x'Qx + the bar terms, which
         * the encoding carries in the shadow and would otherwise vanish here) */
        double po = t->cfix;
        for (int j = 0; j < nvar; j++) po += t->c[j] * t->x[j];
        if (t->has_qobj) po += 0.5 * task_xQx(t, t->x);
        for (int k = 0; k < t->nbarC; k++) {
            int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b]; double tr = 0.0;
            for (int e = 0; e < t->sym_nnz[m]; e++) {
                int p = t->sym_subi[m][e], q = t->sym_subj[m][e];
                double av = t->sym_val[m][e];
                tr += av * t->barx[b][p * d + q];
                if (p != q) tr += av * t->barx[b][q * d + p];
            }
            po += t->barC_coef[k] * tr;
        }
        t->pobj = po;
        t->dobj = sh->dobj;   /* deviation: no meaningful dobj for the QCQP path */
        t->has_sol = 1;
        t->solsta = sh->solsta;
    } else if (sh->has_sol) {
        t->has_sol = 1;
        t->solsta = sh->solsta;
    }
    PRIMAL_deletetask(&sh);
    PRIMAL_deleteenv(&shenv);
    return rcs;
}

/* =====================================================================
 * Basis solve (PRIMAL_solvebasis): evaluates the basis given by skx/skc.
 * SK_LOW/SK_UPR variables are fixed at their bounds; rows with skc BAS have
 * a nonbasic slack... PRIMAL convention: skc[i]=BAS means the slack of row i
 * is basic (row NOT active); skc[i]=LOW/UPR = row active at the bound
 * (constraint satisfied with equality). skx[j]=BAS = basic variable. It solves
 * the system: A_B x_B = b - A_N x_N with A_B = active rows x basic variables;
 * then duals: y = (A_B')^{-1} c_B on the active constraints, z = c - A'y on
 * the variables; optimality check (z>=0 at lower, z<=0 at upper for min). If
 * not optimal: solve from scratch with PRIMAL_optimize.
 * ===================================================================== */
/* Builds a basis of the standard form (`sf->m` column indices) from the basis
 * in general form given by skx (variables) and skc (constraints): every basic
 * variable enters with its std columns (`vars[j].col/tau`), every basic
 * constraint with its slack (`SFCK_SLACK` of row r). EQ rows have no slack and
 * VARUB are not constraints: they are skipped. Returns 1 if it obtains exactly
 * sf->m distinct columns, 0 otherwise (the caller falls back on optimize).
 * Foundation of the warm-start of PRIMAL_solvebasis. */
static int stdform_basis_from_keys(const StdForm *sf, const int *skx, const int *skc,
                                   int nvar, int ncon, int *basis)
{
    int m = sf->m, n = sf->n, nb = 0;
    char *taken = (char *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!taken) return 0;
    for (int j = 0; j < nvar && nb < m; j++) {
        if (skx[j] != PRIMAL_SK_BAS) continue;
        const SfrVar *v = &sf->vars[j];
        if (v->fixed) continue;
        for (int t = 0; t < v->ncols && nb < m; t++) {
            int c = v->col[t];
            if (c >= 0 && c < n && !taken[c]) { basis[nb++] = c; taken[c] = 1; }
        }
    }
    for (int i = 0; i < ncon && nb < m; i++) {
        if (skc[i] != PRIMAL_SK_BAS) continue;
        for (int r = 0; r < m; r++) {
            if (sf->rows[r].kind == SFRK_EQ || sf->rows[r].kind == SFRK_VARUB) continue;
            if (sf->rows[r].orig != i) continue;
            for (int k = 0; k < n; k++)
                if (sf->cols[k].kind == SFCK_SLACK && sf->cols[k].idx == r && !taken[k]) {
                    basis[nb++] = k; taken[k] = 1; break;
                }
            break;
        }
    }
    free(taken);
    return nb == m;
}

/* Solves from a user-provided basis (LP only): builds the standard form, checks
 * the basis, and either optimizes from it or re-solves the model when it does
 * not measure (documented deviation: no basis certificate). */
PRIMALrescodee PRIMAL_solvebasis(PRIMALtask_t t) {
    if (!t) return PRIMAL_RES_ERR_NULL;
    int nvar = t->numvar, ncon = t->numcon;
    if (t->has_qobj || t->has_qcon > 0 || t->numcones > 0 || t->numbarvar > 0)
        return PRIMAL_RES_ERR_ARG;   /* documented deviation: LP only */
    if (!t->skc || !t->skx) return PRIMAL_RES_ERR_ARG;   /* no basis given */

    int s = (t->sense == PRIMAL_OPTIMIZE_MAXIMIZE) ? -1 : 1;

    /* effective bounds */
    double *lx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *ux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *lc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *x = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    int *bvi = (int *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(int));
    int *bri = (int *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(int));
    if (!lx || !ux || !lc || !uc || !x || !bvi || !bri) {
        free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int j = 0; j < nvar; j++) bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    for (int i = 0; i < ncon; i++) bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc[i], &uc[i]);

    /* x_N: nonbasic variables at their bounds */
    int nbas = 0;
    for (int j = 0; j < nvar; j++) {
        switch (t->skx[j]) {
            case PRIMAL_SK_LOW: x[j] = lx[j]; break;
            case PRIMAL_SK_UPR: x[j] = ux[j]; break;
            case PRIMAL_SK_BAS: bvi[nbas++] = j; x[j] = 0.0; break;
            default: free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
                     return PRIMAL_RES_ERR_ARG;   /* SUPBAS/UNDEF not supported */
        }
    }
    /* active rows: those with skc LOW/UPR (constraint anchored at its bound, nonbasic slack) */
    int nact = 0;
    for (int i = 0; i < ncon; i++) {
        if (t->skc[i] == PRIMAL_SK_LOW) bri[nact++] = i;
        else if (t->skc[i] == PRIMAL_SK_UPR) bri[nact++] = i;
        else if (t->skc[i] != PRIMAL_SK_BAS) {
            free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
            return PRIMAL_RES_ERR_ARG;
        }
    }
    /* the basis must be square: |BAS vars| == |active rows| (otherwise the
     * basis is overdetermined: we solve anyway in the min-norm sense only if
     * nbas == nact) */
    if (nbas != nact) {
        free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
        return PRIMAL_RES_ERR_ARG;
    }

    /* solve A_B x_B = rhs - A_N x_N */
    if (nbas > 0) {
        double *B = (double *)malloc((size_t)nbas * (size_t)nbas * sizeof(double));
        double *rhs = (double *)malloc((size_t)nbas * sizeof(double));
        if (!B || !rhs) { free(B); free(rhs); free(lx); free(ux); free(lc); free(uc);
                           free(x); free(bvi); free(bri); return PRIMAL_RES_ERR_ALLOC; }
        for (int r = 0; r < nact; r++) {
            int i = bri[r];
            /* target: active bound of the row (LOW -> lc, UPR -> uc) */
            double target = (t->skc[i] == PRIMAL_SK_UPR) ? uc[i] : lc[i];
            /* rhs = target - sum_j A_ij x_j (nonbasic only) */
            double acc = target;
            for (int j = 0; j < nvar; j++) {
                if (t->skx[j] == PRIMAL_SK_BAS) continue;
                double aij;
                PRIMAL_getaij(t, i, j, &aij);
                acc -= aij * x[j];
            }
            rhs[r] = acc;
            for (int c = 0; c < nbas; c++)
                PRIMAL_getaij(t, i, bvi[c], &B[r * nbas + c]);
        }
        LuFact *f = dmat_lu_factor(B, nbas);
        int singular = 0;
        if (!f) singular = 1;
        else if (dmat_lu_solve(f, rhs) != 0) { singular = 1; dmat_lu_free(f); f = NULL; }
        if (singular) {
            /* minimal regularization: the basis is degenerate but the solution
             * may exist; fallback: least-squares resolution not implemented ->
             * explicit error */
            free(B); free(rhs); free(lx); free(ux); free(lc); free(uc);
            free(x); free(bvi); free(bri);
            return PRIMAL_RES_ERR_ARG;
        }
        dmat_lu_free(f);
        for (int c = 0; c < nbas; c++) x[bvi[c]] = rhs[c];
        free(B); free(rhs);
    }

    /* duals: y in ORIGINAL form (clone convention: c + A'y + z = 0 with the
     * costs as written). Stationarity of the basic variables:
     *   c_B + A_B' y_act = 0  ->  A_B' y_act = -c_B
     * (non-basic variables do not contribute: z absorbs them) */
    double *y = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    if (!y) { free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
              return PRIMAL_RES_ERR_ALLOC; }
    if (nbas > 0) {
        double *Bt = (double *)malloc((size_t)nbas * (size_t)nbas * sizeof(double));
        double *cb = (double *)malloc((size_t)nbas * sizeof(double));
        if (!Bt || !cb) { free(Bt); free(cb); free(y); free(lx); free(ux); free(lc);
                          free(uc); free(x); free(bvi); free(bri); return PRIMAL_RES_ERR_ALLOC; }
        for (int c = 0; c < nbas; c++) cb[c] = -t->c[bvi[c]];
        for (int r = 0; r < nact; r++)
            for (int c = 0; c < nbas; c++) {
                double aij;
                PRIMAL_getaij(t, bri[r], bvi[c], &aij);
                Bt[c * nbas + r] = aij;   /* B transposed: row c = column r */
            }
        LuFact *f = dmat_lu_factor(Bt, nbas);
        if (f && dmat_lu_solve(f, cb) == 0) {
            for (int r = 0; r < nact; r++) y[bri[r]] = cb[r];
            dmat_lu_free(f);
        } else {
            if (f) dmat_lu_free(f);
            /* degenerate dual basis: y=0 on the active (fallback) */
        }
        free(Bt); free(cb);
    }
    /* z = -(c + A'y) on the nonbasic, 0 on the basic (original form) */
    double *z = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    if (!z) { free(y); free(lx); free(ux); free(lc); free(uc); free(x); free(bvi); free(bri);
              return PRIMAL_RES_ERR_ALLOC; }
    for (int j = 0; j < nvar; j++) {
        if (t->skx[j] == PRIMAL_SK_BAS) { z[j] = 0.0; continue; }
        double av = 0.0;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) av += c->val[k] * y[c->sub[k]];
        z[j] = -(t->c[j] + av);
    }

    /* check primal feasibility of the basic variables */
    int pfeas = 1;
    for (int c = 0; c < nbas && pfeas; c++) {
        int j = bvi[c];
        if (x[j] < lx[j] - 1e-7 * (1.0 + fabs(lx[j]))) pfeas = 0;
        if (x[j] > ux[j] + 1e-7 * (1.0 + fabs(ux[j]))) pfeas = 0;
    }
    /* check dual feasibility: reduced r_j = -z_j (from the system c+A'y+z=0).
     * lower optimal <=> r <= 0 (max) / r >= 0 (min)  <=> s*z <= 0;
     * upper  optimal <=> s*z >= 0. (verified on an optimal/non-optimal basis) */
    int dfeas = 1;
    for (int j = 0; j < nvar && dfeas; j++) {
        if (t->skx[j] == PRIMAL_SK_BAS) continue;
        if (t->skx[j] == PRIMAL_SK_LOW) { if (s * z[j] > 1e-7) dfeas = 0; }
        if (t->skx[j] == PRIMAL_SK_UPR) { if (s * z[j] < -1e-7) dfeas = 0; }
    }

    PRIMALrescodee rc = PRIMAL_RES_OK;
    if (!pfeas || !dfeas) {
        /* The basis is not a certificate: the reference answers by
         * RE-OPTIMIZING from it. We build the standard form, map the basis from
         * skx/skc and run the revised simplex from that basis; if it succeeds
         * we publish the re-optimized point, otherwise we solve from scratch
         * (declared deviation: no basis repair). */
        int warm_ok = 0;
        {
            int nv = nvar;
            size_t nnz = 0;
            for (int j = 0; j < nv; j++) nnz += (size_t)t->cols[j].nz;
            int *cp = (int *)malloc((size_t)(nv + 1) * sizeof(int));
            int *sub = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
            double *val = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
            double *cint = (double *)malloc((size_t)(nv > 0 ? nv : 1) * sizeof(double));
            if (cp && sub && val && cint) {
                int p = 0; cp[0] = 0;
                for (int j = 0; j < nv; j++) {
                    for (int k = 0; k < t->cols[j].nz; k++) { sub[p] = t->cols[j].sub[k]; val[p] = t->cols[j].val[k]; p++; }
                    cp[j + 1] = p;
                    cint[j] = s * t->c[j];
                }
                StdForm *sf = stdform_build(nv, ncon, cint, NULL, NULL, NULL, 0,
                                            lx, ux, lc, uc, cp, sub, val);
                if (sf) {
                    int *sb = (int *)malloc((size_t)(sf->m > 0 ? sf->m : 1) * sizeof(int));
                    if (sb && stdform_basis_from_keys(sf, (const int *)t->skx, (const int *)t->skc, nv, ncon, sb)) {
                        double *dA = stdform_dense_A(sf);
                        double *xt = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
                        double *ystd = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
                        double *ymin = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
                        if (dA && xt && ystd && ymin) {
                            /* primal-feasible basis -> revised (primal); basis
                             * dual-feasible but primal-infeasible -> dual. */
                            int st = 0;
                            if (pfeas)
                                st = simplex_revised_solve_std(dA, sf->m, sf->n, sf->b, sf->c, sb,
                                                               iter_cap(t->max_iter_simplex), xt, ystd);
                            if (st != 0 && dfeas)
                                st = simplex_dual_solve_std(dA, sf->m, sf->n, sf->b, sf->c, sb,
                                                            iter_cap(t->max_iter_simplex), xt, sb, ystd);
                            if (st == 0) {
                                stdform_map_x(sf, xt, x);
                                stdform_map_y(sf, ystd, ymin);
                                for (int i = 0; i < ncon; i++) y[i] = s * ymin[i];
                                for (int j = 0; j < nvar; j++) {
                                    if (t->skx[j] == PRIMAL_SK_BAS) { z[j] = 0.0; continue; }
                                    double av = 0.0;
                                    const Col *cc = &t->cols[j];
                                    for (int k = 0; k < cc->nz; k++) av += cc->val[k] * y[cc->sub[k]];
                                    z[j] = -(t->c[j] + av);
                                }
                                warm_ok = 1;
                            }
                        }
                        free(dA); free(xt); free(ystd); free(ymin);
                    }
                    free(sb);
                    stdform_free(sf);
                }
            }
            free(cp); free(sub); free(val); free(cint);
        }
        if (!warm_ok) {
            rc = PRIMAL_optimize(t);
            free(y); free(z); free(lx); free(ux); free(lc); free(uc);
            free(x); free(bvi); free(bri);
            return rc;
        }
        /* warm_ok: falls through to publication */
    }
    {
        /* store the basic solution */
        PRIMALrescodee rp = opt_prepare(t);
        if (rp != PRIMAL_RES_OK) {
            free(y); free(z); free(lx); free(ux); free(lc); free(uc);
            free(x); free(bvi); free(bri);
            return rp;
        }
        memcpy(t->x, x, (size_t)nvar * sizeof(double));
        for (int i = 0; i < ncon; i++) {
            double yy = y[i];
            t->y[i] = yy;
            t->slc[i] = yy < 0.0 ? yy : 0.0;
            t->suc[i] = yy > 0.0 ? yy : 0.0;
        }
        for (int j = 0; j < nvar; j++) {
            double zz = z[j];
            t->slx[j] = zz < 0.0 ? zz : 0.0;
            t->sux[j] = zz > 0.0 ? zz : 0.0;
        }
        double po = t->cfix;
        for (int j = 0; j < nvar; j++) po += t->c[j] * x[j];
        t->pobj = po;
        t->dobj = po;
        t->has_sol = 1;
        t->solsta = PRIMAL_SOL_STA_OPTIMAL;
        t->result_stale = 0;
        {
            char pb[64];
            snprintf(pb, sizeof pb, "basis solved (%d basic)", nbas);
            tprog(t, pb);
            cb_fire(t, PRIMAL_CALLBACK_PRIMAL_SIMPLEX);
        }
    }
    free(y); free(z); free(lx); free(ux); free(lc); free(uc);
    free(x); free(bvi); free(bri);
    return rc;
}

/* ---- solution I/O (reference writesolution/readsolution and variants) ----
 * Declared deviation: the reference FORMAT was not read, so here there is a
 * text format of our own ("PRIMAL-SOLUTION 1", one line per vector
 * `tag n v1 ... vn`) that round-trips, and a binary dump of our own for
 * `writebsolution`/`readbsolution`. `writesolutionfile`/`readsolutionfile` are
 * the same calls (a single format). */
