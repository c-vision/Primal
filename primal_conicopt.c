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
/* primal_conicopt.c - exp/power tangent cuts, optimize_conic.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* Sum of the stored A entries in row i, column j (duplicates included). */
static double conic_aij(PRIMALtask_t t, int i, int j) {
    const Col *cc = &t->cols[j];
    double s = 0.0;
    for (int q = 0; q < cc->nz; q++)
        if (cc->sub[q] == i) s += cc->val[q];
    return s;
}

/* kinds of R_+ rows / equalities (for dual mapping) */
enum { CR_ROWLO = 0, CR_ROWUP = 1, CR_VARLO = 2, CR_VARUP = 3,
       CR_VARNEG = 4, CR_CUT = 5, CR_UNUSED = 6, CR_PSDCUT = 7,
       EQ_FXROW = 0, EQ_FXVAR = 1, EQ_RQUAD_U = 2, EQ_RQUAD_V = 3,
       EQ_RQUAD_W = 4, EQ_RQUAD_T = 5, EQ_RQUAD_R = 6, EQ_LINK = 7 };

/* ---- nonlinear cones (PEXP/DEXP/PPOW/RPOW): tangent cuts ----
 * The PEXP/DEXP/PPOW/RPOW branch is not handled by a native conic solver:
 * their set is {phi(x) <= 0} with phi convex (hypograph), so it is
 * outer-approximated by tangent cuts (linearization of phi at the current
 * point) and re-solving the SOCP: monotone convergence to the optimum (outer
 * approximation of a convex set).
 * Conventions:
 *   PEXP(m0,m1,m2): m0 >= m1*exp(m2/m1), m1 >= 0
 *   DEXP(m0,m1,m2): (−m0,−m1,−m2) ∈ PEXP  (dual: m0 <= m1*exp(m2/m1), m1<=0)
 *   PPOW(m0,m1,m2;a): m0^a*m1^(1-a) >= |m2|, m0,m1 >= 0, a=coneparam
 *   RPOW(m0,m1,m2;a): sqrt2*m0^a*m1^(1-a) >= |m2|  (a=1/2 -> 2*m0*m1 >= m2^2,
 *                      coincides with RQUAD) */
#define EXPP_MAXROUND 60
#define EXPP_WCAP     10.0   /* cap on v/u in exp cuts (avoids overflow) */
#define EXPP_FCAP     50.0   /* cap on ln(f0) in pow cuts */
static double expp_fval(int type, double alpha, double u, double v) {
    switch (type) {
        case PRIMAL_CT_PEXP: {
            if (u > 1e-12) { double w = v / u; if (w > EXPP_WCAP) w = EXPP_WCAP; return u * exp(w); }
            return (v <= 0.0) ? 0.0 : INFINITY;
        }
        case PRIMAL_CT_DEXP: {  /* (−m0,−m1,−m2) ∈ PEXP: t=−m0, u=−m1, v=−m2 */
            if (u > 1e-12) { double w = v / u; if (w > EXPP_WCAP) w = EXPP_WCAP; return u * exp(w); }
            return (v <= 0.0) ? 0.0 : INFINITY;
        }
        case PRIMAL_CT_PPOW: {
            double p = 1.0 / alpha;
            if (u <= 1e-12) return (fabs(v) < 1e-12) ? 0.0 : INFINITY;
            double l = p * log(fabs(v) > 1e-300 ? fabs(v) : 1e-300)
                     + (1.0 - p) * log(u);
            if (l > EXPP_FCAP) l = EXPP_FCAP;
            return exp(l);
        }
        default: return 0.0;
    }
}

/* adds a tangent cut for a nonlinear cone at the current point.
 * returns the number of rows written (1 or 2) at position rr of G/h. */
int expp_add_cut(int type, double alpha, const int *mem,
                        const double *xs,
                        int *cutcol, double *cuta, double *cuth) {
    int nc = 0;
    if (type == PRIMAL_CT_PEXP || type == PRIMAL_CT_DEXP) {
        int sg = (type == PRIMAL_CT_PEXP) ? 1 : -1;   /* DEXP: negated variables */
        double u = sg * xs[mem[1]], v = sg * xs[mem[2]];
        if (u < 1e-2) u = 1e-2;          /* clamp: well-conditioned tangents */
        double w = v / u; if (w > EXPP_WCAP) { w = EXPP_WCAP; v = w * u; }
        double f0 = u * exp(w);
        double fu = exp(w) * (1.0 - w);
        double fv = exp(w);
        double k = f0 - fu * u - fv * v;
        /* row: sg*m0 - fu*(sg*m1) - fv*(sg*m2) - k >= 0 */
        cutcol[0] = mem[0]; cuta[0] = (double)sg;
        cutcol[1] = mem[1]; cuta[1] = -fu * (double)sg;
        cutcol[2] = mem[2]; cuta[2] = -fv * (double)sg;
        cuth[0] = -k;
        nc = 1;
    } else if (type == PRIMAL_CT_PPOW) {
        double p = 1.0 / alpha;
        double u = xs[mem[1]], v = xs[mem[2]];
        if (u < 1e-2) u = 1e-2;
        double f0 = expp_fval(PRIMAL_CT_PPOW, alpha, u, v);
        double fu = (1.0 - p) * f0 / u;
        double fv = (v != 0.0) ? p * f0 / v : 0.0;
        double k = f0 - fu * u - fv * v;
        /* t - fu*u - fv*v - k >= 0 */
        cutcol[0] = mem[0]; cuta[0] = 1.0;
        cutcol[1] = mem[1]; cuta[1] = -fu;
        cutcol[2] = mem[2]; cuta[2] = -fv;
        cuth[0] = -k;
        nc = 1;
    } else { /* RPOW: |m2| <= sqrt2*m0^a*m1^(1-a), cut of the concave function g */
        double m0 = xs[mem[0]], m1 = xs[mem[1]];
        if (m0 < 1e-2) m0 = 1e-2;
        if (m1 < 1e-2) m1 = 1e-2;
        double g0 = sqrt(2.0) * pow(m0, alpha) * pow(m1, 1.0 - alpha);
        double gx = alpha * g0 / m0, gy = (1.0 - alpha) * g0 / m1;
        double cT = g0 - gx * m0 - gy * m1;   /* T(x) = gx*m0 + gy*m1 + cT */
        cutcol[0] = mem[0]; cuta[0] = gx;
        cutcol[1] = mem[1]; cuta[1] = gy;
        cutcol[2] = mem[2]; cuta[2] = -1.0;
        cutcol[3] = mem[0]; cuta[3] = gx;
        cutcol[4] = mem[1]; cuta[4] = gy;
        cutcol[5] = mem[2]; cuta[5] = 1.0;
        cuth[0] = cT; cuth[1] = cT;
        nc = 2;
    }
    /* normalize each row so the cut LP is not ill-conditioned by the exp scale */
    for (int rr = 0; rr < nc; rr++) {
        double nrm = 0.0;
        for (int e = 0; e < 3; e++) { double a = fabs(cuta[3*rr+e]); if (a > nrm) nrm = a; }
        if (nrm > 0.0) { for (int e = 0; e < 3; e++) cuta[3*rr+e] /= nrm; cuth[rr] /= nrm; }
    }
    return nc;
}

static PRIMALrescodee optimize_conic_impl(PRIMALtask_t t, int s);
/* Wrapper: fires the BEGIN/END conic callbacks around the implementation. */
PRIMALrescodee optimize_conic(PRIMALtask_t t, int s) {
    iter_cb_begin(t);
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_CONIC);
    PRIMALrescodee r = optimize_conic_impl(t, s);
    cb_fire(t, PRIMAL_CALLBACK_END_CONIC);
    iter_cb_end();
    return r;
}
/* Builds the conic outer approximation (tangent cuts for nonlinear cones,
 * big-M rows for PSD bars) and iterates until the stopping test holds. */
static PRIMALrescodee optimize_conic_impl(PRIMALtask_t t, int s) {
    int nvar = t->numvar, ncon = t->numcon;

    if (t->has_qobj)
        return PRIMAL_RES_ERR_ARG;   /* documented deviation: QP + cones unsupported */

    /* ---------- SDP bars: extended variables for the upper triangles ----
     * X_j entries (p<=q) become free variables with big-M bounds; the PSD
     * cone is outer-approximated by tangent cuts on -lambda_min added in
     * the outer loop (like the nonlinear-cone cuts). */
    int nb = t->numbarvar;
    int *barOff = NULL, *barPq = NULL;
    int nbarvar = 0;   /* total bar entries */
    int maxbardim = 1;
    if (nb > 0) {
        barOff = (int *)malloc((size_t)nb * sizeof(int));
        barPq = (int *)malloc((size_t)nb * sizeof(int));
        if (!barOff || !barPq) { free(barOff); free(barPq); return PRIMAL_RES_ERR_ALLOC; }
        for (int j = 0; j < nb; j++) {
            int d = t->barDim[j];
            barPq[j] = d * (d + 1) / 2;
            nbarvar += barPq[j];
            if (d > maxbardim) maxbardim = d;
        }
    }

    /* ---------- sizes ---------- */
    int nR = 0, neq = 0, naux = 0, nSocTot = 0, nNlin = 0, nCutMax = 0;
    for (int j = 0; j < nvar; j++) {
        if (t->bkx[j] == PRIMAL_BK_LO || t->bkx[j] == PRIMAL_BK_RA) nR++;
        if (t->bkx[j] == PRIMAL_BK_UP || t->bkx[j] == PRIMAL_BK_RA) nR++;
        if (t->bkx[j] == PRIMAL_BK_FX) nR += 2;   /* v >= bl and -v >= -bl:
                                                  avoids E-row duplication when
                                                  the fixed var also appears in
                                                  user equality rows (KKT would
                                                  be rank-deficient) */
    }
    for (int i = 0; i < ncon; i++) {
        if (t->bkc[i] == PRIMAL_BK_LO || t->bkc[i] == PRIMAL_BK_RA) nR++;
        if (t->bkc[i] == PRIMAL_BK_UP || t->bkc[i] == PRIMAL_BK_RA) nR++;
        if (t->bkc[i] == PRIMAL_BK_FX) neq++;
    }
    for (int k = 0; k < t->numcones; k++) {
        int m = t->cone_nmem[k];
        if (t->cone_type[k] == PRIMAL_CT_QUAD) {
            nSocTot += m;
        } else if (t->cone_type[k] == PRIMAL_CT_RQUAD) {
            naux += m + 2;  neq += m + 2;  nSocTot += m;
        } else {
            /* nonlinear: 3 auxiliary variables (signed members), 3
             * link equalities, 2 sign rows on the auxiliaries, tangent cuts on
             * the auxiliaries */
            nNlin++;
            naux += 3; neq += 3; nR += 2; /* sign rows on A0, A1 */
            nCutMax += (EXPP_MAXROUND + 1) * ((t->cone_type[k] == PRIMAL_CT_RPOW) ? 2 : 1);
        }
    }
    int ntot = nvar + naux + nbarvar;
    /* big-M bounds on every bar entry: 2 R_+ rows each (>= -M, <= M) */
    nR += 2 * nbarvar;
    /* PSD cut slots: one per (bar, round) like the SDP path (SDP_MAXROUND),
     * stored in the cut block of G/h (R_+ rows) */
    int nCutPsd = nb * (SDP_MAXROUND + 1);
    int K = nR + nSocTot + nCutMax + nCutPsd;
    if (K == 0) return PRIMAL_RES_ERR_ARG;
    if (nb > 0) {
        /* One block of compressed upper-triangle columns per bar, after the aux
         * columns. Naming every bar the SAME base (as this line did) makes a
         * second bar overwrite the first: the rows of bar 1 then read bar 0's
         * variables, which is not the model -- and for equal bar rows it
         * duplicates them, so the KKT system is rank-deficient too. */
        int bo = nvar + naux;
        for (int j = 0; j < nb; j++) { barOff[j] = bo; bo += barPq[j]; }
    }

    double *E = neq ? (double *)calloc((size_t)neq * (size_t)ntot, sizeof(double)) : NULL;
    double *d = neq ? (double *)calloc((size_t)neq, sizeof(double)) : NULL;
    double *G = (double *)calloc((size_t)K * (size_t)ntot, sizeof(double));
    double *h = (double *)malloc((size_t)K * sizeof(double));
    double *c = (double *)calloc((size_t)ntot, sizeof(double));
    /* solver cones: R_+ block + one per user QUAD/RQUAD (nonlinear cones are
     * cut-based, no solver cone) + the cut R_+ cone => numcones + 3 max */
    SocpCone *cones = (SocpCone *)calloc((size_t)(t->numcones + 3), sizeof(SocpCone));
    int *cmem = (int *)malloc((size_t)(nSocTot > 0 ? nSocTot : 1) * sizeof(int));
    int *rowKind = (int *)malloc((size_t)K * sizeof(int));
    int *rowIdx  = (int *)malloc((size_t)K * sizeof(int));
    int *eqKind  = (int *)malloc((size_t)(neq > 0 ? neq : 1) * sizeof(int));
    int *eqIdx   = (int *)malloc((size_t)(neq > 0 ? neq : 1) * sizeof(int));
    int *eqAux   = (int *)malloc((size_t)(neq > 0 ? neq : 1) * sizeof(int));
    if (!G || !h || !c || !cones || !cmem || !rowKind || !rowIdx ||
        !eqKind || !eqIdx || !eqAux || (neq && (!E || !d))) {
        free(E); free(d); free(G); free(h); free(c); free(cones); free(cmem);
        free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int q = 0; q < neq; q++) eqAux[q] = -1;
    for (int j = 0; j < nvar; j++) c[j] = s * t->c[j];

    int r = 0, e = 0, co = 0;   /* R_+ row, equality, soc-member cursors */

    /* ---------- variable bounds ---------- */
    for (int j = 0; j < nvar; j++) {
        if (t->bkx[j] == PRIMAL_BK_LO || t->bkx[j] == PRIMAL_BK_RA) {
            G[r * ntot + j] = 1.0; h[r] = -t->blx[j];
            rowKind[r] = CR_VARLO; rowIdx[r] = j; r++;
        }
        if (t->bkx[j] == PRIMAL_BK_UP || t->bkx[j] == PRIMAL_BK_RA) {
            G[r * ntot + j] = -1.0; h[r] = t->bux[j];
            rowKind[r] = CR_VARUP; rowIdx[r] = j; r++;
        }
        if (t->bkx[j] == PRIMAL_BK_FX) {
            /* fixed var as two opposite R_+ rows (KKT conditioning: avoids a
             * duplicate E column when the var also appears in EQ rows) */
            G[r * ntot + j] = 1.0; h[r] = -t->blx[j];
            rowKind[r] = CR_VARLO; rowIdx[r] = j; r++;
            G[r * ntot + j] = -1.0; h[r] = t->blx[j];
            rowKind[r] = CR_VARUP; rowIdx[r] = j; r++;
        }
    }
    /* ---------- SDP bar entries: big-M rows + compressed matrices ------ */
    double **symPq = NULL;
    if (nb > 0) {
        symPq = (double **)malloc((size_t)(t->nsym > 0 ? t->nsym : 1) * sizeof(double *));
        if (!symPq) {
            free(E); free(d); free(G); free(h); free(c); free(cones); free(cmem);
            free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux);
            free(barOff); free(barPq);
            return PRIMAL_RES_ERR_ALLOC;
        }
        int ok = 1;
        for (int m = 0; m < t->nsym && ok; m++) {
            int dd = t->sym_dim[m];
            double *M = (double *)calloc((size_t)dd * (size_t)dd, sizeof(double));
            if (!M) { ok = 0; break; }
            for (int q2 = 0; q2 < t->sym_nnz[m]; q2++) {
                int si = t->sym_subi[m][q2], sj = t->sym_subj[m][q2];
                double v = t->sym_val[m][q2];
                M[si * dd + sj] += v;
                if (si != sj) M[sj * dd + si] += v;
            }
            int pq = dd * (dd + 1) / 2;
            symPq[m] = (double *)malloc((size_t)pq * sizeof(double));
            if (!symPq[m]) { free(M); ok = 0; break; }
            for (int p = 0; p < dd; p++)
                for (int q2 = p; q2 < dd; q2++)
                    symPq[m][bar_pack(dd, p, q2)] =
                        (p == q2) ? M[p * dd + q2] : 2.0 * M[p * dd + q2];
            free(M);
        }
        if (!ok) {
            for (int m = 0; m < t->nsym; m++) free(symPq[m]);
            free(symPq);
            free(E); free(d); free(G); free(h); free(c); free(cones); free(cmem);
            free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux);
            free(barOff); free(barPq);
            return PRIMAL_RES_ERR_ALLOC;
        }
        /* big-M rows on bar entries: +/- (entry) <= M. The cap is what makes the
         * first solve of this build answerable, not a bound of the model, so an
         * answer that sits on it is settled the same way the cut route settles
         * its own: after the loop, bar_cap_ray asks the MODEL whether it has a
         * recession direction and the measurement decides the verdict. */
        for (int j = 0; j < nb; j++) {
            for (int p = 0; p < barPq[j]; p++) {
                int v = barOff[j] + p;
                G[r * ntot + v] = 1.0; h[r] = SDP_BIGM;
                rowKind[r] = CR_VARLO; rowIdx[r] = v; r++;
                G[r * ntot + v] = -1.0; h[r] = SDP_BIGM;
                rowKind[r] = CR_VARUP; rowIdx[r] = v; r++;
            }
        }
        /* barC into the objective */
        for (int k = 0; k < t->nbarC; k++) {
            int b = t->barC_bar[k], m = t->barC_sym[k], dd = t->barDim[b];
            int pq = dd * (dd + 1) / 2;
            for (int p = 0; p < pq; p++)
                c[barOff[b] + p] += s * t->barC_coef[k] * symPq[m][p];
        }
    }
    /* ---------- linear rows (bar A terms appended to each row) ---------- */
    for (int i = 0; i < ncon; i++) {
        if (t->bkc[i] == PRIMAL_BK_FX) {
            for (int j = 0; j < nvar; j++) E[e * ntot + j] = conic_aij(t, i, j);
            for (int k = 0; k < t->nbarA; k++) {
                if (t->barA_con[k] != i) continue;
                int b = t->barA_bar[k], m = t->barA_sym[k], dd = t->barDim[b];
                int pq = dd * (dd + 1) / 2;
                for (int p = 0; p < pq; p++)
                    E[e * ntot + barOff[b] + p] += t->barA_coef[k] * symPq[m][p];
            }
            d[e] = t->blc[i];
            eqKind[e] = EQ_FXROW; eqIdx[e] = i; e++;
            continue;
        }
        if (t->bkc[i] == PRIMAL_BK_LO || t->bkc[i] == PRIMAL_BK_RA) {
            for (int j = 0; j < nvar; j++) G[r * ntot + j] = conic_aij(t, i, j);
            for (int k = 0; k < t->nbarA; k++) {
                if (t->barA_con[k] != i) continue;
                int b = t->barA_bar[k], m = t->barA_sym[k], dd = t->barDim[b];
                int pq = dd * (dd + 1) / 2;
                for (int p = 0; p < pq; p++)
                    G[r * ntot + barOff[b] + p] += t->barA_coef[k] * symPq[m][p];
            }
            h[r] = -t->blc[i];
            rowKind[r] = CR_ROWLO; rowIdx[r] = i; r++;
        }
        if (t->bkc[i] == PRIMAL_BK_UP || t->bkc[i] == PRIMAL_BK_RA) {
            for (int j = 0; j < nvar; j++) G[r * ntot + j] = -conic_aij(t, i, j);
            for (int k = 0; k < t->nbarA; k++) {
                if (t->barA_con[k] != i) continue;
                int b = t->barA_bar[k], m = t->barA_sym[k], dd = t->barDim[b];
                int pq = dd * (dd + 1) / 2;
                for (int p = 0; p < pq; p++)
                    G[r * ntot + barOff[b] + p] -= t->barA_coef[k] * symPq[m][p];
            }
            h[r] = t->buc[i];
            rowKind[r] = CR_ROWUP; rowIdx[r] = i; r++;
        }
    }
    /* ---------- user cones ---------- */
    int ncones_solver = 0;
    if (nR > 0) {
        cones[ncones_solver].type = 0;
        cones[ncones_solver].nmem = nR;
        cones[ncones_solver].mem = NULL;
        ncones_solver++;
    }
    int *auxBase = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    int nauxDone = 0;   /* auxiliary variables cursor */
    int *nlType = (int *)malloc((size_t)(nNlin > 0 ? nNlin : 1) * sizeof(int));
    double *nlAlpha = (double *)malloc((size_t)(nNlin > 0 ? nNlin : 1) * sizeof(double));
    int *nlMem = (int *)malloc((size_t)(3 * (nNlin > 0 ? nNlin : 1)) * sizeof(int));
    if (!auxBase || !nlType || !nlAlpha || !nlMem) {
        free(E); free(d); free(G); free(h); free(c); free(cones); free(cmem);
        free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux); free(auxBase); free(nlType); free(nlAlpha); free(nlMem);
        return PRIMAL_RES_ERR_ALLOC;
    }
    /* ---------- user cones: nonlinear-cone pre-pass ----------
     * the nonnegativity rows of the members must be in the R_+ block
     * BEFORE the cut slots and the identity rows of the QUAD/RQUAD members */
    nNlin = 0;
    /* auxiliary base for ALL cones, in order of appearance */
    for (int k = 0; k < t->numcones; k++) {
        int ct = t->cone_type[k], m = t->cone_nmem[k];
        if (ct == PRIMAL_CT_RQUAD) { auxBase[k] = nvar + nauxDone; nauxDone += m + 2; }
        else if (ct == PRIMAL_CT_QUAD) auxBase[k] = -1;
        else { auxBase[k] = nvar + nauxDone; nauxDone += 3; }
    }
    nauxDone = 0;
    for (int k = 0; k < t->numcones; k++) {
        int ct = t->cone_type[k];
        if (ct != PRIMAL_CT_PEXP && ct != PRIMAL_CT_DEXP &&
            ct != PRIMAL_CT_PPOW && ct != PRIMAL_CT_RPOW) continue;
        const int *mem = t->cone_mem[k];
        int sg = (ct == PRIMAL_CT_DEXP) ? -1 : 1;
        int A0 = auxBase[k];
        int nl = nNlin++;
        /* DEXP = PEXP in the auxiliary space (auxiliaries = signed members) */
        nlType[nl] = (ct == PRIMAL_CT_DEXP) ? PRIMAL_CT_PEXP : ct;
        nlAlpha[nl] = t->cone_param[k];
        for (int i = 0; i < 3; i++) {
            nlMem[3 * nl + i] = A0 + i;
            /* link equality: A_i - sg*m_i = 0 */
            eqKind[e] = EQ_LINK; eqIdx[e] = mem[i]; eqAux[e] = A0 + i;
            E[e * ntot + A0 + i] = 1.0; E[e * ntot + mem[i]] = -(double)sg;
            d[e] = 0.0; e++;
        }
        nauxDone += 3;
        /* sign rows on the auxiliaries: A0 >= 0, A1 >= 0 */
        for (int i = 0; i < 2; i++) {
            G[r * ntot + A0 + i] = 1.0; h[r] = 0.0;
            rowKind[r] = CR_VARLO; rowIdx[r] = A0 + i; r++;
        }
    }

    for (int k = 0; k < t->numcones; k++) {
        int m = t->cone_nmem[k];
        const int *mem = t->cone_mem[k];
        if (t->cone_type[k] == PRIMAL_CT_QUAD) {
            auxBase[k] = -1;
            cones[ncones_solver].type = 1;
            cones[ncones_solver].nmem = m;
            cones[ncones_solver].mem = cmem + co;
            for (int i = 0; i < m; i++) {
                cmem[co] = mem[i];
                G[r * ntot + mem[i]] = 1.0;   /* s = x_member */
                h[r] = 0.0;
                rowKind[r] = CR_VARLO; rowIdx[r] = mem[i]; r++;
                co++;
            }
            ncones_solver++;
        } else if (t->cone_type[k] == PRIMAL_CT_RQUAD) {
            /* RQUAD members: u=mem[0], v=mem[1], w_i=mem[2+i]
             * aux vars: U=nvar+nauxDone, V=+1, W_i=+2.., T=+m, R=+m+1 */
            int U = auxBase[k], V = U + 1, W0 = U + 2;
            int T = W0 + (m - 2), R = T + 1;
            /* equalities: U-u, V-v, W_i-w_i, sqrt2*T-u-v, sqrt2*R-u+v */
            eqKind[e] = EQ_RQUAD_U; eqIdx[e] = mem[0];
            E[e * ntot + U] = 1.0; E[e * ntot + mem[0]] = -1.0; d[e] = 0.0; e++;
            eqKind[e] = EQ_RQUAD_V; eqIdx[e] = mem[1];
            E[e * ntot + V] = 1.0; E[e * ntot + mem[1]] = -1.0; d[e] = 0.0; e++;
            for (int i = 2; i < m; i++) {
                eqKind[e] = EQ_RQUAD_W; eqIdx[e] = mem[i];
                E[e * ntot + W0 + i - 2] = 1.0; E[e * ntot + mem[i]] = -1.0; d[e] = 0.0; e++;
            }
            eqKind[e] = EQ_RQUAD_T; eqIdx[e] = -1;
            E[e * ntot + T] = sqrt(2.0); E[e * ntot + U] = -1.0; E[e * ntot + V] = -1.0; d[e] = 0.0; e++;
            eqKind[e] = EQ_RQUAD_R; eqIdx[e] = -1;
            E[e * ntot + R] = sqrt(2.0); E[e * ntot + U] = -1.0; E[e * ntot + V] = 1.0; d[e] = 0.0; e++;
            /* SOC over (T, R, W_1..W_{m-2}) with identity G rows */
            cones[ncones_solver].type = 1;
            cones[ncones_solver].nmem = m;
            cones[ncones_solver].mem = cmem + co;
            cmem[co] = T;
            G[r * ntot + T] = 1.0; h[r] = 0.0;
            rowKind[r] = CR_VARLO; rowIdx[r] = T; r++; co++;
            cmem[co] = R;
            G[r * ntot + R] = 1.0; h[r] = 0.0;
            rowKind[r] = CR_VARLO; rowIdx[r] = R; r++; co++;
            for (int i = 2; i < m; i++) {
                cmem[co] = W0 + i - 2;
                G[r * ntot + W0 + i - 2] = 1.0; h[r] = 0.0;
                rowKind[r] = CR_VARLO; rowIdx[r] = W0 + i - 2; r++; co++;
            }
            ncones_solver++;
        }
        /* nonlinear cones: no solver cone; signs and equalities already in the
         * pre-pass, tangent cuts handled in the outer loop */
    }

    int nCutAll = nCutMax + nCutPsd;
    int r_cut = r;   /* the cuts (conic + PSD) occupy [r_cut, r_cut+nCutAll) */
    if (getenv("GMB_DBG")) {
        fprintf(stderr, "  [phase] conic asm=%.4fs\n",
                (double)(clock() - g_phase_t0) / (double)CLOCKS_PER_SEC);
        g_phase_t0 = clock();
    }
    /* R_+ cone for the cuts: after the QUAD/RQUAD cones (positional layout) */
    int cutCone = -1;
    if (nCutAll > 0) {
        cutCone = ncones_solver;
        cones[cutCone].type = 0;
        cones[cutCone].mem = NULL;
        ncones_solver++;
    }

    /* ---------- tangent cuts + PSD cuts + iterative solve ---------- */
    int *cutcol = (int *)malloc((size_t)(3 * (nCutAll > 0 ? nCutAll : 1)) * sizeof(int));
    double *cuta = (double *)malloc((size_t)(3 * (nCutAll > 0 ? nCutAll : 1)) * sizeof(double));
    double *cuth = (double *)malloc((size_t)(nCutAll > 0 ? nCutAll : 1) * sizeof(double));
    double *xs = (double *)calloc((size_t)(ntot > 0 ? ntot : 1), sizeof(double));
    double *ys = (double *)calloc((size_t)(neq > 0 ? neq : 1), sizeof(double));
    double *lm = (double *)calloc((size_t)(K > 0 ? K : 1), sizeof(double));
    /* PSD scratch: eigen for each bar matrix */
    double *psd_eval = nb > 0 ? (double *)malloc((size_t)maxbardim * sizeof(double)) : NULL;
    double *psd_evec = nb > 0 ? (double *)malloc((size_t)maxbardim * (size_t)maxbardim * sizeof(double)) : NULL;
    double *psd_X = nb > 0 ? (double *)malloc((size_t)maxbardim * (size_t)maxbardim * sizeof(double)) : NULL;
    double *psd_cut_val = (nb > 0 && nCutPsd > 0)
        ? (double *)calloc((size_t)nCutPsd * (size_t)ntot, sizeof(double)) : NULL;
    double *psd_cut_rhs = (nb > 0 && nCutPsd > 0)
        ? (double *)malloc((size_t)nCutPsd * sizeof(double)) : NULL;
    int ncuts = 0;          /* nonlinear-cone cuts (3-col format) */
    int ncuts_psd = 0;      /* PSD cuts (dense over bar entries) */
    PRIMALrescodee rcs = PRIMAL_RES_OK;
    if (!cutcol || !cuta || !cuth || !xs || !ys || !lm || !auxBase ||
        (nb > 0 && (!psd_eval || !psd_evec || !psd_X || !psd_cut_val || !psd_cut_rhs))) {
        rcs = PRIMAL_RES_ERR_ALLOC;
    } else {
        double tol_out = (t->tol_co_pfeas < t->tol_co_dfeas) ? t->tol_co_pfeas : t->tol_co_dfeas;
        double tolv_psd = 1e-9 * (1.0 + (double)maxbardim);
        for (int round = 0;; round++) {
            /* cut rows: slots [r_cut, r_cut+nCutAll) */
            if (nCutAll > 0)
                memset((void *)(G + (size_t)r_cut * ntot), 0,
                       (size_t)(nCutAll * ntot) * sizeof(double));
            /* round 0: problems without cuts are often unbounded (the nonlinear
             * member variables have no G rows): add an initial tangent cut for
             * every cone right away */
            if (ncuts == 0 && nNlin > 0) {
                int nw = 0;
                double *sxv = (double *)calloc((size_t)ntot, sizeof(double));
                for (int k = 0; k < nNlin; k++) {
                    /* tangent at (u,v)=(1,0): the members are the auxiliaries */
                    sxv[nlMem[3 * k + 0]] = 1.0;
                    sxv[nlMem[3 * k + 1]] = 1.0;
                    sxv[nlMem[3 * k + 2]] = (nlType[k] == PRIMAL_CT_PPOW) ? 1.0 : 0.0;
                    nw += expp_add_cut(nlType[k], nlAlpha[k], nlMem + 3 * k, sxv,
                                       cutcol + 3 * nw, cuta + 3 * nw, cuth + nw);
                    sxv[nlMem[3 * k + 0]] = 0.0;
                    sxv[nlMem[3 * k + 1]] = 0.0;
                }
                free(sxv);
                ncuts = nw;
            }

            if (cutCone >= 0) cones[cutCone].nmem = ncuts + ncuts_psd;
            int rr = r_cut;
            for (int q = 0; q < ncuts; q++) {
                for (int i = 0; i < 3; i++)
                    G[(size_t)rr * ntot + cutcol[3 * q + i]] += cuta[3 * q + i];
                h[rr] = cuth[q];
                rowKind[rr] = CR_CUT; rowIdx[rr] = -1;
                rr++;
            }
            for (int q = 0; q < ncuts_psd; q++) {
                const double *cv = psd_cut_val + (size_t)q * (size_t)ntot;
                for (int v = 0; v < ntot; v++)
                    if (cv[v] != 0.0) G[(size_t)rr * ntot + v] += cv[v];
                h[rr] = psd_cut_rhs[q];
                rowKind[rr] = CR_PSDCUT; rowIdx[rr] = -1;
                rr++;
            }
            int Nsys = ntot + neq;
            for (int kk = 0; kk < ncones_solver; kk++) Nsys += cones[kk].nmem;
            /* Many small cones give a sparse augmented system well below 800:
             * the 100-cone SOCP benchmark is about 3x faster on that path.
             * Keep dense LU for a single large, dense-ish cone. */
            int sparse_conic = (Nsys >= 300 && ncones_solver >= 16) ||
                               (Nsys >= 800) || (getenv("GMB_SOCP_SPARSE") != NULL);
            int st = sparse_conic
                ? socp_solve_sparse(ntot, neq, E, d, c, ncones_solver, cones, G, h,
                                    t->tol_co_gap, tol_out, iter_cap(t->max_iter_intpnt), xs, ys, lm)
                : socp_solve(ntot, neq, E, d, c, ncones_solver, cones, G, h,
                             t->tol_co_gap, tol_out, iter_cap(t->max_iter_intpnt), xs, ys, lm);
            if (st != 0) {
                /* No point is published here. The iterate socp_solve left in
                 * xs/ys/lm is copied into t->x/barx/barsj by the publication
                 * block below, which this branch does not reach, so raising
                 * has_sol made the getters answer OK on the all-zero buffer
                 * opt_prepare left -- and getprimalinfeas, reading that same
                 * buffer, answered how much x = 0 violates the model (measured
                 * 2.0 on the bounded bar of T100 C). The verdict travels in rc. */
                t->solsta = PRIMAL_SOL_STA_UNKNOWN;
                rcs = (st == 1) ? PRIMAL_RES_TRM_MAX_ITER : PRIMAL_RES_ERR_ARG;
                break;
            }
            {
                char pb[96];
                snprintf(pb, sizeof pb, "conic round %d, cuts %d+%d", round, ncuts, ncuts_psd);
                tprog(t, pb);
                cb_fire(t, PRIMAL_CALLBACK_CONIC);
            }
            /* Worst RELATIVE violation of the nonlinear cones. Each block is
             * read in its own homogeneous units (the t-space form of PPOW
             * shrinks a small t by 1/(a t^(a-1)): measured 6.4x on
             * regression_regularized, which accepted a 3.1e-8 violation against
             * a 1e-8 tolerance) and normalised by the block's own size, which
             * is how rel_pri normalises by 1+|b|. */
            double viol = 0.0;
            for (int k = 0; k < nNlin; k++) {
                const int *m = nlMem + 3 * k;
                double m0 = xs[m[0]], m1 = xs[m[1]], m2 = xs[m[2]];
                double a = nlAlpha[k], sc = fabs(m0), vk;
                if (fabs(m1) > sc) sc = fabs(m1);
                if (fabs(m2) > sc) sc = fabs(m2);
                if (nlType[k] == PRIMAL_CT_PEXP) {
                    vk = expp_fval(nlType[k], a, m1, m2) - m0;
                    if (-m1 > vk) vk = -m1;
                } else {
                    double p0 = m0 > 1e-8 ? m0 : 1e-8, p1 = m1 > 1e-8 ? m1 : 1e-8;
                    double g = (nlType[k] == PRIMAL_CT_RPOW ? sqrt(2.0) : 1.0)
                             * pow(p0, a) * pow(p1, 1.0 - a);
                    vk = fabs(m2) - g;
                    if (-m0 > vk) vk = -m0;
                    if (-m1 > vk) vk = -m1;
                }
                vk /= 1.0 + sc;
                if (vk > viol) viol = vk;
            }
            /* PSD violation: lambda_min(X_j) >= -tol for every bar */
            int psd_viol = 0;
            if (nb > 0) {
                for (int j = 0; j < nb && !psd_viol; j++) {
                    int dd = t->barDim[j];
                    for (int p = 0; p < dd; p++)
                        for (int q2 = 0; q2 < dd; q2++)
                            psd_X[p * dd + q2] = xs[barOff[j] +
                                bar_pack(dd, (p < q2 ? p : q2), (p < q2 ? q2 : p))];
                    dmat_eig_jacobi(dd, psd_X, psd_eval, psd_evec);
                    int imin = 0;
                    for (int q2 = 1; q2 < dd; q2++)
                        if (psd_eval[q2] < psd_eval[imin]) imin = q2;
                    if (psd_eval[imin] < -tolv_psd) psd_viol = 1;
                }
            }
            if (getenv("GMB_DBG")) fprintf(stderr,
                "round=%d ncuts=%d psd=%d rel_viol=%.3g tol=%.3g\n",
                round, ncuts, ncuts_psd, viol, tol_out);
            if (!(viol <= tol_out) || psd_viol) {
                /* not (yet) inside the cones: add cuts and repeat */
                if (round >= EXPP_MAXROUND) { rcs = PRIMAL_RES_TRM_MAX_ITER; break; }
                for (int k = 0; k < nNlin; k++) {
                    int nc = expp_add_cut(nlType[k], nlAlpha[k], nlMem + 3 * k,
                                          xs,
                                          cutcol + 3 * ncuts, cuta + 3 * ncuts, cuth + ncuts);
                    if (ncuts + ncuts_psd + nc > nCutAll) { rcs = PRIMAL_RES_ERR_ALLOC; break; }
                    ncuts += nc;
                }
                if (rcs != PRIMAL_RES_OK) break;
                /* PSD tangent cuts per violated bar */
                for (int j = 0; j < nb; j++) {
                    int dd = t->barDim[j];
                    for (int p = 0; p < dd; p++)
                        for (int q2 = 0; q2 < dd; q2++)
                            psd_X[p * dd + q2] = xs[barOff[j] +
                                bar_pack(dd, (p < q2 ? p : q2), (p < q2 ? q2 : p))];
                    dmat_eig_jacobi(dd, psd_X, psd_eval, psd_evec);
                    int imin = 0;
                    for (int q2 = 1; q2 < dd; q2++)
                        if (psd_eval[q2] < psd_eval[imin]) imin = q2;
                    double lam = psd_eval[imin];
                    if (lam >= -tolv_psd) continue;
                    if (ncuts + ncuts_psd >= nCutAll) { rcs = PRIMAL_RES_ERR_ALLOC; break; }
                    /* tangent to -lambda_min at X0: <UU',X> >= <UU',X0> - lam */
                    double *cv = psd_cut_val + (size_t)ncuts_psd * (size_t)ntot;
                    memset(cv, 0, (size_t)ntot * sizeof(double));
                    double proj0 = 0.0;
                    for (int p = 0; p < dd; p++)
                        for (int q2 = p; q2 < dd; q2++) {
                            double uu = psd_evec[p * dd + imin], vv = psd_evec[q2 * dd + imin];
                            double cval = (p == q2) ? uu * vv : 2.0 * uu * vv;
                            cv[barOff[j] + bar_pack(dd, p, q2)] = cval;
                            proj0 += cval * xs[barOff[j] + bar_pack(dd, p, q2)];
                        }
                    psd_cut_rhs[ncuts_psd] = proj0 - lam;
                    ncuts_psd++;
                }
                if (rcs != PRIMAL_RES_OK) break;
                continue;
            }
            break;  /* converged */
        }
    }
    /* ---------- the cap is not a constraint of the model ----------
     * Same question the outer approximation asks of its own answer, same code
     * deciding it: a bar entry on +-SDP_BIGM stopped on a bound this build
     * invented. Anything but OK here leaves the route on its non-answer
     * cleanup below -- no point, no publication. */
    if (rcs == PRIMAL_RES_OK && nb > 0)
        rcs = bar_cap_verdict(t, s, symPq, nb, barOff, barPq, xs);
    if (rcs != PRIMAL_RES_OK) {
        free(xs); free(ys); free(lm); free(E); free(d); free(G); free(h); free(c);
        free(cones); free(cmem); free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux);
        free(auxBase); free(nlType); free(nlAlpha); free(nlMem);
        free(cutcol); free(cuta); free(cuth);
        if (nb > 0) {
            for (int m = 0; m < t->nsym; m++) free(symPq[m]);
            free(symPq); free(psd_eval); free(psd_evec); free(psd_X);
            free(psd_cut_val); free(psd_cut_rhs);
            free(barOff); free(barPq);
        }
        return rcs;
    }

    /* ---------- primal ---------- */
    for (int j = 0; j < nvar; j++) t->x[j] = xs[j];
    /* bar primal: map the compressed entries to the dense matrices */
    for (int j = 0; j < nb; j++) {
        int dd = t->barDim[j];
        double *X = t->barx[j];
        for (int k = 0; k < dd * dd; k++) X[k] = 0.0;
        for (int p = 0; p < dd; p++)
            for (int q2 = 0; q2 < dd; q2++)
                X[p * dd + q2] = xs[barOff[j] +
                    bar_pack(dd, (p < q2 ? p : q2), (p < q2 ? q2 : p))];
    }
    /* bar dual (approx): Z_j = C_j - sum_i y_i A^i, same convention as the
     * SDP path (computed after t->y is final, see below) */
    /* ---------- duals ---------- */
    double *ymin = (double *)calloc((size_t)(ncon > 0 ? ncon : 1), sizeof(double));
    double *zmin = (double *)calloc((size_t)ntot, sizeof(double));
    if (!ymin || !zmin) {
        free(ymin); free(zmin); free(xs); free(ys); free(lm); free(E); free(d);
        free(G); free(h); free(c); free(cones); free(cmem);
        free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux); free(auxBase); free(nlType); free(nlAlpha); free(nlMem); free(cutcol); free(cuta); free(cuth);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int q = 0; q < r; q++) {
        double lam = lm[q];
        if (lam == 0.0 || rowIdx[q] >= nvar) continue;  /* aux rows: dual discarded */
        switch (rowKind[q]) {
            case CR_ROWLO: ymin[rowIdx[q]] -= lam; break;
            case CR_ROWUP: ymin[rowIdx[q]] += lam; break;
            case CR_VARLO: zmin[rowIdx[q]] -= lam; break;
            case CR_VARUP: zmin[rowIdx[q]] += lam; break;
            case CR_VARNEG: zmin[rowIdx[q]] += lam; break;
            case CR_CUT:
                {
                    int ci = q - r_cut;
                    if (ci >= 0 && ci < nCutAll) {
                        for (int i = 0; i < 3; i++) {
                            int col = cutcol[3 * ci + i];
                            if (col >= 0 && col < ntot)
                                zmin[col] -= cuta[3 * ci + i] * lam;
                        }
                    }
                }
                break;
            case CR_PSDCUT:
                /* A PSD tangent lives on bar columns only, and the bar dual is
                 * assembled below from t->y and the model's own barA/barC terms
                 * -- the same convention the cut route publishes, where tangent
                 * multipliers are deliberately not in Z_j. Reading this row as a
                 * nonlinear cut would index cutcol slots that were never written. */
                break;
            default: break;
        }
    }
    for (int q = 0; q < e; q++) {
        double ye = ys[q];
        if (ye == 0.0) continue;
        switch (eqKind[q]) {
            case EQ_FXVAR: zmin[eqIdx[q]] += ye; break;
            case EQ_LINK:
                zmin[eqAux[q]] += ye;
                zmin[eqIdx[q]] -= ye;
                break;
            case EQ_RQUAD_U: zmin[eqIdx[q]] -= ye; break;
            case EQ_RQUAD_V: zmin[eqIdx[q]] -= ye; break;
            case EQ_RQUAD_W: zmin[eqIdx[q]] -= ye; break;
            default: break;  /* T,R equalities: aux only */
        }
    }
    /* FX rows: ymin_i = +y_eq (E row = a_i) */
    for (int q = 0; q < e; q++)
        if (eqKind[q] == EQ_FXROW) ymin[eqIdx[q]] += ys[q];

    for (int i = 0; i < ncon; i++) {
        double yy = s * ymin[i];
        t->y[i]   = yy;
        t->slc[i] = yy < 0.0 ? yy : 0.0;
        t->suc[i] = yy > 0.0 ? yy : 0.0;
    }
    /* Deviation, measured by T101 I: a cone member's dual lives in the multipliers
     * of the TANGENT rows, and those are the outer approximation, not the model --
     * lifting them would be the hole T89 warns about for Z_j. So a member of a cone
     * publishes slx+sux = 0 here even where the block's dual is (5/3,-4/3,1), and
     * the cone's dual feasibility is read from c and y instead (cone_dual_worst). */
    for (int j = 0; j < nvar; j++) {
        double zz = s * zmin[j];
        t->slx[j] = zz < 0.0 ? zz : 0.0;
        t->sux[j] = zz > 0.0 ? zz : 0.0;
    }

    /* ---------- bar duals (after t->y is final: same convention as SDP) --- */
    for (int j = 0; j < nb; j++) {
        int dd = t->barDim[j];
        double *Z = t->barsj[j];
        for (int k = 0; k < dd * dd; k++) Z[k] = 0.0;
        for (int k = 0; k < t->nbarC; k++) {
            if (t->barC_bar[k] != j) continue;
            int m = t->barC_sym[k];
            double cf = t->barC_coef[k];
            for (int q2 = 0; q2 < t->sym_nnz[m]; q2++) {
                int si = t->sym_subi[m][q2], sj = t->sym_subj[m][q2];
                double v = cf * t->sym_val[m][q2];
                Z[si * dd + sj] += v;
                if (si != sj) Z[sj * dd + si] += v;
            }
        }
        for (int k = 0; k < t->nbarA; k++) {
            if (t->barA_bar[k] != j) continue;
            int m = t->barA_sym[k];
            double cf = t->y[t->barA_con[k]] * t->barA_coef[k];
            for (int q2 = 0; q2 < t->sym_nnz[m]; q2++) {
                int si = t->sym_subi[m][q2], sj = t->sym_subj[m][q2];
                double v = cf * t->sym_val[m][q2];
                Z[si * dd + sj] += v;
                if (si != sj) Z[sj * dd + si] += v;
            }
        }
    }

    /* ---------- objective ---------- */
    double po = t->cfix;
    for (int j = 0; j < nvar; j++) po += t->c[j] * t->x[j];
    /* barC contribution: <C_j, X_j> via the compressed form */
    for (int k = 0; k < t->nbarC; k++) {
        int b = t->barC_bar[k], m = t->barC_sym[k], dd = t->barDim[b];
        double tr = 0.0;
        for (int p = 0; p < dd; p++)
            for (int q2 = p; q2 < dd; q2++)
                tr += symPq[m][bar_pack(dd, p, q2)] * t->barx[b][p * dd + q2];
        po += t->barC_coef[k] * tr;
    }
    t->pobj = po;

    double dob = 0.0;
    for (int q = 0; q < neq; q++) dob += d[q] * ys[q];
    for (int q = 0; q < K; q++)  dob += h[q] * lm[q];
    /* cfix is a term of the objective as written (pobj starts from it above), so
     * it sits OUTSIDE the sense factor: inside, -s*(dob+cfix) misses by 2*cfix. */
    t->dobj = -s * dob + t->cfix;

    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_OPTIMAL;
    tlog(t, "conic optimal solution found\n");

    free(ymin); free(zmin); free(xs); free(ys); free(lm); free(E); free(d);
    free(G); free(h); free(c); free(cones); free(cmem);
    free(rowKind); free(rowIdx); free(eqKind); free(eqIdx); free(eqAux); free(auxBase); free(nlType); free(nlAlpha); free(nlMem); free(cutcol); free(cuta); free(cuth);
    if (nb > 0) {
        for (int m = 0; m < t->nsym; m++) free(symPq[m]);
        free(symPq); free(psd_eval); free(psd_evec); free(psd_X);
        free(psd_cut_val); free(psd_cut_rhs);
        free(barOff); free(barPq);
    }
    return PRIMAL_RES_OK;
}

/* allocate/reset solution buffers for any task (dispatcher + shadow tasks) */
