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
/* primal_sdptask.c - conic/SDP conversion, unified IPM leg, SDP outer approximation.
 * Verbatim split of primal.c: no logic change. Shares primal_priv.h.
 */
#include "primal_priv.h"

/* =====================================================================
 * SDP path: outer approximation with tangent cuts on lambda_min
 * =====================================================================
 * Standard form: min c'x + sum_j <C_j, X_j>
 *                s.t. linear rows over x (+ terms <A^k, X_j>), X_j >= 0.
 * Each X_j is represented by its upper-triangle entries
 * (p <= q) as free variables; the PSD cone is approximated from outside
 * with linear cuts tangent to the convex function -lambda_min:
 *   -lambda_min(X) >= -lambda_min(X0) - <U U', X - X0>   (U eigenvector)
 * i.e.  <U U', X> >= <U U', X0> - lambda_min(X0).
 * Each round solves an LP (stdform + simplex); cuts are added at the
 * violated points until lambda_min(X_j) >= -tol. */


/* Pack upper-triangle position (p <= q) of a d x d symmetric matrix
 * into a flat index. Used by the bar-variable conversions. */
int bar_pack(int d, int p, int q) { return p * d - p * (p - 1) / 2 + (q - p); }

/* add the bar terms of original row k into Abar[r*nb+j] (dense symmetric). */
static void sdp_bar_row(PRIMALtask_t t, int k, int r, int nb, double **symPq, double **Abar) {
    for (int kb = 0; kb < t->nbarA; kb++) {
        if (t->barA_con[kb] != k) continue;
        int b = t->barA_bar[kb], m = t->barA_sym[kb], d = t->barDim[b];
        double coef = t->barA_coef[kb];
        double *M = Abar[(size_t)r * nb + b];
        for (int p = 0; p < d; p++) for (int q = p; q < d; q++) {
            double sv = (p == q) ? symPq[m][bar_pack(d, p, q)] : 0.5 * symPq[m][bar_pack(d, p, q)];
            double v = coef * sv;
            M[p * d + q] += v; M[q * d + p] = M[p * d + q];
        }
    }
}

/* SDP/conic via primal-dual interior point (Nesterov-Todd).  Converts the task
 * (scalar bounds + ranged rows + PSD bar vars + every cone: QUAD/RQUAD as SOC
 * blocks, PEXP/DEXP/PPOW/RPOW as barrier blocks) to the standard form handled
 * by sdp_ipm(): nonneg scalar slacks + equalities + PSD/SOC/exp blocks.  Ranged
 * rows are split into >= and <=; ranged scalar variables into a nonneg shift
 * plus a cap row.  Used instead of the tangent-cut outer approximation (which
 * is limited by its dense LP master). */
static PRIMALrescodee optimize_sdp_ipm_impl(PRIMALtask_t t, int s);
/* Run the native SDP/conic interior-point route with conic begin/end
 * callbacks around the implementation. Returns its result code. */
PRIMALrescodee optimize_sdp_ipm(PRIMALtask_t t, int s) {
    iter_cb_begin(t);
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_CONIC);
    PRIMALrescodee r = optimize_sdp_ipm_impl(t, s);
    cb_fire(t, PRIMAL_CALLBACK_END_CONIC);
    iter_cb_end();
    return r;
}
/* Convert the task to conic standard form and solve it with the
 * unified interior-point method. Builds SOC/exp blocks and maps
 * the solution back to the user space. */
static PRIMALrescodee optimize_sdp_ipm_impl(PRIMALtask_t t, int s) {
    int nvar = t->numvar, ncon = t->numcon, nb = t->numbarvar;
    /* A quadratic objective or a quadratic row is not representable here: the
     * conversion would drop it and answer on a model that no longer has it.
     * Measured on a bar + x'x<=2 task, which used to come back UNBOUNDED.  This
     * is an invariant, not an unsupported shape: the dispatcher sends those
     * models to quad_encode_task, which carries the bar blocks with it (T86). */
    if (t->has_qobj || t->has_qcon > 0) return PRIMAL_RES_ERR_ARG;

    /* ---- conic blocks: one per cone, its components are new variables z_i
     * linked to the member variables by equality rows (the scalar conversion
     * below keeps x_{mem}):
     *   QUAD  z_a = x_{mem[a]};
     *   RQUAD z_0=(u+v)/sqrt2, z_1=(u-v)/sqrt2, z_{2+j}=w_j;
     *   PEXP/PPOW/RPOW  z_a = x_{mem[a]}  (barrier block, expcone.c);
     *   DEXP            z_a = -x_{mem[a]} (DEXP = -PEXP). ---- */
    int nsoc = 0, nSocVar = 0, nep = 0;
    int *socdim = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    int *socOf  = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    int *socCone = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    int *ekind  = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    double *ealpha = (double *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(double));
    int *eOf  = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    int *eSgn = (int *)malloc((size_t)(t->numcones > 0 ? t->numcones : 1) * sizeof(int));
    if (!socdim || !socOf || !socCone || !ekind || !ealpha || !eOf || !eSgn) {
        free(socdim); free(socOf); free(socCone); free(ekind); free(ealpha); free(eOf); free(eSgn);
        return PRIMAL_RES_ERR_ALLOC;
    }
    for (int k = 0; k < t->numcones; k++) {
        int ct = t->cone_type[k];
        if (ct == PRIMAL_CT_QUAD || ct == PRIMAL_CT_RQUAD) {
            socOf[nsoc] = nSocVar; socdim[nsoc] = t->cone_nmem[k];
            socCone[nsoc] = k; nSocVar += t->cone_nmem[k]; nsoc++;
        } else if (ct == PRIMAL_CT_PEXP || ct == PRIMAL_CT_DEXP ||
                   ct == PRIMAL_CT_PPOW || ct == PRIMAL_CT_RPOW) {
            eOf[nep] = k;
            ekind[nep] = (ct == PRIMAL_CT_PPOW) ? EXPCONE_PPOW :
                         (ct == PRIMAL_CT_RPOW) ? EXPCONE_RPOW : EXPCONE_PEXP;
            ealpha[nep] = t->cone_param[k];
            eSgn[nep] = (ct == PRIMAL_CT_DEXP) ? -1 : 1;
            nep++;
        } else { free(socdim); free(socOf); free(socCone); free(ekind); free(ealpha); free(eOf); free(eSgn);
            return PRIMAL_RES_ERR_ARG; }
    }

    /* ---- compressed upper-triangle coefficients of the matrix store ---- */
    double **symPq = (double **)malloc((size_t)(t->nsym > 0 ? t->nsym : 1) * sizeof(double *));
    if (!symPq) { free(socdim); free(socOf); free(socCone); free(ekind); free(ealpha); free(eOf); free(eSgn);
        return PRIMAL_RES_ERR_ALLOC; }
    for (int m = 0; m < t->nsym; m++) symPq[m] = NULL;
    for (int m = 0; m < t->nsym; m++) {
        int d = t->sym_dim[m];
        double *M = (double *)calloc((size_t)d * d, sizeof(double));
        if (!M) goto fail_sym;
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            int si = t->sym_subi[m][e], sj = t->sym_subj[m][e]; double v = t->sym_val[m][e];
            M[si * d + sj] += v; if (si != sj) M[sj * d + si] += v;
        }
        int pq = d * (d + 1) / 2;
        symPq[m] = (double *)malloc((size_t)pq * sizeof(double));
        if (!symPq[m]) { free(M); goto fail_sym; }
        for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
            symPq[m][bar_pack(d, p, q)] = (p == q) ? M[p * d + q] : 2.0 * M[p * d + q];
        free(M);
    }

    /* ---- objective matrices Cbar[j] (sign applied) ---- */
    double **Cbar = (double **)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double *));
    if (!Cbar) goto fail_sym;
    for (int j = 0; j < nb; j++) {
        int d = t->barDim[j];
        Cbar[j] = (double *)calloc((size_t)d * d, sizeof(double));
        if (!Cbar[j]) goto fail_cbar;
    }
    for (int k = 0; k < t->nbarC; k++) {
        int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b]; double coef = s * t->barC_coef[k];
        double *M = Cbar[b];
        for (int p = 0; p < d; p++) for (int q = p; q < d; q++) {
            double sv = (p == q) ? symPq[m][bar_pack(d, p, q)] : 0.5 * symPq[m][bar_pack(d, p, q)];
            double v = coef * sv; M[p * d + q] += v; M[q * d + p] = M[p * d + q];
        }
    }

    /* ---- scalar row matrix Arow (ncon x nvar) ---- */
    double *Arow = (double *)calloc((size_t)(ncon > 0 ? ncon : 1) * (size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    if (!Arow) goto fail_cbar;
    for (int j = 0; j < nvar; j++) { const Col *col = &t->cols[j];
        for (int e = 0; e < col->nz; e++) Arow[col->sub[e] * nvar + j] += col->val[e]; }

    /* ---- represent each scalar var as const + sum coef * newvar (>= 0) ---- */
    int nv2 = 0, m2 = 0;
    int *vN = (int *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(int));
    int *vIdx = (int *)calloc((size_t)(nvar > 0 ? nvar : 1) * 2, sizeof(int));
    double *vCoef = (double *)calloc((size_t)(nvar > 0 ? nvar : 1) * 2, sizeof(double));
    double *vConst = (double *)calloc((size_t)(nvar > 0 ? nvar : 1), sizeof(double));
    int *varRow = (int *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(int));
    int *varSlack = (int *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(int));
    double *varCap = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    int *rowOf = (int *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(int));
    int *rowSlack = (int *)malloc((size_t)(ncon > 0 ? ncon : 1) * 2 * sizeof(int));
    if (!vN || !vIdx || !vCoef || !vConst || !varRow || !varSlack || !varCap || !rowOf || !rowSlack) goto fail_row;
    for (int i = 0; i < nvar; i++) { varRow[i] = -1; varSlack[i] = -1; varCap[i] = 0.0; }
    for (int i = 0; i < nvar; i++) {
        int bk = t->bkx[i];
        double lo = t->blx[i], up = t->bux[i];
        if (bk == PRIMAL_BK_RA) {   /* an infinite side degenerates to FR/LO/UP */
            if (!isfinite(lo) && !isfinite(up)) bk = PRIMAL_BK_FR;
            else if (!isfinite(lo)) bk = PRIMAL_BK_UP;
            else if (!isfinite(up)) bk = PRIMAL_BK_LO;
        }
        if (bk == PRIMAL_BK_FR) { vN[i] = 2; vIdx[2*i] = nv2++; vIdx[2*i+1] = nv2++; vCoef[2*i] = 1.0; vCoef[2*i+1] = -1.0; }
        else if (bk == PRIMAL_BK_LO) { vN[i] = 1; vIdx[2*i] = nv2++; vCoef[2*i] = 1.0; vConst[i] = lo; }
        else if (bk == PRIMAL_BK_UP) { vN[i] = 1; vIdx[2*i] = nv2++; vCoef[2*i] = -1.0; vConst[i] = up; }
        else if (bk == PRIMAL_BK_RA) {
            /* x = lo + u, u >= 0 capped by the equality row u + s = up - lo. */
            vN[i] = 1; vIdx[2*i] = nv2++; vCoef[2*i] = 1.0; vConst[i] = lo;
            varSlack[i] = nv2++; varRow[i] = m2++; varCap[i] = up - lo;
        }
        else { vN[i] = 0; vConst[i] = lo; }
    }
    for (int k = 0; k < ncon; k++) {
        int bk = t->bkc[k];
        if (bk == PRIMAL_BK_FR) { rowOf[k] = -1; rowSlack[2*k] = rowSlack[2*k+1] = -1; continue; }
        rowOf[k] = m2;
        if (bk == PRIMAL_BK_RA) { m2 += 2; rowSlack[2*k] = nv2++; rowSlack[2*k+1] = nv2++; }
        else if (bk == PRIMAL_BK_FX) { m2 += 1; rowSlack[2*k] = rowSlack[2*k+1] = -1; }
        else { m2 += 1; rowSlack[2*k] = nv2++; rowSlack[2*k+1] = -1; }
    }
    int mTot = m2 + nSocVar + 3 * nep;
    double *E2 = (double *)calloc((size_t)(mTot > 0 ? mTot : 1) * (size_t)(nv2 > 0 ? nv2 : 1), sizeof(double));
    double *b2 = (double *)calloc((size_t)(mTot > 0 ? mTot : 1), sizeof(double));
    double *c2 = (double *)calloc((size_t)(nv2 > 0 ? nv2 : 1), sizeof(double));
    double **Abar = (double **)calloc((size_t)(mTot > 0 ? mTot : 1) * (size_t)(nb > 0 ? nb : 1), sizeof(double *));
    double **Csoc = (double **)calloc((size_t)(nsoc > 0 ? nsoc : 1), sizeof(double *));
    double **Asoc = (double **)calloc((size_t)(mTot > 0 ? mTot : 1) * (size_t)(nsoc > 0 ? nsoc : 1), sizeof(double *));
    double **Cexp = (double **)calloc((size_t)(nep > 0 ? nep : 1), sizeof(double *));
    double **Aexp = (double **)calloc((size_t)(mTot > 0 ? mTot : 1) * (size_t)(nep > 0 ? nep : 1), sizeof(double *));
    if (!E2 || !b2 || !c2 || !Abar || !Csoc || !Asoc || !Cexp || !Aexp) goto fail_exp;
    for (int r = 0; r < mTot; r++) for (int j = 0; j < nb; j++) {
        int d = t->barDim[j];
        Abar[(size_t)r * nb + j] = (double *)calloc((size_t)d * d, sizeof(double));
        if (!Abar[(size_t)r * nb + j]) goto fail_abar;
    }
    for (int i = 0; i < nsoc; i++) {
        Csoc[i] = (double *)calloc((size_t)socdim[i], sizeof(double));
        if (!Csoc[i]) goto fail_abar;
        for (int r = 0; r < mTot; r++) {
            Asoc[(size_t)r * nsoc + i] = (double *)calloc((size_t)socdim[i], sizeof(double));
            if (!Asoc[(size_t)r * nsoc + i]) goto fail_abar;
        }
    }
    for (int i = 0; i < nep; i++) {
        Cexp[i] = (double *)calloc(3, sizeof(double));
        if (!Cexp[i]) goto fail_abar;
        for (int r = 0; r < mTot; r++) {
            Aexp[(size_t)r * nep + i] = (double *)calloc(3, sizeof(double));
            if (!Aexp[(size_t)r * nep + i]) goto fail_abar;
        }
    }
    for (int i = 0; i < nvar; i++) { double ci = s * t->c[i];
        for (int p = 0; p < vN[i]; p++) c2[vIdx[2*i+p]] += ci * vCoef[2*i+p]; }
    for (int k = 0; k < ncon; k++) {
        if (rowOf[k] < 0) continue;
        int r = rowOf[k], bk = t->bkc[k];
        double bb = 0.0;
        for (int i = 0; i < nvar; i++) { double a = Arow[k * nvar + i]; if (a == 0.0) continue;
            bb += a * vConst[i];
            for (int p = 0; p < vN[i]; p++) E2[(size_t)r * nv2 + vIdx[2*i+p]] += a * vCoef[2*i+p]; }
        sdp_bar_row(t, k, r, nb, symPq, Abar);
        if (bk == PRIMAL_BK_RA) {
            int r2 = r + 1;
            for (int i = 0; i < nvar; i++) { double a = Arow[k * nvar + i]; if (a == 0.0) continue;
                for (int p = 0; p < vN[i]; p++) E2[(size_t)r2 * nv2 + vIdx[2*i+p]] += a * vCoef[2*i+p]; }
            sdp_bar_row(t, k, r2, nb, symPq, Abar);
            E2[(size_t)r * nv2 + rowSlack[2*k]] = -1.0; b2[r] = t->blc[k] - bb;
            E2[(size_t)r2 * nv2 + rowSlack[2*k+1]] = 1.0; b2[r2] = t->buc[k] - bb;
        } else if (bk == PRIMAL_BK_LO) { E2[(size_t)r * nv2 + rowSlack[2*k]] = -1.0; b2[r] = t->blc[k] - bb; }
        else if (bk == PRIMAL_BK_UP) { E2[(size_t)r * nv2 + rowSlack[2*k]] = 1.0; b2[r] = t->buc[k] - bb; }
        else { b2[r] = t->blc[k] - bb; }   /* FX */
    }
    /* ---- cap rows of the ranged scalar variables (see the variable loop) ---- */
    for (int i = 0; i < nvar; i++) {
        if (varRow[i] < 0) continue;
        int r = varRow[i];
        E2[(size_t)r * nv2 + vIdx[2*i]] = 1.0;
        E2[(size_t)r * nv2 + varSlack[i]] = 1.0;
        b2[r] = varCap[i];
    }
    /* ---- SOC linking rows ----
     * QUAD  (t, x...): z_a = x_{mem[a]}            -> z_a - x = 0
     * RQUAD (u, v, w...): z_0=(u+v)/sqrt2, z_1=(u-v)/sqrt2, z_{2+j}=w_j
     *   -> sqrt2*z_0 - u - v = 0, sqrt2*z_1 - u + v = 0, z_{2+j} - w_j = 0 */
    for (int i = 0; i < nsoc; i++) {
        const int *mem = t->cone_mem[socCone[i]];   /* block i <-> cone socCone[i]:
                                                       exp blocks are not SOC ones */
        int mm = socdim[i];
        if (t->cone_type[socCone[i]] == PRIMAL_CT_QUAD) {
            for (int a = 0; a < mm; a++) {
                int r = m2 + socOf[i] + a, v = mem[a];
                Asoc[(size_t)r * nsoc + i][a] = 1.0;
                b2[r] = vConst[v];
                for (int p = 0; p < vN[v]; p++) E2[(size_t)r * nv2 + vIdx[2*v+p]] -= vCoef[2*v+p];
            }
        } else {   /* RQUAD */
            const double sq2 = sqrt(2.0);
            for (int a = 0; a < mm; a++) {
                int r = m2 + socOf[i] + a;
                /* lhsC = coefficient this row puts on the member variable, on the
                 * same side as z.  The constant part of that variable goes to the
                 * right-hand side with the OPPOSITE sign, as every other row here
                 * (data rows, QUAD, exp/power linking) does. */
                double lhsC = 0.0;
                if (a == 0 || a == 1) {
                    int vs[2] = { mem[0], mem[1] };
                    double cs[2] = { -1.0, (a == 0) ? -1.0 : 1.0 };
                    Asoc[(size_t)r * nsoc + i][a] = sq2;
                    for (int t2 = 0; t2 < 2; t2++) { int v = vs[t2];
                        lhsC += cs[t2] * vConst[v];
                        for (int p = 0; p < vN[v]; p++) E2[(size_t)r * nv2 + vIdx[2*v+p]] += cs[t2] * vCoef[2*v+p]; }
                } else {
                    int v = mem[a];
                    Asoc[(size_t)r * nsoc + i][a] = 1.0;
                    lhsC = -vConst[v];
                    for (int p = 0; p < vN[v]; p++) E2[(size_t)r * nv2 + vIdx[2*v+p]] -= vCoef[2*v+p];
                }
                b2[r] = -lhsC;
            }
        }
    }
    /* ---- exp/power linking rows: z_a = sg * x_{mem[a]}, sg = -1 for DEXP
     * (DEXP = -PEXP) and +1 for PEXP/PPOW/RPOW ---- */
    for (int i = 0; i < nep; i++) {
        const int *mem = t->cone_mem[eOf[i]];
        const double sg = (double)eSgn[i];
        for (int a = 0; a < 3; a++) {
            int r = m2 + nSocVar + 3 * i + a, v = mem[a];
            Aexp[(size_t)r * nep + i][a] = 1.0;
            b2[r] = sg * vConst[v];
            for (int p = 0; p < vN[v]; p++) E2[(size_t)r * nv2 + vIdx[2*v+p]] -= sg * vCoef[2*v+p];
        }
    }

    /* ---- solve ---- */
    double *x2 = (double *)calloc((size_t)(nv2 > 0 ? nv2 : 1), sizeof(double));
    double *y2 = (double *)calloc((size_t)(mTot > 0 ? mTot : 1), sizeof(double));
    double **Xb = (double **)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double *));
    double **Sb = (double **)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double *));
    double **Zsoc = (double **)calloc((size_t)(nsoc > 0 ? nsoc : 1), sizeof(double *));
    double **Ssoc = (double **)calloc((size_t)(nsoc > 0 ? nsoc : 1), sizeof(double *));
    double **Zexp = (double **)calloc((size_t)(nep > 0 ? nep : 1), sizeof(double *));
    double **Sexp = (double **)calloc((size_t)(nep > 0 ? nep : 1), sizeof(double *));
    int alloc_ok = x2 && y2 && Xb && Sb && Zsoc && Ssoc && Zexp && Sexp;
    for (int j = 0; j < nb && alloc_ok; j++) {
        int d = t->barDim[j];
        Xb[j] = (double *)calloc((size_t)d * d, sizeof(double));
        Sb[j] = (double *)calloc((size_t)d * d, sizeof(double));
        if (!Xb[j] || !Sb[j]) alloc_ok = 0;
    }
    for (int i = 0; i < nsoc && alloc_ok; i++) {
        Zsoc[i] = (double *)calloc((size_t)socdim[i], sizeof(double));
        Ssoc[i] = (double *)calloc((size_t)socdim[i], sizeof(double));
        if (!Zsoc[i] || !Ssoc[i]) alloc_ok = 0;
    }
    for (int i = 0; i < nep && alloc_ok; i++) {
        Zexp[i] = (double *)calloc(3, sizeof(double));
        Sexp[i] = (double *)calloc(3, sizeof(double));
        if (!Zexp[i] || !Sexp[i]) alloc_ok = 0;
    }
    /* primal warm start: map `warm_x` into the converted space (NaN =
     * unset; a boundary value gives 0, which the IPM ignores to stay
     * inside the nonnegative cone). The dual (`puty`) is not yet
     * threaded through: it is the declared open half of this warm start. */
    double *xwarm2 = NULL;
    if (t->has_warm && t->warm_x && nv2 > 0) {
        xwarm2 = (double *)malloc((size_t)nv2 * sizeof(double));
        if (xwarm2) {
            for (int i = 0; i < nv2; i++) xwarm2[i] = NAN;
            for (int i = 0; i < nvar; i++) {
                if (t->warm_x[i] != t->warm_x[i]) continue;
                for (int p = 0; p < vN[i]; p++) {
                    double v = vCoef[2*i+p] * (t->warm_x[i] - vConst[i]);
                    xwarm2[vIdx[2*i+p]] = v > 0.0 ? v : 0.0;
                }
            }
        }
    }
    double *ywarm2 = NULL;
    if (t->has_warm && t->warm_y && mTot > 0) {
        ywarm2 = (double *)malloc((size_t)mTot * sizeof(double));
        if (ywarm2) {
            for (int k = 0; k < mTot; k++) ywarm2[k] = NAN;
            for (int k = 0; k < ncon; k++)
                if (rowOf[k] >= 0 && t->bkc[k] != PRIMAL_BK_RA &&
                    t->warm_y[k] == t->warm_y[k])
                    ywarm2[rowOf[k]] = -s * t->warm_y[k];
        }
    }
    PRIMALrescodee rc = PRIMAL_RES_OK;
    if (!alloc_ok) rc = PRIMAL_RES_ERR_ALLOC;
    else {
        int fb = 0;
        int st = sdp_ipm(mTot, nv2, E2, b2, c2, nb, t->barDim,
                         (const double *const *)Cbar, (const double *const *)Abar,
                         nsoc, socdim, (const double *const *)Csoc, (const double *const *)Asoc,
                         nep, ekind, ealpha, (const double *const *)Cexp, (const double *const *)Aexp,
                         iter_cap(t->max_iter_intpnt), 1e-7, t->tol_co_pfeas, t->tol_co_dfeas, t->tol_co_gap,
                         t->tol_near_rel,
                         x2, Xb, y2, Sb, Zsoc, Ssoc, Zexp, Sexp, xwarm2, ywarm2, &fb);
        /* Retain SOC duals only for a successful native answer. A subsequent
         * fallback must not inherit the certificate of a different point. */
        if (st == 0 && nsoc == t->numcones && nsoc > 0) {
            int shared = 0;
            for (int k = 0; k < nsoc && !shared; k++)
                for (int a = 0; a < socdim[k] && !shared; a++)
                    for (int q = 0; q < k && !shared; q++)
                        for (int b = 0; b < socdim[q]; b++)
                            if (t->cone_mem[k][a] == t->cone_mem[q][b]) shared = 1;
            if (shared) {
                t->soc_dual = (SocDual *)malloc((size_t)nSocVar * sizeof(SocDual));
                if (!t->soc_dual) { st = 2; fb = 0; }
                else {
                    int p = 0;
                    for (int k = 0; k < nsoc; k++) for (int a = 0; a < socdim[k]; a++) {
                        SocDual *v = &t->soc_dual[p++];
                        v->cone = k; v->pos = a; v->var = t->cone_mem[k][a];
                        v->value = Ssoc[k][a];
                        if (t->cone_type[k] == PRIMAL_CT_RQUAD && a < 2)
                            v->value = (Ssoc[k][0] + (a == 0 ? Ssoc[k][1] : -Ssoc[k][1])) / sqrt(2.0);
                    }
                    t->nsoc_dual = nSocVar;
                }
            }
        }
        if (st == 0 || fb) {
            for (int i = 0; i < nvar; i++) { double v = vConst[i];
                for (int p = 0; p < vN[i]; p++) v += vCoef[2*i+p] * x2[vIdx[2*i+p]];
                t->x[i] = v; }
            for (int j = 0; j < nb; j++) { int d = t->barDim[j];
                for (int a = 0; a < d * d; a++) t->barx[j][a] = Xb[j][a]; }
            for (int k = 0; k < ncon; k++) t->y[k] = (rowOf[k] >= 0) ? -s * y2[rowOf[k]] : 0.0;
            for (int j = 0; j < nb; j++) { int d = t->barDim[j];
                for (int a = 0; a < d * d; a++) t->barsj[j][a] = Sb[j][a]; }
            double pobj = t->cfix;
            for (int i = 0; i < nvar; i++) pobj += t->c[i] * t->x[i];
            for (int k = 0; k < t->nbarC; k++) {
                int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b]; double tr = 0.0;
                for (int p = 0; p < d; p++) for (int q = p; q < d; q++)
                    tr += symPq[m][bar_pack(d, p, q)] * t->barx[b][p * d + q];
                pobj += t->barC_coef[k] * tr;
            }
            double dob = 0.0;
            for (int k = 0; k < mTot; k++) dob += b2[k] * y2[k];
            /* b2'y2 is the dual of the SHIFTED problem: a variable written as
             * x_i = vConst_i + coef*x2 loses its c_i*vConst_i from the
             * converted objective, and the fixed term cfix is not in it at all.
             * Both must come back for dobj to be the dual of the user's model. */
            double kfix = t->cfix;
            for (int i = 0; i < nvar; i++) kfix += t->c[i] * vConst[i];
            t->pobj = pobj; t->dobj = s * dob + kfix;
            t->has_sol = 1; t->solsta = PRIMAL_SOL_STA_OPTIMAL;
            /* fb: the point is published as a fallback, but the native verdict
             * stays "unresolved" -- the dispatcher tries the cuts and delivers
             * this only if they do not answer either. */
            rc = (st == 0) ? PRIMAL_RES_OK : PRIMAL_RES_TRM_MAX_ITER;
            if (st == 0) tlog(t, "optimal solution found (SDP IPM)\n");
        } else {
            rc = (st == 2) ? PRIMAL_RES_ERR_ALLOC : PRIMAL_RES_TRM_MAX_ITER;
            /* No solution is published here: opt_prepare zeroed x/pobj/dobj and
             * this route filled none of them, so has_sol=1 would hand the
             * getters an all-zero vector and an objective of 0 as if they were
             * the solver's answer -- the policy T87 set for getprimalobj. */
            t->solsta = PRIMAL_SOL_STA_UNKNOWN;
        }
    }
    for (int j = 0; j < nb; j++) { free(Xb ? Xb[j] : NULL); free(Sb ? Sb[j] : NULL); }
    for (int i = 0; i < nsoc; i++) { free(Zsoc ? Zsoc[i] : NULL); free(Ssoc ? Ssoc[i] : NULL); }
    for (int i = 0; i < nep; i++) { free(Zexp ? Zexp[i] : NULL); free(Sexp ? Sexp[i] : NULL); }
    free(Xb); free(Sb); free(Zsoc); free(Ssoc); free(Zexp); free(Sexp); free(x2); free(y2);
    free(xwarm2); free(ywarm2);
    for (int r = 0; r < mTot; r++) for (int j = 0; j < nb; j++) free(Abar[(size_t)r * nb + j]);
    free(Abar); free(E2); free(b2); free(c2);
    for (int i = 0; i < nsoc; i++) { free(Csoc[i]); for (int r = 0; r < mTot; r++) free(Asoc[(size_t)r * nsoc + i]); }
    for (int i = 0; i < nep; i++) { free(Cexp[i]); for (int r = 0; r < mTot; r++) free(Aexp[(size_t)r * nep + i]); }
    free(Csoc); free(Asoc); free(Cexp); free(Aexp);
    free(socdim); free(socOf); free(socCone); free(ekind); free(ealpha); free(eOf); free(eSgn);
    free(vN); free(vIdx); free(vCoef); free(vConst); free(Arow);
    free(varRow); free(varSlack); free(varCap);
    free(rowOf); free(rowSlack);
    for (int j = 0; j < nb; j++) free(Cbar[j]);
    free(Cbar);
    for (int m = 0; m < t->nsym; m++) free(symPq[m]);
    free(symPq);
    return rc;

fail_abar:
    for (int r = 0; r < mTot; r++) for (int j = 0; j < nb; j++) free(Abar[(size_t)r * nb + j]);
    if (Csoc) for (int i = 0; i < nsoc; i++) free(Csoc[i]);
    if (Asoc) for (int i = 0; i < nsoc; i++) for (int r = 0; r < mTot; r++) free(Asoc[(size_t)r * nsoc + i]);
    if (Cexp) for (int i = 0; i < nep; i++) free(Cexp[i]);
    if (Aexp) for (int i = 0; i < nep; i++) for (int r = 0; r < mTot; r++) free(Aexp[(size_t)r * nep + i]);
fail_exp:
    free(Csoc); free(Asoc); free(Cexp); free(Aexp);
    free(E2); free(b2); free(c2); free(Abar);
fail_row:
    free(vN); free(vIdx); free(vCoef); free(vConst); free(Arow);
    free(varRow); free(varSlack); free(varCap);
    free(rowOf); free(rowSlack);
fail_cbar:
    for (int j = 0; j < nb; j++) free(Cbar[j]);
    free(Cbar);
fail_sym:
    for (int m = 0; m < t->nsym; m++) free(symPq[m]);
    free(symPq);
    free(socdim); free(socOf); free(socCone); free(ekind); free(ealpha); free(eOf); free(eSgn);
    return PRIMAL_RES_ERR_ALLOC;
}

/* Dense row-major -> the column-major triplets stdform_build takes. Returns 0
 * on allocation failure, leaving the three outputs NULL. */
static int dense_to_csc(const double *A, int nrow, int ntot,
                        int **pptr, int **psub, double **pval) {
    int nz = 0;
    *pptr = NULL; *psub = NULL; *pval = NULL;
    for (int rr = 0; rr < nrow; rr++)
        for (int jj = 0; jj < ntot; jj++)
            if (A[(size_t)rr * ntot + jj] != 0.0) nz++;
    int *ptr = (int *)malloc((size_t)(ntot + 1) * sizeof(int));
    int *sub = (int *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int));
    double *val = (double *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(double));
    if (!ptr || !sub || !val) { free(ptr); free(sub); free(val); return 0; }
    int w = 0;
    for (int jj = 0; jj < ntot; jj++) {
        ptr[jj] = w;
        for (int rr = 0; rr < nrow; rr++)
            if (A[(size_t)rr * ntot + jj] != 0.0) {
                sub[w] = rr; val[w] = A[(size_t)rr * ntot + jj]; w++;
            }
    }
    ptr[ntot] = w;
    *pptr = ptr; *psub = sub; *pval = val;
    return 1;
}


/* The outer approximation caps the bar entries (SDP_BIGM) so that its very
 * first LP -- which has no cuts yet -- has a finite answer. A solution sitting
 * ON that cap is the answer of the capped problem, and nothing about it is a
 * statement on the model. This is where such a point is settled: the direction
 * is measured where it has to hold, on the model as written,
 *   a_i' rho = 0 on an equality row, <= 0 on a row capped above, >= 0 on a row
 *     bounded below -- so x + t rho keeps every row satisfied for every t >= 0;
 *   rho_j >= 0 under a lower bound, <= 0 under an upper one, 0 under both;
 *   rho_bar PSD, so x + t rho keeps every bar inside the cone;
 *   rho in K for every cone of the task -- the recession cone of a convex cone
 *     is the cone itself -- which is why an untestable cone type fails here;
 *   c' rho < 0, so the objective runs to -infinity along it.
 * Those five are what a primal ray of the model means. The LP is only where a
 * candidate comes from: a candidate that does not measure proves nothing, and
 * 0 is returned so that no status at all is claimed for the model. rho is
 * normalized here to max |.| = 1, which is what makes the tolerances below
 * relative to the model's own coefficients instead of to the cap. */
static int sdp_ray_measures(PRIMALtask_t t, int nvar, int nb, int ntot,
                            const int *barOff, const double *Arow, int nrowmodel,
                            const double *c, double *rho) {
    double mx = 0.0, scale = 1.0;
    for (int j = 0; j < ntot; j++) if (isfinite(rho[j]) && fabs(rho[j]) > mx) mx = fabs(rho[j]);
    if (!(mx > 0.0)) return 0;
    for (int j = 0; j < ntot; j++) rho[j] /= mx;
    for (int rr = 0; rr < nrowmodel; rr++)
        for (int j = 0; j < ntot; j++) {
            double a = fabs(Arow[(size_t)rr * ntot + j]);
            if (a > scale) scale = a;
        }
    double tol = 1e-8 * scale;
    int ii = 0;
    for (int rr = 0; rr < nrowmodel; rr++) {
        while (ii < t->numcon && t->bkc[ii] == PRIMAL_BK_FR) ii++;  /* the row loop skipped these */
        if (ii >= t->numcon) return 0;
        double v = 0.0;
        for (int j = 0; j < ntot; j++) v += Arow[(size_t)rr * ntot + j] * rho[j];
        if (t->bkc[ii] != PRIMAL_BK_LO && v >  tol) return 0;   /* the row has an upper side */
        if (t->bkc[ii] != PRIMAL_BK_UP && v < -tol) return 0;   /* ... or a lower one */
        ii++;
    }
    for (int j = 0; j < nvar; j++) {
        double lo, up;
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lo, &up);
        if (isfinite(up) && rho[j] >  tol) return 0;
        if (isfinite(lo) && rho[j] < -tol) return 0;
    }
    for (int j = 0; j < nb; j++) {
        int d = t->barDim[j];
        double *R = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
        if (!R) return 0;
        for (int p = 0; p < d; p++)
            for (int q = p; q < d; q++) {
                double v = rho[barOff[j] + bar_pack(d, p, q)];
                R[p * d + q] = v; R[q * d + p] = v;
            }
        double emax = 0.0, e = bar_min_eig(d, R, &emax);
        free(R);
        if (!(e >= -t->semi_tol_approx * (1.0 + emax))) return 0;
    }
    /* The cones of the model, which the candidate LP does not carry: an LP
     * cannot write them. Each is a closed convex cone, so its recession cone is
     * the cone itself and the direction has to lie inside it. The block is
     * normalised by its own largest entry because membership is scale invariant
     * while rho is normalised as a whole -- otherwise a cone sitting on small
     * coordinates of a big vector would be judged against the wrong scale. */
    for (int k = 0; k < t->numcones; k++) {
        int ct = t->cone_type[k], nk = t->cone_nmem[k];
        if (ct != PRIMAL_CT_QUAD && ct != PRIMAL_CT_RQUAD && ct != PRIMAL_CT_PEXP &&
            ct != PRIMAL_CT_DEXP && ct != PRIMAL_CT_PPOW && ct != PRIMAL_CT_RPOW)
            return 0;                  /* not testable here: claim nothing */
        if (nk <= 0 || !t->cone_mem[k]) return 0;
        double *cv = (double *)malloc((size_t)nk * sizeof(double));
        if (!cv) return 0;
        double sc = 0.0;
        for (int i = 0; i < nk; i++) {
            cv[i] = rho[t->cone_mem[k][i]];
            if (fabs(cv[i]) > sc) sc = fabs(cv[i]);
        }
        if (sc > 0.0) for (int i = 0; i < nk; i++) cv[i] /= sc;
        int in = cone_signed_slack(ct, t->cone_param[k], cv, nk) >= -1e-8;
        free(cv);
        if (!in) return 0;
    }
    double cr = 0.0, cs = 1.0;
    for (int j = 0; j < ntot; j++) { cr += c[j] * rho[j]; cs += fabs(c[j]); }
    return cr < -1e-8 * cs;
}

/* Where a verdict about a model is settled when the route's own numbers cannot
 * give one: an answer sitting on a bound WE put there, or a conic run that ran
 * out of iterations. Either way the candidate is asked from the MODEL as
 * written -- its rows, its bounds, its cost, one compressed block of columns per
 * bar -- never from the rows a route accumulated on the way, because those bind
 * the escape directions too and would tie the verdict to a trajectory.
 * The cones cannot be written by an LP, so the candidate carries a POLYHEDRAL
 * INNER approximation of each quad/rotated block (t >= sum |u_i| for QUAD,
 * (t+u) >= |t-u| + sqrt(2) sum |v_i| for RQUAD: 2^(nk-1) sign rows, each a
 * subset of the cone because the 1-norm dominates the 2-norm). A narrower
 * candidate set can only MISS a recession direction, never invent one, and what
 * decides is still the measurement. An exponential or power block has no
 * polyhedral inner approximation worth writing -- a finite set of its rays
 * generates a cone strictly inside it -- so such a block is simply absent from
 * the candidate and can only make the answer "no verdict".
 * mode 1 asks whether the model is INFEASIBLE, and that needs the OPPOSITE
 * relaxation: a set that CONTAINS each cone, because an LP found infeasible on a
 * subset of the model says nothing about the model. Only the LINEAR CONSEQUENCES
 * of membership are written -- the domain faces (t >= 0 and u >= 0, negated for
 * DEXP) and, for QUAD, t >= |u_i| for each free member. Infeasibility of that
 * relaxation is infeasibility of the model; feasibility of it is not feasibility
 * of the model, so a "no" stays silent, as it does for mode 0.
 * 1 = the witness the mode asks about measured, 0 = none measured (nothing is
 * claimed about the model), -1 = out of memory. */
int model_lp_witness(PRIMALtask_t t, int s, double **symPq, int mode,
                            const char *tag, double *rayout, double *dualout) {
    int nvar = t->numvar, ncon = t->numcon, nb = t->numbarvar;
    int ntot = nvar, nrowmodel = 0, nrowcone = 0, nrowlp;
    for (int j = 0; j < nb; j++) ntot += t->barDim[j] * (t->barDim[j] + 1) / 2;
    for (int i = 0; i < ncon; i++) if (t->bkc[i] != PRIMAL_BK_FR) nrowmodel++;
    for (int k = 0; k < t->numcones; k++) {
        int ct = t->cone_type[k], nk = t->cone_nmem[k];
        if (nk < 2) continue;
        if (mode == 0) {
            if ((ct == PRIMAL_CT_QUAD || ct == PRIMAL_CT_RQUAD) && nk - 1 <= 8)
                nrowcone += 1 << (nk - 1);
        } else if (ct == PRIMAL_CT_QUAD) {
            nrowcone += 1 + 2 * (nk - 1);
        } else if (ct == PRIMAL_CT_RQUAD || ct == PRIMAL_CT_PEXP ||
                   ct == PRIMAL_CT_DEXP   || ct == PRIMAL_CT_PPOW ||
                   ct == PRIMAL_CT_RPOW) {
            nrowcone += 2;
        }
    }
    nrowlp = nrowmodel + nrowcone;
    if (nrowlp <= 0 || ntot <= 0) return 0;

    int *barOff = (int *)malloc((size_t)(nb > 0 ? nb : 1) * sizeof(int));
    double *A = (double *)calloc((size_t)nrowlp * (size_t)ntot, sizeof(double));
    double *c = (double *)calloc((size_t)ntot, sizeof(double));
    double *lx = (double *)malloc((size_t)ntot * sizeof(double));
    double *ux = (double *)malloc((size_t)ntot * sizeof(double));
    double *lc = (double *)malloc((size_t)nrowlp * sizeof(double));
    double *uc = (double *)malloc((size_t)nrowlp * sizeof(double));
    if (!barOff || !A || !c || !lx || !ux || !lc || !uc) {
        free(barOff); free(A); free(c); free(lx); free(ux); free(lc); free(uc);
        return -1;
    }
    {
        int bo = nvar;
        for (int j = 0; j < nb; j++) {
            int d = t->barDim[j];
            barOff[j] = bo;
            bo += d * (d + 1) / 2;
        }
    }
    for (int j = 0; j < nvar; j++) {
        if (mode == 0) c[j] = s * t->c[j];   /* mode 1 asks feasibility: cost 0 */
        switch (t->bkx[j]) {
            case PRIMAL_BK_FR: lx[j] = -INF; ux[j] = INF; break;
            case PRIMAL_BK_LO: lx[j] = t->blx[j]; ux[j] = INF; break;
            case PRIMAL_BK_UP: lx[j] = -INF; ux[j] = t->bux[j]; break;
            case PRIMAL_BK_RA: lx[j] = t->blx[j]; ux[j] = t->bux[j]; break;
            default:        lx[j] = ux[j] = t->blx[j]; break;
        }
    }
    /* The cap is what this function exists to question: here the bar entries are
     * free, except that a PSD matrix has a nonnegative diagonal. */
    for (int j = nvar; j < ntot; j++) { lx[j] = -INF; ux[j] = INF; }
    for (int j = 0; j < nb; j++) {
        int d = t->barDim[j];
        for (int p = 0; p < d; p++) lx[barOff[j] + bar_pack(d, p, p)] = 0.0;
    }
    for (int k = 0; k < t->nbarC; k++) {
        int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b];
        int pq = d * (d + 1) / 2;
        for (int e = 0; e < pq; e++)
            c[barOff[b] + e] += s * t->barC_coef[k] * symPq[m][e];
    }
    {
        int r = 0;
        for (int i = 0; i < ncon; i++) {
            if (t->bkc[i] == PRIMAL_BK_FR) continue;   /* the row loop has no content */
            for (int j = 0; j < nvar; j++) {
                const Col *col = &t->cols[j];
                for (int k = 0; k < col->nz; k++)
                    if (col->sub[k] == i) A[(size_t)r * ntot + j] += col->val[k];
            }
            for (int k = 0; k < t->nbarA; k++) {
                if (t->barA_con[k] != i) continue;
                int b = t->barA_bar[k], m = t->barA_sym[k], d = t->barDim[b];
                int pq = d * (d + 1) / 2;
                for (int e = 0; e < pq; e++)
                    A[(size_t)r * ntot + barOff[b] + e] += t->barA_coef[k] * symPq[m][e];
            }
            lc[r] = (t->bkc[i] == PRIMAL_BK_UP) ? -INF : t->blc[i];
            uc[r] = (t->bkc[i] == PRIMAL_BK_LO) ? INF : t->buc[i];
            if (t->bkc[i] == PRIMAL_BK_FX) { lc[r] = uc[r] = t->blc[i]; }
            r++;
        }
    }

    /* The rows each mode asks about: mode 0 the INNER set of each quad block
     * (one row per sign pattern, homogeneous, so that a direction satisfying
     * them all is inside K and K's recession cone -- which for a cone is K
     * itself -- is not approximated up); mode 1 the consequences that membership
     * IMPLIES, which is the other direction and the only one that can prove an
     * infeasibility. */
    {
        int r = nrowmodel;
        for (int k = 0; k < t->numcones && r < nrowlp; k++) {
            int ct = t->cone_type[k], nk = t->cone_nmem[k];
            const int *mi = t->cone_mem[k];
            if (nk < 2 || !mi) continue;
            for (int p = 0; p < nk; p++)
                if (mi[p] < 0 || mi[p] >= nvar) { nk = 0; break; }
            if (nk == 0) continue;
            if (mode == 1) {
                /* What membership IMPLIES, linearly. QUAD: t >= 0 and t >= |u_i|,
                 * one row per sign. The other kinds have nothing beyond their
                 * domain faces t >= 0, u >= 0 (and PEXP adds none on v), and
                 * DEXP = -PEXP, so its faces are those two negated -- which is
                 * all `g` carries. A superset of each cone: infeasible here is
                 * infeasible in the model. */
                double g = (ct == PRIMAL_CT_DEXP) ? -1.0 : 1.0;
                if (ct == PRIMAL_CT_QUAD) {
                    for (int i = 1; i < nk && r < nrowlp; i++) {
                        for (int sg = 0; sg < 2 && r < nrowlp; sg++, r++) {
                            double *row = A + (size_t)r * (size_t)ntot;
                            row[mi[0]] += 1.0;
                            row[mi[i]] += sg ? -1.0 : 1.0;
                            lc[r] = 0.0; uc[r] = INF;
                        }
                    }
                    if (r < nrowlp) {
                        A[(size_t)r * ntot + mi[0]] += 1.0;
                        lc[r] = 0.0; uc[r] = INF; r++;
                    }
                } else if (ct == PRIMAL_CT_RQUAD || ct == PRIMAL_CT_PEXP ||
                           ct == PRIMAL_CT_DEXP   || ct == PRIMAL_CT_PPOW ||
                           ct == PRIMAL_CT_RPOW) {
                    for (int i = 0; i < 2 && r < nrowlp; i++, r++) {
                        A[(size_t)r * ntot + mi[i]] += g;
                        lc[r] = 0.0; uc[r] = INF;
                    }
                }
                continue;
            }
            if ((ct != PRIMAL_CT_QUAD && ct != PRIMAL_CT_RQUAD) || nk - 1 > 8)
                continue;      /* one row per sign pattern: this is a DENSE LP, and
                                * 2^(nk-1) is the cost of the inner set */
            int nsign = 1 << (nk - 1);
            for (int q = 0; q < nsign && r < nrowlp; q++, r++) {
                double *row = A + (size_t)r * (size_t)ntot;
                if (ct == PRIMAL_CT_QUAD) {
                    /* t >= |u_i| for every sign pattern at once is t >= sum|u_i|,
                     * and that is a subset of QUAD by the 1-norm dominating the
                     * 2-norm. */
                    row[mi[0]] += 1.0;
                    for (int i = 1; i < nk; i++)
                        row[mi[i]] += (q & (1 << (i - 1))) ? -1.0 : 1.0;
                } else {
                    /* RQUAD's inner set is (t+u) >= |t-u| + sqrt(2) sum|v_i|: the
                     * two signs of s0 pick which of t/u the left side collapses
                     * onto (2u >= ... or 2t >= ...), so the 2^(nk-1) rows together
                     * ask both. */
                    double s0 = (q & 1) ? -1.0 : 1.0;
                    row[mi[0]] += 1.0 - s0; row[mi[1]] += 1.0 + s0;
                    for (int i = 2; i < nk; i++)
                        row[mi[i]] += (q & (1 << (i - 1))) ? -sqrt(2.0) : sqrt(2.0);
                }
                lc[r] = 0.0; uc[r] = INF;      /* row >= 0 */
            }
        }
    }

    int rayed = 0, oom = 0;
    int *ptr = NULL, *sub = NULL;
    double *val = NULL;
    StdForm *sf = NULL;
    if (dense_to_csc(A, nrowlp, ntot, &ptr, &sub, &val))
        sf = stdform_build(ntot, nrowlp, c, NULL, NULL, NULL, 0,
                           lx, ux, lc, uc, ptr, sub, val);
    else oom = 1;
    free(ptr); free(sub); free(val);
    if (sf) {
        double *dA = stdform_dense_A(sf);
        double *xt = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
        double *yd = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
        double *dray = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
        double *pr = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
        double *rho = (double *)malloc((size_t)ntot * sizeof(double));
        if (!dA || !xt || !yd || !dray || !pr || !rho) oom = 1;
        else {
            int st = simplex_solve_std(dA, sf->m, sf->n, sf->b, sf->c,
                                       iter_cap(t->max_iter_simplex), xt, yd, dray, pr);
            char cb[96];
            snprintf(cb, sizeof cb, "%s: candidate LP status %d\n", tag, st);
            tlog(t, cb);
            if (mode == 1) {
                if (st == 1) {
                    rayed = 1;               /* the relaxation IS the verdict */
                    /* The candidate dual vector over the constructed LP's rows,
                     * the first nrowmodel of which are the model's own rows. The
                     * caller measures it against the model's cones; the
                     * cone-consequence rows are simply not part of the model and
                     * are dropped here (their multipliers have no model image). */
                    if (dualout) {
                        double *ymin = (double *)calloc((size_t)(sf->ncon > 0 ? sf->ncon : 1), sizeof(double));
                        if (ymin) {
                            stdform_map_y(sf, dray, ymin);
                            for (int r = 0; r < nrowmodel; r++) dualout[r] = ymin[r];
                            free(ymin);
                        }
                    }
                }
            } else if (st == 2) {
                stdform_map_dir(sf, pr, rho);
                if (sdp_ray_measures(t, nvar, nb, ntot, barOff, A, nrowmodel, c, rho)) {
                    rayed = 1;
                    /* The direction measured in the model's own space. For a
                     * bar-free model ntot == numvar and this is a primal ray the
                     * user can read; with bars the compressed block has no image
                     * as `numvar` scalars (the T85 deviation), so the caller
                     * publishes nothing and the vector stays internal. */
                    /* `rayout` holds `numvar` entries: with bars `ntot > numvar` and
                     * the compressed block has no image in that space (T85),
                     * so only the scalar part is copied -- and for a model
                     * with bars `conic_publish_pray` refuses it anyway. */
                    if (rayout && ntot == nvar)
                        for (int j = 0; j < nvar; j++) rayout[j] = rho[j];
                }
            }
        }
        free(dA); free(xt); free(yd); free(dray); free(pr); free(rho);
        stdform_free(sf);
    } else if (!oom) {
        char cb[96];
        snprintf(cb, sizeof cb, "%s: the candidate LP could not be built\n", tag);
        tlog(t, cb);
        oom = 1;
    }
    free(barOff); free(A); free(c); free(lx); free(ux); free(lc); free(uc);
    return oom ? -1 : rayed;
}

/* An answer that touches the SDP_BIGM cap is the answer of a problem this solver
 * invented to have a first iterate at all, not of the model, and nothing about
 * it is a statement on the model. Both routes that cap the bar entries -- the
 * tangent-cut outer approximation and the conic build -- settle such a point
 * here, so that the VERDICT does not depend on which of them answered.
 * `z` is the route's solution vector in the compressed bar space. Returns
 * PRIMAL_RES_OK when no entry is on the cap, which means there is nothing to
 * settle and the route publishes what it has. */
PRIMALrescodee bar_cap_verdict(PRIMALtask_t t, int s, double **symPq,
                                      int nb, const int *barOff, const int *barPq,
                                      const double *z) {
    int capped = 0;
    for (int j = 0; j < nb && !capped; j++)
        for (int p = 0; p < barPq[j] && !capped; p++)
            if (fabs(z[barOff[j] + p]) >= SDP_BIGM * (1.0 - 1e-9)) capped = 1;
    if (!capped) return PRIMAL_RES_OK;

    int rayed = model_lp_witness(t, s, symPq, 0, "bar cap", NULL, NULL);
    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
    if (rayed < 0) return PRIMAL_RES_ERR_ALLOC;
    if (rayed > 0) {
        /* A ray of the model, measured on its own rows: the objective has no
         * finite value here. The direction is not handed out -- it lives partly
         * inside a bar, and PRIMAL_getprimalray has numvar scalars as its object
         * (the same deviation T85 declared for the conic paths). */
        t->prosta = PRIMAL_PRO_STA_DUAL_INFEAS;
        tlog(t, "dual infeasible (unbounded): recession direction measured\n");
        return PRIMAL_RES_ERR_UNBOUNDED;
    }
    /* No direction of the model measures, so nothing is claimed. Note what this
     * also means: a model whose infimum is merely not ATTAINED inside a bar --
     * max -x0 under [[x0,1],[1,x1]] >= 0 needs x1 -> infinity, so its diagonal
     * parks on the cap while no recession direction exists -- is answered "no
     * verdict", not with a value. Declared limitation. */
    tlog(t, "answer sits on the bar cap: not an answer of the model\n");
    return PRIMAL_RES_TRM_MAX_ITER;
}

static PRIMALrescodee optimize_sdp_impl(PRIMALtask_t t, int s);
/* Run the tangent-cut SDP outer-approximation route with conic begin/end
 * callbacks around the implementation. Returns its result code. */
PRIMALrescodee optimize_sdp(PRIMALtask_t t, int s) {
    iter_cb_begin(t);
    cb_fire(t, PRIMAL_CALLBACK_BEGIN_CONIC);
    PRIMALrescodee r = optimize_sdp_impl(t, s);
    cb_fire(t, PRIMAL_CALLBACK_END_CONIC);
    iter_cb_end();
    return r;
}
/* Solve the SDP outer approximation by tangent cuts on lambda_min.
 * Builds the compressed bar space, iterates the LP master with new
 * cuts, and publishes the primal/dual point on convergence. */
static PRIMALrescodee optimize_sdp_impl(PRIMALtask_t t, int s) {
    int nvar = t->numvar, ncon = t->numcon;
    int nb = t->numbarvar;
    PRIMALrescodee rc = PRIMAL_RES_OK;

    if (t->has_qobj || t->has_qcon > 0)
        return PRIMAL_RES_ERR_ARG;   /* documented deviation */

    /* ---------- map of the compressed bar variables (p <= q) ---------- */
    int *barOff = (int *)malloc((size_t)(nb > 0 ? nb : 1) * sizeof(int));
    int *barPq  = (int *)malloc((size_t)(nb > 0 ? nb : 1) * sizeof(int));
    if (!barOff || !barPq) { free(barOff); free(barPq); return PRIMAL_RES_ERR_ALLOC; }
    int ntot = nvar;
    for (int j = 0; j < nb; j++) {
        int d = t->barDim[j];
        barOff[j] = ntot;
        barPq[j] = d * (d + 1) / 2;
        ntot += barPq[j];
    }
    int maxdim = 1;
    for (int j = 0; j < nb; j++) if (t->barDim[j] > maxdim) maxdim = t->barDim[j];

    /* compressed coefficient of a store entry (si,sj):
     * symmetry -> off-diagonal counts twice in <M, X> */
    /* precompute the compressed form for every matrix of the store */
    double **symPq = (double **)malloc((size_t)(t->nsym > 0 ? t->nsym : 1) * sizeof(double *));
    if (!symPq) { free(barOff); free(barPq); return PRIMAL_RES_ERR_ALLOC; }
    int symPqOk = 1;
    for (int m = 0; m < t->nsym && symPqOk; m++) {
        int d = t->sym_dim[m];
        double *M = (double *)calloc((size_t)d * (size_t)d, sizeof(double));
        if (!M) { symPqOk = 0; break; }
        for (int e = 0; e < t->sym_nnz[m]; e++) {
            int si = t->sym_subi[m][e], sj = t->sym_subj[m][e];
            double v = t->sym_val[m][e];
            M[si * d + sj] += v;
            if (si != sj) M[sj * d + si] += v;   /* implicit symmetry */
        }
        int pq = d * (d + 1) / 2;
        symPq[m] = (double *)malloc((size_t)pq * sizeof(double));
        if (!symPq[m]) { free(M); symPqOk = 0; break; }
        for (int p = 0; p < d; p++)
            for (int q = p; q < d; q++)
                symPq[m][bar_pack(d, p, q)] =
                    (p == q) ? M[p * d + q] : 2.0 * M[p * d + q];
        free(M);
    }
    if (!symPqOk) {
        for (int m = 0; m < t->nsym; m++) free(symPq[m]);
        free(symPq); free(barOff); free(barPq);
        return PRIMAL_RES_ERR_ALLOC;
    }

    /* ---------- cuts: seed ----------
     * No seed cut: the initial LP is still bounded by the big-M
     * bounds on the bar entries (SDP_BIGM), and an initial cut of the form
     * <I,X> >= d-1 is NOT valid for the PSD cone (it would exclude
     * legitimate PSD points, e.g. X = J/4 with tr = 0.5 < 1). */
    typedef struct { int bar; int n; int *sub; double *val; double rhs; } CutRow;
    int ncuts = 0, cutcap = nb + 4;
    CutRow *cuts = (CutRow *)calloc((size_t)cutcap, sizeof(CutRow));
    if (!cuts) {
        for (int m = 0; m < t->nsym; m++) free(symPq[m]);
        free(symPq); free(barOff); free(barPq);
        return PRIMAL_RES_ERR_ALLOC;
    }


    {
        double *eval = (double *)malloc((size_t)maxdim * sizeof(double));
        double *evec = (double *)malloc((size_t)maxdim * (size_t)maxdim * sizeof(double));
        double *Xf = (double *)malloc((size_t)maxdim * (size_t)maxdim * sizeof(double));
        double *zsol = (double *)malloc((size_t)ntot * sizeof(double));
        double *yb = (double *)calloc((size_t)(ncon + nb * (SDP_MAXROUND + 1) + 1), sizeof(double));
        double tolv = 1e-9 * (1.0 + (double)maxdim);
        int solved = 0, status = 0;
        double pobj = t->cfix;
        /* Stagnation of the cut loop. A tangent cut is a FIRST-ORDER separator:
         * when the iterate sits on a face where -lambda_min has no descent
         * direction, every new cut is nearly parallel to the last one and the
         * violation stops shrinking. Measured on lyapunov_roa (28 rows, two
         * bars): the FEASIBLE levels finish at round 0 in 0.02 s, and every
         * INFEASIBLE one burns all 200 rounds in ~20 s without ever getting
         * close -- half the bisection steps were spent proving a negative that
         * the cuts cannot reach. So the loop also watches the violation: when
         * it fails to improve for a number of consecutive rounds, the cuts are
         * not converging and spending the remaining budget is arithmetic, not
         * progress. Same verdict as running out of rounds (TRM_MAX_ITER, no
         * solution claimed), two orders of magnitude earlier. */


        if (!eval || !evec || !Xf || !zsol || !yb) { rc = PRIMAL_RES_ERR_ALLOC; }
        else {
            for (int round = 0; round <= SDP_MAXROUND && rc == PRIMAL_RES_OK; round++) {
                {
                    char pb[96];
                    snprintf(pb, sizeof pb, "sdp round %d, cuts %d", round, ncuts);
                    tprog(t, pb);
                    cb_fire(t, PRIMAL_CALLBACK_CONIC);
                }
                int nrow = ncon + ncuts;

                /* ---------- LP: dense rows -> CSC ---------- */
                double *Arow = (double *)calloc((size_t)nrow * (size_t)ntot, sizeof(double));
                double *c = (double *)calloc((size_t)ntot, sizeof(double));
                double *lx = (double *)malloc((size_t)ntot * sizeof(double));
                double *ux = (double *)malloc((size_t)ntot * sizeof(double));
                double *lc = (double *)malloc((size_t)nrow * sizeof(double));
                double *uc = (double *)malloc((size_t)nrow * sizeof(double));
                if (!Arow || !c || !lx || !ux || !lc || !uc) {
                    free(Arow); free(c); free(lx); free(ux); free(lc); free(uc);
                    rc = PRIMAL_RES_ERR_ALLOC; break;
                }
                for (int jj = 0; jj < nvar; jj++) {
                    c[jj] = s * t->c[jj];
                    switch (t->bkx[jj]) {
                        case PRIMAL_BK_FR: lx[jj] = -INF; ux[jj] = INF; break;
                        case PRIMAL_BK_LO: lx[jj] = t->blx[jj]; ux[jj] = INF; break;
                        case PRIMAL_BK_UP: lx[jj] = -INF; ux[jj] = t->bux[jj]; break;
                        case PRIMAL_BK_RA: lx[jj] = t->blx[jj]; ux[jj] = t->bux[jj]; break;
                        default:        lx[jj] = ux[jj] = t->blx[jj]; break;
                    }
                }
                /* documented big-M: bar entries limited to +-1e6 so that
         * the initial relaxation is bounded; the cuts restore it */
        for (int jj = nvar; jj < ntot; jj++) { lx[jj] = -SDP_BIGM; ux[jj] = SDP_BIGM; }
                for (int k = 0; k < t->nbarC; k++) {
                    int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b];
                    int pq = d * (d + 1) / 2;
                    for (int e = 0; e < pq; e++)
                        c[barOff[b] + e] += s * t->barC_coef[k] * symPq[m][e];
                }
                int r = 0;
                for (int i = 0; i < ncon; i++) {
                    if (t->bkc[i] == PRIMAL_BK_FR) continue;   /* row without content */
                    for (int jj = 0; jj < nvar; jj++) {
                        const Col *col = &t->cols[jj];
                        for (int k = 0; k < col->nz; k++)
                            if (col->sub[k] == i) Arow[r * ntot + jj] += col->val[k];
                    }
                    for (int k = 0; k < t->nbarA; k++) {
                        if (t->barA_con[k] != i) continue;
                        int b = t->barA_bar[k], m = t->barA_sym[k], d = t->barDim[b];
                        int pq = d * (d + 1) / 2;
                        for (int e = 0; e < pq; e++)
                            Arow[r * ntot + barOff[b] + e] +=
                                t->barA_coef[k] * symPq[m][e];
                    }
                    lc[r] = (t->bkc[i] == PRIMAL_BK_UP) ? -INF : t->blc[i];
                    uc[r] = (t->bkc[i] == PRIMAL_BK_LO) ? INF : t->buc[i];
                    if (t->bkc[i] == PRIMAL_BK_FX) { lc[r] = uc[r] = t->blc[i]; }
                    r++;
                }
                /* the model's own rows are the ones written so far; the cuts are
                 * appended after them, and the cut space is not the model. */
                for (int k = 0; k < ncuts; k++) {
                    for (int e = 0; e < cuts[k].n; e++)
                        Arow[r * ntot + cuts[k].sub[e]] += cuts[k].val[e];
                    lc[r] = cuts[k].rhs; uc[r] = INF;
                    r++;
                }
                nrow = r;

                int *ptr = NULL, *sub = NULL;
                double *val = NULL;
                if (!dense_to_csc(Arow, nrow, ntot, &ptr, &sub, &val)) {
                    free(Arow); free(c); free(lx); free(ux); free(lc); free(uc);
                    rc = PRIMAL_RES_ERR_ALLOC; break;
                }
                /* If the answer turns out to sit on the bar cap, the candidate is
                 * NOT asked from these rows: they are the cut space, and its
                 * tangents bind the escape directions. bar_cap_ray rebuilds the
                 * model's own rows from the task. */

                /* ---------- solve LP ---------- */
                StdForm *sf = stdform_build(ntot, nrow, c, NULL, NULL, NULL, 0, lx, ux, lc, uc, ptr, sub, val);
                if (!sf) {
                    free(Arow); free(ptr); free(sub); free(val);
                    free(c); free(lx); free(ux); free(lc); free(uc);
                    rc = PRIMAL_RES_ERR_ARG; break;
                }
                double *xt = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
                double *ystd = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
                double *dA = stdform_dense_A(sf);
                if (!xt || !ystd || !dA) {
                    free(xt); free(ystd); free(dA); stdform_free(sf);
                    free(Arow); free(ptr); free(sub); free(val);
                    free(c); free(lx); free(ux); free(lc); free(uc);
                    rc = PRIMAL_RES_ERR_ALLOC; break;
                }
                /* no rays here: this solve is in cut space, whose rows are the
                 * tangents, not the model's — a witness of it is not one of it */
                status = simplex_solve_std(dA, sf->m, sf->n, sf->b, sf->c,
                                           iter_cap(t->max_iter_simplex), xt, ystd, NULL, NULL);
                free(dA);
                stdform_map_x(sf, xt, zsol);
                stdform_map_y(sf, ystd, yb);

                if (status != 0) {
                    /* Neither a certificate nor a point comes out of this
                     * route: no *_CER member (T98), and no x either -- the
                     * buffer would read as an answer of all zeros. The verdict
                     * is carried by prosta, which no longer needs has_sol. */
                    t->solsta = PRIMAL_SOL_STA_UNKNOWN;
                    t->prosta = (status == 1) ? PRIMAL_PRO_STA_PRIM_INFEAS
                                : (status == 2) ? PRIMAL_PRO_STA_DUAL_INFEAS
                                : PRIMAL_PRO_STA_UNKNOWN;
                    for (int jj = 0; jj < nvar; jj++) t->x[jj] = 0.0;
                    tlog(t, status == 1 ? "primal infeasible\n" :
                          status == 2 ? "dual infeasible (unbounded)\n" : "iteration limit\n");
                    free(xt); free(ystd); stdform_free(sf);
                    free(Arow); free(ptr); free(sub); free(val);
                    free(c); free(lx); free(ux); free(lc); free(uc);
                    rc = (status == 1) ? PRIMAL_RES_ERR_INFEASIBLE
                         : (status == 2) ? PRIMAL_RES_ERR_UNBOUNDED
                         : PRIMAL_RES_TRM_MAX_ITER;
                    break;
                }

                /* ---------- PSD violation ---------- */
                int anycut = 0;
                double viol = 0.0;
                for (int j = 0; j < nb; j++) {
                    int d = t->barDim[j];
                    for (int p = 0; p < d; p++)
                        for (int q = 0; q < d; q++)
                            Xf[p * d + q] = zsol[barOff[j] + bar_pack(d, (p < q ? p : q), (p < q ? q : p))];
                    dmat_eig_jacobi(d, Xf, eval, evec);
                    int imin = 0;
                    for (int k = 1; k < d; k++) if (eval[k] < eval[imin]) imin = k;
                    double lam = eval[imin];
                    { double vj = tolv - lam; if (vj > viol) viol = vj; }
                    if (lam < -tolv) {
                        /* tangent cut: <UU', X> >= <UU', X0> - lam */
                        if (ncuts >= cutcap) {
                            int nc = cutcap * 2;
                            CutRow *cn = (CutRow *)realloc(cuts, (size_t)nc * sizeof(CutRow));
                            if (!cn) { rc = PRIMAL_RES_ERR_ALLOC; break; }
                            cuts = cn; cutcap = nc;
                        }
                        CutRow *cw = &cuts[ncuts];
                        cw->bar = j; cw->n = barPq[j];
                        cw->sub = (int *)malloc((size_t)cw->n * sizeof(int));
                        cw->val = (double *)calloc((size_t)cw->n, sizeof(double));
                        if (!cw->sub || !cw->val) { rc = PRIMAL_RES_ERR_ALLOC; break; }
                        for (int e = 0; e < cw->n; e++) cw->sub[e] = barOff[j] + e;
                        double proj0 = 0.0;
                        for (int p = 0; p < d; p++)
                            for (int q = p; q < d; q++) {
                                double u = evec[p * d + imin], v = evec[q * d + imin];
                                double cval = (p == q) ? u * v : 2.0 * u * v;
                                cw->val[bar_pack(d, p, q)] = cval;
                                proj0 += cval * zsol[barOff[j] + bar_pack(d, p, q)];
                            }
                        cw->rhs = proj0 - lam;
                        ncuts++;
                        anycut = 1;
                    }
                }

                /* Per-round progress measure, so the stall rule can be read off a
                 * trace instead of only from the line where it fires.
                 *
                 * viol is tolv - lambda_min, a MARGIN to the cone boundary, not a
                 * violation: it is about tolv when the block sits just outside,
                 * about tolv again when it is well inside, and it only falls as
                 * lambda_min rises. So "progress" means "this number is going
                 * down", and its absolute value says nothing about how far the
                 * block is from feasible. Do not read it as a residual. */
                if (getenv("GMB_DBG")) fprintf(stderr,
                    "  [cut] round=%d ncuts=%d viol=%.4g viol_prev=%.4g ratio=%.3g\n",
                    round, ncuts, viol, viol_prev,
                    viol_prev < 1e299 && viol_prev > 0.0 ? viol / viol_prev : 0.0);

                /* ---------- does the cut loop still make progress? ----------
                 * viol is the worst (tolv - lambda_min) over the bars -- a margin
                 * to the boundary, so a tangent cut that does not push it down
                 * measurably is not separating the iterate.
                 * Count the rounds without a real improvement and leave early:
                 * the verdict is the same one running out of rounds produces
                 * (TRM_MAX_ITER, no solution claimed), reached in a fraction of
                 * the budget. */
                /* Per-round progress measure: how far the worst bar is from the
                 * cone boundary, and how it moved. The loop itself is NOT
                 * stopped by this -- it runs to SDP_MAXROUND and reports
                 * TRM_MAX_ITER when the cuts cannot separate the iterate.
                 *
                 * viol is tolv - lambda_min, a MARGIN to the boundary, not a
                 * violation: it is about tolv when the block sits just outside,
                 * about tolv again when it is well inside, and it only falls as
                 * lambda_min rises. So "progress" means "this number is going
                 * down", and its absolute value says nothing about how far the
                 * block is from feasible. Do not read it as a residual.
                 *
                 * Why there is no early stop here: a tangent cut on
                 * -lambda_min is not monotone. On maxcut_sdp through this route
                 * the violation goes 1e6 -> 2.2e6 -> 2.56e6 over eight rounds
                 * and then collapses to 5.7 and to 0.28 by round 20, where the
                 * cut route solves the model in 0.06 s. A counter of stalled
                 * rounds shorter than that plateau kills a legitimate
                 * convergence and returns TRM_MAX_ITER on a solvable model
                 * (measured: 8 rounds -> obj=0.0 rc=1007 FAIL, 20 rounds ->
                 * obj=-4.0 rc=0 OK); a longer one does not fix the premise, it
                 * only moves the failure (lyapunov_roa: 1.0 s at 8, 239 s at
                 * 20, 191 s once the comparison is fixed round over round).
                 * Worth ~17% on the single sample that reaches this route, so
                 * it is not worth a stopping rule whose premise is wrong. This
                 * line and GMB_NO_SDP_IPM are what is left of that attempt,
                 * and they are the parts with value: the loop used to be a
                 * black box. */
                if (getenv("GMB_DBG")) fprintf(stderr,
                    "  [cut] round=%d ncuts=%d viol=%.4g\n", round, ncuts, viol);

                /* ---------- the cap is not a constraint of the model ----------
                 * The LP caps every bar entry at +-SDP_BIGM so that the very first
                 * one -- with no cuts yet -- has an answer. A "converged" point
                 * that sits on that cap is not converged: the LP stopped on a
                 * bound the model does not have. */
                PRIMALrescodee caprc = PRIMAL_RES_OK;
                if (status == 0 && !anycut)
                    caprc = bar_cap_verdict(t, s, symPq, nb, barOff, barPq, zsol);
                free(xt); free(ystd); stdform_free(sf);
                free(c); free(lx); free(ux); free(lc); free(uc);
                free(Arow); free(ptr); free(sub); free(val);
                if (caprc != PRIMAL_RES_OK) { rc = caprc; break; }

                if (anycut) continue;

                /* ---------- solution ---------- */
                solved = 1;
                for (int jj = 0; jj < nvar; jj++) t->x[jj] = zsol[jj];
                for (int j = 0; j < nb; j++) {
                    int d = t->barDim[j];
                    for (int p = 0; p < d; p++)
                        for (int q = 0; q < d; q++)
                            t->barx[j][p * d + q] =
                                zsol[barOff[j] + bar_pack(d, (p < q ? p : q), (p < q ? q : p))];
                }
                for (int i = 0; i < ncon; i++) t->y[i] = s * yb[i];
                for (int i = 0; i < ncon; i++) {
                    t->slc[i] = t->y[i] < 0.0 ? t->y[i] : 0.0;
                    t->suc[i] = t->y[i] > 0.0 ? t->y[i] : 0.0;
                }
                for (int jj = 0; jj < nvar; jj++) {
                    double zz = 0.0;
                    for (int i = 0; i < ncon; i++) {
                        const Col *col = &t->cols[jj];
                        for (int k = 0; k < col->nz; k++)
                            if (col->sub[k] == i) zz += col->val[k] * t->y[i];
                    }
                    zz = -(s * t->c[jj] + zz);
                    t->slx[jj] = zz < 0.0 ? zz : 0.0;
                    t->sux[jj] = zz > 0.0 ? zz : 0.0;
                }
                pobj = t->cfix;
                for (int jj = 0; jj < nvar; jj++) pobj += t->c[jj] * t->x[jj];
                for (int k = 0; k < t->nbarC; k++) {
                    int b = t->barC_bar[k], m = t->barC_sym[k], d = t->barDim[b];
                    double tr = 0.0;
                    for (int p = 0; p < d; p++)
                        for (int q = p; q < d; q++)
                            tr += symPq[m][bar_pack(d, p, q)] *
                                  t->barx[b][p * d + q];
                    pobj += t->barC_coef[k] * tr;
                }
                /* approximate bar dual: Z_j = C_j - sum_i y_i A^i.
                 * For the LP relaxation C_j - sum_i y_i A^i =
                 * sum_k lambda_k U_k U_k' (LP dual feasibility over the free
                 * variables) => Z_j is PSD by construction. */
                for (int j = 0; j < nb; j++) {
                    int d = t->barDim[j];
                    double *Z = t->barsj[j];
                    for (int k = 0; k < t->nbarC; k++) {
                        if (t->barC_bar[k] != j) continue;
                        int m = t->barC_sym[k];
                        double cf = t->barC_coef[k];
                        for (int e = 0; e < t->sym_nnz[m]; e++) {
                            int si = t->sym_subi[m][e], sj2 = t->sym_subj[m][e];
                            double v = cf * t->sym_val[m][e];
                            Z[si * d + sj2] += v;
                            if (si != sj2) Z[sj2 * d + si] += v;
                        }
                    }
                    for (int k = 0; k < t->nbarA; k++) {
                        if (t->barA_bar[k] != j) continue;
                        int m = t->barA_sym[k];
                        double cf = t->y[t->barA_con[k]] * t->barA_coef[k];
                        for (int e = 0; e < t->sym_nnz[m]; e++) {
                            int si = t->sym_subi[m][e], sj2 = t->sym_subj[m][e];
                            double v = cf * t->sym_val[m][e];
                            Z[si * d + sj2] += v;
                            if (si != sj2) Z[sj2 * d + si] += v;
                        }
                    }
                }
                t->pobj = pobj;
                t->dobj = pobj;   /* documented deviation: approximate bar dual */
                t->has_sol = 1;
                t->solsta = PRIMAL_SOL_STA_OPTIMAL;
                tlog(t, "optimal solution found\n");
                break;
            }
            if (rc == PRIMAL_RES_OK && !solved) {
                /* The cut loop ran out of rounds: what t->x holds is not a
                 * solution of the model but of an outer approximation that is
                 * still missing the cone, and here it never even reached one --
                 * measured on the 20x20 rank-one face model, which answered
                 * rc=1007 with getxx OK and x=0, pobj=0 against an optimum of
                 * -1.  Same rule as the conic IPM refusal above: no solution is
                 * claimed when there is none (PRIMAL_getsolsta answers UNKNOWN
                 * on its own), and the getters refuse. */
                tlog(t, "iteration limit (tagli PSD)\n");
                rc = PRIMAL_RES_TRM_MAX_ITER;
            }
        }
        free(eval); free(evec); free(Xf); free(zsol); free(yb);
    }

    for (int k = 0; k < ncuts; k++) { free(cuts[k].sub); free(cuts[k].val); }
    free(cuts);
    for (int m = 0; m < t->nsym; m++) free(symPq[m]);
    free(symPq);
    free(barOff); free(barPq);
    return rc;
}

/* =====================================================================
 * MIP path: depth-first branch & bound over LP relaxations (simplex).
 * Branching: most fractional integer variable.  The integrality threshold is
 * PRIMAL_DPAR_MIP_TOL_INTHER and the node cap PRIMAL_IPAR_MIP_MAX_NODES
 * (relaxations solved); hitting the cap returns PRIMAL_RES_TRM_MAX_ITER with
 * the incumbent (if any) stored.  A node is pruned when its relaxation bound
 * reaches the incumbent within the gap tolerance, the larger of
 * PRIMAL_DPAR_MIP_TOL_ABS_GAP and PRIMAL_DPAR_MIP_TOL_REL_GAP*(1+|incumbent|)
 * (the relative one is off at 0 by default, see README).  QP + integers and
 * cones + integers are documented deviations: PRIMAL_RES_ERR_ARG.
 * Duals are not reported for MIP solutions (y=z=0, dobj=pobj), matching
 * the practical use of PRIMAL's MIP solver.
 * ===================================================================== */


/* LP/QP relaxation of the task with bounds (lx,ux)/(lc,uc); min-space
 * objective (sum s*c*x) returned in *pmin (linear part only: the QP
 * objective is added by the caller for the incumbent comparison? no —
 * for QP relaxations pmin includes the quadratic term of the ORIGINAL
 * variables). Status: 0 ok, 1 infeas, 2 unbounded, 3 iter limit. */

/* The LP routing (crash basis + revised simplex, tableau fallback) is
 * defined further below; the MIP relaxation must use the SAME engine as
 * the LP path, otherwise the tableau (which gives a suboptimal optimum
 * on some forms) makes the B&B diverge. */

int mip_relax(PRIMALtask_t t, int s, const double *lx, const double *ux,
                     const double *lc, const double *uc,
                     double *xout, double *pmin) {
    int nvar = t->numvar, ncon = t->numcon;
    double *ci = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!ci) return 3;
    for (int j = 0; j < nvar; j++) ci[j] = s * t->c[j];

    /* min-form Q (only the objective; QP relaxations use the IPM) */
    double *qv = t->has_qobj ? scaled_qvals(t, s, NULL) : NULL;
    if (t->has_qobj && !qv) { free(ci); return 3; }

    int *ptr = NULL, *sub = NULL;
    double *aval = NULL;
    if (!build_csc(t, &ptr, &sub, &aval)) { free(ci); free(qv); return 3; }

    /* Node bound tightening: the MIP publishes no duals, so tightening the
     * implied bounds is sound. An empty interval closes the node. */
    double *tlx = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    double *tux = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!tlx || !tux) { free(tlx); free(tux); free(ci); free(qv); free(ptr); free(sub); free(aval); return 3; }
    memcpy(tlx, lx, (size_t)nvar * sizeof(double));
    memcpy(tux, ux, (size_t)nvar * sizeof(double));
    if (!getenv("GMB_NO_BOUND_TIGHTEN"))
        bound_tighten(t, nvar, ncon, tlx, tux, lc, uc, NULL, NULL, NULL, NULL);
    for (int j = 0; j < nvar; j++)
        if (tlx[j] > tux[j] + 1e-12 * (1.0 + fabs(tlx[j]))) {
            free(tlx); free(tux); free(ci); free(qv); free(ptr); free(sub); free(aval);
            return 1;
        }
    /* Row/column equilibration as in the LP path: the simplex on the unscaled
     * problem can stop at a suboptimal optimum (reproducible bug: a binary
     * knapsack with [0,1] bounds gave -8.5 instead of -9). Copies because
     * scale_equilibrate modifies in place; the solution is descaled via x = D x'. */
    int nnz = ptr[nvar];
    double *lc_s = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *uc_s = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
    double *av_s = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    double *ds   = (double *)malloc((size_t)(nvar > 0 ? nvar : 1) * sizeof(double));
    if (!lc_s || !uc_s || !av_s || !ds) {
        free(lc_s); free(uc_s); free(av_s); free(ds);
        free(tlx); free(tux); free(ci); free(qv); free(ptr); free(sub); free(aval);
        return 3;
    }
    for (int i = 0; i < ncon; i++) { lc_s[i] = lc[i]; uc_s[i] = uc[i]; }
    for (int q = 0; q < nnz; q++) av_s[q] = aval[q];
    if (t->scaling && !t->has_qobj) {
        double *rs = (double *)malloc((size_t)(ncon > 0 ? ncon : 1) * sizeof(double));
        if (!rs) { free(lc_s); free(uc_s); free(av_s); free(ds);
                   free(tlx); free(tux); free(ci); free(qv); free(ptr); free(sub); free(aval); return 3; }
        scale_equilibrate(nvar, ncon, ptr, sub, av_s, lc_s, uc_s, tlx, tux, ci, NULL, rs, ds);
        free(rs);
    } else {
        for (int j = 0; j < nvar; j++) ds[j] = 1.0;
    }

    StdForm *sf = stdform_build(nvar, ncon, ci, t->qt_i, t->qt_j, qv,
                                t->has_qobj ? t->qt_n : 0, tlx, tux, lc_s, uc_s, ptr, sub, av_s);
    free(lc_s); free(uc_s); free(av_s);
    free(tlx); free(tux);
    free(ptr); free(sub); free(aval); free(qv);
    if (!sf) { free(ci); free(ds); return 3; }

    double *xt   = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    double *ystd = (double *)calloc((size_t)(sf->m > 0 ? sf->m : 1), sizeof(double));
    double *zst  = (double *)calloc((size_t)(sf->n > 0 ? sf->n : 1), sizeof(double));
    if (!xt || !ystd || !zst) {
        free(xt); free(ystd); free(zst); stdform_free(sf); free(ci); return 3;
    }

    int status;
    if (t->has_qobj) {
        double *dA = stdform_dense_A(sf), *dQ = stdform_dense_Q(sf);
        if (!dA || !dQ) { free(dA); free(dQ); free(xt); free(ystd); free(zst); stdform_free(sf); free(ci); return 3; }
        status = ipm_solve_std(dA, dQ, sf->m, sf->n, sf->b, sf->c,
                               t->tol_qo_gap, t->tol_qo_pfeas, t->tol_qo_dfeas, iter_cap(t->max_iter_intpnt),
                               xt, ystd, zst, NULL, NULL);
        free(dA); free(dQ);
    } else {
        /* Same engine as the LP path (method 0): crash basis + revised, with
         * tableau fallback. The tableau alone can stop at a suboptimal
         * optimum (see the forward-declaration comment). */
        status = solve_std_routed(sf->Aptr, sf->Arow, sf->Aval, NULL, NULL, NULL,
                                  sf->m, sf->n, sf->b, sf->c, t, xt, ystd, zst,
                                  NULL, NULL, 0, NULL, NULL);
    }
    if (status == 0) {
        stdform_map_x(sf, xt, xout);
        for (int j = 0; j < nvar; j++) xout[j] *= ds[j];   /* descale the columns */
        double p = 0.0;
        for (int j = 0; j < nvar; j++) p += (s * t->c[j]) * xout[j];
        if (t->has_qobj) {
            double qq = task_xQx(t, xout);
            p += 0.5 * s * qq;   /* min-form quadratic term (s on 1/2 x'Qx) */
        }
        *pmin = p;
    }
    free(xt); free(ystd); free(zst); stdform_free(sf); free(ci); free(ds);
    return status;
}

