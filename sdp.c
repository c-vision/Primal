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

/* sdp.c - primal-dual interior point for conic problems (standard form)
 *
 *   min  c'x + sum_j <C_j, X_j> + sum_i <c_i, z_i>
 *   s.t. A x + sum_j <A_kj, X_j> + sum_i <a_ki, z_i> = b_k
 *        x >= 0,  X_j >= 0 (PSD),  z_i in SOC
 *
 * Unified conic path: R_+, SOC and PSD blocks in the same interior point.
 * R_+ and PSD blocks enter through their Nesterov-Todd scaling and the
 * normal-equations Schur complement on the m equality rows:
 *     (E Theta E') dy = -rp - E Theta(r~ + rd),
 *     dz = Theta(E' dy + r~ + rd),   ds = -Theta^-1 dz + r~,
 * with Theta_i = x_i/s_i (R_+), Theta_j(V) = W_j V W_j (PSD, W_j S_j W_j = X_j).
 * SOC blocks are NOT scaled by Q(w): the NT scaling point diverges at the cone
 * boundary (z o s -> 0) so the direction becomes unusable near the optimum.
 * Instead each SOC block keeps its cone equation in the arrow-matrix form used
 * by socp.c, robust at the boundary,
 *     A(z) dz + A(s) ds = -A(z) s + sigma*mu*e   (e = (1,0,..,0)),
 * and the full augmented KKT  [Kpp Esoc'; Esoc Theta_soc]  is solved for
 * (dy, dz_soc); then ds = -rd - E'dy and ds_soc follows from the same equation.
 * Mehrotra predictor-corrector throughout.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "sdp.h"
#include "linalg.h"
#include "expcone.h"

/* per-iteration callback hook (implemented in primal.c) */
extern int primal_cb_iter_on;
void primal_cb_iter(int code);


/* Iterates the exp/power line search remembers the merit of, to let a step that
 * worsens it briefly (the only way out of a basin far from the path). */
#define MWIN 4

/* ---------- second-order-cone Jordan-algebra helpers ---------- */
/* Signed distance of v to the boundary of SOC, in the cone's own units: v[0] is
 * how far past the vertex along the axis the ball of radius |v_{1:}| reaches.
 * Zero on the boundary, negative outside; the ratio of two blocks' values is
 * meaningless across different k, which is why this is reported per block. */
static double soc_margin(const double *v, int k) { double nb = 0; for (int i = 1; i < k; i++) nb += v[i] * v[i]; return v[0] - sqrt(nb); }
/* Largest t in [0,1] with v + t*dv still in the interior.  |t*s| = 0 is a
 * QUADRATIC in t (A t^2 + B t + C, the cone's own defining difference), so the
 * boundary is hit at an exact root -- no bisection, and no conservatism, unlike
 * the exp cones where the epigraph is transcendental.  A <= 0 means the
 * direction never returns to the boundary (or runs parallel to it), which is
 * t = 1. */
static double soc_step(const double *v, const double *dv, int k) {
    double n2 = 0, nd2 = 0, cr = 0;
    for (int i = 1; i < k; i++) { n2 += v[i] * v[i]; nd2 += dv[i] * dv[i]; cr += v[i] * dv[i]; }
    double A = nd2 - dv[0] * dv[0], B = 2 * (cr - v[0] * dv[0]), C = n2 - v[0] * v[0];
    if (fabs(A) < 1e-300) { if (B < -1e-300) { double t = -C / B; return t < 1 ? t : 1; } return 1; }
    double disc = B * B - 4 * A * C; if (disc < 0) return 1;
    double sq = sqrt(disc), r2 = (-B + sq) / (2 * A);
    return (r2 > 0 && r2 < 1) ? r2 : 1;
}
/* max step t in [0,1] keeping z + t*dz in the exp/power cone interior. */
static double expcone_maxstep(int kind, double alpha, const double *z, const double *dz) {
    double lo = 0.0, hi = 1.0, fv, x[3];
    for (int a = 0; a < 3; a++) x[a] = z[a] + dz[a];
    if (expcone_barrier(kind, alpha, x, &fv) == 0) return 1.0;
    for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi);
        for (int a = 0; a < 3; a++) x[a] = z[a] + mid * dz[a];
        if (expcone_barrier(kind, alpha, x, &fv) == 0) lo = mid; else hi = mid;
    }
    return lo;
}
/* Same for the dual variable, which lives in the DUAL cone K* (these cones are
 * not self-dual, so testing membership in K would clamp the dual step to 0). */
static double expcone_dual_maxstep(int kind, double alpha, const double *s, const double *ds) {
    double lo = 0.0, hi = 1.0, x[3];
    for (int a = 0; a < 3; a++) x[a] = s[a] + ds[a];
    if (expcone_dual_in(kind, alpha, x)) return 1.0;
    for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi);
        for (int a = 0; a < 3; a++) x[a] = s[a] + mid * ds[a];
        if (expcone_dual_in(kind, alpha, x)) lo = mid; else hi = mid;
    }
    return lo;
}
/* Arrow matrix A(v) (socp.c convention) and its action. */
static void soc_arrow(const double *v, int k, double *A) {
    memset(A, 0, sizeof(double) * (size_t)k * k);
    A[0] = v[0];
    for (int i = 1; i < k; i++) { A[i] = v[i]; A[i * k] = v[i]; A[i * k + i] = v[0]; }
}
/* o = A w, with A already built by soc_arrow (dense k x k, no symmetry assumed:
 * the callers feed it A(z) and A(s), which are symmetric, and w any vector). */
static void soc_arrow_mul(const double *A, const double *w, int k, double *o) {
    for (int i = 0; i < k; i++) { double t = 0; for (int j = 0; j < k; j++) t += A[i * k + j] * w[j]; o[i] = t; }
}
/* ---------- symmetric (PSD) helpers ---------- */
/* R = f(A) for a symmetric A by eigendecomposition, f applied to the
 * eigenvalues: kind 0 f = sqrt(max(lam,0)) (the NT scaling point needs it on a
 * numerically semidefinite product, where a small negative eigenvalue is
 * round-off, not a direction to leave the cone), kind 1 f = 1/sqrt(lam) and
 * kind 2 f = 1/lam -- the last two are floored rather than skipped because an
 * eigenvalue at 0 makes the inverse genuinely undefined and a large finite
 * entry keeps the LU solvable; the divergence then shows up as a stalled
 * iterate, which the caller measures, instead of as an Inf in the matrix.
 * Leaves R untouched when the eigensolver cannot be given workspace. */
static void sym_fun(int d, const double *A, int kind, double *R) {
    double *ev = (double *)malloc((size_t)d * sizeof(double));
    double *V  = (double *)malloc((size_t)d * d * sizeof(double));
    double *Ac = (double *)malloc((size_t)d * d * sizeof(double));
    if (!ev || !V || !Ac) { free(ev); free(V); free(Ac); return; }
    memcpy(Ac, A, sizeof(double) * (size_t)d * d);
    dmat_eig_jacobi(d, Ac, ev, V);
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) {
            double lam = ev[k], f;
            if (kind == 0)      f = lam > 0.0 ? sqrt(lam) : 0.0;
            else if (kind == 1) f = 1.0 / sqrt(lam > 1e-12 ? lam : 1e-12);
            else                f = 1.0 / (fabs(lam) > 1e-12 ? fabs(lam) : 1e-12);
            s += f * V[i * d + k] * V[j * d + k];
        }
        R[i * d + j] = s;
    }
    free(ev); free(V); free(Ac);
}
/* Dense row-major product C = A B (d x d).  Small by design: it is called per
 * PSD block per Newton pass to build W = X#S and its inverses, where a block is
 * a few dozen across at most. */
static void mmul(int d, const double *A, const double *B, double *C) {
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) { double s = 0.0; for (int k = 0; k < d; k++) s += A[i * d + k] * B[k * d + j]; C[i * d + j] = s; }
}
/* Smallest eigenvalue of a symmetric matrix -- the cone margin of a PSD block.
 * Copies first because dmat_eig_jacobi overwrites its argument: the callers feed
 * it the PUBLISHED Xbar/Sbar blocks, and a spectrally deformed block would be
 * reported as the solution. */
static double min_eig(int d, const double *A) {
    double *ev = (double *)malloc((size_t)d * sizeof(double));
    double *V  = (double *)malloc((size_t)d * d * sizeof(double));
    double *Ac = (double *)malloc((size_t)d * d * sizeof(double));
    double lo = 0.0;
    if (ev && V && Ac) { memcpy(Ac, A, sizeof(double) * (size_t)d * d); dmat_eig_jacobi(d, Ac, ev, V); lo = ev[0]; for (int k = 1; k < d; k++) if (ev[k] < lo) lo = ev[k]; }
    free(ev); free(V); free(Ac); return lo;
}
/* tr(A B) = <A,B>, the inner product the PSD cone pairs blocks with.  B is read
 * with its indices swapped, not because B is symmetric (the blocks are, but a
 * scratch product fed here is not always) but because tr(A B) is what the conic
 * Lagrangian uses: sum_ij A_ij B_ji.  Using A_ij B_ij instead would agree on
 * symmetric inputs and silently mis-read every other one. */
static double trAB(int d, const double *A, const double *B) { double s = 0.0; for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) s += A[i * d + j] * B[j * d + i]; return s; }

/* ---- flat snapshot of the whole iterate (scalar pair, multipliers, PSD/SOC/
 * exp blocks), used by the exp/power mu-continuation to keep the best point.
 * ipm_state() copies iterate -> buf when out != 0, buf -> iterate when 0. ---- */
static size_t ipm_state_size(int n, int m, int nb, const int *dims,
                             int nsoc, const int *socdims, int Ke) {
    size_t s = 2 * (size_t)n + (size_t)m + 2 * (size_t)Ke;
    for (int j = 0; j < nb; j++) s += 2 * (size_t)dims[j] * dims[j];
    for (int i = 0; i < nsoc; i++) s += 2 * (size_t)socdims[i];
    return s;
}
#define IPM_STATE_ARGS int n, int m, int nb, const int *dims, int nsoc, const int *socdims, int Ke, \
    double *xs, double *ss, double *y, double *const *Xbar, double *const *Sbar, \
    double *const *Zsoc, double *const *Ssoc, double *ez, double *es
static void ipm_state(double *buf, int out, IPM_STATE_ARGS) {
    size_t o = 0;
#define IPM_CP(p, cnt) do { size_t bytes_ = (size_t)(cnt) * sizeof(double);                       \
        if (out) memcpy(buf + o, (p), bytes_); else memcpy((p), buf + o, bytes_);                 \
        o += (size_t)(cnt); } while (0)
    IPM_CP(xs, n); IPM_CP(ss, n); IPM_CP(y, m);
    for (int j = 0; j < nb; j++) { IPM_CP(Xbar[j], dims[j] * dims[j]); IPM_CP(Sbar[j], dims[j] * dims[j]); }
    for (int i = 0; i < nsoc; i++) { IPM_CP(Zsoc[i], socdims[i]); IPM_CP(Ssoc[i], socdims[i]); }
    IPM_CP(ez, Ke); IPM_CP(es, Ke);
#undef IPM_CP
}
/* The iterate as seen from inside sdp_ipm (its locals carry these names). */
#define IPM_STATE_PASS n, m, nb, dims, nsoc, socdims, Ke, \
    xs, ss, y, Xbar, Sbar, Zsoc, Ssoc, ez, es

/* Search directions of one pass: primal, dual and multipliers, per block. */
typedef struct {
    const double *dx, *ds, *Dx, *Ds, *Dzsoc, *Dssoc, *Dez, *Des, *dy;
    const int *soff;
    size_t dmax2;
} ipmdir;

/* x += ap*dx for the primal side, += ad*d* for the dual side, in every block. */
static void ipm_apply(double ap, double ad, const ipmdir *D, IPM_STATE_ARGS) {
    for (int i = 0; i < n; i++) { xs[i] += ap * D->dx[i]; ss[i] += ad * D->ds[i]; }
    for (int j = 0; j < nb; j++) { int d = dims[j];
        for (int a = 0; a < d * d; a++) Xbar[j][a] += ap * D->Dx[(size_t)j * D->dmax2 + a];
        for (int a = 0; a < d * d; a++) Sbar[j][a] += ad * D->Ds[(size_t)j * D->dmax2 + a]; }
    for (int i = 0; i < nsoc; i++) { int kk = socdims[i];
        for (int a = 0; a < kk; a++) Zsoc[i][a] += ap * D->Dzsoc[D->soff[i] + a];
        for (int a = 0; a < kk; a++) Ssoc[i][a] += ad * D->Dssoc[D->soff[i] + a]; }
    for (int a = 0; a < Ke; a++) { ez[a] += ap * D->Dez[a]; es[a] += ad * D->Des[a]; }
    for (int k = 0; k < m; k++) y[k] += ad * D->dy[k];
}

/* ---- residual kernel.  The main loop and the merit line search must measure
 * the same three numbers, so the computation lives here once: primal rows,
 * dual rows (scalar, bar, SOC, exp) and the complementarity measure. ---- */
typedef struct { double pfeas, dfeas, mu; } ipmres;

typedef struct {
    int n, m, nb, nsoc, nep, Ke, ntot;
    double bn, cn;                    /* |b|_inf, |c|_inf: MOSEK's normalisers */
    const double *A, *b, *c;
    const int *dims, *socdims;
    const double *const *Cbar, *const *Abar;
    const double *const *Csoc, *const *Asoc;
    const double *const *Cexp, *const *Aexp;
    const int *ekind; const double *ealpha;
    double *xs, *ss, *y, *rp, *rdx, *scr;
    double *const *Xbar, *const *Sbar, *const *Zsoc, *const *Ssoc;
    double *ez, *es;
} ipmc;

static ipmres ipm_resid(const ipmc *P) {
    const int n = P->n, m = P->m, nb = P->nb, nsoc = P->nsoc, nep = P->nep;
    const int Ke = P->Ke;
    const double *A = P->A, *b = P->b, *c = P->c;
    const int *dims = P->dims, *socdims = P->socdims;
    double *xs = P->xs, *ss = P->ss, *y = P->y, *t1 = P->scr;
    double *rp = P->rp, *rdx = P->rdx, *ez = P->ez, *es = P->es;
    ipmres R; double mu = 0.0, pf = 0.0, df = 0.0;
    for (int k = 0; k < m; k++) {
        double t = -b[k];
        for (int i = 0; i < n; i++) t += A[k * n + i] * xs[i];
        for (int j = 0; j < nb; j++) t += trAB(dims[j], P->Abar[k * nb + j], P->Xbar[j]);
        for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; const double *ak = P->Asoc[k * nsoc + i];
            for (int a = 0; a < kk; a++) t += ak[a] * P->Zsoc[i][a]; }
        for (int i = 0; i < nep; i++) { const double *ak = P->Aexp[k * nep + i];
            for (int a = 0; a < 3; a++) t += ak[a] * ez[3 * i + a]; }
        rp[k] = t;
    }
    for (int i = 0; i < n; i++) { double t = ss[i] - c[i]; for (int k = 0; k < m; k++) t += A[k * n + i] * y[k]; rdx[i] = t; }
    for (int i = 0; i < n; i++) mu += xs[i] * ss[i];
    for (int j = 0; j < nb; j++) mu += trAB(dims[j], P->Xbar[j], P->Sbar[j]);
    for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; for (int a = 0; a < kk; a++) mu += P->Zsoc[i][a] * P->Ssoc[i][a]; }
    for (int a = 0; a < Ke; a++) mu += ez[a] * es[a];
    mu /= (double)(P->ntot > 0 ? P->ntot : 1);

    for (int k = 0; k < m; k++) { double a = fabs(rp[k]); if (a > pf) pf = a; }
    for (int i = 0; i < n; i++) { double a = fabs(rdx[i]); if (a > df) df = a; }
    for (int j = 0; j < nb; j++) { int d = dims[j];
        for (int a = 0; a < d * d; a++) t1[a] = -P->Cbar[j][a] + P->Sbar[j][a];
        for (int k = 0; k < m; k++) { double yk = y[k]; const double *Akj = P->Abar[k * nb + j]; for (int a = 0; a < d * d; a++) t1[a] += yk * Akj[a]; }
        for (int a = 0; a < d * d; a++) { double v = fabs(t1[a]); if (v > df) df = v; } }
    for (int i = 0; i < nsoc; i++) { int kk = socdims[i];
        for (int a = 0; a < kk; a++) t1[a] = -P->Csoc[i][a] + P->Ssoc[i][a];
        for (int k = 0; k < m; k++) { double yk = y[k]; const double *ak = P->Asoc[k * nsoc + i]; for (int a = 0; a < kk; a++) t1[a] += yk * ak[a]; }
        for (int a = 0; a < kk; a++) { double v = fabs(t1[a]); if (v > df) df = v; } }
    for (int i = 0; i < nep; i++) {
        double rde[3];
        for (int a = 0; a < 3; a++) rde[a] = -P->Cexp[i][a] + es[3 * i + a];
        for (int k = 0; k < m; k++) { const double *ak = P->Aexp[k * nep + i]; double yk = y[k]; for (int a = 0; a < 3; a++) rde[a] += yk * ak[a]; }
        for (int a = 0; a < 3; a++) { double v = fabs(rde[a]); if (v > df) df = v; }
    }
    R.pfeas = pf; R.dfeas = df; R.mu = mu;
    return R;
}

/* ---- measured quality of the current iterate, in MOSEK's three relative
 * numbers.  The gap is computed from the two objectives, NOT from mu: in this
 * standard form pobj - dobj does equal <x,s> = mu*ntot when the iterate is
 * dual feasible, and measuring it closes the loop on that identity instead of
 * assuming it.  pobj = <c,x> + sum <C_i,z_i> over the cone blocks, dobj = b'y.
 * The normalisers are MOSEK's (1 + |b|_inf) and (1 + |c|_inf), so a residual
 * stays a residual on a problem whose data happen to be tiny. ---- */
typedef struct { double pri, dual, gap; } ipmqual;

static ipmqual ipm_quality(const ipmc *P, const ipmres *R) {
    double pobj = 0.0, dobj = 0.0;
    ipmqual Q;
    for (int i = 0; i < P->n; i++) pobj += P->c[i] * P->xs[i];
    for (int j = 0; j < P->nb; j++) pobj += trAB(P->dims[j], P->Cbar[j], P->Xbar[j]);
    for (int i = 0; i < P->nsoc; i++) { int kk = P->socdims[i]; const double *ci = P->Csoc[i];
        for (int a = 0; a < kk; a++) pobj += ci[a] * P->Zsoc[i][a]; }
    for (int i = 0; i < P->nep; i++) { const double *ci = P->Cexp[i];
        for (int a = 0; a < 3; a++) pobj += ci[a] * P->ez[3 * i + a]; }
    for (int k = 0; k < P->m; k++) dobj += P->b[k] * P->y[k];
    Q.pri = R->pfeas / (1.0 + P->bn);
    Q.dual = R->dfeas / (1.0 + P->cn);
    Q.gap = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
    return Q;
}

/* ---- signed slack of a point with respect to the cone it lives in: positive
 * inside, negative outside.  The UNITS differ per cone -- a bar slack is an
 * eigenvalue, an exp PEXP slack is t - u*exp(v/u) (degree 1) and a PPOW/RPOW
 * one is rot*t^(2a)*u^(2(1-a)) - v^2 (degree 2) -- so what these answer is the
 * SIGN and the order of depth, not a comparable distance.  They exist because
 * rel_pri measures only the equality rows: a point sitting outside K still
 * reads rel_pri = 0, and membership is enforced by the line-search test alone.
 * Each helper is the min over the functions that DEFINE the cone (u > 0 and
 * the epigraph for PEXP; t > 0, u > 0 and the hypograph for PPOW/RPOW), so a
 * negative value is a violation, not a shallow interior point. */
static double exp_blk_slack(int kind, double alpha, const double *z) {
    double t = z[0], u = z[1], v = z[2], a;
    if (kind == EXPCONE_PEXP) {
        a = u;
        if (u > 0.0) { double G = t - u * exp(v / u); if (G < a) a = G; }
        return a;
    }
    a = (t < u) ? t : u;
    if (t > 0.0 && u > 0.0) {
        double rot = (kind == EXPCONE_RPOW) ? 2.0 : 1.0;
        double G = rot * pow(t, 2.0 * alpha) * pow(u, 2.0 * (1.0 - alpha)) - v * v;
        if (G < a) a = G;
    }
    return a;
}
/* Same for K*: PEXP* = {s0 > 0, -s2 > 0, s1 - s2 + s2*ln(-s2/s0) > 0};
 * the power cones need (s0/a)^a (s1/(1-a))^(1-a) >= (1|sqrt 2) times |s2|. */
static double exp_blk_dual_slack(int kind, double alpha, const double *s) {
    double s0 = s[0], s1 = s[1], s2 = s[2], a;
    if (kind == EXPCONE_PEXP) {
        a = (s0 < -s2) ? s0 : -s2;
        if (s0 > 0.0 && s2 < 0.0) {
            double q = s1 - s2 + s2 * log(-s2 / s0);
            if (q < a) a = q;
        }
        return a;
    }
    a = (s0 < s1) ? s0 : s1;
    if (s0 > 0.0 && s1 > 0.0) {
        double lhs = pow(s0 / alpha, alpha) * pow(s1 / (1.0 - alpha), 1.0 - alpha);
        double d = lhs - (kind == EXPCONE_RPOW ? sqrt(2.0) : 1.0) * fabs(s2);
        if (d < a) a = d;
    }
    return a;
}
/* Worst signed slack over every block of the product cone, primal and dual:
 * R_+^n, each PSD block (smallest eigenvalue), each SOC/RQUAD block, each
 * exp/power block and its dual. */
static void ipm_cone_slacks(const ipmc *P, double *pri, double *dual) {
    const int n = P->n, nb = P->nb, nsoc = P->nsoc, nep = P->nep;
    double a = HUGE_VAL, b = HUGE_VAL;
#define MINQ(v) do { if ((v) < a) a = (v); } while (0)
#define MINQD(v) do { if ((v) < b) b = (v); } while (0)
    for (int i = 0; i < n; i++) { MINQ(P->xs[i]); MINQD(P->ss[i]); }
    for (int j = 0; j < nb; j++) { int d = P->dims[j];
        MINQ(min_eig(d, P->Xbar[j])); MINQD(min_eig(d, P->Sbar[j])); }
    for (int i = 0; i < nsoc; i++) { int kk = P->socdims[i];
        MINQ(soc_margin(P->Zsoc[i], kk)); MINQD(soc_margin(P->Ssoc[i], kk)); }
    for (int i = 0; i < nep; i++) {
        MINQ(exp_blk_slack(P->ekind[i], P->ealpha[i], P->ez + 3 * i));
        MINQD(exp_blk_dual_slack(P->ekind[i], P->ealpha[i], P->es + 3 * i)); }
#undef MINQ
#undef MINQD
    *pri = a; *dual = b;
}

/* Compensated dot product (Neumaier summation + FMA product residual).  On a
 * degenerate SDP the KKT residual loses digits to the double summation, and the
 * iterative refinement downstream can only correct the LU error it can actually
 * see -- a residual computed in plain double caps that.  Two doubles of
 * precision for the price of a few FMA (long double is 8 bytes on arm64). */
static double nsum_prod(const double *a, const double *b, int n) {
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double p = a[i] * b[i];
        double e = fma(a[i], b[i], -p);
        double t = s + p;
        double bb = (fabs(s) >= fabs(p)) ? ((s - t) + p) : ((p - t) + s);
        c += bb + e;
        s = t;
    }
    return s + c;
}

/* Rescue Newton solve for a PSD-only path whose Schur direction failed.
 * Retain scalar and svec(PSD) primal increments in [0 E; E' -H], where
 * H_i=s_i/x_i and H_j(D)=W_j^-1 D W_j^-1.  Eliminating these increments
 * gives the usual E H^-1 E' Schur system, but can erase its small directions
 * near a PSD face.  Solving the augmented equations retains A dx=-rp.
 * svec uses sqrt(2) on off-diagonals, preserving the trace inner product.
 * Recover dual increments from dual feasibility, avoiding a second ill-
 * conditioned congruence through W^-1.  The fixed workspace limit (512
 * rows) keeps this exceptional dense solve bounded; larger models retain
 * the existing non-convergence/fallback behavior.  No model is reduced.
 */
static int sdp_aug_direction(int m,int n,int nb,const int *dims,size_t d2,
 const double *A,const double *const *Ab,const double *const *Cb,
 const double *xs,const double *ss,const double *const *Sb,const double *y,
 const double *rp,const double *rd,const double *Wi,const double *Xi,double target,
 double *dy,double *dx,double *ds,double *Dx,double *Ds){
 /* Bound this dense rescue workspace before dimension products/allocations. */
 if(m>512||n>512-m)return 0;
 int q=n;
 for(int j=0;j<nb;j++){
  int d=dims[j];if(d<1||d>31)return 0;
  int packed=d*(d+1)/2;if(packed>512-m-q)return 0;q+=packed;
 }
 int N=m+q;
 double *K=calloc((size_t)N*N,sizeof(double)),*r=calloc(N,sizeof(double));
 double *cs=calloc(N,sizeof(double)),*org=calloc(N,sizeof(double)),*cor=calloc(N,sizeof(double));
 int *bj=calloc(q,sizeof(int)),*bi=calloc(q,sizeof(int)),*bl=calloc(q,sizeof(int));
 if(!K||!r||!cs||!org||!cor||!bj||!bi||!bl){free(K);free(r);free(cs);free(org);free(cor);free(bj);free(bi);free(bl);return 0;}
 for(int k=0;k<m;k++)r[k]=-rp[k];
 for(int i=0;i<n;i++){
  for(int k=0;k<m;k++)K[(size_t)k*N+m+i]=K[(size_t)(m+i)*N+k]=A[(size_t)k*n+i];
  K[(size_t)(m+i)*N+m+i]=-ss[i]/xs[i];r[m+i]=-rd[i]+ss[i]-target/xs[i];
 }
 int z=n;
 for(int j=0;j<nb;j++){int d=dims[j];for(int a=0;a<d;a++)for(int b=a;b<d;b++,z++){
  bj[z]=j;bi[z]=a;bl[z]=b;double f=a==b?1:sqrt(2.0);
  double v=Cb[j][a*d+b]-target*Xi[(size_t)j*d2+a*d+b];
  for(int k=0;k<m;k++){double av=f*Ab[k*nb+j][a*d+b];K[(size_t)k*N+m+z]=K[(size_t)(m+z)*N+k]=av;v-=Ab[k*nb+j][a*d+b]*y[k];}
  r[m+z]=f*v;
 }}
 for(int u=n;u<q;u++)for(int v=n;v<q;v++)if(bj[u]==bj[v]){
  int j=bj[u],d=dims[j],a=bi[u],b=bl[u],c=bi[v],e=bl[v];const double *W=Wi+(size_t)j*d2;
  double h=W[a*d+c]*W[e*d+b];if(c!=e)h+=W[a*d+e]*W[c*d+b];
  if(c!=e)h/=sqrt(2.0);if(a!=b)h*=sqrt(2.0);
  K[(size_t)(m+u)*N+m+v]=-h;
 }
 for(int i=0;i<N;i++){double mx=0;for(int j=0;j<N;j++)if(fabs(K[(size_t)i*N+j])>mx)mx=fabs(K[(size_t)i*N+j]);double rs=mx>0?1/mx:1;r[i]*=rs;for(int j=0;j<N;j++)K[(size_t)i*N+j]*=rs;}
 for(int j=0;j<N;j++){double mx=0;for(int i=0;i<N;i++)if(fabs(K[(size_t)i*N+j])>mx)mx=fabs(K[(size_t)i*N+j]);cs[j]=mx>0?1/mx:1;for(int i=0;i<N;i++)K[(size_t)i*N+j]*=cs[j];}
 memcpy(org,r,(size_t)N*sizeof(double));LuFact *lf=dmat_lu_factor(K,N);int ok=lf!=NULL;
 if(ok){dmat_lu_solve_comp(lf,r);
  for(int it=0;it<3;it++){
   double before=0,after=0;
   for(int i=0;i<N;i++){cor[i]=org[i]-nsum_prod(K+(size_t)i*N,r,N);if(fabs(cor[i])>before)before=fabs(cor[i]);}
   dmat_lu_solve_comp(lf,cor);for(int i=0;i<N;i++)r[i]+=cor[i];
   for(int i=0;i<N;i++){double e=fabs(org[i]-nsum_prod(K+(size_t)i*N,r,N));if(e>after)after=e;}
   if(!(after<before)){for(int i=0;i<N;i++)r[i]-=cor[i];break;}
  }
  for(int i=0;i<N;i++)if(!isfinite(r[i]))ok=0;
 }
 if(ok){
  for(int i=0;i<N;i++)r[i]*=cs[i];for(int k=0;k<m;k++)dy[k]=r[k];
  for(int i=0;i<n;i++){dx[i]=r[m+i];double v=-rd[i];for(int k=0;k<m;k++)v-=A[k*n+i]*dy[k];ds[i]=v;}
  for(int u=n;u<q;u++){int j=bj[u],d=dims[j],a=bi[u],b=bl[u];double v=r[m+u]/(a==b?1:sqrt(2.0));Dx[(size_t)j*d2+a*d+b]=Dx[(size_t)j*d2+b*d+a]=v;}
  for(int j=0;j<nb;j++){int d=dims[j];for(int a=0;a<d*d;a++){double v=Cb[j][a]-Sb[j][a];for(int k=0;k<m;k++)v-=Ab[k*nb+j][a]*(y[k]+dy[k]);Ds[(size_t)j*d2+a]=v;}}
 }
 dmat_lu_free(lf);free(K);free(r);free(cs);free(org);free(cor);free(bj);free(bi);free(bl);return ok;
}

/* ---- Experimental HSD (homogeneous self-dual) embedding, PSD+scalar only ----
 * Opt-in via GMB_SDP_HSD (default off, so unset behaves exactly as before):
 * the free-variable-as-difference-of-two-nonnegatives trick used throughout
 * the tree (primal.c splits every free scalar this way before calling
 * sdp_ipm) lets both halves drift together along the resulting null
 * direction with nothing but the log-barrier to restrain them -- measured on
 * a degenerate SOS certificate (M1, jcpaik/p2-kkt-flag-sos) both halves reach
 * ~1e6 while their difference, the variable that matters, sits at the wrong
 * value.  The self-dual embedding adds a homogenizing tau and an
 * infeasibility certificate kappa plus a normalizing theta row, which bounds
 * the whole trajectory instead of letting a null direction run away.  Kept
 * separate from the main IPM above (PSD+scalar blocks only, nsoc == 0 &&
 * nep == 0): it does not touch SOC or exp/power, and even where it applies it
 * is a second, unproven code path -- opt-in until it is validated the way
 * the primary path was (three ways: hand value, full KKT via public getters,
 * BAS vs ITR cross-check). */

/* Double-double (error-free transformations): the Newton system here mixes
 * the same wildly different magnitudes that motivated dmat_lu_solve_comp,
 * and a homogeneous embedding adds tau/kappa/theta columns that couple every
 * row to the model's own scale -- two doubles of mantissa is cheap next to
 * getting the direction wrong. */
typedef struct { double hi, lo; } hdd;

static hdd hdd_make(double x) { hdd r = { x, 0.0 }; return r; }

static hdd hdd_add(hdd a, hdd b) {
    double s = a.hi + b.hi, v = s - a.hi;
    double e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    hdd r = { s + e, e - ((s + e) - s) };
    return r;
}

static hdd hdd_neg(hdd a) { hdd r = { -a.hi, -a.lo }; return r; }
static hdd hdd_sub(hdd a, hdd b) { return hdd_add(a, hdd_neg(b)); }

static hdd hdd_mul(hdd a, hdd b) {
    double p = a.hi * b.hi;
    double e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    hdd r = { p + e, e - ((p + e) - p) };
    return r;
}

static hdd hdd_div(hdd a, hdd b) {
    double q = a.hi / b.hi;
    hdd x = hdd_make(q);
    for (int it = 0; it < 2; it++) {
        hdd rem = hdd_sub(a, hdd_mul(b, x));
        x = hdd_add(x, hdd_make(rem.hi / b.hi));
    }
    return x;
}

/* Dense LU with partial pivoting, double-double throughout (factors a COPY,
 * unlike dmat_lu_factor/dmat_lu_solve which this does not share -- the HSD
 * Newton system is rebuilt and refactored every iteration just like the
 * primary path's Nsys, so there is no factorisation to keep between calls).
 * Returns -1 on an exactly zero pivot or allocation failure, 0 on success,
 * solution written back into v. */
static int hsd_ddsolve(const double *M, double *v, int n) {
    hdd *a = (hdd *)malloc((size_t)n * (size_t)n * sizeof(hdd));
    hdd *b = (hdd *)malloc((size_t)n * sizeof(hdd));
    if (!a || !b) { free(a); free(b); return -1; }
    for (int i = 0; i < n * n; i++) a[i] = hdd_make(M[i]);
    for (int i = 0; i < n; i++) b[i] = hdd_make(v[i]);
    for (int k = 0; k < n; k++) {
        int p = k;
        double mx = fabs(a[(size_t)k * n + k].hi);
        for (int i = k + 1; i < n; i++) {
            double z = fabs(a[(size_t)i * n + k].hi);
            if (z > mx) { mx = z; p = i; }
        }
        if (mx < 1e-300) { free(a); free(b); return -1; }
        if (p != k) {
            for (int j = 0; j < n; j++) {
                hdd t = a[(size_t)k * n + j];
                a[(size_t)k * n + j] = a[(size_t)p * n + j];
                a[(size_t)p * n + j] = t;
            }
            hdd t = b[k]; b[k] = b[p]; b[p] = t;
        }
        for (int i = k + 1; i < n; i++) {
            hdd f = hdd_div(a[(size_t)i * n + k], a[(size_t)k * n + k]);
            for (int j = k + 1; j < n; j++)
                a[(size_t)i * n + j] = hdd_sub(a[(size_t)i * n + j], hdd_mul(f, a[(size_t)k * n + j]));
            b[i] = hdd_sub(b[i], hdd_mul(f, b[k]));
            a[(size_t)i * n + k] = hdd_make(0.0);
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        hdd t = b[i];
        for (int j = i + 1; j < n; j++) t = hdd_sub(t, hdd_mul(a[(size_t)i * n + j], b[j]));
        b[i] = hdd_div(t, a[(size_t)i * n + i]);
    }
    for (int i = 0; i < n; i++) v[i] = b[i].hi + b[i].lo;
    free(a); free(b);
    return 0;
}

/* Residuals and objective of the homogeneous embedding at (x,s,X,S,y,tau,
 * kappa,theta).  bb/bc/BC are the fixed offset vectors (b - A x0 etc., built
 * once from the artificial starting point in hsd_psd) and bz is the
 * embedding's own scalar offset; rg is the duality-gap residual (pobj - dobj
 * + kappa + bz*theta) and rn the normalizing-row residual -- these two close
 * the embedding the way rp/rd close the ordinary primal-dual system. */
typedef struct { double pf, df, mu, rg, rn, pobj, dobj; } hsd_stat;

static hsd_stat hsd_measure(int m, int n, int nb, const int *dims, size_t d2,
        const double *A, const double *b, const double *c,
        const double *const *C, const double *const *AB,
        const double *bb, const double *bc, const double *BC, double bz,
        const double *x, const double *s, const double *const *X, const double *const *S,
        const double *y, double tau, double kap, double theta,
        double *rp, double *rd, double *rB, int nu) {
    hsd_stat R = { 0, 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < m; k++) {
        double v = -b[k] * tau + bb[k] * theta;
        for (int i = 0; i < n; i++) v += A[(size_t)k * n + i] * x[i];
        for (int j = 0; j < nb; j++) v += trAB(dims[j], AB[(size_t)k * nb + j], X[j]);
        rp[k] = v; if (fabs(v) > R.pf) R.pf = fabs(v);
        R.dobj += b[k] * y[k];
    }
    for (int i = 0; i < n; i++) {
        double v = s[i] - c[i] * tau + bc[i] * theta;
        for (int k = 0; k < m; k++) v += A[(size_t)k * n + i] * y[k];
        rd[i] = v; if (fabs(v) > R.df) R.df = fabs(v);
        R.mu += x[i] * s[i]; R.pobj += c[i] * x[i];
    }
    for (int j = 0; j < nb; j++) {
        int d = dims[j];
        double *r = rB + (size_t)j * d2;
        for (int a = 0; a < d * d; a++) r[a] = S[j][a] - C[j][a] * tau + BC[(size_t)j * d2 + a] * theta;
        for (int k = 0; k < m; k++) for (int a = 0; a < d * d; a++) r[a] += AB[(size_t)k * nb + j][a] * y[k];
        for (int a = 0; a < d * d; a++) if (fabs(r[a]) > R.df) R.df = fabs(r[a]);
        R.mu += trAB(d, X[j], S[j]); R.pobj += trAB(d, C[j], X[j]);
    }
    R.mu = (R.mu + tau * kap) / nu;
    R.rg = R.pobj - R.dobj + kap + bz * theta;
    R.rn = -bz * tau - nu;
    for (int k = 0; k < m; k++) R.rn += bb[k] * y[k];
    for (int i = 0; i < n; i++) R.rn -= bc[i] * x[i];
    for (int j = 0; j < nb; j++) R.rn -= trAB(dims[j], BC + (size_t)j * d2, X[j]);
    if (fabs(R.rg) > R.pf) R.pf = fabs(R.rg);
    if (fabs(R.rn) > R.pf) R.pf = fabs(R.rn);
    return R;
}

/* PSD+scalar homogeneous self-dual IPM.  N = m + 2 (multiplier block plus
 * tau, theta columns); kappa is eliminated in closed form from the tau row,
 * same as the primary path eliminates ds/dz from dy.  Mehrotra predictor
 * (pass 0, sigma=0) then corrector (pass 1) exactly like sdp_ipm's main loop;
 * the corrector's own direction gets up to 5 rounds of double-double
 * iterative refinement (mirroring the primary path's refine_on loop, but in
 * dd arithmetic throughout rather than compensated residual + double
 * solve). Accept/backtrack on the same pf+df+mu merit as sdp_ipm. */
static int hsd_psd(int m, int n, const double *A, const double *b, const double *c,
        int nb, const int *dims, const double *const *C, const double *const *AB,
        int max_iter, double rtol_pri, double rtol_dual, double rtol_gap,
        double *x, double *const *X, double *y, double *const *S) {
    int dmax = 1, nu = n + 1;
    for (int j = 0; j < nb; j++) { if (dims[j] > dmax) dmax = dims[j]; nu += dims[j]; }
    size_t d2 = (size_t)dmax * dmax, bd2 = (size_t)nb * d2;
    int N = m + 2;
    double *xx = (double *)calloc((size_t)n, sizeof(double)), *ss = (double *)calloc((size_t)n, sizeof(double));
    double *bb = (double *)calloc((size_t)m, sizeof(double)), *bc = (double *)calloc((size_t)n, sizeof(double));
    double *BC = (double *)calloc(bd2, sizeof(double)), *rp = (double *)calloc((size_t)m, sizeof(double));
    double *rd = (double *)calloc((size_t)n, sizeof(double)), *rB = (double *)calloc(bd2, sizeof(double));
    double *W = (double *)calloc(bd2, sizeof(double)), *Wi = (double *)calloc(bd2, sizeof(double)), *Xi = (double *)calloc(bd2, sizeof(double));
    double *WA = (double *)calloc((size_t)m * bd2, sizeof(double));
    double *WC = (double *)calloc(bd2, sizeof(double)), *WB = (double *)calloc(bd2, sizeof(double));
    double *base = (double *)calloc(bd2, sizeof(double)), *dx = (double *)calloc((size_t)n, sizeof(double));
    double *ds = (double *)calloc((size_t)n, sizeof(double)), *dX = (double *)calloc(bd2, sizeof(double));
    double *dS = (double *)calloc(bd2, sizeof(double)), *dy = (double *)calloc((size_t)m, sizeof(double));
    double *K = (double *)calloc((size_t)N * N, sizeof(double)), *rhs = (double *)calloc((size_t)N, sizeof(double));
    double *t1 = (double *)calloc(d2, sizeof(double)), *t2 = (double *)calloc(d2, sizeof(double));
    double *t3 = (double *)calloc(d2, sizeof(double)), *g = (double *)calloc(d2, sizeof(double));
    double *row = (double *)calloc((size_t)N, sizeof(double)), *col = (double *)calloc((size_t)N, sizeof(double));
    double *snap = (double *)calloc((size_t)(2 * n) + 2 * bd2 + (size_t)m + 3, sizeof(double));
    if (!xx || !ss || !bb || !bc || !BC || !rp || !rd || !rB || !W || !Wi || !Xi || !WA || !WC || !WB ||
        !base || !dx || !ds || !dX || !dS || !dy || !K || !rhs || !t1 || !t2 || !t3 || !g || !row || !col || !snap)
        goto fail;
    for (int i = 0; i < n; i++) { xx[i] = 10; ss[i] = 1; bc[i] = c[i] - 1; }
    for (int j = 0; j < nb; j++) {
        int d = dims[j];
        for (int a = 0; a < d * d; a++) X[j][a] = S[j][a] = 0;
        for (int a = 0; a < d; a++) X[j][a * d + a] = S[j][a * d + a] = 1;
        for (int a = 0; a < d * d; a++) BC[(size_t)j * d2 + a] = C[j][a] - S[j][a];
    }
    double tau = 1, kap = 1, theta = 1, p0 = 0, bz;
    double best_viol = 1e300;
    for (int i = 0; i < n; i++) p0 += c[i] * xx[i];
    for (int j = 0; j < nb; j++) p0 += trAB(dims[j], C[j], X[j]);
    bz = -p0 - 1;
    for (int k = 0; k < m; k++) {
        bb[k] = b[k];
        for (int i = 0; i < n; i++) bb[k] -= A[(size_t)k * n + i] * xx[i];
        for (int j = 0; j < nb; j++) bb[k] -= trAB(dims[j], AB[(size_t)k * nb + j], X[j]);
        y[k] = 0;
    }
    double bn = 0, cn = 0;
    for (int k = 0; k < m; k++) if (fabs(b[k]) > bn) bn = fabs(b[k]);
    for (int i = 0; i < n; i++) if (fabs(c[i]) > cn) cn = fabs(c[i]);
    int status = 1;
    double mwin[MWIN]; int nwin = 0, iwin = 0;
    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(34);
        hsd_stat R = hsd_measure(m, n, nb, dims, d2, A, b, c, C, AB, bb, bc, BC, bz,
                                  xx, ss, (const double *const *)X, (const double *const *)S,
                                  y, tau, kap, theta, rp, rd, rB, nu);
        double pri = R.pf / (tau * (1 + bn)), dual = R.df / (tau * (1 + cn));
        double gap = fabs(R.pobj - R.dobj) / (tau + fabs(R.pobj) + fabs(R.dobj));
        double viol = pri / rtol_pri;
        if (dual / rtol_dual > viol) viol = dual / rtol_dual;
        if (gap / rtol_gap > viol) viol = gap / rtol_gap;
        /* best-iterate selector (same role as the primary path's snapshot):
         * the non-monotone window can let the residuals oscillate, so the
         * point that must be published is the one with the smallest violation,
         * not the last one. */
        if (isfinite(viol) && viol < best_viol) {
            best_viol = viol;
            double *sp = snap;
            memcpy(sp, xx, (size_t)n * sizeof(double)); sp += n;
            memcpy(sp, ss, (size_t)n * sizeof(double)); sp += n;
            for (int j = 0; j < nb; j++) { int d = dims[j]; memcpy(sp + (size_t)j * d2, X[j], (size_t)d * d * sizeof(double)); }
            sp += bd2;
            for (int j = 0; j < nb; j++) { int d = dims[j]; memcpy(sp + (size_t)j * d2, S[j], (size_t)d * d * sizeof(double)); }
            sp += bd2;
            memcpy(sp, y, (size_t)m * sizeof(double)); sp += m;
            sp[0] = tau; sp[1] = kap; sp[2] = theta;
        }
        if (getenv("GMB_DBG") && (it % 10 == 0 || it == max_iter - 1))
            fprintf(stderr, "[HSD] it=%d tau=%.7g kap=%.7g th=%.7g mu=%.3g pri=%.3g dual=%.3g gap=%.3g pf=%.3g df=%.3g viol=%.3g\n",
                    it, tau, kap, theta, R.mu, pri, dual, gap, R.pf, R.df, viol);
        if (isfinite(pri) && isfinite(dual) && isfinite(gap) && tau > 1e-12 &&
            pri <= rtol_pri && dual <= rtol_dual && gap <= rtol_gap) {
            status = 0; break;
        }
        if (!(R.mu > 0 && isfinite(R.mu) && tau > 1e-16 && kap > 0 && theta > 0)) break;
        for (int j = 0; j < nb; j++) {
            int d = dims[j]; double *wj = W + (size_t)j * d2;
            sym_fun(d, X[j], 0, t1); mmul(d, t1, S[j], t2); mmul(d, t2, t1, t3);
            sym_fun(d, t3, 1, t2); mmul(d, t1, t2, t3); mmul(d, t3, t1, wj);
            sym_fun(d, wj, 2, Wi + (size_t)j * d2); sym_fun(d, X[j], 2, Xi + (size_t)j * d2);
            mmul(d, wj, C[j], t1); mmul(d, t1, wj, WC + (size_t)j * d2);
            mmul(d, wj, BC + (size_t)j * d2, t1); mmul(d, t1, wj, WB + (size_t)j * d2);
            for (int k = 0; k < m; k++) { mmul(d, wj, AB[(size_t)k * nb + j], t1); mmul(d, t1, wj, WA + ((size_t)k * nb + j) * d2); }
        }
        double sigma = 0;
        for (int pass = 0; pass < 2; pass++) {
            memset(K, 0, (size_t)N * N * sizeof(double));
            memset(rhs, 0, (size_t)N * sizeof(double));
            for (int j = 0; j < nb; j++) {
                int d = dims[j]; int off = (int)((size_t)j * d2); double *q = base + off;
                for (int a = 0; a < d * d; a++) q[a] = rB[off + a] - S[j][a] + sigma * R.mu * Xi[off + a];
                mmul(d, W + off, q, t1); mmul(d, t1, W + off, q);
            }
            for (int k = 0; k < m; k++) {
                double v = -rp[k], ht = -b[k], hq = bb[k];
                for (int i = 0; i < n; i++) {
                    double t = xx[i] / ss[i], gi = rd[i] - ss[i] + sigma * R.mu / xx[i];
                    v -= A[(size_t)k * n + i] * t * gi;
                    ht -= A[(size_t)k * n + i] * t * c[i]; hq += A[(size_t)k * n + i] * t * bc[i];
                }
                for (int j = 0; j < nb; j++) {
                    int d = dims[j]; size_t off = (size_t)j * d2;
                    v -= trAB(d, AB[(size_t)k * nb + j], base + off);
                    ht -= trAB(d, AB[(size_t)k * nb + j], WC + off);
                    hq += trAB(d, AB[(size_t)k * nb + j], WB + off);
                }
                rhs[k] = v; K[(size_t)k * N + m] = ht; K[(size_t)k * N + m + 1] = hq;
                for (int l = 0; l < m; l++) {
                    double z = 0;
                    for (int i = 0; i < n; i++) z += A[(size_t)k * n + i] * (xx[i] / ss[i]) * A[(size_t)l * n + i];
                    for (int j = 0; j < nb; j++) z += trAB(dims[j], AB[(size_t)k * nb + j], WA + ((size_t)l * nb + j) * d2);
                    K[(size_t)k * N + l] = z;
                }
            }
            rhs[m] = -R.rg - (sigma * R.mu - tau * kap) / tau;
            rhs[m + 1] = -R.rn;
            K[(size_t)m * N + m] = -kap / tau; K[(size_t)m * N + m + 1] = bz;
            K[(size_t)(m + 1) * N + m] = -bz;
            for (int k = 0; k < m; k++) { K[(size_t)m * N + k] = -b[k]; K[(size_t)(m + 1) * N + k] = bb[k]; }
            for (int i = 0; i < n; i++) {
                double t = xx[i] / ss[i], gi = rd[i] - ss[i] + sigma * R.mu / xx[i];
                rhs[m] -= c[i] * t * gi; rhs[m + 1] += bc[i] * t * gi;
                K[(size_t)m * N + m] -= c[i] * t * c[i]; K[(size_t)m * N + m + 1] += c[i] * t * bc[i];
                K[(size_t)(m + 1) * N + m] += bc[i] * t * c[i]; K[(size_t)(m + 1) * N + m + 1] -= bc[i] * t * bc[i];
                for (int k = 0; k < m; k++) {
                    K[(size_t)m * N + k] += c[i] * t * A[(size_t)k * n + i];
                    K[(size_t)(m + 1) * N + k] -= bc[i] * t * A[(size_t)k * n + i];
                }
            }
            for (int j = 0; j < nb; j++) {
                int d = dims[j]; size_t off = (size_t)j * d2;
                rhs[m] -= trAB(d, C[j], base + off); rhs[m + 1] += trAB(d, BC + off, base + off);
                K[(size_t)m * N + m] -= trAB(d, C[j], WC + off); K[(size_t)m * N + m + 1] += trAB(d, C[j], WB + off);
                K[(size_t)(m + 1) * N + m] += trAB(d, BC + off, WC + off); K[(size_t)(m + 1) * N + m + 1] -= trAB(d, BC + off, WB + off);
                for (int k = 0; k < m; k++) {
                    K[(size_t)m * N + k] += trAB(d, C[j], WA + ((size_t)k * nb + j) * d2);
                    K[(size_t)(m + 1) * N + k] -= trAB(d, BC + off, WA + ((size_t)k * nb + j) * d2);
                }
            }
            for (int k = 0; k < m; k++) {
                double mx = 0; for (int l = 0; l < N; l++) if (fabs(K[(size_t)k * N + l]) > mx) mx = fabs(K[(size_t)k * N + l]);
                row[k] = mx > 0 ? 1 / mx : 1;
            }
            for (int k = m; k < N; k++) {
                double mx = 0; for (int l = 0; l < N; l++) if (fabs(K[(size_t)k * N + l]) > mx) mx = fabs(K[(size_t)k * N + l]);
                row[k] = mx > 0 ? 1 / mx : 1;
            }
            for (int k = 0; k < N; k++) { rhs[k] *= row[k]; for (int l = 0; l < N; l++) K[(size_t)k * N + l] *= row[k]; }
            for (int l = 0; l < N; l++) {
                double mx = 0; for (int k = 0; k < N; k++) if (fabs(K[(size_t)k * N + l]) > mx) mx = fabs(K[(size_t)k * N + l]);
                col[l] = mx > 0 ? 1 / mx : 1;
                for (int k = 0; k < N; k++) K[(size_t)k * N + l] *= col[l];
            }
            if (hsd_ddsolve(K, rhs, N)) goto finish;
            for (int k = 0; k < N; k++) rhs[k] *= col[k];
            for (int k = 0; k < m; k++) dy[k] = rhs[k];
            double dtau = rhs[m], dtheta = rhs[m + 1], dkap = (sigma * R.mu - tau * kap - kap * dtau) / tau;
            for (int ir = 0; ir < 5; ir++) {
                for (int i = 0; i < n; i++) {
                    double v = rd[i] - ss[i] + sigma * R.mu / xx[i] - c[i] * dtau + bc[i] * dtheta;
                    for (int k = 0; k < m; k++) v += A[(size_t)k * n + i] * dy[k];
                    dx[i] = (xx[i] / ss[i]) * v;
                    ds[i] = -(ss[i] / xx[i]) * dx[i] - ss[i] + sigma * R.mu / xx[i];
                }
                for (int j = 0; j < nb; j++) {
                    int d = dims[j]; size_t off = (size_t)j * d2;
                    for (int a = 0; a < d * d; a++) g[a] = rB[off + a] - S[j][a] + sigma * R.mu * Xi[off + a] - C[j][a] * dtau + BC[off + a] * dtheta;
                    for (int k = 0; k < m; k++) for (int a = 0; a < d * d; a++) g[a] += AB[(size_t)k * nb + j][a] * dy[k];
                    mmul(d, W + off, g, t1); mmul(d, t1, W + off, dX + off);
                    /* dS = -Wi dX Wi - S + sigma*mu*Xi algebraically simplifies to
                     * -g - S + sigma*mu*Xi when Wi is EXACTLY W's inverse (dX = W g
                     * W, so Wi dX Wi = Wi W g W Wi = g).  Computing it via the
                     * matrix round-trip instead loses digits proportional to
                     * cond(W): on M1 (degree 8) W's eigenvalues span emin=3.5e-05
                     * to emax=6.7e+06 (cond ~1.9e11), and going through Wi*dX*Wi
                     * amplifies g by cond(W) and then divides it back out, losing
                     * ~cond(W)*eps of the result -- measured as the dual Newton
                     * equation (which the scalar block satisfies to 1e-16) failing
                     * by ~1e-3 to 1e-2 on the bar block, which is exactly why any
                     * accepted step blew up the dual residual and forced alpha
                     * toward machine epsilon.  The direct form needs no inverse at
                     * all, so it has no such cancellation to lose digits to. */
                    for (int a = 0; a < d * d; a++) dS[off + a] = -g[a] - S[j][a] + sigma * R.mu * Xi[off + a];
                }
                double maxerr = 0;
                for (int k = 0; k < m; k++) {
                    double v = rp[k] - b[k] * dtau + bb[k] * dtheta;
                    for (int i = 0; i < n; i++) v += A[(size_t)k * n + i] * dx[i];
                    for (int j = 0; j < nb; j++) v += trAB(dims[j], AB[(size_t)k * nb + j], dX + (size_t)j * d2);
                    rhs[k] = -v * row[k]; if (fabs(v) > maxerr) maxerr = fabs(v);
                }
                double eg = R.rg + dkap + bz * dtheta;
                for (int k = 0; k < m; k++) eg -= b[k] * dy[k];
                for (int i = 0; i < n; i++) eg += c[i] * dx[i];
                for (int j = 0; j < nb; j++) eg += trAB(dims[j], C[j], dX + (size_t)j * d2);
                rhs[m] = -eg * row[m]; if (fabs(eg) > maxerr) maxerr = fabs(eg);
                double en = R.rn - bz * dtau;
                for (int k = 0; k < m; k++) en += bb[k] * dy[k];
                for (int i = 0; i < n; i++) en -= bc[i] * dx[i];
                for (int j = 0; j < nb; j++) en -= trAB(dims[j], BC + (size_t)j * d2, dX + (size_t)j * d2);
                rhs[m + 1] = -en * row[m + 1]; if (fabs(en) > maxerr) maxerr = fabs(en);
                if (maxerr < 1e-13 || ir == 4) break;
                if (hsd_ddsolve(K, rhs, N)) goto finish;
                for (int k = 0; k < m; k++) dy[k] += rhs[k] * col[k];
                dtau += rhs[m] * col[m]; dtheta += rhs[m + 1] * col[m + 1];
                dkap = (sigma * R.mu - tau * kap - kap * dtau) / tau;
            }
            double alpha = 1;
            for (int i = 0; i < n; i++) {
                if (dx[i] < 0 && -xx[i] / dx[i] < alpha) alpha = -xx[i] / dx[i];
                if (ds[i] < 0 && -ss[i] / ds[i] < alpha) alpha = -ss[i] / ds[i];
            }
            if (dtau < 0 && -tau / dtau < alpha) alpha = -tau / dtau;
            if (dkap < 0 && -kap / dkap < alpha) alpha = -kap / dkap;
            if (dtheta < 0 && -theta / dtheta < alpha) alpha = -theta / dtheta;
            for (int j = 0; j < nb; j++) {
                int d = dims[j]; size_t off = (size_t)j * d2;
                sym_fun(d, X[j], 1, t1); mmul(d, t1, dX + off, t2); mmul(d, t2, t1, t3);
                double e = min_eig(d, t3); if (e < 0 && -1 / e < alpha) alpha = -1 / e;
                sym_fun(d, S[j], 1, t1); mmul(d, t1, dS + off, t2); mmul(d, t2, t1, t3);
                e = min_eig(d, t3); if (e < 0 && -1 / e < alpha) alpha = -1 / e;
            }
            if (alpha > 1) alpha = 1;
            alpha *= 0.99;
            if (pass == 0) {
                double mup = (tau + alpha * dtau) * (kap + alpha * dkap);
                for (int i = 0; i < n; i++) mup += (xx[i] + alpha * dx[i]) * (ss[i] + alpha * ds[i]);
                for (int j = 0; j < nb; j++) {
                    int d = dims[j]; size_t off = (size_t)j * d2;
                    for (int a = 0; a < d * d; a++) { t1[a] = X[j][a] + alpha * dX[off + a]; t2[a] = S[j][a] + alpha * dS[off + a]; }
                    mup += trAB(d, t1, t2);
                }
                mup /= nu;
                sigma = pow(fmax(0, mup / R.mu), 3);
                if (sigma < 0.01) sigma = 0.01;
                if (sigma > 1) sigma = 1;
            } else {
                double old = R.pf + R.df + R.mu;
                double ref = old;
                for (int q = 0; q < MWIN; q++) if (q < nwin && mwin[q] > ref) ref = mwin[q];
                ref += 1e-10 * ref + 1e-16;
                int accepted = 0;
                for (int bt = 0; bt < 60; bt++, alpha *= 0.5) {
                    if (alpha < 1e-14) break;
                    int ok = 1;
                    for (int i = 0; i < n; i++) if (xx[i] + alpha * dx[i] <= 0 || ss[i] + alpha * ds[i] <= 0) { ok = 0; break; }
                    if (tau + alpha * dtau <= 0 || kap + alpha * dkap <= 0 || theta + alpha * dtheta <= 0) ok = 0;
                    for (int j = 0; j < nb && ok; j++) {
                        int d = dims[j]; size_t off = (size_t)j * d2;
                        for (int a = 0; a < d * d; a++) t1[a] = X[j][a] + alpha * dX[off + a];
                        if (min_eig(d, t1) <= 0) ok = 0;
                        for (int a = 0; a < d * d; a++) t1[a] = S[j][a] + alpha * dS[off + a];
                        if (min_eig(d, t1) <= 0) ok = 0;
                    }
                    if (!ok) continue;
                    for (int i = 0; i < n; i++) { xx[i] += alpha * dx[i]; ss[i] += alpha * ds[i]; }
                    for (int k = 0; k < m; k++) y[k] += alpha * dy[k];
                    for (int j = 0; j < nb; j++) {
                        int d = dims[j]; size_t off = (size_t)j * d2;
                        for (int a = 0; a < d * d; a++) { X[j][a] += alpha * dX[off + a]; S[j][a] += alpha * dS[off + a]; }
                    }
                    tau += alpha * dtau; kap += alpha * dkap; theta += alpha * dtheta;
                    hsd_stat T = hsd_measure(m, n, nb, dims, d2, A, b, c, C, AB, bb, bc, BC, bz,
                                              xx, ss, (const double *const *)X, (const double *const *)S,
                                              y, tau, kap, theta, rp, rd, rB, nu);
                    if (isfinite(T.mu) && T.pf + T.df + T.mu <= ref) { accepted = 1; break; }
                    for (int i = 0; i < n; i++) { xx[i] -= alpha * dx[i]; ss[i] -= alpha * ds[i]; }
                    for (int k = 0; k < m; k++) y[k] -= alpha * dy[k];
                    for (int j = 0; j < nb; j++) {
                        int d = dims[j]; size_t off = (size_t)j * d2;
                        for (int a = 0; a < d * d; a++) { X[j][a] -= alpha * dX[off + a]; S[j][a] -= alpha * dS[off + a]; }
                    }
                    tau -= alpha * dtau; kap -= alpha * dkap; theta -= alpha * dtheta;
                }
                if (getenv("GMB_DBG"))
                    fprintf(stderr, "[HSD] step it=%d alpha=%.4g sigma=%.4g dtau=%.4g dkap=%.4g dtheta=%.4g accepted=%d\n",
                            it, alpha, sigma, dtau, dkap, dtheta, accepted);
                if (accepted) { mwin[iwin] = old; iwin = (iwin + 1) % MWIN; if (nwin < MWIN) nwin++; }
                else goto finish;
            }
        }
    }
finish:
    if (getenv("GMB_DBG")) fprintf(stderr, "[HSD] finish best_viol=%.4g status=%d\n", best_viol, status);
    if (status != 0 && best_viol <= 1.0) {
        double *sp = snap;
        memcpy(xx, sp, (size_t)n * sizeof(double)); sp += n;
        memcpy(ss, sp, (size_t)n * sizeof(double)); sp += n;
        for (int j = 0; j < nb; j++) { int d = dims[j]; memcpy(X[j], sp + (size_t)j * d2, (size_t)d * d * sizeof(double)); }
        sp += bd2;
        for (int j = 0; j < nb; j++) { int d = dims[j]; memcpy(S[j], sp + (size_t)j * d2, (size_t)d * d * sizeof(double)); }
        sp += bd2;
        memcpy(y, sp, (size_t)m * sizeof(double)); sp += m;
        tau = sp[0]; kap = sp[1]; theta = sp[2];
        status = 0;
    }
    if (status == 0) {
        double inv = 1 / tau;
        for (int i = 0; i < n; i++) x[i] = xx[i] * inv;
        for (int k = 0; k < m; k++) y[k] *= inv;
        for (int j = 0; j < nb; j++) { int d = dims[j]; for (int a = 0; a < d * d; a++) { X[j][a] *= inv; S[j][a] *= inv; } }
    }
    free(xx); free(ss); free(bb); free(bc); free(BC); free(rp); free(rd); free(rB);
    free(W); free(Wi); free(Xi); free(WA); free(WC); free(WB); free(base);
    free(dx); free(ds); free(dX); free(dS); free(dy); free(K); free(rhs);
    free(t1); free(t2); free(t3); free(g); free(row); free(col); free(snap);
    return status;
fail:
    free(xx); free(ss); free(bb); free(bc); free(BC); free(rp); free(rd); free(rB);
    free(W); free(Wi); free(Xi); free(WA); free(WC); free(WB); free(base);
    free(dx); free(ds); free(dX); free(dS); free(dy); free(K); free(rhs);
    free(t1); free(t2); free(t3); free(g); free(row); free(col); free(snap);
    return 2;
}


/**
 * Primal-dual interior point method for conic problems (SDP/SOC/exp-power/LP/QP).
 *
 * @param m       [in]  Number of equality constraints.
 * @param n       [in]  Number of scalar variables.
 * @param A       [in]  Equality constraint matrix (m x n, row-major).
 * @param b       [in]  RHS for equalities (size m).
 * @param c       [in]  Linear objective coefficients (size n).
 * @param nb      [in]  Number of PSD blocks.
 * @param dims    [in]  Dimensions of PSD blocks (size nb).
 * @param Cbar    [in]  Cost matrices for PSD blocks (array of nb, each dim_j*dim_j row-major).
 * @param Abar    [in]  Constraint matrices for PSD blocks (array of m*nb, each dim_j*dim_j).
 * @param nsoc    [in]  Number of SOC blocks.
 * @param socdims [in]  Dimensions of SOC blocks (size nsoc).
 * @param Csoc    [in]  Cost vectors for SOC blocks (array of nsoc, each k_i).
 * @param Asoc    [in]  Constraint matrices for SOC blocks (array of m*nsoc, each k_i).
 * @param nep     [in]  Number of exp/power blocks.
 * @param ekind   [in]  Cone kinds for exp/power blocks (size nep).
 * @param ealpha  [in]  Power parameters for PPOW/RPOW blocks (size nep).
 * @param Cexp    [in]  Cost vectors for exp/power blocks (array of nep, each 3).
 * @param Aexp    [in]  Constraint matrices for exp/power blocks (array of m*nep, each 3).
 * @param max_iter [in] Maximum iterations.
 * @param tol      [in]  Absolute Newton tolerance (unused; rtol_* used instead).
 * @param rtol_pri [in]  Relative primal feasibility tolerance.
 * @param rtol_dual [in] Relative dual feasibility tolerance.
 * @param rtol_gap [in]  Relative duality gap tolerance.
 * @param near_rel [in]  Near-optimal factor (multiplies rtol_* for verdict).
 * @param x        [out] Primal scalar solution (size n). May be NULL.
 * @param Xbar     [out] Primal PSD solutions (array of nb, each dim_j*dim_j). May be NULL.
 * @param y        [out] Dual for equalities (size m). May be NULL.
 * @param Sbar     [out] Dual PSD solutions (array of nb, each dim_j*dim_j). May be NULL.
 * @param Zsoc     [out] Primal SOC solutions (array of nsoc, each k_i). May be NULL.
 * @param Ssoc     [out] Dual SOC solutions (array of nsoc, each k_i). May be NULL.
 * @param Zexp     [out] Primal exp/power solutions (array of nep, each 3). May be NULL.
 * @param Sexp     [out] Dual exp/power solutions (array of nep, each 3). May be NULL.
 * @param xwarm    [in]  Optional warm start x (size n). NaN entries -> default.
 * @param ywarm    [in]  Optional warm start y (size m). NaN entries -> default.
 * @param fb_ok    [out] Optional flag set to 1 if a near-optimal fallback candidate was saved.
 *
 * @return 0 = optimal, 1 = max-iter, 2 = memory, 3 = singular.
 *
 * @note Solves unified conic problem:
 *       min  c'x + sum_j <C_j, X_j> + sum_i <c_i, z_i>
 *       s.t. A x + sum_j <A_kj, X_j> + sum_i <a_ki, z_i> = b_k
 *            x >= 0,  X_j >= 0 (PSD),  z_i in SOC/exp/power.
 *
 *       Unified conic path: R_+, SOC and PSD blocks in the same IPM.
 *       - R_+ and PSD use Nesterov-Todd scaling + normal equations on m rows.
 *       - SOC uses arrow-matrix augmented KKT (NT scaling diverges at boundary).
 *       - Exp/power blocks have Hessian-NT rows in the same KKT (mu/sigma-free).
 *
 *       Mehrotra predictor-corrector with fraction-to-boundary step lengths.
 *       Objective cuts (PRIMAL_DPAR_LOWER_OBJ_CUT/UPPER_OBJ_CUT) supported.
 *       Wall-clock deadline from PRIMAL_DPAR_OPTIMIZER_MAX_TIME.
 *
 *       GMB_SDP_HSD environment variable enables experimental homogeneous
 *       self-dual embedding for PSD-only models.
 *
 *       Warm start: xwarm/ywarm (NaN entries = default). xwarm projected to
 *       interior; ywarm mapped via -s*ywarm.
 *
 *       Fallback candidate (fb_ok): saves a near-optimal point that passes
 *       the near-optimal factor but not the strict tolerances. Returned when
 *       cuts also fail.
 *
 * @example
 * // See sdp.h for parameter format
 *
 * The body is sdp_ipm_run; sdp_ipm (after it) runs it once as it always
 * ran, and a second time in the exp/power SECANT mode only when that first
 * run fails -- see the comment there.
 */
static int sdp_ipm_run(int secant, int m, int n, const double *A, const double *b, const double *c,
            int nb, const int *dims,
            const double *const *Cbar, const double *const *Abar,
            int nsoc, const int *socdims,
            const double *const *Csoc, const double *const *Asoc,
            int nep, const int *ekind, const double *ealpha,
            const double *const *Cexp, const double *const *Aexp,
            int max_iter, double tol, double rtol_pri, double rtol_dual, double rtol_gap,
            double near_rel,
            double *x, double *const *Xbar, double *y, double *const *Sbar,
            double *const *Zsoc, double *const *Ssoc,
            double *const *Zexp, double *const *Sexp, const double *xwarm,
            const double *ywarm, int *fb_ok)
{
    if (fb_ok) *fb_ok = 0;
    /* Opt-in homogeneous self-dual embedding, PSD+scalar only (see hsd_psd's
     * comment above): default off (unset), so this changes nothing unless
     * the caller explicitly asks for the second, unproven code path. */
    if (getenv("GMB_SDP_HSD") && nb > 0 && nsoc == 0 && nep == 0)
        return hsd_psd(m, n, A, b, c, nb, dims, Cbar, Abar,
                        max_iter, rtol_pri, rtol_dual, rtol_gap, x, Xbar, y, Sbar);
    if (m < 0 || n < 0 || nb < 0 || nsoc < 0 || nep < 0) return 1;
    int dmax = 1; for (int j = 0; j < nb; j++) if (dims[j] > dmax) dmax = dims[j];
    int kmax = 1; for (int i = 0; i < nsoc; i++) if (socdims[i] > kmax) kmax = socdims[i];
    size_t dmax2 = (size_t)dmax * dmax;
    size_t tsz = dmax2 > (size_t)kmax ? dmax2 : (size_t)kmax;
    int K = 0; for (int i = 0; i < nsoc; i++) K += socdims[i];
    int Ke = 3 * nep;
    int *soff = (int *)malloc((size_t)(nsoc > 0 ? nsoc + 1 : 1) * sizeof(int));
    if (!soff) return 2;
    soff[0] = 0; for (int i = 0; i < nsoc; i++) soff[i + 1] = soff[i] + socdims[i];

    size_t nbd2 = (size_t)(nb > 0 ? nb : 1) * dmax2;
    double *Ws = (double *)malloc(nbd2 * sizeof(double));
    double *Wi = (double *)malloc(nbd2 * sizeof(double));
    double *Xi = (double *)malloc(nbd2 * sizeof(double));
    double *t1 = (double *)malloc(tsz * sizeof(double));
    double *t2 = (double *)malloc(tsz * sizeof(double));
    double *t3 = (double *)malloc(tsz * sizeof(double));
    double *Mt = (double *)malloc((size_t)(m > 0 ? m : 1) * (size_t)(nb > 0 ? nb : 1) * dmax2 * sizeof(double));
    double *xs = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *ss = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *zsoc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *ssoc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *Dzsoc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    double *Dssoc = (double *)malloc((size_t)(K > 0 ? K : 1) * sizeof(double));
    /* SOC arrow scratch: A(z), A(s) are kmax^2 (soc_arrow), the rest kmax. */
    double *Az = (double *)malloc((size_t)kmax * (size_t)kmax * sizeof(double));
    double *As = (double *)malloc((size_t)kmax * (size_t)kmax * sizeof(double));
    double *rds = (double *)malloc((size_t)kmax * sizeof(double));
    double *rcs = (double *)malloc((size_t)kmax * sizeof(double));
    double *soc_e = (double *)malloc((size_t)kmax * sizeof(double));
    double *ez = (double *)malloc((size_t)(Ke > 0 ? Ke : 1) * sizeof(double));
    double *es = (double *)malloc((size_t)(Ke > 0 ? Ke : 1) * sizeof(double));
    double *Dez = (double *)malloc((size_t)(Ke > 0 ? Ke : 1) * sizeof(double));
    double *Des = (double *)malloc((size_t)(Ke > 0 ? Ke : 1) * sizeof(double));
    double *rp = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    double *rdx = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *dy = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    double *dx = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *ds = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *Dx = (double *)malloc(nbd2 * sizeof(double));
    double *Ds = (double *)malloc(nbd2 * sizeof(double));
    double *snap = (double *)malloc(
                     ipm_state_size(n, m, nb, dims, nsoc, socdims, Ke) * sizeof(double));
    /* Candidato di fallback (percorso separato, T171/risk_parity): salvato quando
     * il merito assoluto non accetta mai il punto ma la terna RELATIVA sta nel
     * fattore near-optimal.  Non e' have_best: il verdetto nativo resta "non
     * risolto" e i tagli restano la prima scelta. */
    double *fb_snap = (double *)malloc(
                     ipm_state_size(n, m, nb, dims, nsoc, socdims, Ke) * sizeof(double));
    double *trial = (nep > 0 || nb > 0) ? (double *)malloc(
                     ipm_state_size(n, m, nb, dims, nsoc, socdims, Ke) * sizeof(double)) : NULL;
    int have_best = 0, stall = 0, lost = 0, frozen = 0, fb_saved = 0;
    double prev_feas = HUGE_VAL; int use_rescue = 0;
    double gap_best = 0.0, viol_best = 0.0, merit_prev = 0.0, viol_prev = 0.0;
    double mwin[MWIN]; int nwin = 0, iwin = 0;
    int ok = Ws && Wi && Xi && t1 && t2 && t3 && Mt && xs && ss && zsoc && ssoc && Dzsoc && Dssoc &&
             Az && As && rds && rcs && soc_e &&
             ez && es && Dez && Des && rp && rdx && dy && dx && ds && Dx && Ds && fb_snap &&
             snap && ((nep == 0 && nb == 0) || trial);
    int status = 1;
    if (!ok) { status = 2; goto done; }

    /* Starting point: strictly inside every cone at once, because the barrier
     * terms are only defined there.  Ones on the diagonal for the PSD blocks and
     * the axis vertex (1,0,..,0) for SOC are the cheapest such points; the
     * exp/power blocks are the pair z and s = -grad f(z), which makes each
     * block's starting complementarity <z,s> equal to the barrier's degree
     * (nu = 2 for PEXP, 3 for the power cones) instead of some number that
     * would then set mu -- and mu is what every tolerance on the path is
     * relative to. */
    for (int i = 0; i < n; i++) { xs[i] = 1.0; ss[i] = 1.0; }
    /* warm start: un punto primale fornito dall'utente sovrascrive le colonne
     * scalari (NaN = non impostato; 0 lascerebbe il punto fuori dal cono
     * nonnegativo, quindi si accetta solo > 0). L'IPM e' infeasible-start, quindi
     * il punto non deve essere ammissibile -- deve solo stare nei coni. */
    if (xwarm)
        for (int i = 0; i < n; i++)
            if (isfinite(xwarm[i]) && xwarm[i] > 0.0) xs[i] = xwarm[i];
    for (int j = 0; j < nb; j++) { int d = dims[j];
        for (int a = 0; a < d * d; a++) { Xbar[j][a] = 0.0; Sbar[j][a] = 0.0; }
        for (int a = 0; a < d; a++) { Xbar[j][a * d + a] = 1.0; Sbar[j][a * d + a] = 1.0; } }
    for (int i = 0; i < nsoc; i++) { int k = socdims[i];
        for (int a = 0; a < k; a++) { Zsoc[i][a] = 0.0; Ssoc[i][a] = 0.0; }
        Zsoc[i][0] = 1.0; Ssoc[i][0] = 1.0; }
    for (int i = 0; i < nep; i++) {
        double *zi = ez + 3 * i, *si = es + 3 * i, g[3];
        if (ekind[i] == EXPCONE_PEXP) { zi[0] = 2.0; zi[1] = 1.0; zi[2] = 0.0; }
        else { zi[0] = 1.0; zi[1] = 1.0; zi[2] = 0.0; }
        expcone_grad(ekind[i], ealpha[i], zi, g);
        for (int a = 0; a < 3; a++) si[a] = -g[a];
    }
    for (int k = 0; k < m; k++) y[k] = 0.0;
    /* warm start duale: come il primale, un punto fornito dall'utente (NaN =
     * non impostato). L'IPM e' infeasible-start, quindi non deve misurare. */
    if (ywarm)
        for (int k = 0; k < m; k++)
            if (isfinite(ywarm[k])) y[k] = ywarm[k];

    ipmc C;
    C.n = n; C.m = m; C.nb = nb; C.nsoc = nsoc; C.nep = nep; C.Ke = Ke;
    C.A = A; C.b = b; C.c = c; C.dims = dims; C.socdims = socdims;
    C.Cbar = Cbar; C.Abar = Abar; C.Csoc = Csoc; C.Asoc = Asoc; C.Cexp = Cexp; C.Aexp = Aexp;
    C.ekind = ekind; C.ealpha = ealpha;
    C.xs = xs; C.ss = ss; C.y = y; C.rp = rp; C.rdx = rdx; C.scr = t1;
    C.Xbar = Xbar; C.Sbar = Sbar; C.Zsoc = Zsoc; C.Ssoc = Ssoc; C.ez = ez; C.es = es;
    int ntot = n; for (int j = 0; j < nb; j++) ntot += dims[j]; ntot += nsoc; ntot += Ke;
    C.ntot = ntot;
    C.bn = 0.0; C.cn = 0.0;
    for (int k = 0; k < m; k++) { double a = fabs(b[k]); if (a > C.bn) C.bn = a; }
    for (int i = 0; i < n; i++) { double a = fabs(c[i]); if (a > C.cn) C.cn = a; }

    ipmdir D;
    D.dx = dx; D.ds = ds; D.Dx = Dx; D.Ds = Ds; D.Dzsoc = Dzsoc; D.Dssoc = Dssoc;
    D.Dez = Dez; D.Des = Des; D.dy = dy; D.soff = soff; D.dmax2 = dmax2;

    for (int it = 0; it < max_iter; it++) {
        if (primal_cb_iter_on) primal_cb_iter(34);
        ipmres R = ipm_resid(&C);
        double mu = R.mu;
        if (!isfinite(mu)) { status = 1; break; }
        double pfeas = R.pfeas, dfeas = R.dfeas;
        { double feasc = pfeas + dfeas;
          use_rescue = (it > 0 && feasc > 1.05 * prev_feas);
          prev_feas = feasc; }
        ipmqual Q = ipm_quality(&C, &R);
        double viol = Q.pri / rtol_pri;
        if (Q.dual / rtol_dual > viol) viol = Q.dual / rtol_dual;
        if (Q.gap  / rtol_gap  > viol) viol = Q.gap  / rtol_gap;
        /* A collapsed step repeats the same merit forever: measured on T47, the
         * dual ratio test returns ad = 0 once s sits exactly on the boundary of
         * K*, and the loop then spends its whole budget recomputing the same
         * iterate.  Stop on that, accepted point or not. */
        double merit = pfeas + dfeas + mu;
        /* "Collassato" va giudicato su DUE scale: il merito assoluto puo' essere
         * fermo al pavimento di rappresentazione di pfeas (risk_parity:
         * pfeas=6.9e-6 con |b|~1e4) mentre la terna RELATIVA -- che e' cio' che
         * il gate giudica -- continua a scendere.  Congelare su una sola
         * ferma il percorso prima del punto che il fallback vuole. */
        if (it > 0 && !(merit < 0.9995 * merit_prev) && !(viol < 0.9995 * viol_prev)) {
            if (++frozen >= 6) {
                if (getenv("GMB_DBG")) fprintf(stderr,
                    "  [frozen] merit stalled at it=%d merit=%.3g\n", it, merit);
                /* Candidato di fallback: il punto congelato, se la terna RELATIVA
                 * e' nel fattore near-optimal.  Il verdetto nativo resta "non
                 * risolto": i tagli restano la prima scelta, questo punto esce
                 * solo se anch'essi non rispondono (risk_parity: pfeas=6.9e-6
                 * assoluto, floor di |b|~1e4, ma rel_pri=6.9e-10, rel_gap=1.37e-7). */
                if (!have_best && !fb_saved && viol <= near_rel) {
                    fb_saved = 1;
                    ipm_state(fb_snap, 1, IPM_STATE_PASS);
                }
                goto refine;
            }
        } else frozen = 0;
        merit_prev = merit; viol_prev = viol;
        if (getenv("GMB_DBG")) fprintf(stderr,
            "it=%d pfeas=%.3g dfeas=%.3g mu=%.3g | rel_pri=%.3g rel_dual=%.3g rel_gap=%.3g\n",
            it, pfeas, dfeas, mu, Q.pri, Q.dual, Q.gap);
        if (pfeas < tol && dfeas < tol && mu < tol) {
            /* A central-path iterate sits at O(sqrt(mu)) from the optimum when
             * the optimum is on the cone boundary, so the acceptance tolerance
             * alone leaves ~1e-4 of primal error.  Keep reducing mu while it
             * pays, and remember the point with the smallest violation of the
             * triple the route gate will judge it on.  This is the whole
             * stopping rule now: the bar/SOC route used to break here on the
             * absolute mu floor instead, and since that floor is 1e-7 while the
             * declared gap tolerance is 1e-8 RELATIVE, it handed back points at
             * rel_gap 2.1e-8 .. 1.53e-7 (measured on sdo1/sdo2/sdo_lmi/
             * dual_sdo_l1/maxcut_sdp/nearestcorrelation/sparsecholesky).
             * Ranking by gap instead -- which is what this did -- throws points
             * away: on market_impact the polish reached it=27 at
             * rel_pri=1.03e-9, rel_dual=5.3e-15, rel_gap=3.67e-9, all three
             * inside the declared 1e-8, and delivered it=25 at rel_pri=1.82e-8
             * because its gap was 3.33e-9 against 3.59e-9.  One criterion, in
             * the selector and in the gate. */
            int better = !have_best || viol < 0.99 * viol_best ||
                         (viol <= 1.01 * viol_best && Q.gap < 0.5 * gap_best);
            if (better) {
                have_best = 1; gap_best = Q.gap; viol_best = viol; stall = 0;
                ipm_state(snap, 1, IPM_STATE_PASS);
            } else if (++stall >= 4) { status = 0; break; }
            if (mu < tol * tol) { status = 0; break; }   /* floor of double */
            lost = 0;
        }
        /* Once a point is acceptable the polish only pays while it stays in
         * tolerance: a mu-target step whose affine part dominates cannot
         * restore the residual it just lost, so hold mu fixed (pure centering,
         * sigma = 1) and let Newton bring the iterate back on the mu-central
         * path.  Abort if that does not work after a few tries. */
        int polish_center = 0;
        if (nep > 0 && have_best && !(pfeas < tol && dfeas < tol)) {
            polish_center = 1;
            if (++lost >= 8) { status = 0; break; }
        }

        double sigma = 0.0;
        /* Two passes of the same system = Mehrotra.  Pass 0 is the AFFINE
         * predictor: sigma = 0, so the right side carries the residuals and
         * nothing else, and its only product is the centreing factor -- the
         * ratio (predicted complementary product)/(current mu), cubed because
         * that product is quadratic in the step.  Pass 1 re-solves with
         * sigma*mu on the right.  Nothing is applied between the passes, so the
         * second one rebuilds an IDENTICAL matrix and factors it again: the
         * redundancy is real (two LU factorisations per iteration) and kept
         * because sharing the factorisation would mean hoisting a third buffer
         * and its lifetime out of the block that builds the system, for a
         * saving that is not what limits these solves. */
        for (int pass = 0; pass < 2; pass++) {
            int aug_dx = 0;   /* dx already solved in the augmented system */
            /* per-block scaling */
            for (int j = 0; j < nb; j++) { int d = dims[j];
                sym_fun(d, Xbar[j], 0, t1); mmul(d, t1, Sbar[j], t2); mmul(d, t2, t1, t3);
                sym_fun(d, t3, 1, t2); mmul(d, t1, t2, t3); mmul(d, t3, t1, Ws + j * dmax2);
                sym_fun(d, Ws + j * dmax2, 2, Wi + j * dmax2); sym_fun(d, Xbar[j], 2, Xi + j * dmax2); }
            for (int k = 0; k < m; k++) for (int j = 0; j < nb; j++) { int d = dims[j];
                const double *Wj = Ws + j * dmax2; mmul(d, Wj, Abar[k * nb + j], t1); mmul(d, t1, Wj, Mt + ((size_t)k * nb + j) * dmax2); }
            /* ---- augmented KKT: [ Kpp  Esoc' ; Esoc  Theta_soc ] ----
             * R_+ and PSD enter via their normal-equations Schur complement Kpp;
             * each SOC block keeps its cone equation A(s)dz + A(z)ds = rc in the
             * arrow-matrix form used by socp.c (robust at the cone boundary). */
            {
                /* SECANT mode: R_+ block in AUGMENTED form (exp/power, no PSD):
                 * dx stays an unknown, with row  -(s/x) dx + A' dy = -g,
                 * instead of being eliminated into A diag(x/s) A'.  Measured on
                 * logistic_large (137 scalars: 96 halves of split free
                 * variables plus v_i, W, all interior at the optimum): every
                 * x/s grows to 3e5..6.7e11, the normal-equations block swamps
                 * the multiplier rows, and dx = (x/s)(A'dy + g) re-amplifies
                 * dy's rounding by the same 1e11 -- the direction then broke
                 * the very rows it linearises, |rp + A d| up to 15x |rp|
                 * (it=23: 3.7e-8 -> 4.7e-7), and pfeas climbed at ap = 1.  In
                 * the augmented form the large x/s is a TINY diagonal, and the
                 * error of a split pair falls along dx+ + dx-, which A
                 * annihilates (|rp + A d| back to 1e-12 on every iteration).
                 * Not the default: alone it loses logistic (mosek_comparison),
                 * which the elimination solves. */
                int augx = secant && nep > 0 && nb == 0 && n > 0;
                int bx = m + K + Ke;
                int Nsys = bx + (augx ? n : 0);
                double *Sys = (double *)calloc((size_t)Nsys * (size_t)Nsys, sizeof(double));
                double *Srhs = (double *)calloc((size_t)(Nsys > 0 ? Nsys : 1), sizeof(double));
                if (!Sys || !Srhs) { free(Sys); free(Srhs); status = 2; goto done; }
                if (augx) for (int i = 0; i < n; i++) {
                    int r = bx + i;
                    for (int k = 0; k < m; k++) {
                        Sys[(size_t)r * Nsys + k] = A[k * n + i];
                        Sys[(size_t)k * Nsys + r] = A[k * n + i];
                    }
                    Sys[(size_t)r * Nsys + r] = -ss[i] / xs[i];
                    Srhs[r] = -((-ss[i] + sigma * mu / xs[i]) + rdx[i]);
                }
                for (int k = 0; k < m; k++) {
                    double rk = -rp[k];
                    if (!augx) for (int i = 0; i < n; i++) { double th = xs[i] / ss[i];
                        double g = (-ss[i] + sigma * mu / xs[i]) + rdx[i]; rk -= A[k * n + i] * th * g; }
                    for (int j = 0; j < nb; j++) { int d = dims[j]; const double *M = Mt + ((size_t)k * nb + j) * dmax2;
                        double *g = t1;
                        for (int a = 0; a < d * d; a++) g[a] = -Cbar[j][a] + sigma * mu * Xi[j * dmax2 + a];
                        for (int kk2 = 0; kk2 < m; kk2++) { double yk = y[kk2]; const double *Akj = Abar[kk2 * nb + j]; for (int a = 0; a < d * d; a++) g[a] += yk * Akj[a]; }
                        rk -= trAB(d, M, g); }
                    Srhs[k] = rk;
                    for (int l = 0; l < m; l++) {
                        double s = 0.0;
                        if (!augx) for (int i = 0; i < n; i++) s += A[k * n + i] * (xs[i] / ss[i]) * A[l * n + i];
                        for (int j = 0; j < nb; j++) s += trAB(dims[j], Mt + ((size_t)k * nb + j) * dmax2, Abar[l * nb + j]);
                        Sys[(size_t)k * Nsys + l] = s;
                    }
                }
                for (int i = 0; i < nsoc; i++) { int kk = socdims[i];
                    const double *zi = Zsoc[i], *si = Ssoc[i];
                    soc_arrow(zi, kk, Az); soc_arrow(si, kk, As);
                    for (int a = 0; a < kk; a++) rds[a] = -Csoc[i][a] + si[a];
                    for (int k = 0; k < m; k++) { const double *ak = Asoc[k * nsoc + i]; double yk = y[k]; for (int a = 0; a < kk; a++) rds[a] += yk * ak[a]; }
                    for (int a = 0; a < kk; a++) soc_e[a] = (a == 0) ? 1.0 : 0.0;
                    soc_arrow_mul(Az, si, kk, rcs);
                    for (int a = 0; a < kk; a++) rcs[a] = -rcs[a] + sigma * mu * soc_e[a];
                    int bi = m + soff[i];
                    for (int a = 0; a < kk; a++) {
                        for (int b = 0; b < kk; b++) Sys[(size_t)(bi + a) * Nsys + (bi + b)] = As[a * kk + b];
                        for (int k = 0; k < m; k++) { const double *ak = Asoc[k * nsoc + i];
                            double t = 0; for (int b = 0; b < kk; b++) t += Az[a * kk + b] * ak[b];
                            Sys[(size_t)(bi + a) * Nsys + k] = -t; }
                        double t = 0; for (int b = 0; b < kk; b++) t += Az[a * kk + b] * rds[b];
                        Srhs[bi + a] = rcs[a] + t;
                    }
                    for (int k = 0; k < m; k++) { const double *ak = Asoc[k * nsoc + i]; for (int a = 0; a < kk; a++) Sys[(size_t)k * Nsys + (bi + a)] = ak[a]; }
                }
                /* exp/power blocks: cone equation Thin dz + ds/scale = r~, with
                 * Thin the Hessian-NT scaling evaluated at the mu-free dual
                 * direction (O(1), so the affine predictor is non-degenerate)
                 * and r~ = (-s - sigma*mu*grad f(z))/scale.  scale = <z,s>/nu
                 * shrinks with mu, so the whole 3-row block is divided by it: a
                 * left row scaling that leaves (dy, dz, ds) untouched but lifts
                 * the block out of the mu orders it was sinking under Jacobi
                 * equilibration (which normalises each row by its own maximum,
                 * and that maximum is the raw O(1) coupling entry). */
                for (int i = 0; i < nep; i++) {
                    double Th[9], Tin[9], rt[3], rde[3], sc = 0.0, inv;
                    /* SECANT mode: the primal-dual secant scaling (both
                     * secant conditions exact, see expcone.c) in place of the
                     * Hessian-NT chain; it falls back to expcone_scaling
                     * itself wherever it is not defined. */
                    int brow = secant
                        ? expcone_scaling_da(ekind[i], ealpha[i], ez + 3 * i, es + 3 * i,
                                              sigma * mu, Th, Tin, rt, &sc)
                        : expcone_scaling(ekind[i], ealpha[i], ez + 3 * i, es + 3 * i,
                                              sigma * mu, Th, Tin, rt, &sc);
                    if (brow < 0) {
                        if (getenv("GMB_DBG")) fprintf(stderr,
                            "  exp block %d scaling failed: z=(%g,%g,%g) s=(%g,%g,%g) sInK*=%d sigma=%.3g mu=%.3g\n",
                            i, ez[3*i], ez[3*i+1], ez[3*i+2], es[3*i], es[3*i+1], es[3*i+2],
                            expcone_dual_in(ekind[i], ealpha[i], es + 3 * i), sigma, mu);
                        free(Sys); free(Srhs); status = 1; goto refine;
                    }
                    inv = 1.0 / sc;
                    if (getenv("GMB_DBG")) {
                        double em[EXPCONE_EV_N];
                        expcone_nt_metrics(ekind[i], ealpha[i], ez + 3 * i, es + 3 * i, em);
                        fprintf(stderr,
                            "[expm] mu=%.3g sig=%.3g blk=%d kind=%d gap=%.3g condz=%.3g conds=%.3g"
                            " rel=%.3g mz=%.3g ms=%.3g wn=%.3g row=%s\n",
                            mu, sigma, i, ekind[i], em[EXPCONE_EV_GAP], em[EXPCONE_EV_CONDZ],
                            em[EXPCONE_EV_CONDS], em[EXPCONE_EV_REL], em[EXPCONE_EV_MZ],
                            em[EXPCONE_EV_MS], em[EXPCONE_EV_WN], brow ? "nt" : "hz");
                    }
                    for (int a = 0; a < 3; a++) rde[a] = (-Cexp[i][a] + es[3 * i + a]) * inv;
                    for (int k = 0; k < m; k++) { const double *ak = Aexp[k * nep + i]; double yk = y[k]; for (int a = 0; a < 3; a++) rde[a] += yk * ak[a] * inv; }
                    int bi = m + K + 3 * i;
                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) Sys[(size_t)(bi + a) * Nsys + (bi + b)] = Tin[a * 3 + b];
                        for (int k = 0; k < m; k++) Sys[(size_t)(bi + a) * Nsys + k] = -Aexp[k * nep + i][a] * inv;
                        Srhs[bi + a] = rt[a] + rde[a];
                    }
                    for (int k = 0; k < m; k++) { const double *ak = Aexp[k * nep + i]; for (int a = 0; a < 3; a++) Sys[(size_t)k * Nsys + (bi + a)] = ak[a]; }
                }
                /* Diagonal equilibration (Jacobi row+column) of the augmented
                 * system.  Measured on the exp/power cases, the three row
                 * groups live on different scales: the multiplier block is
                 * A diag(x/s) A' = O(1/(sigma*mu)), a cone-equation row carries
                 * the Hessian scaling O(1/(sigma*mu)) (an arrow matrix O(1) for
                 * SOC), and the coupling columns are the raw A entries — a
                 * spread that reaches 1e11 by mu ~ 1e-8.  Scaling every row and
                 * column to max modulus 1 buys about two orders on the polish
                 * residuals (T36-type case: 1e-5 -> 2.7e-8 of variable error).
                 * Only the exp/power blocks are equilibrated: the PSD/SOC-only
                 * path is validated as it stands. */
                double *rr = (double *)malloc((size_t)Nsys * sizeof(double));
                double *cc = (double *)malloc((size_t)Nsys * sizeof(double));
                if (!rr || !cc) { free(rr); free(cc); free(Sys); free(Srhs); status = 2; goto done; }
                for (int i = 0; i < Nsys; i++) { rr[i] = 1.0; cc[i] = 1.0; }
                if (nep > 0 || nb > 0) {
                    for (int i = 0; i < Nsys; i++) {
                        double mx = 0.0;
                        for (int j = 0; j < Nsys; j++) { double a = fabs(Sys[(size_t)i * Nsys + j]); if (a > mx) mx = a; }
                        rr[i] = mx > 0.0 ? 1.0 / mx : 1.0;
                        Srhs[i] *= rr[i];
                        for (int j = 0; j < Nsys; j++) Sys[(size_t)i * Nsys + j] *= rr[i];
                    }
                    for (int j = 0; j < Nsys; j++) {
                        double mx = 0.0;
                        for (int i = 0; i < Nsys; i++) { double a = fabs(Sys[(size_t)i * Nsys + j]); if (a > mx) mx = a; }
                        cc[j] = mx > 0.0 ? 1.0 / mx : 1.0;
                        for (int i = 0; i < Nsys; i++) Sys[(size_t)i * Nsys + j] *= cc[j];
                    }
                }
                free(rr);
                /* Iterative refinement of the same factorisation, POLISH ONLY.
                 * Deep on the path the accuracy of the direction is what
                 * limits progress: the LU is backward stable but its SOLUTION
                 * carries cond(A)*eps, and at mu ~ 1e-8 the exp/power rows are
                 * conditioned to ~1e8, leaving 5e-8 of the rows unsatisfied --
                 * the primal residual the polish was paying for mu
                 * (market_impact: mu=1.4e-10 and still 2.2e-8 infeasible).
                 * Far from the path the limit is the cone step instead, and the
                 * exactly solved affine direction is more aggressive than the
                 * ratio test can use (measured: refinement always on collapses
                 * the PPOW oracle's primal margin onto dK at it=7, after which
                 * the line search rejects every step).  A round is kept only
                 * while it shrinks the residual: past cond ~ 1/eps the refined
                 * direction solves a noise-perturbed system (measured on T47,
                 * condz 2e14, where it froze the dual step at zero). */
                int refine_on = (nep > 0 && have_best) || (nb > 0 && nep == 0);
                double *refi = refine_on ? (double *)malloc(3 * (size_t)Nsys * sizeof(double)) : NULL;
                double *gsc = refi, *rcor = refi ? refi + Nsys : NULL, *dcor = refi ? refi + 2 * Nsys : NULL;
                if (refine_on && !refi) { free(cc); free(Sys); free(Srhs); status = 2; goto done; }
                if (refine_on) memcpy(gsc, Srhs, (size_t)Nsys * sizeof(double));
                LuFact *f = dmat_lu_factor(Sys, Nsys);
                if (!f) { free(cc); free(refi); free(Sys); free(Srhs); status = 3; goto refine; }
                /* Compensated triangular sweeps for the PSD/SOC-only path
                 * (nep == 0): the exp/power rows (socp.c, sdp.c's own Thin
                 * block) are validated on the plain solve, see
                 * dmat_lu_solve_comp's comment in linalg.c. */
                if (nep == 0) dmat_lu_solve_comp(f, Srhs); else dmat_lu_solve(f, Srhs);
                for (int rif = 0; rif < 3 && refine_on; rif++) {
                    double rn = 0.0, rn2 = 0.0;
                    for (int a = 0; a < Nsys; a++) { const double *row = Sys + (size_t)a * Nsys;
                        double t = nsum_prod(row, Srhs, Nsys);
                        rcor[a] = gsc[a] - t; if (fabs(rcor[a]) > rn) rn = fabs(rcor[a]); }
                    memcpy(dcor, rcor, (size_t)Nsys * sizeof(double));
                    if ((nep == 0 ? dmat_lu_solve_comp(f, dcor) : dmat_lu_solve(f, dcor)) != 0) break;
                    for (int b = 0; b < Nsys; b++) Srhs[b] += dcor[b];
                    for (int a = 0; a < Nsys; a++) { const double *row = Sys + (size_t)a * Nsys;
                        double t = nsum_prod(row, Srhs, Nsys);
                        t = fabs(gsc[a] - t); if (t > rn2) rn2 = t; }
                    if (!(rn2 < rn)) { for (int b = 0; b < Nsys; b++) Srhs[b] -= dcor[b]; break; }
                }
                dmat_lu_free(f);
                for (int k = 0; k < m; k++) dy[k] = Srhs[k] * cc[k];
                for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; for (int a = 0; a < kk; a++) Dzsoc[soff[i] + a] = Srhs[m + soff[i] + a] * cc[m + soff[i] + a]; }
                for (int a = 0; a < Ke; a++) Dez[a] = Srhs[m + K + a] * cc[m + K + a];
                if (augx) for (int i = 0; i < n; i++) dx[i] = Srhs[bx + i] * cc[bx + i];
                aug_dx = augx;
                free(cc);
                free(refi);
                free(Sys); free(Srhs);
            }
            /* dz */
            if (!aug_dx) for (int i = 0; i < n; i++) { double e = rdx[i]; for (int k = 0; k < m; k++) e += A[k * n + i] * dy[k]; e += (-ss[i] + sigma * mu / xs[i]); dx[i] = (xs[i] / ss[i]) * e; }
            for (int j = 0; j < nb; j++) { int d = dims[j]; double *e = t1;
                for (int a = 0; a < d * d; a++) e[a] = 0.0;
                for (int k = 0; k < m; k++) { double yk = dy[k]; const double *Akj = Abar[k * nb + j]; for (int a = 0; a < d * d; a++) e[a] += yk * Akj[a]; }
                for (int a = 0; a < d * d; a++) e[a] += -Cbar[j][a] + sigma * mu * Xi[j * dmax2 + a];
                for (int kk2 = 0; kk2 < m; kk2++) { double yk = y[kk2]; const double *Akj = Abar[kk2 * nb + j]; for (int a = 0; a < d * d; a++) e[a] += yk * Akj[a]; }
                const double *Wj = Ws + j * dmax2; mmul(d, Wj, e, t2); mmul(d, t2, Wj, Dx + j * dmax2); }
            /* ds.  With dx from the augmented system, take ds from the dual
             * row  A'dy + ds = -rdx  (as the SOC/exp blocks below do), not
             * from -(s/x) dx: at a bound s/x reaches 1e10, and dx's last-bit
             * rounding came back as 1e-5 of dual infeasibility (gp1, it=18). */
            if (aug_dx) for (int i = 0; i < n; i++) { double t = -rdx[i]; for (int k = 0; k < m; k++) t -= A[k * n + i] * dy[k]; ds[i] = t; }
            else for (int i = 0; i < n; i++) ds[i] = -(ss[i] / xs[i]) * dx[i] + (-ss[i] + sigma * mu / xs[i]);
            for (int j = 0; j < nb; j++) { int d = dims[j]; const double *Wij = Wi + j * dmax2;
                mmul(d, Wij, Dx + j * dmax2, t1); mmul(d, t1, Wij, t2);
                for (int a = 0; a < d * d; a++) Ds[j * dmax2 + a] = -t2[a] - Sbar[j][a] + sigma * mu * Xi[j * dmax2 + a]; }
            for (int i = 0; i < nsoc; i++) { int kk = socdims[i];
                for (int a = 0; a < kk; a++) { double t = -(-Csoc[i][a] + Ssoc[i][a]); for (int k = 0; k < m; k++) t -= Asoc[k * nsoc + i][a] * y[k]; for (int k = 0; k < m; k++) t -= Asoc[k * nsoc + i][a] * dy[k]; Dssoc[soff[i] + a] = t; } }
            for (int i = 0; i < nep; i++) {
                double rde[3];
                for (int a = 0; a < 3; a++) rde[a] = -Cexp[i][a] + es[3 * i + a];
                for (int k = 0; k < m; k++) { const double *ak = Aexp[k * nep + i]; double yk = y[k]; for (int a = 0; a < 3; a++) rde[a] += yk * ak[a]; }
                for (int a = 0; a < 3; a++) { double t = -rde[a]; for (int k = 0; k < m; k++) t -= Aexp[k * nep + i][a] * dy[k]; Des[3 * i + a] = t; }
            }
            if (pass == 0) {
                double ap = 1.0, ad = 1.0;
                for (int i = 0; i < n; i++) { if (dx[i] < 0) { double t = -xs[i] / dx[i]; if (t < ap) ap = t; } if (ds[i] < 0) { double t = -ss[i] / ds[i]; if (t < ad) ad = t; } }
                for (int j = 0; j < nb; j++) { int d = dims[j];
                    sym_fun(d, Xbar[j], 1, t1); mmul(d, t1, Dx + j * dmax2, t2); mmul(d, t2, t1, t3); double lp = min_eig(d, t3); if (lp < 0) { double t = 1.0 / (-lp); if (t < ap) ap = t; }
                    sym_fun(d, Sbar[j], 1, t1); mmul(d, t1, Ds + j * dmax2, t2); mmul(d, t2, t1, t3); double ld = min_eig(d, t3); if (ld < 0) { double t = 1.0 / (-ld); if (t < ad) ad = t; } }
                for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; double a = soc_step(Zsoc[i], Dzsoc + soff[i], kk); if (a < ap) ap = a; a = soc_step(Ssoc[i], Dssoc + soff[i], kk); if (a < ad) ad = a; }
                for (int i = 0; i < nep; i++) { double a = expcone_maxstep(ekind[i], ealpha[i], ez + 3 * i, Dez + 3 * i); if (a < ap) ap = a; a = expcone_dual_maxstep(ekind[i], ealpha[i], es + 3 * i, Des + 3 * i); if (a < ad) ad = a; }
                if (ap > 1) ap = 1; if (ad > 1) ad = 1;
                double mua = 0;
                for (int i = 0; i < n; i++) mua += (xs[i] + ap * dx[i]) * (ss[i] + ad * ds[i]);
                for (int j = 0; j < nb; j++) { int d = dims[j]; for (int a = 0; a < d * d; a++) t1[a] = Xbar[j][a] + ap * Dx[j * dmax2 + a]; for (int a = 0; a < d * d; a++) t2[a] = Sbar[j][a] + ad * Ds[j * dmax2 + a]; mua += trAB(d, t1, t2); }
                for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; for (int a = 0; a < kk; a++) mua += (Zsoc[i][a] + ap * Dzsoc[soff[i] + a]) * (Ssoc[i][a] + ad * Dssoc[soff[i] + a]); }
                for (int a = 0; a < Ke; a++) mua += (ez[a] + ap * Dez[a]) * (es[a] + ad * Des[a]);
                mua /= (double)(ntot > 0 ? ntot : 1);
                sigma = (mu > 0) ? (mua / mu) : 0; sigma = sigma * sigma * sigma;
                if (!(sigma > 0.1)) sigma = 0.1; if (sigma > 1) sigma = 1;
                if (polish_center) sigma = 1.0;
            } else {
                /* Rescue for a MULTI-BLOCK PSD path whose Schur elimination lost
                 * the small directions near a PSD face: when the previous step
                 * made the feasibility GROW, rebuild the corrector direction from
                 * the augmented system, which retains A dx = -rp.  Gated to
                 * nb > 1: single-block models keep their ordinary result and
                 * fall back to the tangent cuts as before.  DECLARED deviation:
                 * the gate (nb > 1, 5% growth) is a measured heuristic, not a
                 * Schur conditioning test. */
                if (use_rescue && nb > 1 && nep == 0 && nsoc == 0) {
                    int rr = sdp_aug_direction(m, n, nb, dims, dmax2, A, Abar, Cbar,
                               xs, ss, (const double *const *)Sbar, y, rp, rdx, Wi, Xi, sigma * mu,
                               dy, dx, ds, Dx, Ds);
                    if (getenv("GMB_DBG")) fprintf(stderr, "  [rescue] it=%d ret=%d\n", it, rr);
                }
                double ap = 1.0, ad = 1.0;
                for (int i = 0; i < n; i++) { if (dx[i] < 0) { double t = -xs[i] / dx[i]; if (t < ap) ap = t; } if (ds[i] < 0) { double t = -ss[i] / ds[i]; if (t < ad) ad = t; } }
                for (int j = 0; j < nb; j++) { int d = dims[j];
                    sym_fun(d, Xbar[j], 1, t1); mmul(d, t1, Dx + j * dmax2, t2); mmul(d, t2, t1, t3); double lp = min_eig(d, t3); if (lp < 0) { double t = 1.0 / (-lp); if (t < ap) ap = t; }
                    sym_fun(d, Sbar[j], 1, t1); mmul(d, t1, Ds + j * dmax2, t2); mmul(d, t2, t1, t3); double ld = min_eig(d, t3); if (ld < 0) { double t = 1.0 / (-ld); if (t < ad) ad = t; } }
                for (int i = 0; i < nsoc; i++) { int kk = socdims[i]; double a = soc_step(Zsoc[i], Dzsoc + soff[i], kk); if (a < ap) ap = a; a = soc_step(Ssoc[i], Dssoc + soff[i], kk); if (a < ad) ad = a; }
                for (int i = 0; i < nep; i++) { double a = expcone_maxstep(ekind[i], ealpha[i], ez + 3 * i, Dez + 3 * i); if (a < ap) ap = a; a = expcone_dual_maxstep(ekind[i], ealpha[i], es + 3 * i, Des + 3 * i); if (a < ad) ad = a; }
                if (ap > 1) ap = 1; if (ad > 1) ad = 1;
                /* SECANT mode: one step length for both sides.  The secant
                 * scaling is a primal-dual (NT-type) scaling, whose
                 * complementarity prediction assumes it: with ap != ad the
                 * cross term of (z + ap dz)(s + ad ds) no longer cancels,
                 * and on gp1 one such step (ap=0.16, ad=0.71) lifted mu 38x. */
                if (secant) { if (ad < ap) ap = ad; else ad = ap; }
                if (getenv("GMB_DBG")) fprintf(stderr,"  [corr] sigma=%.3g ap=%.4f ad=%.4f\n",sigma,ap,ad);
                /* Fraction of the maximal boundary step the CORRECTOR may take.
                 * Far from the optimum (mu large) it sits on its 0.90 floor --
                 * take 90% of what the cone ratio test allows, stay well inside.
                 * As mu falls it rises to the 0.999 cap: only an iterate that is
                 * already accurate is let that close to a face, because there
                 * the optimum itself may lie on one. */
                double tau = 1.0 - mu * 0.01; if (tau < 0.90) tau = 0.90; if (tau > 0.999) tau = 0.999;
                ap *= tau; ad *= tau;
                /* The cone backtracking above only asks that the trial point stay
                 * INSIDE the cones; it says nothing about the equations, so the
                 * polish was free to take steps that trade primal feasibility for
                 * centrality.  Measured on gp1, market_impact and logistic_large:
                 * mu fell to 1e-9 and the dual rows to 1e-14 while pfeas GREW
                 * (3.7e-6 -> 5.3e-5, 6.5e-5 -> 1.5e-4, 1e-9 -> 3.8e-3), which is
                 * exactly what the acceptance test refuses and what then pushed
                 * those cases onto the tangent cuts.  The step is therefore also
                 * required to decrease the l1 merit of the system the corrector
                 * actually linearised -- rows, dual rows and the PERTURBED
                 * complementarity X S = sigma*mu*e, whose residual at the trial
                 * point is |mu_trial - sigma*mu| (with mu itself in place of that
                 * term the direction is not even descent away from the path).
                 * The test is non-monotone over a window of MWIN iterates
                 * (Gripsrud-Tunsvik-Osterre): enforced monotonicity is what kills
                 * the solve, because a PPOW oracle at it=3 improves its infeasibil-
                 * ity 1.874 -> 1.800 while its mu doubles (one block sits at
                 * gap=1.2e-2 against the global mu=1.8e-1, so the common path
                 * target pulls it back up) and that excursion is what then carries
                 * the merit to 4e-1.  The window cannot excuse the pathology
                 * either: those runaways happen at merits of 1e-4 and below, where
                 * the window max has long since fallen to that size.  Nothing
                 * accepted means the direction is spent: stop advancing and report
                 * the best point already in the snapshot. */
                double merit0 = pfeas + dfeas + (1.0 - sigma) * mu, ref = merit0;
                for (int q = 0; q < MWIN; q++) if (q < nwin && mwin[q] > ref) ref = mwin[q];
                ref += 1e-10 * ref + 1e-16;
                int took = 0, stalled = 0;
                for (int bt = 0; bt < 60; bt++) { int good = 1;
                    for (int j = 0; j < nb && good; j++) { int d = dims[j]; for (int a = 0; a < d * d; a++) t1[a] = Xbar[j][a] + ap * Dx[j * dmax2 + a]; if (min_eig(d, t1) < 0) good = 0; }
                    for (int j = 0; j < nb && good; j++) { int d = dims[j]; for (int a = 0; a < d * d; a++) t1[a] = Sbar[j][a] + ad * Ds[j * dmax2 + a]; if (min_eig(d, t1) < 0) good = 0; }
                    for (int i = 0; i < nsoc && good; i++) { int kk = socdims[i];
                        for (int a = 0; a < kk; a++) t1[a] = Zsoc[i][a] + ap * Dzsoc[soff[i] + a];
                        if (soc_margin(t1, kk) < 0) good = 0;
                        for (int a = 0; a < kk; a++) t1[a] = Ssoc[i][a] + ad * Dssoc[soff[i] + a];
                        if (soc_margin(t1, kk) < 0) good = 0; }
                    for (int i = 0; i < nep && good; i++) { double xx[3], fv;
                        for (int a = 0; a < 3; a++) xx[a] = ez[3 * i + a] + ap * Dez[3 * i + a];
                        if (expcone_barrier(ekind[i], ealpha[i], xx, &fv)) good = 0;
                        for (int a = 0; a < 3; a++) xx[a] = es[3 * i + a] + ad * Des[3 * i + a];
                        if (!expcone_dual_in(ekind[i], ealpha[i], xx)) good = 0; }
                    if (good && (nep > 0 || nb > 0)) {
                        ipm_state(trial, 1, IPM_STATE_PASS);
                        ipm_apply(ap, ad, &D, IPM_STATE_PASS);
                        ipmres T = ipm_resid(&C);
                        good = isfinite(T.mu) &&
                               T.pfeas + T.dfeas + fabs(T.mu - sigma * mu) <= ref;
                        if (good) { took = 1; break; }
                        ipm_state(trial, 0, IPM_STATE_PASS);
                    }
                    if (good) break;
                    ap *= 0.5; ad *= 0.5;
                    if ((nep > 0 || nb > 0) && ap < 1e-12) { stalled = 1; break; } }
                if (stalled) {
                    if (getenv("GMB_DBG")) fprintf(stderr,
                        "  [ls] no descent step at it=%d ap=%.3g merit=%.3g ref=%.3g\n",
                        it, ap, merit0, ref);
                    goto refine;
                }
                /* `took` is set only by the exp/power branch, so this line is the
                 * ordinary apply on the PSD/SOC path (a `good` trial point breaks
                 * out without setting it).  For an exp/power model it is reached
                 * either by the same accept or by 60 halvings, where
                 * ap = ap0*2^-60 ~ 1e-18 and the call moves the iterate by
                 * nothing.  Read, not just measured: a third exit would be the
                 * frozen-case `goto refine` above. */
                if (!took) ipm_apply(ap, ad, &D, IPM_STATE_PASS);
                mwin[iwin] = merit0; iwin = (iwin + 1) % MWIN;
                if (nwin < MWIN) nwin++;
            }
        }
    }
refine:
    if (have_best) {
        ipm_state(snap, 0, IPM_STATE_PASS);
        status = 0;
    } else if (fb_saved && nep > 0) {
        /* Il punto congelato entra negli output come CANDIDATO solo per i
         * modelli exp/power (nep>0): li' il pavimento di residuo e' noto
         * (T90/T91/logistic_large/risk_parity) e i tagli restano la prima
         * scelta.  Sui modelli PSD/SOC vale la policy di T96: un "non risolto"
         * non pubblica.  status resta "non risolto", cosi' il gate lo giudica. */
        ipm_state(fb_snap, 0, IPM_STATE_PASS);
    }
    /* Route selection, on the MEASURED quality of the point about to be handed
     * back and against the task's OWN interior-point tolerances: mu is the
     * average complementarity product, so it is scale-free but says nothing
     * about the problem it came from (the same mu is a 1e-12 gap on one model
     * and a 1e-3 on another), and a boundary optimum is approached as
     * O(sqrt(mu)), so judging a point by mu both over-states its error and
     * ignores the residuals.  The residual kernel is re-run on the restored
     * point rather than trusting what the polish stored.  A point that misses
     * its tolerances is only as good as the mu that produced it, and there the
     * tangent-cut outer approximation is the better answer: report "not solved"
     * and let the caller route there.  expcone_route_probe prints which route
     * answered. */
    if (nep > 0 || status == 0 || getenv("GMB_DBG")) {
        ipmqual Q = { 0.0, 0.0, 0.0 }; int good = 0, good_near = 0, have = 0;
        double kpri = HUGE_VAL, kdual = HUGE_VAL;
        if (status == 0) {
            ipmres R = ipm_resid(&C);
            Q = ipm_quality(&C, &R);
            ipm_cone_slacks(&C, &kpri, &kdual);
            have = 1;
            good = isfinite(Q.pri) && isfinite(Q.dual) && isfinite(Q.gap) &&
                   Q.pri <= rtol_pri && Q.dual <= rtol_dual && Q.gap <= rtol_gap;
            /* The reference's near-optimal rule (INTPNT_CO_TOL_NEAR_REL, default
             * 1000): a point that misses the prescribed accuracy is checked
             * against the SAME termination criteria with every tolerance
             * multiplied by the factor, and if it passes it is declared optimal.
             * There is no "near optimal" status to invent -- MSKsolsta has no
             * NEAR_ member -- so what the factor decides is the verdict. */
            good_near = good || (isfinite(Q.pri) && isfinite(Q.dual) && isfinite(Q.gap) &&
                   Q.pri <= rtol_pri * near_rel && Q.dual <= rtol_dual * near_rel &&
                   Q.gap <= rtol_gap * near_rel);
        }
        /* Where an alternative algorithm exists -- exp/power, the tangent-cut
         * outer approximation -- the DECLARED tolerances keep choosing the
         * route: the alternative measures better on the cases that reach it, and
         * since both answers are declared optimal the factor cannot make the
         * delivered point worse than it was.  Where this IPM is the only conic
         * route the factor is the verdict: a point outside even the effective
         * tolerance is reported "not solved" instead of published unjudged. */
        if (status == 0 && !good && (nep > 0 || !good_near)) status = 1;
        if (getenv("GMB_DBG") && have) fprintf(stderr,
            "  [route] status=%d rel_pri=%.3g rel_dual=%.3g rel_gap=%.3g"
            " (tol %g/%g/%g near %g) blocks=%d\n",
            status, Q.pri, Q.dual, Q.gap, rtol_pri, rtol_dual, rtol_gap,
            near_rel, nep);
        /* No restored point means no measurement: printing the initialisers
         * would claim a triple of zeros on a solve that never got one (seen on
         * risk_parity, which stops short of the polish). */
        if (getenv("GMB_DBG") && !have) fprintf(stderr,
            "  [route] status=%d triple not measured (no point restored)"
            " (tol %g/%g/%g near %g) blocks=%d\n",
            status, rtol_pri, rtol_dual, rtol_gap, near_rel, nep);
        /* Neither rel_pri (equality rows only) nor rel_dual says whether the
         * point is inside the cones it is supposed to live in, so cone
         * membership is measured here and printed on its own line: a negative
         * value is a published violation.  Printed also when the gate then
         * hands the model to the tangent cuts -- that is the point the native
         * route would have published. */
        if (getenv("GMB_DBG") && have) fprintf(stderr,
            "  [cones] pri_slack=%.3g dual_slack=%.3g\n", kpri, kdual);
    }
    for (int i = 0; i < n; i++) x[i] = xs[i];
    for (int i = 0; i < nep; i++) { for (int a = 0; a < 3; a++) { Zexp[i][a] = ez[3 * i + a]; Sexp[i][a] = es[3 * i + a]; } }
done:
    /* Un punto e' negli output (accettato-e-declassato, o congelato) ma il
     * verdetto nativo e' "non risolto": il chiamante puo' pubblicarlo come
     * fallback se anche i tagli non rispondono. */
    if (fb_ok) *fb_ok = (status != 0 && fb_saved && nep > 0) ? 1 : 0;
    free(soff); free(Ws); free(Wi); free(Xi); free(t1); free(t2); free(t3); free(Mt);
    free(xs); free(ss); free(zsoc); free(ssoc); free(Dzsoc); free(Dssoc);
    free(Az); free(As); free(rds); free(rcs); free(soc_e);
    free(ez); free(es); free(Dez); free(Des);
    free(rp); free(rdx); free(dy); free(dx); free(ds); free(Dx); free(Ds); free(snap); free(trial); free(fb_snap);
    return status;
}

/* The native solve, run as it always ran; an exp/power model without PSD
 * blocks that it does NOT solve gets one more native attempt in SECANT mode
 * (augmented R_+ block, primal-dual secant scaling, common step length --
 * the three comments marked SECANT in sdp_ipm_run) before the caller hands
 * it to the tangent cuts.  A second attempt rather than a new default
 * because the measured corpus splits: the secant mode solves logistic_large
 * (status=0, rel_gap 4.7e-11; the first run stalls at 6.9e-8), but on gp1 a
 * non-strictly-complementary block (z and s both on their boundaries) is
 * handled by the Hessian-NT chain and not by the secant, which stops at
 * rel_gap 1.6e-8.  Anything the first run solves is therefore returned
 * untouched, and a failed second attempt restores the first run's outputs
 * (fallback point and fb_ok included), so the retry can only turn a failure
 * into a success.  GMB_NO_EXP_RETRY disables it; GMB_EXP_DA runs the secant
 * mode alone. */
int sdp_ipm(int m, int n, const double *A, const double *b, const double *c,
            int nb, const int *dims,
            const double *const *Cbar, const double *const *Abar,
            int nsoc, const int *socdims,
            const double *const *Csoc, const double *const *Asoc,
            int nep, const int *ekind, const double *ealpha,
            const double *const *Cexp, const double *const *Aexp,
            int max_iter, double tol, double rtol_pri, double rtol_dual, double rtol_gap,
            double near_rel,
            double *x, double *const *Xbar, double *y, double *const *Sbar,
            double *const *Zsoc, double *const *Ssoc,
            double *const *Zexp, double *const *Sexp, const double *xwarm,
            const double *ywarm, int *fb_ok)
{
#define SDP_IPM_RUN(sec) sdp_ipm_run((sec), m, n, A, b, c, nb, dims, Cbar, Abar, nsoc, socdims, \
        Csoc, Asoc, nep, ekind, ealpha, Cexp, Aexp, max_iter, tol, rtol_pri, rtol_dual, rtol_gap, \
        near_rel, x, Xbar, y, Sbar, Zsoc, Ssoc, Zexp, Sexp, xwarm, ywarm, fb_ok)
    int first = getenv("GMB_EXP_DA") ? 1 : 0;
    int st = SDP_IPM_RUN(first);
    if (st != 0 && !first && nep > 0 && nb == 0 && !getenv("GMB_NO_EXP_RETRY")) {
        size_t ns = (size_t)n + (size_t)m + 6 * (size_t)nep, o = 0;
        for (int i = 0; i < nsoc; i++) ns += 2 * (size_t)socdims[i];
        double *save = (double *)malloc(ns * sizeof(double));
        if (save) {
            int fb0 = fb_ok ? *fb_ok : 0;
#define SDP_CP(p, cnt, out) do { size_t k_ = (size_t)(cnt);                                   \
            if (out) memcpy(save + o, (p), k_ * sizeof(double));                               \
            else memcpy((p), save + o, k_ * sizeof(double)); o += k_; } while (0)
#define SDP_ALL(out) do { o = 0; SDP_CP(x, n, out); SDP_CP(y, m, out);                       \
            for (int i = 0; i < nsoc; i++) { SDP_CP(Zsoc[i], socdims[i], out); SDP_CP(Ssoc[i], socdims[i], out); } \
            for (int i = 0; i < nep; i++) { SDP_CP(Zexp[i], 3, out); SDP_CP(Sexp[i], 3, out); } } while (0)
            SDP_ALL(1);
            if (getenv("GMB_DBG")) fprintf(stderr, "  [retry] exp/power secant mode after status=%d\n", st);
            if (SDP_IPM_RUN(1) == 0) st = 0;
            else { SDP_ALL(0); if (fb_ok) *fb_ok = fb0; }
#undef SDP_ALL
#undef SDP_CP
            free(save);
        }
    }
#undef SDP_IPM_RUN
    return st;
}
