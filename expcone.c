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

/* expcone.c - see expcone.h. */
#include <stddef.h>
#include <math.h>
#include "expcone.h"
#include "linalg.h"

/* PEXP: t >= u e^{v/u}, u > 0.  f = -log(G) - log(u), G = t - u e^{v/u}. */
static int pexp_core(const double *x, double *f, double *g, double *H) {
    double t = x[0], u = x[1], v = x[2];
    if (!(u > 0.0)) return -1;
    double w = v / u, ew = exp(w);
    double G = t - u * ew;
    if (!(G > 0.0)) return -1;
    double gu = ew * (w - 1.0);      /* dG/du */
    double gv = -ew;                 /* dG/dv */
    if (f) *f = -log(G) - log(u);
    if (g) {
        g[0] = -1.0 / G;
        g[1] = -gu / G - 1.0 / u;
        g[2] = -gv / G;
    }
    if (H) {
        double dG[3] = { 1.0, gu, gv };
        double d2G[9] = {
            0.0,            0.0,           0.0,
            0.0,            -ew * w * w / u, ew * w / u,
            0.0,            ew * w / u,     -ew / u
        };
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double h = dG[i] * dG[j] / (G * G) - d2G[i * 3 + j] / G;
            if (i == 1 && j == 1) h += 1.0 / (u * u);
            H[i * 3 + j] = h;
        }
    }
    return 0;
}

/* PPOW/RPOW: rot*t^{2a} u^{2(1-a)} - v^2 > 0, t,u > 0 (rot=1 PPOW, 2 RPOW). */
static int ppow_core(const double *x, double alpha, double rot, double *f, double *g, double *H) {
    double t = x[0], u = x[1], v = x[2];
    if (!(t > 0.0) || !(u > 0.0) || !(alpha > 0.0) || !(alpha < 1.0)) return -1;
    double p = pow(t, 2.0 * alpha) * pow(u, 2.0 * (1.0 - alpha));
    double D = rot * p - v * v;
    if (!(D > 0.0)) return -1;
    double dt = rot * 2.0 * alpha * p / t;            /* dD/dt */
    double du = rot * 2.0 * (1.0 - alpha) * p / u;    /* dD/du */
    double dv = -2.0 * v;                             /* dD/dv */
    if (f) *f = -log(D) - (1.0 - alpha) * log(t) - alpha * log(u);
    if (g) {
        g[0] = -dt / D - (1.0 - alpha) / t;
        g[1] = -du / D - alpha / u;
        g[2] = -dv / D;
    }
    if (H) {
        double dD[3] = { dt, du, dv };
        double d2D[9] = {
            rot * 2.0 * alpha * (2.0 * alpha - 1.0) * p / (t * t),  rot * 4.0 * alpha * (1.0 - alpha) * p / (t * u), 0.0,
            rot * 4.0 * alpha * (1.0 - alpha) * p / (t * u),        rot * 2.0 * (1.0 - alpha) * (1.0 - 2.0 * alpha) * p / (u * u), 0.0,
            0.0,                                                    0.0,                                            -2.0
        };
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double h = dD[i] * dD[j] / (D * D) - d2D[i * 3 + j] / D;
            if (i == 0 && j == 0) h += (1.0 - alpha) / (t * t);
            if (i == 1 && j == 1) h += alpha / (u * u);
            H[i * 3 + j] = h;
        }
    }
    return 0;
}

/**
 * Computes the log-barrier value for an exponential/power cone.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param x      [in]  Point in the cone (t, u, v for PEXP/DEXP; t, u, v for PPOW/RPOW).
 * @param f      [out] Pointer to receive the barrier value (may be NULL).
 *
 * @return 0 on success, -1 if x is not in the interior of the cone.
 *
 * @note PEXP: f = -log(t - u*exp(v/u)) - log(u), defined for u > 0, t > u*exp(v/u).
 *       PPOW/RPOW: f = -log(D) - (1-a)*log(t) - a*log(u), D = rot*t^{2a}*u^{2(1-a)} - v^2.
 *       Returns -1 if x is not in the interior of the cone.
 */
int expcone_barrier(int kind, double alpha, const double *x, double *f) {
    if (kind == EXPCONE_PEXP) return pexp_core(x, f, NULL, NULL);
    if (kind == EXPCONE_PPOW) return ppow_core(x, alpha, 1.0, f, NULL, NULL);
    if (kind == EXPCONE_RPOW) return ppow_core(x, alpha, 2.0, f, NULL, NULL);
    return -1;
}
/**
 * Computes the gradient of the log-barrier for an exponential/power cone.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param x      [in]  Point in the cone.
 * @param g      [out] Gradient vector (size 3). Must not be NULL.
 *
 * @return 0 on success, -1 if x is not in the interior of the cone.
 *
 * @note Gradient of the log-barrier function.
 *       PEXP: g = (-1/G, -(ew*(w-1)+1/u)/G, ew/G) where w=v/u, G=t-u*exp(v/u).
 *       PPOW/RPOW: g = -(dD/D) - (1-a)/t etc. where D = rot*t^{2a}*u^{2(1-a)} - v^2.
 */
int expcone_grad(int kind, double alpha, const double *x, double *g) {
    if (kind == EXPCONE_PEXP) return pexp_core(x, NULL, g, NULL);
    if (kind == EXPCONE_PPOW) return ppow_core(x, alpha, 1.0, NULL, g, NULL);
    if (kind == EXPCONE_RPOW) return ppow_core(x, alpha, 2.0, NULL, g, NULL);
    return -1;
}
/**
 * Computes the Hessian of the log-barrier for an exponential/power cone.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param x      [in]  Point in the cone.
 * @param H      [out] Hessian matrix (3x3 row-major). Must not be NULL.
 *
 * @return 0 on success, -1 if x is not in the interior of the cone.
 *
 * @note Hessian of the log-barrier function (3x3 symmetric matrix).
 *       For PEXP: H_ij = (dG_i dG_j)/G^2 - d2G_ij/G + diag(0, 1/u^2, 0).
 *       For PPOW/RPOW: H_ij = (dD_i dD_j)/D^2 - d2D_ij/D + diag((1-a)/t^2, a/u^2, 0).
 *       Returns -1 if x is not in the interior of the cone.
 */
int expcone_hess(int kind, double alpha, const double *x, double *H) {
    if (kind == EXPCONE_PEXP) return pexp_core(x, NULL, NULL, H);
    if (kind == EXPCONE_PPOW) return ppow_core(x, alpha, 1.0, NULL, NULL, H);
    if (kind == EXPCONE_RPOW) return ppow_core(x, alpha, 2.0, NULL, NULL, H);
    return -1;
}

/* 3x3 inverse (row-major); returns 0 on success, -1 if singular. */
static int mat3inv(const double *M, double *Mi) {
    double a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], i = M[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!(det > 0.0) && !(det < 0.0)) return -1;
    double id = 1.0 / det;
    Mi[0] = (e * i - f * h) * id; Mi[1] = (c * h - b * i) * id; Mi[2] = (b * f - c * e) * id;
    Mi[3] = (f * g - d * i) * id; Mi[4] = (a * i - c * g) * id; Mi[5] = (c * d - a * f) * id;
    Mi[6] = (d * h - e * g) * id; Mi[7] = (b * g - a * h) * id; Mi[8] = (a * e - b * d) * id;
    return 0;
}

/* C = A B, 3x3 row-major. */
static void mm3(const double *A, const double *B, double *C) {
    double t[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double s = 0.0;
        for (int k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
        t[i * 3 + j] = s;
    }
    for (int i = 0; i < 9; i++) C[i] = t[i];
}

/* M symmetric positive definite -> Out = V diag(lam^p) V'. -1 if not PD. */
static int spd_pow3(const double *M, double p, double *Out) {
    double A[9], ev[3], V[9];
    for (int i = 0; i < 9; i++) A[i] = 0.5 * (M[i] + M[(i % 3) * 3 + i / 3]);
    dmat_eig_jacobi(3, A, ev, V);
    for (int k = 0; k < 3; k++) if (!(ev[k] > 0.0) || !isfinite(ev[k])) return -1;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double s = 0.0;
        for (int k = 0; k < 3; k++) s += pow(ev[k], p) * V[i * 3 + k] * V[j * 3 + k];
        Out[i * 3 + j] = s;
    }
    return 0;
}

/* Power cones only.  With xi = rot*p/D > 1 (p = t^{2a}u^{2(1-a)},
 * D = rot p - v^2), the coordinates of the scaling point are
 *   t(xi) = (2 a xi + 1 - a)/s0,  u(xi) = (2(1-a) xi + a)/s1,
 * and p must also equal 4 xi (xi-1)/(rot s2^2).  gap(xi) is the log of the
 * ratio of the two expressions: +inf at xi -> 1+, decreasing through 0. */
static double pow_gap(double xi, double alpha, double rot, const double *s) {
    double b = 1.0 - alpha, s0 = s[0], s1 = s[1], s2 = s[2];
    double t = (2.0 * alpha * xi + b) / s0;
    double u = (2.0 * b * xi + alpha) / s1;
    double lhs = 4.0 * xi * (xi - 1.0) / (rot * s2 * s2);
    if (!(t > 0.0) || !(u > 0.0) || !(lhs > 0.0)) return -INFINITY;
    return 2.0 * alpha * log(t) + 2.0 * b * log(u) - log(lhs);
}

/* Membership in the interior of K*.  The defining expressions are differences
 * of terms that cancel on the boundary, so the error of the result is a
 * multiple of the TERMS: evaluated as given, the same direction answered
 * differently at the two scales it is used at (logistic_large block 22 at
 * mu=1.9e-5: bracket +2.9e-17 for s, -7.3e-10 for s/gap with gap=1.3e-7, the
 * round-off multiplied by 1/gap).  Normalising by the 1-norm puts every term at
 * O(1), so the verdict depends on the direction and not on how the caller
 * happened to scale it.  It cannot decide a direction that is at the boundary
 * within round-off -- no verdict exists for that one in double precision -- so a
 * caller that treats "outside" as fatal must not use it that way: see nt_eval,
 * which degrades the row instead. */
/**
 * Checks membership in the dual cone K*.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param s      [in]  Point in the dual space (size 3).
 *
 * @return 1 if s is in the interior of K*, 0 otherwise.
 *
 * @note Dual cone membership:
 *       PEXP*:  s0 > 0, s2 < 0, s1 - s2 + s2*ln(-s2/s0) > 0.
 *       PPOW/RPOW dual:  s0 > 0, s1 > 0, (s0/a)^a * (s1/(1-a))^(1-a) >= (1|sqrt(2))*|s2|.
 *       Normalizes by 1-norm for scale invariance.
 */
int expcone_dual_in(int kind, double alpha, const double *s) {
    double nrm = fabs(s[0]) + fabs(s[1]) + fabs(s[2]);
    double s0, s1, s2;
    if (!(nrm > 0.0) || !isfinite(nrm)) return 0;
    s0 = s[0] / nrm; s1 = s[1] / nrm; s2 = s[2] / nrm;
    if (!isfinite(s0) || !isfinite(s1) || !isfinite(s2)) return 0;
    if (kind == EXPCONE_PEXP) {
        if (!(s0 > 0.0) || !(s2 < 0.0)) return 0;
        return s1 - s2 + s2 * log(-s2 / s0) > 0.0;
    }
    if (kind != EXPCONE_PPOW && kind != EXPCONE_RPOW) return 0;
    if (!(alpha > 0.0) || !(alpha < 1.0)) return 0;
    if (!(s0 > 0.0) || !(s1 > 0.0)) return 0;
    {   double lhs = pow(s0 / alpha, alpha) * pow(s1 / (1.0 - alpha), 1.0 - alpha);
        double rhs = (kind == EXPCONE_RPOW ? sqrt(2.0) : 1.0) * fabs(s2);
        return lhs > rhs;
    }
}

/**
 * Computes the dual cone scaling point W such that -∇f(W) = s.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param s      [in]  Dual vector in K* (size 3).
 * @param W      [out] Scaling point W (size 3).
 *
 * @return 0 on success, -1 if s not in K* or W not found.
 *
 * @note Finds W such that -∇f(W) = s.
 *       PEXP: W = (1/s0 + exp(w)/q, 1/q, w/q) where w = ln(-s2/s0), q = s1 - s2 + s2*w.
 *       PPOW/RPOW: Solves for xi via binary search on pow_gap(xi) > 0,
 *                  then W = (2a*xi+1-a)/s0, (2(1-a)*xi+a)/s1, -0.5*s2*D.
 *       Verifies by computing grad at W and checking ||grad + s|| < 1e-6.
 */
int expcone_dual_point(int kind, double alpha, const double *s, double *W) {
    double g[3], res = 0.0, sn = 1.0 + fabs(s[0]) + fabs(s[1]) + fabs(s[2]);
    if (!expcone_dual_in(kind, alpha, s)) return -1;
    if (kind == EXPCONE_PEXP) {
        double s0 = s[0], s1 = s[1], s2 = s[2];
        double w = log(-s2 / s0), q = s1 - s2 + s2 * w;
        if (!(q > 0.0)) return -1;
        W[0] = 1.0 / s0 + exp(w) / q; W[1] = 1.0 / q; W[2] = w / q;
    } else {
        double rot = (kind == EXPCONE_RPOW) ? 2.0 : 1.0;
        double s0 = s[0], s1 = s[1], s2 = s[2], xi = 1.0;
        if (fabs(s2) > 1e-13 * (fabs(s0) + fabs(s1) + 1.0)) {
            double lo = 1.0 + 1e-13, hi = 2.0;
            if (pow_gap(lo, alpha, rot, s) > 0.0) {
                int nb = 0;
                while (pow_gap(hi, alpha, rot, s) > 0.0) { if (++nb > 200) return -1; hi *= 2.0; }
                for (int it = 0; it < 200; it++) {
                    double mid = sqrt(lo * hi);
                    if (pow_gap(mid, alpha, rot, s) > 0.0) lo = mid; else hi = mid;
                }
                xi = sqrt(lo * hi);
            }   /* else |s2| is small enough that xi = 1 is exact to working precision */
        }
        {
            double p, D;
            W[0] = (2.0 * alpha * xi + 1.0 - alpha) / s0;
            W[1] = (2.0 * (1.0 - alpha) * xi + alpha) / s1;
            p = pow(W[0], 2.0 * alpha) * pow(W[1], 2.0 * (1.0 - alpha));
            D = rot * p / xi;
            W[2] = -0.5 * s2 * D;
        }
    }
    if (expcone_grad(kind, alpha, W, g)) return -1;
    for (int i = 0; i < 3; i++) res = fabs(g[i] + s[i]) > res ? fabs(g[i] + s[i]) : res;
    return (res < 1e-6 * sn) ? 0 : -1;
}

/* Degree of the barrier: <x, grad f(x)> = -nu for every block. */
static double expcone_nu(int kind) { return kind == EXPCONE_PEXP ? 2.0 : 3.0; }

static void eig3(const double *M, double *ev, double *V) {
    double A[9];
    for (int i = 0; i < 9; i++) A[i] = 0.5 * (M[i] + M[(i % 3) * 3 + i / 3]);
    dmat_eig_jacobi(3, A, ev, V);
}
/* lambda_max/lambda_min; -1 when the matrix is not positive definite. */
static double spread3(const double *ev) {
    double lo = ev[0], hi = ev[0];
    for (int k = 1; k < 3; k++) { if (ev[k] < lo) lo = ev[k]; if (ev[k] > hi) hi = ev[k]; }
    if (!(lo > 0.0) || !isfinite(lo) || !isfinite(hi)) return -1.0;
    return hi / lo;
}

/* Relative distance of x to the boundary of K: the cone's defining terms over
 * the 1-norm of x, so it is homogeneous of degree 0 (scale free) and the
 * smallest of them is where the metric starts to blow up.  Negative outside. */
static double expcone_margin(int kind, double alpha, const double *x) {
    double n = fabs(x[0]) + fabs(x[1]) + fabs(x[2]), a, b;
    if (!(n > 0.0)) return -1.0;
    if (kind == EXPCONE_PEXP) {
        double u = x[1], w;
        if (!(u > 0.0)) return -1.0;
        w = x[2] / u;
        if (!(w < 700.0)) return -1.0;
        a = (x[0] - u * exp(w)) / n;
        b = u / n;
        return b < a ? b : a;
    }
    if (kind == EXPCONE_PPOW || kind == EXPCONE_RPOW) {
        double rot = (kind == EXPCONE_RPOW) ? 2.0 : 1.0, t = x[0], u = x[1], p;
        if (!(t > 0.0) || !(u > 0.0) || !(alpha > 0.0) || !(alpha < 1.0)) return -1.0;
        p = pow(t, 2.0 * alpha) * pow(u, 2.0 * (1.0 - alpha));
        a = (rot * p - x[2] * x[2]) / (n * n);            /* degree 2 over n^2 */
        b = t / n; if (b < a) a = b;
        b = u / n; if (b < a) a = b;
        return a;
    }
    return -1.0;
}
/* Same for the dual cone K* (its argument is a dual vector). */
static double expcone_dual_margin(int kind, double alpha, const double *s) {
    double n = fabs(s[0]) + fabs(s[1]) + fabs(s[2]), a;
    if (!(n > 0.0)) return -1.0;
    if (kind == EXPCONE_PEXP) {
        double s0 = s[0], s1 = s[1], s2 = s[2];
        if (!(s0 > 0.0) || !(s2 < 0.0)) return -1.0;
        a = (s1 - s2 + s2 * log(-s2 / s0)) / n;
        s0 /= n; if (s0 < a) a = s0;
        s2 = -s2 / n; if (s2 < a) a = s2;
        return a;
    }
    if (kind == EXPCONE_PPOW || kind == EXPCONE_RPOW) {
        double s0 = s[0], s1 = s[1], lhs, rot;
        if (!(s0 > 0.0) || !(s1 > 0.0) || !(alpha > 0.0) || !(alpha < 1.0)) return -1.0;
        rot = (kind == EXPCONE_RPOW) ? sqrt(2.0) : 1.0;
        lhs = pow(s0 / alpha, alpha) * pow(s1 / (1.0 - alpha), 1.0 - alpha)
              - rot * fabs(s[2]);
        a = lhs / n;
        s0 /= n; if (s0 < a) a = s0;
        s1 /= n; if (s1 < a) a = s1;
        return a;
    }
    return -1.0;
}

/* Everything the scaling decision needs, evaluated once. */
typedef struct {
    double Hz[9], Hs[9], Mid[9], W[3], sn[3];
    double gap, condz, conds, rel, mz, ms, wn;
    int have_s;                       /* W, Hs, Mid and conds/rel are usable */
} expnt;

static int nt_eval(int kind, double alpha, const double *z, const double *s, expnt *M) {
    double Hw[9], ev[3], V[9], tmp[9], zn;
    M->have_s = 0;
    M->condz = M->conds = M->rel = M->mz = M->ms = M->wn = -1.0;
    M->gap = 0.0;                                   /* accumulated below */
    if (expcone_hess(kind, alpha, z, M->Hz)) return -1;
    for (int i = 0; i < 3; i++) M->gap += z[i] * s[i];
    M->gap /= expcone_nu(kind);
    if (!(M->gap > 0.0) || !isfinite(M->gap)) return -1;
    /* The dual side enters through its DIRECTION: s itself is proportional to
     * the running mu, and at mu = 1e-12 its own strict inequalities and the
     * residual check of expcone_dual_point stop meaning anything. */
    for (int i = 0; i < 3; i++) M->sn[i] = s[i] / M->gap;
    eig3(M->Hz, ev, V);
    M->condz = spread3(ev);
    M->mz = expcone_margin(kind, alpha, z);
    M->ms = expcone_dual_margin(kind, alpha, M->sn);
    /* A dual direction outside int K* is not a scaling CHOICE, it is the
     * situation at a tight optimum: s belongs to the boundary of K* there, so
     * the Hessian-NT chain is simply unavailable and the row degrades to the
     * barrier Hessian at z (have_s stays 0).  Only z itself being unusable is
     * fatal, and that is what the expcone_hess and gap tests above report. */
    if (!expcone_dual_in(kind, alpha, M->sn)) return 0;
    if (expcone_dual_point(kind, alpha, M->sn, M->W)) return 0;
    if (expcone_hess(kind, alpha, M->W, Hw) || mat3inv(Hw, M->Hs)) return 0;
    eig3(M->Hs, ev, V);
    M->conds = spread3(ev);
    if (!spd_pow3(M->Hs, 0.5, tmp)) {
        double S[9];
        mm3(tmp, M->Hz, M->Mid);                    /* H_s^{1/2} H_z */
        mm3(M->Mid, tmp, S);                         /* ... times H_s^{1/2} */
        for (int i = 0; i < 9; i++) M->Mid[i] = S[i];
        eig3(M->Mid, ev, V);
        M->rel = spread3(ev);
    }
    zn = fabs(z[0]) + fabs(z[1]) + fabs(z[2]);
    if (zn > 0.0) M->wn = (fabs(M->W[0]) + fabs(M->W[1]) + fabs(M->W[2])) / zn;
    M->have_s = (M->conds > 0.0 && M->rel > 0.0);
    return 0;
}

/**
 * Computes the NT metrics for an exp/power cone.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param z      [in]  Primal point (size 3).
 * @param s      [in]  Dual point (size 3).
 * @param m      [out] Array of 7 doubles receiving metrics:
 *                   [0] gap = <z,s>/nu, [1] condz, [2] conds, [3] rel,
 *                   [4] mz (primal margin), [5] ms (dual margin), [6] wn.
 *
 * @return 0 on success, -1 if metrics cannot be computed.
 *
 * @note Metrics for the scaling decision (nt_eval):
 *       gap = <z,s>/nu, condz = cond(Hessian at z), conds = cond(Hessian at W),
 *       rel = spread(H_s^{1/2} H_z H_s^{1/2}), mz/ms = primal/dual margins,
 *       wn = ||W||_1 / ||z||_1.
 *       Returns -1 if z or s not in interior, or metrics unavailable.
 */
int expcone_nt_metrics(int kind, double alpha, const double *z, const double *s, double *m) {
    expnt M; int rc = nt_eval(kind, alpha, z, s, &M);
    m[EXPCONE_EV_GAP] = M.gap;   m[EXPCONE_EV_CONDZ] = M.condz;
    m[EXPCONE_EV_CONDS] = M.conds; m[EXPCONE_EV_REL] = M.rel;
    m[EXPCONE_EV_MZ] = M.mz;     m[EXPCONE_EV_MS] = M.ms;
    m[EXPCONE_EV_WN] = M.wn;
    return rc;
}

/* The metric enters the row UNWEIGHTED.  Multiplying the row by scale gives
 *   (scale*M(z,sn)) dz + ds = -s - smu*grad f(z),   sn = s/scale,
 * and on the central path (s = -mu*grad f(z), scale = mu, sn = -grad f(z)) that
 * coefficient is exactly mu*grad^2 f(z): the metric of the CURRENT path
 * parameter, with sigma*mu only on the right, which is what the affine
 * predictor wants too -- one matrix for predictor and corrector, different
 * residuals (Mehrotra).  The centreing factor cancels: driving the target to
 * sigma*mu scales sn by 1/sigma and the metric is degree -1 in its dual
 * argument, so for M = grad^2 f(z) the product is exactly sigma-free, and for
 * the chain the cancellation is first order (its own mu-independence is what
 * sn = s/scale buys, and T78 measures it on the paired direction).  A damping
 * theta = clamp(smu/scale, 0.1, 1) -- re-weighting the matrix by sigma and
 * flooring it to keep the block invertible -- was tried in
 * place of that cancellation and is what is being removed here: the floor makes
 * the row approximate an equation that is not the one being followed, and on
 * the three degenerate cases (gp1, market_impact, the T47 CBF) the polish then
 * accepted NO point at all and the answer came from the tangent cuts.
 * Unweighted the same three reach mu = 1.4e-10, 4.0e-9 and a 4.75e-6 radial
 * KKT (better than the cuts' 5.4e-5); the only case that loses is T36, whose
 * optimum sits on the cone boundary: 6.4e-6 instead of 9.1e-11, still inside
 * the 1e-4 the case is checked to (bench/expcone_metric_grid.py). */
/**
 * Computes the scaling matrix Theta and related quantities for exp/power cones.
 *
 * @param kind   [in]  Cone type: EXPCONE_PEXP, EXPCONE_PPOW, or EXPCONE_RPOW.
 * @param alpha  [in]  Power parameter (for PPOW/RPOW only).
 * @param z      [in]  Primal point (size 3).
 * @param s      [in]  Dual point (size 3).
 * @param smu    [in]  sigma * mu (centering parameter * complementarity).
 * @param Theta  [out] Scaling matrix = Hessian^{-1} (3x3 row-major). Must not be NULL.
 * @param Thin   [out] Hessian matrix (3x3 row-major). Must not be NULL.
 * @param rt     [out] Right-hand side for the Newton system (size 3). Must not be NULL.
 * @param scale  [out] Optional pointer to receive the scale (mu).
 *
 * @return 1 if Hessian-NT chain used, 0 if barrier Hessian used, -1 on error.
 *
 * @note Computes the scaling for the NT direction:
 *       - If dual point s is in K* and scaling point W exists:
 *         Thin = M(z, s/scale) = M(W) (Hessian-NT chain, mu/sigma-free).
 *         Theta = Thin^{-1}.
 *         rt = -(s + smu*grad f(z)) / gap.
 *       - Else falls back to barrier Hessian: Thin = M(z) = grad^2 f(z).
 *
 *       The NT chain uses sn = s/scale (scale = gap = <z,s>/nu).
 *       Thin is mu- and sigma-free (depends only on the point).
 *       Verified in T78: same Thin for all mu, sigma only in rt.
 *
 * @example
 * double Theta[9], Thin[9], rt[3];
 * int nt = expcone_scaling(EXPCONE_PEXP, 0, z, s, sigma*mu, Theta, Thin, rt, &scale);
 * if (nt) printf("Used NT chain\n");
 */
int expcone_scaling(int kind, double alpha, const double *z, const double *s,
                    double smu, double *Theta, double *Thin, double *rt,
                    double *scale) {
    expnt M;
    double g[3], Hish[9], Mr[9], tmp[9];
    int nt = 0;
    if (scale) *scale = 0.0;
    if (nt_eval(kind, alpha, z, s, &M)) return -1;
    if (expcone_grad(kind, alpha, z, g)) return -1;
    if (M.have_s &&
        !spd_pow3(M.Hs, -0.5, Hish) && !spd_pow3(M.Mid, 0.5, Mr)) {
        /* The Hessian-NT chain.  Its eligibility is the whole criterion and it
         * is not a matter of taste: sn = s/scale in int K* is exactly what
         * gives the scaling point W, hence H_s = (grad^2 f(W))^{-1}, hence the
         * sandwich to square root.  A conditioning gate on top of that (drop
         * the chain when cond(H_s^{1/2} H_z H_s^{1/2}) grows) was measured and
         * removed: at every finite value the oracles PEXP(u=1,v=1),
         * PEXP(u=2,v=1) and PPOW(0.3) fell back to the tangent cuts, while the
         * chain never answered a case worse than the barrier Hessian did. */
        mm3(Hish, Mr, tmp); mm3(tmp, Hish, Thin); nt = 1;
    } else {
        for (int i = 0; i < 9; i++) Thin[i] = M.Hz[i];   /* grad^2 f(z) */
    }
    if (mat3inv(Thin, Theta)) return -1;
    for (int i = 0; i < 3; i++) rt[i] = (-s[i] - smu * g[i]) / M.gap;
    if (scale) *scale = M.gap;
    return nt;
}

/* ============================================================================
 * OPT-IN: primal-dual (Tuncel / Dahl & Andersen 2021) secant scaling.
 * Reachable only through expcone_scaling_da, which sdp.c calls only when the
 * GMB_EXP_DA env var is set.  expcone_scaling above is unchanged.
 *
 * WHICH SPACE THE MATRIX LIVES IN -- the point the earlier attempts missed.
 * Clarabel's Hs enters their KKT as  Hs dz + ds = -d_s  with THEIR z the dual
 * variable: it maps DUAL directions to PRIMAL ones and satisfies Hs z = s
 * (dual point -> primal point), approximating mu*grad^2 f*(z).  Our row is
 *   Thin dz + ds/scale = rt,   dz PRIMAL, ds DUAL,
 * i.e. H := scale*Thin maps PRIMAL -> DUAL, approximating mu*grad^2 f(z)
 * (T78 asserts Thin = grad^2 f(z) at a paired point).  So our Thin is the
 * inverse-space counterpart of their Hs, and their formula has to be
 * MIRRORED -- swap which barrier's gradient and Hessian go where -- not
 * renamed.  Plugging a conjugate-barrier Hessian into our Thin (what the
 * pure-dual-scaling attempts did) puts an (approximate) INVERSE of the right
 * matrix into the row.
 *
 * The mirrored construction.  With W the primal shadow of the dual direction
 * (-grad f(W) = sn, i.e. W = -grad f*(sn), exactly what expcone_dual_point
 * returns) and g = grad f(z), Hz = grad^2 f(z):
 *   Thin = sn sn'/<sn,z> + ds ds'/<ds,dz> + lam a a'
 *   ds = sn + g            dual   deviation from the central path
 *   dz = z - W             primal deviation from the central path
 *   a  = (z x W)/|z x W|,  lam = a' Hz a projected off span{z, W}
 * It is a rank-2 secant (BFGS-type) update of Hz: the two rank-1 terms
 * replace Hz's action on span{z, W} and enforce BOTH secant conditions
 * EXACTLY,
 *   Thin z = sn           (H z = s,                   H maps z onto s)
 *   Thin W = -g           (H (-g*(s)) = -g(z),          shadows onto shadows)
 * which are what make the primal and dual steps symmetric (the NT property,
 * here enforced instead of hoped for).  The third term keeps Hz's curvature
 * in the one remaining direction.  <ds,dz> = <g,dz> = nu*(m-1) with
 * m = -<g,W>/nu >= 1 for any LHSCB (equality exactly on the central path),
 * so the update is well defined off the path and degenerates to 0/0 on it --
 * where the gate falls back to the unchanged expcone_scaling, whose answer
 * there IS Hz (T78).  Nothing here inverts grad^2 f(W) (the step that
 * rejected 980/2640 evaluations on logistic_large with "Hs not PD") and
 * nothing takes a matrix square root; only W is needed, which
 * expcone_dual_point provides for PEXP, PPOW and RPOW alike.
 * ========================================================================= */

static double pd_dot3(const double *a, const double *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static void pd_matvec3(const double *M, const double *v, double *o) {
    for (int i = 0; i < 3; i++) o[i] = M[i*3+0]*v[0] + M[i*3+1]*v[1] + M[i*3+2]*v[2];
}

/* Primal-dual secant scaling (see the block comment above).  Same signature
 * and row contract as expcone_scaling; any point where the update is not
 * defined (on the central path, z parallel to W, sn outside int K*, a
 * non-positive curvature in the third direction) delegates to
 * expcone_scaling unchanged. */
int expcone_scaling_da(int kind, double alpha, const double *z, const double *s,
                       double smu, double *Theta, double *Thin, double *rt,
                       double *scale) {
    const double eps = 2.220446049250313e-16;
    double nu = expcone_nu(kind), gap, sn[3], g[3], Hz[9], W[3];
    gap = pd_dot3(z, s) / nu;
    if (!(gap > 0.0) || !isfinite(gap)) goto fallback;
    for (int i = 0; i < 3; i++) sn[i] = s[i] / gap;
    if (expcone_grad(kind, alpha, z, g) || expcone_hess(kind, alpha, z, Hz)) goto fallback;
    if (!expcone_dual_in(kind, alpha, sn) || expcone_dual_point(kind, alpha, sn, W)) goto fallback;
    {
        double m = -pd_dot3(g, W) / nu;
        double dz[3] = { z[0] - W[0], z[1] - W[1], z[2] - W[2] };
        double ds[3] = { sn[0] + g[0], sn[1] + g[1], sn[2] + g[2] };
        double dsz = pd_dot3(g, dz);                     /* = <ds,dz> = nu*(m-1) */
        double wp[3] = { W[0] - m*z[0], W[1] - m*z[1], W[2] - m*z[2] };
        double Hwp[3]; pd_matvec3(Hz, wp, Hwp);
        double de2 = pd_dot3(wp, Hwp);                   /* W's Hz-norm off z */
        double HW[3]; pd_matvec3(Hz, W, HW);
        double sz = pd_dot3(sn, z);                      /* = nu */
        double cz[3] = { z[1]*W[2] - z[2]*W[1], z[2]*W[0] - z[0]*W[2], z[0]*W[1] - z[1]*W[0] };
        double cn = sqrt(pd_dot3(cz, cz));
        double zn = sqrt(pd_dot3(z, z)), Wn = sqrt(pd_dot3(W, W));
        if (!(dsz / nu > sqrt(eps)) || !(de2 > eps * fabs(pd_dot3(W, HW))) ||
            !(sz > 0.0) || !(cn > eps * zn * Wn))
            goto fallback;
        double a[3] = { cz[0]/cn, cz[1]/cn, cz[2]/cn };
        double Ha[3]; pd_matvec3(Hz, a, Ha);
        double ag = pd_dot3(a, g), ah = pd_dot3(a, Hwp);
        double lam = pd_dot3(a, Ha) - ag * ag / nu - ah * ah / de2;
        if (!(lam > 0.0) || !isfinite(lam)) goto fallback;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            Thin[i*3+j] = sn[i]*sn[j]/sz + ds[i]*ds[j]/dsz + lam*a[i]*a[j];
        if (mat3inv(Thin, Theta)) goto fallback;
        for (int i = 0; i < 3; i++) rt[i] = (-s[i] - smu * g[i]) / gap;
        if (scale) *scale = gap;
        return 1;
    }
fallback:
    return expcone_scaling(kind, alpha, z, s, smu, Theta, Thin, rt, scale);
}
