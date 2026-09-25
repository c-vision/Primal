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

/* expcone.h - barrier, gradient and Hessian of the exponential and power
 * cones, for the non-symmetric-cone interior point.
 *
 * These cones are NOT symmetric: they have no Jordan product, so the NT
 * scaling Q(w) / arrow matrices used for R_+/SOC/PSD do not apply.  The
 * interior point needs a barrier f and its derivatives instead (Skajaa-Ye /
 * Nesterov scaling).
 *
 * Conventions, matching the cone members (t,u,v) = (m0,m1,m2) in primal.c:
 *   PEXP: t >= u*exp(v/u), u >= 0
 *         f = -log(t - u e^{v/u}) - log(u)
 *   PPOW: t^a u^(1-a) >= |v|, t,u >= 0
 *         f = -log(t^{2a} u^{2(1-a)} - v^2) - (1-a) log(t) - a log(u)
 *   RPOW: sqrt2 t^a u^(1-a) >= |v|, t,u >= 0
 *         f = -log(2 t^{2a} u^{2(1-a)} - v^2) - (1-a) log(t) - a log(u)
 *
 * All routines return 0 on success and -1 when x is not in the cone interior
 * (or a parameter is invalid); the caller can then treat the point as
 * infeasible / shorten the step.
 */
#ifndef EXPCONE_H
#define EXPCONE_H

#define EXPCONE_PEXP 0
#define EXPCONE_PPOW 1
#define EXPCONE_RPOW 2

/* barrier value f(x). */
int expcone_barrier(int kind, double alpha, const double *x, double *f);
/* gradient g = grad f(x), length 3. */
int expcone_grad(int kind, double alpha, const double *x, double *g);
/* Hessian H = grad^2 f(x), 3x3 row-major. */
int expcone_hess(int kind, double alpha, const double *x, double *H);

/* These cones are not self-dual, so the dual variable lives in K*, which is
 * again an exp/power cone in closed form (s paired with the members (t,u,v)):
 *   PEXP*: s0 > 0, s2 < 0, s1 - s2 + s2 log(-s2/s0) > 0
 *   PPOW*: s0 > 0, s1 > 0, (s0/a)^a (s1/(1-a))^(1-a) >    |s2|
 *   RPOW*: s0 > 0, s1 > 0, (s0/a)^a (s1/(1-a))^(1-a) > sqrt2 |s2|
 * Returns 1 when s is strictly inside K*, 0 otherwise (invalid kind too). */
int expcone_dual_in(int kind, double alpha, const double *s);
/* Scaling point: the unique W in int K with -grad f(W) = s (the gradient map
 * of a logarithmically homogeneous barrier is a bijection K -> int K*).
 * Returns 0 on success, -1 when s is not in int K*. */
int expcone_dual_point(int kind, double alpha, const double *s, double *W);

/* ---- metric grid for the scaling decision ---------------------------------
 * (z,s) carries an arbitrary mu and the cones are not self-dual, so only
 * ratios of same-degree terms can be compared between problems.  m[] slots:
 *   EXPCONE_EV_GAP  <z,s>/nu            the mu scale (NOT dimensionless)
 *   EXPCONE_EV_CONDZ cond(grad^2 f(z))  >= 1
 *   EXPCONE_EV_CONDS cond(grad^2 f_*(sn)) with sn = s/gap, -1 when no W
 *   EXPCONE_EV_REL   cond(H_s^{1/2} H_z H_s^{1/2})  = 1 exactly at a paired
 *                    point (s = -gap grad f(z)), so it measures how far the
 *                    primal and dual metrics are MISALIGNED - it is also the
 *                    condition number of the matrix the NT square root has to
 *                    take, hence the reliability of the NT chain itself
 *   EXPCONE_EV_MZ    relative distance of z  to the boundary of K
 *   EXPCONE_EV_MS    relative distance of sn to the boundary of K*
 *   EXPCONE_EV_WN    |W|/|z| (kept as evidence: it tracks 1/gap, so it does
 *                    NOT separate the boundary cases from the regular ones)
 * Both margins are degree 0 in their argument (defining function of the cone
 * over the 1-norm), negative outside, and comparable with each other because
 * they are both "how much room is left in the cone", not "where the iterate
 * sits".  Returns -1 when z is not in int K or gap is not positive; slots that
 * cannot be evaluated are set to -1. */
#define EXPCONE_EV_N     7
#define EXPCONE_EV_GAP   0
#define EXPCONE_EV_CONDZ 1
#define EXPCONE_EV_CONDS 2
#define EXPCONE_EV_REL   3
#define EXPCONE_EV_MZ    4
#define EXPCONE_EV_MS    5
#define EXPCONE_EV_WN    6
int expcone_nt_metrics(int kind, double alpha, const double *z, const double *s,
                       double *m);

/* Primal-dual (Hessian-NT) scaling for a cone block, used by the interior
 * point.  The cone row the caller assembles is
 *   Thin dz + ds/scale = rt,
 *   Thin  = M(z,sn),   sn = s/scale,   scale = <z,s>/nu,
 *   rt    = (-s - smu*grad f(z)) / scale,        smu = sigma*mu,
 * M being either the barrier Hessian H_z = grad^2 f(z) or the Hessian-NT chain
 * built from H_s = grad^2 f_*(sn) = (grad^2 f(W))^{-1}:
 *   chain = H_s^{-1/2} (H_s^{1/2} H_z H_s^{1/2})^{1/2} H_s^{-1/2}.
 * These cones are not self-dual and the pairing shrinks with mu
 * (s = -mu*grad f(z) on the path, <z,s> = nu*mu), so the raw Newton metric
 * smu*grad^2 f(z) would enter the KKT system mu orders below its own coupling
 * entries.  Two devices remove exactly that, and nothing more:
 *  - M is evaluated at the mu-free dual DIRECTION sn = s/scale (-> -grad f(z)
 *    on the central path), so M depends on s only through that direction:
 *    replacing s by lambda*s leaves Thin unchanged and multiplies scale by
 *    lambda, and rt is invariant too when smu is scaled along with s (the
 *    central-path case, which is what T78 checks);
 *  - the whole 3-row block is divided by scale, a left row scaling that leaves
 *    (dy, dz, ds) unchanged but lifts the block to O(1).
 * The metric therefore carries NO weight: Thin is a function of the iterate
 * alone, independent of sigma and of mu, and sigma appears only in rt.  That is
 * the Mehrotra splitting - predictor and corrector share one matrix built at
 * the current mu and differ in their residuals - and multiplying the row by
 * scale shows it: the dz coefficient is scale*M(z,sn) = mu*grad^2 f(z) at a
 * paired point, where T78/T80 assert Thin = grad^2 f(z) exactly.  A damping
 * theta = clamp(smu/scale, 0.1, 1) on top of this was measured and dropped
 * (see the comment on the definition).
 * When sn is not in int K* no W exists, the chain is unavailable and M = H_z.
 * Returns the branch used: 1 = NT chain, 0 = barrier Hessian, -1 when z is not
 * in int K or s is not in int K*; *scale is 0.0 on failure. */
int expcone_scaling(int kind, double alpha, const double *z, const double *s,
                    double smu, double *Theta, double *Thin, double *rt,
                    double *scale);

/* ALTERNATE scaling: the primal-dual secant (Tuncel / Dahl & Andersen)
 * scaling, used by sdp.c's SECANT mode only -- the second native attempt on
 * an exp/power model the first run did not solve (or every run, under
 * GMB_EXP_DA).  Clarabel's Hs maps dual -> primal, our Thin primal -> dual,
 * so the construction is MIRRORED, not renamed: Thin is a rank-2 secant
 * update of grad^2 f(z) with Thin z = sn and Thin W = -grad f(z) exact
 * (W = -grad f*(sn)), plus Hz's curvature on the third axis.  Built from
 * our own barriers for PEXP, PPOW and RPOW alike; same signature and row
 * contract as expcone_scaling, to which every point where the update is
 * not defined delegates unchanged -- see expcone.c. */
int expcone_scaling_da(int kind, double alpha, const double *z, const double *s,
                       double smu, double *Theta, double *Thin, double *rt,
                       double *scale);

#endif /* EXPCONE_H */
