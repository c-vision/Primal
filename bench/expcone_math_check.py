#!/usr/bin/env python3
#
# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  A copy of the License
# is in the repository root (LICENSE) and at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Estratto della licenza (Apache License 2.0, §2 "Grant of Copyright License"):
#   "Subject to the terms and conditions of this License, each Contributor
#    hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
#    royalty-free, irrevocable copyright license to reproduce, prepare
#    Derivative Works of, publicly display, publicly perform, sublicense, and
#    distribute the Work and such Derivative Works in Source or Object form."
# 
# Esonero di responsabilita' e assenza di garanzia (Apache License 2.0, §7-§8):
#   [§7] Unless required by applicable law or agreed to in writing, Licensor
#   provides the Work (and each Contributor provides its Contributions) on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#   or implied, including, without limitation, any warranties or conditions of
#   TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
#   PURPOSE.  You are solely responsible for determining the appropriateness of
#   using or redistributing the Work.
#   [§8] In no event and under no legal theory, whether in tort (including
#   negligence), contract, or otherwise, unless required by applicable law or
#   agreed to in writing, shall any Contributor be liable to You for damages,
#   including any direct, indirect, special, incidental, or consequential
#   damages arising as a result of this License or out of the use or inability
#   to use the Work.  This software is provided without any guarantee that it
#   will operate correctly or be free of defects.

"""Derivation check for the exponential/power cone DUAL math used by
expcone.c: membership of int K*, the scaling point W (solution of
-grad f(W) = s) and the Fenchel consistency of the barrier gradients.

Run:  python3 bench/expcone_math_check.py     (exit 1 on any failure)

Cone conventions (primal members x = (t,u,v)):
  PEXP: t >= u*exp(v/u), u > 0            f  = -log(G) - log(u),  G = t - u e^{v/u}
  PPOW: t^a u^(1-a) >= |v|                f  = -log(D) - (1-a)log t - a log u
  RPOW: sqrt2 t^a u^(1-a) >= |v|              D = rot t^{2a} u^{2(1-a)} - v^2, rot=1/2

Derived dual cones (s = (s0,s1,s2) paired with (t,u,v)):
  PEXP*: s0 > 0, s2 < 0, s1 - s2 + s2*log(-s2/s0) > 0
  PPOW*: s0 > 0, s1 > 0, (s0/a)^a (s1/(1-a))^(1-a) >  |s2|
  RPOW*: s0 > 0, s1 > 0, (s0/a)^a (s1/(1-a))^(1-a) > sqrt2 |s2|
"""
import math
import random

ROT = {"PPOW": 1.0, "RPOW": 2.0}


def barrier(kind, a, x):
    t, u, v = x
    if kind == "PEXP":
        if u <= 0.0:
            return None
        g = t - u * math.exp(v / u)
        return None if g <= 0.0 else -math.log(g) - math.log(u)
    if t <= 0.0 or u <= 0.0:
        return None
    d = ROT[kind] * t ** (2 * a) * u ** (2 * (1 - a)) - v * v
    return None if d <= 0.0 else -math.log(d) - (1 - a) * math.log(t) - a * math.log(u)


def grad_fd(kind, a, x, h=1e-6):
    """finite-difference gradient, independent of the closed form"""
    g = []
    for i in range(3):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        fp, fm = barrier(kind, a, xp), barrier(kind, a, xm)
        assert fp is not None and fm is not None, "FD point left the cone"
        g.append((fp - fm) / (2 * h))
    return g


def grad(kind, a, x):
    """closed-form gradient of the primal barrier (mirrors expcone.c)"""
    t, u, v = x
    if kind == "PEXP":
        w = v / u
        ew = math.exp(w)
        G = t - u * ew
        gu = ew * (w - 1.0)
        gv = -ew
        return [-1.0 / G, -gu / G - 1.0 / u, -gv / G]
    rot = ROT[kind]
    p = t ** (2 * a) * u ** (2 * (1 - a))
    D = rot * p - v * v
    dt = rot * 2 * a * p / t
    du = rot * 2 * (1 - a) * p / u
    return [-dt / D - (1 - a) / t, -du / D - a / u, 2 * v / D]


def dual_in(kind, a, s):
    """1 if s is strictly inside the dual cone"""
    s0, s1, s2 = s
    if kind == "PEXP":
        if not (s0 > 0.0 and s2 < 0.0):
            return False
        return s1 - s2 + s2 * math.log(-s2 / s0) > 0.0
    if not (s0 > 0.0 and s1 > 0.0):
        return False
    lhs = (s0 / a) ** a * (s1 / (1 - a)) ** (1 - a)
    rhs = (math.sqrt(2.0) if kind == "RPOW" else 1.0) * abs(s2)
    return lhs > rhs


def dual_point(kind, a, s):
    """W with -grad f(W) = s (the NT scaling point)"""
    s0, s1, s2 = s
    if kind == "PEXP":
        w = math.log(-s2 / s0)
        q = s1 - s2 + s2 * w
        return [1.0 / s0 + math.exp(w) / q, 1.0 / q, w / q]
    if s2 == 0.0:
        return [(1 + a) / s0, (2 - a) / s1, 0.0]
    rot = ROT[kind]

    def gap(xi):  # log(p from t,u) - log(4 xi (xi-1) / (rot s2^2)); decreasing root
        t = (2 * a * xi + 1 - a) / s0
        u = (2 * (1 - a) * xi + a) / s1
        lhs = 4.0 * xi * (xi - 1.0) / (rot * s2 * s2)
        return math.log(t ** (2 * a) * u ** (2 * (1 - a))) - math.log(lhs)

    lo = 1.0 + 1e-13
    assert gap(lo) > 0.0, "no sign change at xi=1"
    hi = 2.0
    while gap(hi) > 0.0:
        hi *= 2.0
        assert hi < 1e18, "no bracket"
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gap(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    xi = math.sqrt(lo * hi)
    t = (2 * a * xi + 1 - a) / s0
    u = (2 * (1 - a) * xi + a) / s1
    p = t ** (2 * a) * u ** (2 * (1 - a))
    D = rot * p / xi
    return [t, u, -s2 * D / 2.0]


def relerr(p, q):
    nq = max(abs(v) for v in q)
    return max(abs(p[i] - q[i]) for i in range(3)) / max(1.0, nq)


def sample(kind, a, rnd):
    if kind == "PEXP":
        return [rnd.uniform(0.5, 8), rnd.uniform(0.2, 4), rnd.uniform(-4, 4)]
    return [rnd.uniform(0.2, 6), rnd.uniform(0.2, 6), rnd.uniform(-3, 3)]


def main():
    rnd = random.Random(20260913)
    kinds = [("PEXP", 0.0), ("PPOW", 0.3), ("PPOW", 0.5), ("PPOW", 0.85),
             ("RPOW", 0.4), ("RPOW", 0.5), ("RPOW", 0.9)]
    stat = {}
    for kind, a in kinds:
        wg = wi = wm = 0.0
        n = 0
        for _ in range(300):
            z = sample(kind, a, rnd)
            if barrier(kind, a, z) is None:
                continue
            n += 1
            g = grad(kind, a, z)
            wg = max(wg, relerr(g, grad_fd(kind, a, z)))
            s = [-v for v in g]
            assert dual_in(kind, a, s), f"-grad f(z) outside K*: {kind} {a} {z} {s}"
            W = dual_point(kind, a, s)
            wi = max(wi, relerr(W, z))
            assert barrier(kind, a, W) is not None, f"W left K: {kind} {a} {s} {W}"
            wm = max(wm, relerr([-v for v in grad(kind, a, W)], s))
        stat[(kind, a)] = (n, wg, wi, wm)
        print(f"{kind}(a={a}): n={n} grad-vs-FD={wg:.2e} involution={wi:.2e} map={wm:.2e}")
    wg = max(v[1] for v in stat.values())
    wi = max(v[2] for v in stat.values())
    wm = max(v[3] for v in stat.values())
    assert wg < 1e-4 and wi < 1e-8 and wm < 1e-8

    # boundary / exterior points must be rejected
    rejected = [("PEXP", 0.0, [1.0, 0.0, 1.0]), ("PEXP", 0.0, [-1.0, 0.0, -1.0]),
                ("PEXP", 0.0, [1.0, -5.0, -1.0]), ("PEXP", 0.0, [1.0, 5.0, 0.0]),
                ("PPOW", 0.3, [1.0, 1.0, 3.0]), ("PPOW", 0.3, [-1.0, 1.0, 0.0]),
                ("RPOW", 0.4, [1.0, 0.2, 0.9]), ("RPOW", 0.5, [0.1, 0.1, 0.5])]
    for kind, a, s in rejected:
        assert not dual_in(kind, a, s), f"accepted exterior point {kind} {a} {s}"

    # reference values of the dual cone (oracle for T80)
    assert relerr(dual_point("PEXP", 0.0, [1.0, 0.0, -1.0]), [2.0, 1.0, 0.0]) < 1e-12
    assert relerr([-v for v in grad("PEXP", 0.0, [2.0, 1.0, 0.0])], [1.0, 0.0, -1.0]) < 1e-12
    # RPOW(1/2) is the rotated quadratic cone: self-dual, 2 s0 s1 >= s2^2
    for s0, s1, s2 in [(2.0, 3.0, 1.0), (1.0, 1.0, 1.4), (5.0, 0.3, 2.0)]:
        inside = 2 * s0 * s1 > s2 * s2
        assert dual_in("RPOW", 0.5, [s0, s1, s2]) == inside, (s0, s1, s2)

    # ---- reference numbers for the C test T80 -----------------------------
    # (the Hessian-NT invariant Theta = grad^2 f(z) at the paired point is
    #  checked in C by T80, where the real closed-form Hessians live)
    print("\nreference values (T80):")
    for kind, a, z in [("PEXP", 0.0, [2.0, 1.0, 0.0]),
                       ("PEXP", 0.0, [5.0, 2.0, 1.0]),
                       ("PPOW", 0.3, [2.0, 1.0, 0.5]),
                       ("RPOW", 0.4, [1.5, 2.0, -0.7])]:
        s = [-v for v in grad(kind, a, z)]
        W = dual_point(kind, a, s)
        print(f"  {kind}(a={a}) z={z} -> s={s} inK*={dual_in(kind,a,s)} W={W}")
    print("OK: dual membership, scaling point and Fenchel consistency verified")


if __name__ == "__main__":
    main()
