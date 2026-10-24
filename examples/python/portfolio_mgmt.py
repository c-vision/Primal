#!/usr/bin/env python3
"""Portfolio management as conic programs (Section 4 of Schmelzer-Hauser-Andersen-Dahl,
"Regression techniques for Portfolio Optimisation using MOSEK", arXiv:1310.3397):

  min variance :  min ||Xw||^2                 s.t. e'w = 1, w >= 0
  tracking     :  min ||Xw - rM||^2            s.t. e'w = 1, w >= 0
  max return   :  max mu'w  s.t. (sigma, Xw) in Q, e'w = 1, w >= 0
  130/30       :  max mu'w  s.t. sum t_i <= 1.6, (t_i, w_i) in Q^2, e'w = 1
  robust       :  max mu0'w - u  s.t. (u, A w) in Q, e'w = 1, w >= 0

Each is solved and verified (budget, bounds, cone feasibility). The data come from
the same 32-bit linear congruential generator as the C example.

    python portfolio_mgmt.py [n] [m]   (default 40 6)

Port of examples/c/portfolio_mgmt.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, CT, SENSE

_state = 314159


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def timed_solve(build):
    """Build, time and solve one model; returns (Result, seconds)."""
    with Model() as t:
        build(t)
        t0 = time.perf_counter()
        r = t.solve()
        return r, time.perf_counter() - t0


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if N < 1 or M < 1:
        return 2
    X = [(rnd() - 0.5) * 0.1 for _ in range(N * M)]
    rM = [(rnd() - 0.5) * 0.08 for _ in range(N)]
    mu = [0.001 + rnd() * 0.02 for _ in range(M)]
    ok = True

    def budget(x):
        s = 0.0
        for j in range(M):
            s += x[j]
        return s

    def build_ls(t, tracking):
        """min ||Xw||^2 (or ||Xw - rM||^2) via QP."""
        t.add_vars(M)
        t.add_cons(1)
        for j in range(M):
            t.var_bounds(j, BK.LO, 0.0, 1.0)
        terms = []
        for a in range(M):
            for b in range(a, M):
                s = 0.0
                for i in range(N):
                    s += X[i * M + a] * X[i * M + b]
                s *= 2.0
                if s != 0.0:
                    terms.append((a, b, s))
        t.qobj(terms)
        cf = 0.0
        for a in range(M):
            c = 0.0
            for i in range(N):
                c += X[i * M + a] * (rM[i] if tracking else 0.0)
            t.cj(a, -2.0 * c)
        if tracking:
            for i in range(N):
                cf += rM[i] * rM[i]
        t.cfix(cf)
        t.a_row(0, list(range(M)), [1.0] * M)
        t.con_bounds(0, BK.FX, 1.0, 1.0)

    r, sec = timed_solve(lambda t: build_ls(t, False))   # min variance
    bud = budget(r.x)
    if r.rc != 0 or abs(bud - 1.0) > 1e-6:
        ok = False
    print("minvar     n=%d m=%d budget=%.6f t=%.4fs" % (N, M, bud, sec))

    r, sec = timed_solve(lambda t: build_ls(t, True))    # tracking
    bud = budget(r.x)
    if r.rc != 0 or abs(bud - 1.0) > 1e-6:
        ok = False
    print("tracking   n=%d m=%d budget=%.6f t=%.4fs" % (N, M, bud, sec))

    sigma = 0.5                                     # max return with risk cone (sigma, Xw) in Q
    W0, R0, TC = 0, M, M + N

    def build_maxret(t):
        t.add_vars(M + N + 1)
        t.add_cons(N + 1)
        t.sense(SENSE.MAX)
        for j in range(M):
            t.cj(W0 + j, mu[j])
            t.var_bounds(W0 + j, BK.LO, 0.0, 1.0)
        for i in range(N):
            t.var_bounds(R0 + i, BK.FR, -math.inf, math.inf)
        t.var_bounds(TC, BK.FX, sigma, sigma)
        for i in range(N):
            sub, val = [], []
            for j in range(M):
                if X[i * M + j] != 0.0:
                    sub.append(W0 + j); val.append(-X[i * M + j])
            sub.append(R0 + i); val.append(1.0)
            t.a_row(i, sub, val)
            t.con_bounds(i, BK.FX, 0.0, 0.0)
        t.a_row(N, list(range(W0, W0 + M)), [1.0] * M)   # budget
        t.con_bounds(N, BK.FX, 1.0, 1.0)
        t.cone(CT.QUAD, [TC] + list(range(R0, R0 + N)))

    r, sec = timed_solve(build_maxret)
    risk, bud = 0.0, 0.0
    if r.rc == 0:
        for i in range(N):
            risk += r.x[R0 + i] * r.x[R0 + i]
        bud = budget(r.x)
        risk = math.sqrt(risk)
    else:
        ok = False
    if risk > sigma * (1.0 + 1e-4) or abs(bud - 1.0) > 1e-6:
        ok = False
    print("maxreturn  n=%d m=%d risk=%.6f budget=%.6f t=%.4fs" % (N, M, risk, bud, sec))

    T0 = M                                          # 130/30: (t_i, w_i) in Q^2, sum t <= 1.6, sum w = 1

    def build_130_30(t):
        t.add_vars(2 * M)
        t.add_cons(2)
        t.sense(SENSE.MAX)
        for j in range(M):
            t.cj(W0 + j, mu[j])
            t.var_bounds(W0 + j, BK.FR, -math.inf, math.inf)
            t.var_bounds(T0 + j, BK.LO, 0.0, math.inf)
            t.cone(CT.QUAD, [T0 + j, W0 + j])
        t.a_row(0, list(range(T0, T0 + M)), [1.0] * M)   # sum t <= 1.6
        t.con_bounds(0, BK.UP, -math.inf, 1.6)
        t.a_row(1, list(range(W0, W0 + M)), [1.0] * M)   # sum w = 1
        t.con_bounds(1, BK.FX, 1.0, 1.0)

    r, sec = timed_solve(build_130_30)
    lev, bud = 0.0, 0.0
    if r.rc == 0:
        for j in range(M):
            lev += r.x[T0 + j]; bud += r.x[W0 + j]
    else:
        ok = False
    if lev > 1.6 + 1e-4 or abs(bud - 1.0) > 1e-6:
        ok = False
    print("130_30     n=%d m=%d leverage=%.6f budget=%.6f t=%.4fs" % (N, M, lev, bud, sec))

    Z0, UV = M, 2 * M                               # robust: max mu0'w - u, (u, A w) in Q

    def build_robust(t):
        t.add_vars(2 * M + 1)
        t.add_cons(M + 1)
        t.sense(SENSE.MAX)
        for j in range(M):
            t.cj(W0 + j, mu[j])
            t.var_bounds(W0 + j, BK.LO, 0.0, 1.0)
        for j in range(M):
            t.var_bounds(Z0 + j, BK.FR, -math.inf, math.inf)
        t.var_bounds(UV, BK.LO, 0.0, math.inf)
        t.cj(UV, -1.0)
        for i in range(M):                          # z_i - (A w)_i = 0
            sub, val = [], []
            for j in range(M):
                a = 0.01 if i == j else (0.002 if j == (i + 1) % M else 0.0)
                if a != 0.0:
                    sub.append(W0 + j); val.append(-a)
            sub.append(Z0 + i); val.append(1.0)
            t.a_row(i, sub, val)
            t.con_bounds(i, BK.FX, 0.0, 0.0)
        t.a_row(M, list(range(W0, W0 + M)), [1.0] * M)   # budget
        t.con_bounds(M, BK.FX, 1.0, 1.0)
        t.cone(CT.QUAD, [UV] + list(range(Z0, Z0 + M)))

    r, sec = timed_solve(build_robust)
    u, normz, bud = 0.0, 0.0, 0.0
    if r.rc == 0:
        u = r.x[UV]
        for j in range(M):
            normz += r.x[Z0 + j] * r.x[Z0 + j]; bud += r.x[W0 + j]
        normz = math.sqrt(normz)
    else:
        ok = False
    if u < normz - 1e-5 or abs(bud - 1.0) > 1e-6:
        ok = False
    print("robust     n=%d m=%d u=%.6f |Aw|=%.6f budget=%.6f t=%.4fs" % (N, M, u, normz, bud, sec))

    print("portfolio  n=%d m=%d %s" % (N, M, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
