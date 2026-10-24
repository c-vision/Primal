#!/usr/bin/env python3
"""The exact `lsq_pos` form of the MosekRegression repository (tschm/MosekRegression,
the code of arXiv:1310.3397):

    min ||X w - rhs||_2  (not squared: (v, X w - rhs) in Q^{N+1})
    s.t.  e'w = 1,  0 <= w <= 1

The independent check is a second solve of the QP min ||Xw - rhs||^2, whose optimum
plus ||rhs||^2 must be v*^2. The data come from the same 32-bit linear congruential
generator as the C example.

    python lsq_pos.py [n] [m]   (default 40 5)

Port of examples/c/lsq_pos.c.
"""

import math
import sys

from primalsolver import Model, BK, CT

_state = 13103397


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def solve_socp(X, rhs, N, M):
    """min v s.t. (v, Xw - rhs) in Q, e'w = 1, 0 <= w <= 1; returns (v, w)."""
    W0, R0, V = 0, M, M + N
    with Model() as t:
        t.add_vars(M + N + 1)
        t.add_cons(N + 1)
        for j in range(M):
            t.var_bounds(W0 + j, BK.RA, 0.0, 1.0)
        for i in range(N):
            t.var_bounds(R0 + i, BK.FR, -math.inf, math.inf)
        t.var_bounds(V, BK.FR, -math.inf, math.inf)
        t.cj(V, 1.0)
        for i in range(N):
            sub, val = [], []
            for j in range(M):
                xij = X[i * M + j]
                if xij != 0.0:
                    sub.append(W0 + j); val.append(-xij)
            sub.append(R0 + i); val.append(1.0)
            t.a_row(i, sub, val)
            t.con_bounds(i, BK.FX, -rhs[i], -rhs[i])
        for j in range(M):
            t.cj(W0 + j, 0.0)
        t.a_row(N, list(range(W0, W0 + M)), [1.0] * M)   # e'w = 1
        t.con_bounds(N, BK.FX, 1.0, 1.0)
        t.cone(CT.QUAD, [V] + list(range(R0, R0 + N)))   # (v, r) in Q
        r = t.solve()
    if r.rc != 0:
        return -1.0, [0.0] * M
    return r.objective, r.x[W0:W0 + M]


def solve_qp(X, rhs, N, M):
    """min ||Xw - rhs||^2 as 1/2 w'Qw + c'w with Q = 2 X'X and c = -2 X'rhs; returns (objective, w)."""
    with Model() as t:
        t.add_vars(M)
        t.add_cons(1)
        for j in range(M):
            t.var_bounds(j, BK.RA, 0.0, 1.0)
        for j in range(M):
            s = 0.0
            for i in range(N):
                s += X[i * M + j] * rhs[i]
            t.cj(j, -2.0 * s)
        for j in range(M):
            for k in range(j, M):
                s = 0.0
                for i in range(N):
                    s += X[i * M + j] * X[i * M + k]
                t.qobj([(j, k, 2.0 * s)])
        t.a_row(0, list(range(M)), [1.0] * M)       # e'w = 1
        t.con_bounds(0, BK.FX, 1.0, 1.0)
        r = t.solve()
    if r.rc != 0:
        return 0.0, [0.0] * M
    return r.objective, r.x[:M]


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if N < 1 or M < 1 or M > 60:
        return 2
    X = [(rnd() - 0.5) * 2.0 for _ in range(N * M)]
    rhs = [(rnd() - 0.5) * 2.0 for _ in range(N)]
    v, ws = solve_socp(X, rhs, N, M)
    q, _ = solve_qp(X, rhs, N, M)
    budget, lo, hi = 0.0, 0.0, 0.0
    for j in range(M):
        budget += ws[j]
        if ws[j] < lo:
            lo = ws[j]
        if ws[j] > hi:
            hi = ws[j]
    rhs2 = 0.0
    for i in range(N):
        rhs2 += rhs[i] * rhs[i]
    v2_from_qp = q + rhs2                           # v*^2 == pobj_qp + ||rhs||^2
    ok = (v >= 0.0 and abs(budget - 1.0) <= 1e-6 and lo >= -1e-6 and hi <= 1.0 + 1e-6
          and abs(v * v - v2_from_qp) <= 1e-5 * (1.0 + abs(v2_from_qp)))
    print("lsq_pos  n=%d m=%d  v*=%.8f  sqrt(QP)=%.8f  e'w=%.2e  w in [%.3f,%.3f] %s"
          % (N, M, v, math.sqrt(abs(v2_from_qp)), budget, lo, hi, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
