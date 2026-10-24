#!/usr/bin/env python3
"""Least-squares regression as a second-order cone program (Schmelzer-Hauser-Andersen-Dahl,
"Regression techniques for Portfolio Optimisation using MOSEK", arXiv:1310.3397, Section 2).

    min ||X w - y||_2  <=>  min v  s.t.  (v, Xw - y) in Q^{n+1}

Two variants: unconstrained (checked against the normal equations solved with the
library's dense LU) and the constrained portfolio form sum(w) = 1, w >= 0 (Section 2.6).
The data come from the same 32-bit linear congruential generator as the C example.

    python regression_ls.py [n] [m]   (default 60 6)

Port of examples/c/regression_ls.c.
"""

import ctypes
import math
import sys
import time

import primalsolver as ps
from primalsolver import Model, BK, CT

_state = 13103397


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


class _DMat(ctypes.Structure):
    _fields_ = [("m", ctypes.c_int), ("n", ctypes.c_int),
                ("v", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))]


def dmat_solve_lu(M, rhs, d):
    """Solve M x = rhs (d x d, row-major) with the library's dense LU (linalg.h), as
    the C sample does; rhs is overwritten with x. Returns the C return code."""
    try:
        lib = ps.LIB
        lib.dmat_new.restype = ctypes.POINTER(_DMat)
        lib.dmat_new.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.dmat_solve_lu.argtypes = [ctypes.POINTER(_DMat), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.dmat_free.argtypes = [ctypes.POINTER(_DMat)]
    except AttributeError:                          # a build that hides internal symbols
        return _solve_lu_py(M, rhs, d)
    p = lib.dmat_new(d, d)
    for i in range(d):
        for j in range(d):
            p.contents.v[i][j] = M[i * d + j]
    b = (ctypes.c_double * d)(*rhs)
    rc = lib.dmat_solve_lu(p, b, d)
    lib.dmat_free(p)
    rhs[:] = list(b)
    return rc


def _solve_lu_py(M, rhs, d):
    """The same elimination as linalg.c's dmat_solve_lu, step for step (it can differ from
    the library in the last bits: the library build contracts a*b+c to FMA)."""
    a = [M[i * d:(i + 1) * d] for i in range(d)]
    for k in range(d):
        piv, maxv = k, abs(a[k][k])
        for i in range(k + 1, d):
            if abs(a[i][k]) > maxv:
                maxv, piv = abs(a[i][k]), i
        if maxv < 1e-14:
            return 1
        if piv != k:
            a[piv], a[k] = a[k], a[piv]
            rhs[piv], rhs[k] = rhs[k], rhs[piv]   # b follows the row swap
        for i in range(k + 1, d):
            f = a[i][k] / a[k][k]
            a[i][k] = f
            for j in range(k + 1, d):
                a[i][j] -= f * a[k][j]
    for i in range(d):
        s = rhs[i]
        for j in range(i):
            s -= a[i][j] * rhs[j]
        rhs[i] = s
    for i in range(d - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, d):
            s -= a[i][j] * rhs[j]
        rhs[i] = s / a[i][i]
    return 0


def solve_ls(constrained, X, y, N, M):
    """vars w (M), r (N), v; rows r_i - (Xw)_i = -y_i; cone (v, r). Returns (rc, v, w, sec)."""
    W0, R0, V = 0, M, M + N
    with Model() as t:
        t.add_vars(M + N + 1)
        t.add_cons(N)
        for j in range(M):
            if constrained:
                t.var_bounds(W0 + j, BK.LO, 0.0, math.inf)
            else:
                t.var_bounds(W0 + j, BK.FR, -math.inf, math.inf)
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
            t.con_bounds(i, BK.FX, -y[i], -y[i])
        t.cone(CT.QUAD, [V] + list(range(R0, R0 + N)))
        if constrained:
            t.add_cons(1)
            t.a_row(N, list(range(W0, W0 + M)), [1.0] * M)
            t.con_bounds(N, BK.FX, 1.0, 1.0)
        t0 = time.perf_counter()
        r = t.solve()
        sec = time.perf_counter() - t0
    return r.rc, r.objective, r.x[W0:W0 + M], sec


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if N < 1 or M < 1:
        return 2
    X = [(rnd() - 0.5) * 2.0 for _ in range(N * M)]
    y = [(rnd() - 0.5) * 2.0 for _ in range(N)]
    ok = True
    rc, v1, w1, sec1 = solve_ls(False, X, y, N, M)  # unconstrained LS
    if rc != 0:
        ok, v1 = False, 0.0
    G, rhs = [0.0] * (M * M), [0.0] * M             # normal-equations reference
    for j in range(M):
        for k in range(M):
            s = 0.0
            for i in range(N):
                s += X[i * M + j] * X[i * M + k]
            G[j * M + k] = s
    for j in range(M):
        s = 0.0
        for i in range(N):
            s += X[i * M + j] * y[i]
        rhs[j] = s
    if dmat_solve_lu(G, rhs, M) == 0:
        if max(abs(w1[j] - rhs[j]) for j in range(M)) > 1e-6:
            ok = False
    rc2, v2, w2, sec2 = solve_ls(True, X, y, N, M)  # constrained (portfolio) LS
    budget, minw = 0.0, 0.0
    if rc2 == 0:
        for j in range(M):
            budget += w2[j]
            if w2[j] < minw:
                minw = w2[j]
    else:
        ok, v2 = False, 0.0
    if ok and (abs(budget - 1.0) > 1e-6 or minw < -1e-6):
        ok = False
    if ok and v2 < v1 - 1e-6:                       # constraints cannot lower the residual
        ok = False
    print("regr_ls    n=%d m=%d uncons=%.6f cons=%.6f t=%.4f/%.4fs %s" % (N, M, v1, v2, sec1, sec2, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
