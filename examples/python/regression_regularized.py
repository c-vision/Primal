#!/usr/bin/env python3
"""Regularised regression (Section 3 of Schmelzer-Hauser-Andersen-Dahl, "Regression
techniques for Portfolio Optimisation using MOSEK", arXiv:1310.3397).

    ridge :  min ||Xw-y||^2 + lambda ||w-w0||^2
    sparse:  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|          (LASSO)
    3/2   :  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|^{3/2}

The squared residual is a quadratic objective; the penalties use the paper's cones
(t_i, p_i) in Q^2 and (t_i, 1, p_i) in POW^{2/3,1/3}. Ridge is checked against
(X'X + lambda I) w = X'y + lambda w0 (the library's dense LU), sparse and 3/2 against
cone feasibility. The data come from the same 32-bit linear congruential generator.

    python regression_regularized.py [n] [m] [lambda]   (default 30 4 0.5)

Port of examples/c/regression_regularized.c.
"""

import ctypes
import math
import sys
import time

import primalsolver as ps
from primalsolver import Model, BK, CT

_state = 27182818


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


REG_RIDGE, REG_SPARSE, REG_POW32 = 0, 1, 2


def put_qobj_residual(t, X, y, N, M):
    """Add the squared-residual QP terms Q = 2 X'X, c = -2 X'y, cfix = y'y."""
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
    for a in range(M):
        s = 0.0
        for i in range(N):
            s += X[i * M + a] * y[i]
        t.cj(a, -2.0 * s)
    yy = 0.0
    for i in range(N):
        yy += y[i] * y[i]
    t.cfix(yy)


def build(t, kind, X, y, w0, N, M, LAM):
    """Build the ridge, sparse or 3/2 regularised regression model."""
    if kind == REG_RIDGE:                           # pure QP: Q = 2(X'X + lam I), c = -2(X'y + lam w0)
        t.add_vars(M)
        for j in range(M):
            t.var_bounds(j, BK.FR, -math.inf, math.inf)
        put_qobj_residual(t, X, y, N, M)
        t.qobj([(j, j, 2.0 * LAM) for j in range(M)])   # accumulate
        for j in range(M):
            t.cj(j, t.getcj(j) - 2.0 * LAM * w0[j])
        cf = t.getcfix()
        w02 = 0.0
        for j in range(M):
            w02 += w0[j] * w0[j]
        t.cfix(cf + LAM * w02)
        return
    W0, P0, T0, ONE = 0, M, 2 * M, 3 * M            # sparse / 3-2: conic penalty on p = w - w0
    t.add_vars(3 * M if kind == REG_SPARSE else 3 * M + 1)
    t.add_cons(M)
    for j in range(M):
        t.var_bounds(W0 + j, BK.FR, -math.inf, math.inf)
    for j in range(M):
        t.var_bounds(P0 + j, BK.FR, -math.inf, math.inf)
    for j in range(M):
        t.var_bounds(T0 + j, BK.LO, 0.0, math.inf)
        t.cj(T0 + j, LAM)
    put_qobj_residual(t, X, y, N, M)
    for j in range(M):                              # p_j - w_j = -w0_j
        t.a_row(j, [W0 + j, P0 + j], [-1.0, 1.0])
        t.con_bounds(j, BK.FX, -w0[j], -w0[j])
    if kind == REG_SPARSE:
        for j in range(M):
            t.cone(CT.QUAD, [T0 + j, P0 + j])
    else:
        t.var_bounds(ONE, BK.FX, 1.0, 1.0)
        for j in range(M):
            t.cone(CT.PPOW, [T0 + j, ONE, P0 + j], 2.0 / 3.0)


def run_variant(kind, X, y, w0, N, M, LAM):
    """Build and solve one variant; returns (rc, x, seconds)."""
    with Model() as t:
        build(t, kind, X, y, w0, N, M, LAM)
        t0 = time.perf_counter()
        r = t.solve()
        return r.rc, r.x, time.perf_counter() - t0


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    if N < 1 or M < 1 or LAM < 0.0:
        return 2
    X = [(rnd() - 0.5) * 2.0 for _ in range(N * M)]
    y = [(rnd() - 0.5) * 2.0 for _ in range(N)]
    w0 = [0.0] * M
    ok = True

    rc, x, sec = run_variant(REG_RIDGE, X, y, w0, N, M, LAM)   # ridge vs (X'X + lam I) w = X'y + lam w0
    w = x[:M] if rc == 0 else [0.0] * M
    if rc != 0:
        ok = False
    G, rhs = [0.0] * (M * M), [0.0] * M
    for j in range(M):
        for k in range(M):
            s = LAM if j == k else 0.0
            for i in range(N):
                s += X[i * M + j] * X[i * M + k]
            G[j * M + k] = s
    for j in range(M):
        s = LAM * w0[j]
        for i in range(N):
            s += X[i * M + j] * y[i]
        rhs[j] = s
    if dmat_solve_lu(G, rhs, M) == 0:
        d = 0.0
        for j in range(M):
            if abs(w[j] - rhs[j]) > d:
                d = abs(w[j] - rhs[j])
        if d > 1e-5:
            ok = False
        print("ridge      n=%d m=%d lam=%.2f wdiff=%.2e t=%.4fs" % (N, M, LAM, d, sec))

    rc, x, sec = run_variant(REG_SPARSE, X, y, w0, N, M, LAM)  # sparse: (t_j, p_j) in Q^2
    worst = 0.0
    if rc == 0:
        for j in range(M):
            slack = abs(x[M + j]) - x[2 * M + j]
            if slack > worst:
                worst = slack
    else:
        ok = False
    if worst > 1e-5:
        ok = False
    print("sparse     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs" % (N, M, LAM, rc, worst, sec))

    rc, x, sec = run_variant(REG_POW32, X, y, w0, N, M, LAM)   # 3/2: t_j >= |p_j|^{3/2}
    worst = 0.0
    if rc == 0:
        for j in range(M):
            slack = math.pow(abs(x[M + j]), 1.5) - x[2 * M + j]
            if slack > worst:
                worst = slack
    else:
        ok = False
    if worst > 1e-4:
        ok = False
    print("pow3_2     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs" % (N, M, LAM, rc, worst, sec))

    print("regr_reg   n=%d m=%d lam=%.2f %s" % (N, M, LAM, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
