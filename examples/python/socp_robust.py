#!/usr/bin/env python3
"""Least-squares regression as a second-order cone program (scaling test for the conic path).

    min ||A x - b||_2  <=>  min t  s.t.  (t, r) in QUAD,  r = A x - b

The optimum t* = ||A x* - b|| is cross-checked against the normal equations
A'A x = A'b, solved with the library's dense LU (as the C sample). The data come from
the same 32-bit linear congruential generator as the C example.

    python socp_robust.py [m] [d]   (default 300 40)

Port of examples/c/socp_robust.c.
"""

import ctypes
import math
import sys
import time

import primalsolver as ps
from primalsolver import Model, BK, CT

_state = 16180339


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


def main() -> int:
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if m < 1 or d < 1:
        return 2
    A = [rnd() * 2.0 - 1.0 for _ in range(m * d)]   # row-major, as the C example
    b = [rnd() * 2.0 - 1.0 for _ in range(m)]
    X0, R0, T = 0, d, d + m                         # vars: x (d), r (m), t (1)
    with Model() as t:
        t.add_vars(d + m + 1)
        for j in range(d):
            t.var_bounds(X0 + j, BK.FR, -math.inf, math.inf)
        for i in range(m):
            t.var_bounds(R0 + i, BK.FR, -math.inf, math.inf)
        t.var_bounds(T, BK.LO, 0.0, math.inf)
        t.cj(T, 1.0)
        t.add_cons(m)
        for i in range(m):                          # r_i - A_i x = -b_i
            t.a_row(i, list(range(X0, X0 + d)) + [R0 + i], [-v for v in A[i * d:i * d + d]] + [1.0])
            t.con_bounds(i, BK.FX, -b[i], -b[i])
        t.cone(CT.QUAD, [T] + list(range(R0, R0 + m)))
        t0 = time.perf_counter()
        r = t.solve()
        sec = time.perf_counter() - t0
    ok = r.rc == 0
    if ok:
        obj = r.objective
        M = [0.0] * (d * d)                         # reference: A'A x = A'b with dense LU
        rhs = [0.0] * d
        for i in range(d):
            for j in range(d):
                s = 0.0
                for k in range(m):
                    s += A[k * d + i] * A[k * d + j]
                M[i * d + j] = s
        for i in range(d):
            s = 0.0
            for k in range(m):
                s += A[k * d + i] * b[k]
            rhs[i] = s
        tref = 0.0
        if dmat_solve_lu(M, rhs, d) == 0:
            ss = 0.0
            for k in range(m):
                res = -b[k]
                for j in range(d):
                    res += A[k * d + j] * rhs[j]
                ss += res * res
            tref = math.sqrt(ss)
        ok = abs(obj - tref) < 1e-4 * (1.0 + abs(tref))
        print("socp_ls    m=%d d=%d cone=%d obj=%.6f ref=%.6f t=%.4fs %s"
              % (m, d, m + 1, obj, tref, sec, "OK" if ok else "FAIL"))
    else:
        print("socp_ls    m=%d d=%d rc=%d FAIL" % (m, d, r.rc))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
