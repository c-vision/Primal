#!/usr/bin/env python3
"""Large sparse convex QP (scaling test for the sparse normal-equations QP interior point).

    min 1/2 x'Qx + c'x   s.t.  A x = b,  x >= 0

Q is a sparse PD matrix (1-D graph Laplacian + 0.5*I, tridiagonal), A has m rows and
b = A*x_ref for a planted feasible x_ref >= 0, so the problem is feasible and strictly
convex. For large n with sparse Q and moderate m this routes to the sparse QP IPM.
Verified: primal feasibility and strong duality. The data come from the same 32-bit
linear congruential generator as the C example.

    python qp_sparse.py [n] [m]   (default 2000 8)

Port of examples/c/qp_sparse.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 13579


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if n < 2 or m_rows < 1:
        return 2
    # Q = 1-D Laplacian + 0.5 I (tridiagonal, PD); lower-triangle triplets
    q = [(j, j, (1.0 if j in (0, n - 1) else 2.0) + 0.5) for j in range(n)]
    q += [(j + 1, j, -1.0) for j in range(n - 1)]
    xref, c = [], []
    for _ in range(n):
        xref.append(rnd() * 5.0 + 0.1 if rnd() < 0.7 else 0.0)   # ~30% at the bound
        c.append(rnd() * 4.0 - 2.0)
    a = [[1.0 if i == 0 else rnd() * 2.0 - 1.0 for _ in range(n)] for i in range(m_rows)]
    b = []
    for i in range(m_rows):
        s = 0.0
        for j in range(n):
            s += a[i][j] * xref[j]
        b.append(s)
    with Model() as m:
        m.add_vars(n)
        m.add_cons(m_rows)
        for j in range(n):
            m.cj(j, c[j])
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.qobj(q)
        cols = list(range(n))
        for i in range(m_rows):
            m.a_row(i, cols, a[i])
            m.con_bounds(i, BK.FX, b[i], b[i])
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, x = r.objective, r.dual, r.x
        maxviol = 0.0
        for i in range(m_rows):
            s = 0.0
            for j in range(n):
                s += a[i][j] * x[j]
            maxviol = max(maxviol, abs(s - b[i]))
        minx = min(0.0, min(x[:n]))
        bn = max([1.0] + [abs(v) for v in b])
        ok = maxviol < 1e-4 * bn and minx > -1e-6 and abs(obj - dobj) < 1e-4 * (1.0 + abs(obj))
    print(f"qp_sparse  n={n} m={m_rows} nnzQ={len(q)} obj={obj:.6f} dobj={dobj:.6f} t={sec:.4f}s "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
