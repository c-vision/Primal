#!/usr/bin/env python3
"""n x n assignment problem (LP, scaling test).

    min sum_ij c_ij x_ij   s.t.  sum_j x_ij = 1,  sum_i x_ij = 1,  x >= 0

The assignment polytope is integral, so the LP optimum is a permutation. Verified:
row/column sums = 1, integrality, strong duality. The costs come from the same
32-bit linear congruential generator as the C example.

    python assignment.py [n]   (default 80)

Port of examples/c/assignment.c.
"""

import math
import sys
import time

from primalsolver import Model, BK


def csum(values, start=0.0):
    """Left-to-right sum, as the C loop (sum() is compensated since Python 3.12)."""
    s = start
    for v in values:
        s += v
    return s


_state = 987654321


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    if n < 1:
        return 2
    nvar = n * n
    c = [rnd() * 100.0 for _ in range(nvar)]
    with Model() as m:
        m.add_vars(nvar)
        m.add_cons(2 * n)
        for k in range(nvar):
            m.cj(k, c[k])
            m.var_bounds(k, BK.LO, 0.0, 1.0)
        for i in range(n):
            m.a_row(i, [i * n + j for j in range(n)], [1.0] * n)
            m.con_bounds(i, BK.FX, 1.0, 1.0)
            m.a_row(n + i, [j * n + i for j in range(n)], [1.0] * n)
            m.con_bounds(n + i, BK.FX, 1.0, 1.0)
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, x = r.objective, r.dual, r.x
        maxrow = maxcol = maxfrac = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                v = x[i * n + j]
                s += v
                maxfrac = max(maxfrac, abs(v - math.floor(v + 0.5)))
            maxrow = max(maxrow, abs(s - 1.0))
        for j in range(n):
            maxcol = max(maxcol, abs(csum(x[i * n + j] for i in range(n)) - 1.0))
        ok = maxrow < 1e-6 and maxcol < 1e-6 and maxfrac < 1e-6 and abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
    print(f"assignment n={n} vars={nvar} cons={2 * n} nnz={2 * nvar} obj={obj:.6f} dobj={dobj:.6f} "
          f"t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
