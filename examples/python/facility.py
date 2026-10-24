#!/usr/bin/env python3
"""Capacitated facility location MILP (scaling test).

    min sum_j f_j y_j + sum_ij c_ij x_ij
      s.t. sum_j x_ij = d_i          (each client served)
           sum_i x_ij <= q_j y_j     (capacity if open)
           y_j in {0,1}, x_ij >= 0

Verified: demands met, capacities respected, y integral, and the MIP objective is
>= the LP relaxation bound (both computed here). The data come from the same 32-bit
linear congruential generator as the C example.

    python facility.py [m] [n]   (default 15 40)

Port of examples/c/facility.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, VAR_TYPE


def csum(values, start=0.0):
    """Left-to-right sum, as the C loop (sum() is compensated since Python 3.12)."""
    s = start
    for v in values:
        s += v
    return s


_state = 55555


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def build(mdl, m, n, integral, f, q, d, c):
    """Build the facility location model (integral=True for the MIP, False for the LP)."""
    mdl.add_vars(m + m * n)                         # y_0..y_{m-1}, then x_ij at m + i*m + j
    mdl.add_cons(n + m)
    for j in range(m):
        mdl.cj(j, f[j])
        mdl.var_bounds(j, BK.RA, 0.0, 1.0)
        if integral:
            mdl.var_type(j, VAR_TYPE.INT_BIN)
    for i in range(n):
        for j in range(m):
            k = m + i * m + j
            mdl.cj(k, c[i * m + j])
            mdl.var_bounds(k, BK.LO, 0.0, math.inf)
    for i in range(n):                              # sum_j x_ij = d_i
        mdl.a_row(i, [m + i * m + j for j in range(m)], [1.0] * m)
        mdl.con_bounds(i, BK.FX, d[i], d[i])
    for j in range(m):                              # sum_i x_ij - q_j y_j <= 0
        mdl.a_row(n + j, [m + i * m + j for i in range(n)] + [j], [1.0] * n + [-q[j]])
        mdl.con_bounds(n + j, BK.UP, -math.inf, 0.0)


def main() -> int:
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if m < 1 or n < 1:
        return 2
    f, q = [], []
    for _ in range(m):
        f.append(100.0 + rnd() * 200.0)
        q.append(40.0 + rnd() * 40.0)
    d = [1.0 + rnd() * 8.0 for _ in range(n)]
    totald = 0.0
    for v in d:
        totald += v
    c = [1.0 + rnd() * 20.0 for _ in range(m * n)]
    totalq = 0.0                                    # scale capacities so total capacity is ample
    for v in q:
        totalq += v
    q = [v * (1.5 * totald / totalq) for v in q]

    obj = 0.0
    with Model() as mdl:
        build(mdl, m, n, True, f, q, d, c)
        t0 = time.perf_counter()
        r = mdl.solve()
        sec = time.perf_counter() - t0
    ok = r.rc == 0
    if ok:
        obj, x = r.objective, r.x
        maxdem = max(abs(csum(x[m + i * m + j] for j in range(m)) - d[i]) for i in range(n))
        maxcap = maxfrac = 0.0
        for j in range(m):
            s = csum(x[m + i * m + j] for i in range(n))
            maxcap = max(maxcap, s - q[j] * x[j])
            maxfrac = max(maxfrac, abs(x[j] - math.floor(x[j] + 0.5)))
        ok = maxdem < 1e-5 and maxcap < 1e-5 and maxfrac < 1e-6
    lp = 0.0                                        # LP relaxation lower bound
    with Model() as rel:
        build(rel, m, n, False, f, q, d, c)
        rr = rel.solve()
        if rr.rc == 0:
            lp = rr.objective
    if obj + 1e-6 < lp:
        ok = False
    print(f"facility   m={m} n={n} vars={m + m * n} cons={n + m} obj={obj:.4f} lp={lp:.4f} t={sec:.4f}s "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
