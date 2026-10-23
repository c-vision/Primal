#!/usr/bin/env python3
"""Large generic sparse LP in inequality form (scaling test for the general ->
standard form conversion feeding the sparse IPM).

    min c'x   s.t.  A x <= b,  x >= 0

Random sparse A (d nonzeros per column) with a PLANTED optimal solution from the KKT
conditions: x* >= 0 with support S, mu* >= 0 (> 0 only on tight rows), nu*_j = 0
(j in S) / > 0 (j not in S); c = nu* - A'mu*, b_i = (Ax*)_i on tight rows and
(Ax*)_i + slack on the rest. Verified: obj == c'x*, feasibility and strong duality.
The data come from the same 32-bit linear congruential generator as the C example.

    python lp_ineq_large.py [rows] [cols] [density]   (default 300 6000 8)

Port of examples/c/lp_ineq_large.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 246813579


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 300     # rows
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000    # cols
    d = int(sys.argv[3]) if len(sys.argv) > 3 else 8       # nnz/col
    if m < 1 or n < 1 or d < 1 or d > m:
        return 2
    arow, aval = [], []                             # sparse A by columns: d distinct rows each
    for _ in range(n):
        used = set()
        for _ in range(d):
            while True:
                i = min(int(rnd() * m), m - 1)
                if i not in used:
                    break
            used.add(i)
            arow.append(i)
            aval.append(2.0 * rnd() - 1.0)          # in [-1,1)
    xs = [rnd() * 10.0 + 0.1 if rnd() < 0.6 else 0.0 for _ in range(n)]   # planted x*, ~60% > 0
    mu, tight = [], []
    for _ in range(m):
        if rnd() < 0.5:
            mu.append(rnd() * 2.0 + 0.1)
            tight.append(True)
        else:
            mu.append(0.0)
            tight.append(False)
    nu = [0.0 if xs[j] > 0.0 else rnd() * 3.0 + 0.1 for j in range(n)]
    axs = [0.0] * m                                 # Ax* (per row)
    for j in range(n):
        if xs[j] != 0.0:
            for k in range(j * d, (j + 1) * d):
                axs[arow[k]] += aval[k] * xs[j]
    b = [axs[i] + (0.0 if tight[i] else rnd() * 5.0 + 0.1) for i in range(m)]
    c = []                                          # c = nu* - A'mu*;  obj* = c'x*
    for j in range(n):
        s = nu[j]
        for k in range(j * d, (j + 1) * d):
            s -= aval[k] * mu[arow[k]]
        c.append(s)
    objstar = 0.0
    for j in range(n):
        objstar += c[j] * xs[j]
    rsub = [[] for _ in range(m)]                   # rows for a_row, columns in increasing order
    rval = [[] for _ in range(m)]
    for j in range(n):
        for k in range(j * d, (j + 1) * d):
            rsub[arow[k]].append(j)
            rval[arow[k]].append(aval[k])
    with Model() as mdl:
        mdl.add_vars(n)
        mdl.add_cons(m)
        for j in range(n):
            mdl.cj(j, c[j])
            mdl.var_bounds(j, BK.LO, 0.0, math.inf)
        for i in range(m):
            mdl.a_row(i, rsub[i], rval[i])
            mdl.con_bounds(i, BK.UP, -math.inf, b[i])
        t0 = time.perf_counter()
        r = mdl.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, x = r.objective, r.dual, r.x
        ax = [0.0] * m
        for j in range(n):
            if x[j] != 0.0:
                for k in range(j * d, (j + 1) * d):
                    ax[arow[k]] += aval[k] * x[j]
        maxviol = max([0.0] + [ax[i] - b[i] for i in range(m)])
        minx = min(0.0, min(x[:n]))
        bn = max([1.0] + [abs(v) for v in b])
        ok = (maxviol < 1e-4 * bn and minx > -1e-6 and abs(obj - objstar) < 1e-4 * (1.0 + abs(objstar))
              and abs(obj - dobj) < 1e-4 * (1.0 + abs(obj)))
    print(f"lp_ineq_large  rows={m} cols={n} nnz={n * d} obj={obj:.6f} obj*={objstar:.6f} dobj={dobj:.6f} "
          f"t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
