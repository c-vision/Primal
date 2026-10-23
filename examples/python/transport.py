#!/usr/bin/env python3
"""Large transportation LP (scaling test).

    min sum_ij c_ij x_ij
      s.t. sum_j x_ij = supply_i   (M rows)
           sum_i x_ij = demand_j   (N rows)
           x_ij >= 0

Feasible by construction (supplies/demands come from a reference flow). Verified:
row/column sums, nonnegativity, strong duality. The data come from the same 32-bit
linear congruential generator as the C example.

    python transport.py [M] [N]   (default 60 60)

Port of examples/c/transport.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 12345


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    mm = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if mm < 1 or nn < 1:
        return 2
    nvar = mm * nn
    c = [0.0] * nvar
    sup = [0.0] * mm
    dem = [0.0] * nn
    for i in range(mm):
        for j in range(nn):
            x0 = rnd() * 10.0
            c[i * nn + j] = rnd() * 5.0 + 0.1
            sup[i] += x0
            dem[j] += x0
    with Model() as m:
        m.add_vars(nvar)
        m.add_cons(mm + nn)
        for k in range(nvar):
            m.cj(k, c[k])
            m.var_bounds(k, BK.LO, 0.0, math.inf)
        for i in range(mm):
            m.a_row(i, [i * nn + j for j in range(nn)], [1.0] * nn)
            m.con_bounds(i, BK.FX, sup[i], sup[i])
        for j in range(nn):
            m.a_row(mm + j, [i * nn + j for i in range(mm)], [1.0] * mm)
            m.con_bounds(mm + j, BK.FX, dem[j], dem[j])
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, x = r.objective, r.dual, r.x
        maxsup = max(abs(sum(x[i * nn + j] for j in range(nn)) - sup[i]) for i in range(mm))
        maxdem = max(abs(sum(x[i * nn + j] for i in range(mm)) - dem[j]) for j in range(nn))
        minx = min(0.0, min(x[:nvar]))
        ok = maxsup < 1e-5 and maxdem < 1e-5 and minx > -1e-6 and abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
    print(f"transport  M={mm} N={nn} vars={nvar} cons={mm + nn} nnz={2 * nvar} obj={obj:.6f} dobj={dobj:.6f} "
          f"t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
