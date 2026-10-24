#!/usr/bin/env python3
"""MOSEK port "lo2" - LP with 4 variables and 3 constraints (one equality, one
upper-bounded, one lower-bounded).

    max 3x0 + x1 + 5x2 + x3
    s.t. 3x0 + x1 + 2x2       = 30
         2x0 + x1 + 3x2 + x3 <= 25
          x0 + 2x1 +  x2 + 2x3 >= 15
         0 <= x0, 0 <= x1 <= 10, x2, x3 >= 0

Check: primal = dual (no duality gap) and BAS vs ITR cross-check.
Port of examples/c/lo2.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE, SOL, SOLSTA


def csum(values, start=0.0):
    """Left-to-right sum, as the C loop (sum() is compensated since Python 3.12)."""
    s = start
    for v in values:
        s += v
    return s


C = [3.0, 1.0, 5.0, 1.0]


def main() -> int:
    with Model() as m:
        m.add_vars(4)
        m.add_cons(3)
        for j in range(4):
            m.cj(j, C[j])
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], [3.0, 1.0, 2.0])
        m.a_row(1, [0, 1, 2, 3], [2.0, 1.0, 3.0, 1.0])
        m.a_row(2, [0, 1, 2, 3], [1.0, 2.0, 1.0, 2.0])
        m.con_bounds(0, BK.FX, 30.0, 30.0)
        m.con_bounds(1, BK.UP, -math.inf, 25.0)
        m.con_bounds(2, BK.LO, 15.0, math.inf)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.RA, 0.0, 10.0)
        m.var_bounds(2, BK.LO, 0.0, math.inf)
        m.var_bounds(3, BK.LO, 0.0, math.inf)
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        xb = (ctypes.c_double * 4)()
        ps.PRIMAL_getxx(m.task, SOL.BAS, xb)
    x, po, dobj = r.x, r.objective, r.dual
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}, {x[3]:.6f})")
    print(f"pobj = {po:.6f}, dobj = {dobj:.6f}, solsta = {r.solsta}")
    ok = r.solsta == SOLSTA.OPTIMAL                 # primal feasibility
    r0 = 3 * x[0] + x[1] + 2 * x[2]
    r1 = 2 * x[0] + x[1] + 3 * x[2] + x[3]
    r2 = x[0] + 2 * x[1] + x[2] + 2 * x[3]
    if abs(r0 - 30.0) > 1e-6 or r1 > 25.0 + 1e-6 or r2 < 15.0 - 1e-6:
        ok = False
    if abs(po - dobj) > 1e-6 * (1 + abs(po)):       # zero duality gap + consistent BAS/ITR
        ok = False
    if abs(po - csum(C[j] * x[j] for j in range(4))) > 1e-6:
        ok = False
    if any(abs(x[j] - xb[j]) > 1e-6 for j in range(4)):
        ok = False
    print(f"{'OK' if ok else 'FAIL'} (optimal certified by pobj = dobj)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
