#!/usr/bin/env python3
"""MOSEK port "sensitivity" - post-optimal analysis (cost sensitivity) on a small LP.

    min x0 + x1   s.t.  x0 + 2*x1 >= 4,  x0, x1 >= 0

Optimum: x* = (0, 2), obj = 2. Cost sensitivity: x0 (nonbasic at the lower bound)
stays at 0 while c0 >= 1/2 -> range [0.5, +inf]; x1 (basic) has the degenerate range
[1, 1]. External check: re-solve with c0 = 0.4 (out of range -> x0 > 0) and
c0 = 0.6 (in range -> x0 stays 0). Port of examples/c/sensitivity.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK


def build(m, c0):
    m.add_vars(2)
    m.add_cons(1)
    m.var_bounds(0, BK.LO, 0.0, math.inf)
    m.var_bounds(1, BK.LO, 0.0, math.inf)
    m.cj(0, c0)
    m.cj(1, 1.0)
    m.a_row(0, [0, 1], [1.0, 2.0])
    m.con_bounds(0, BK.LO, 4.0, math.inf)


def solve_with_c0(c0):
    """Re-solve the LP with a given c0; returns (objective, x0, x1)."""
    with Model() as m:
        build(m, c0)
        r = m.solve()
    if r.rc != 0:
        return -1e30, 0.0, 0.0
    return r.objective, r.x[0], r.x[1]


def main() -> int:
    with Model() as m:
        build(m, 1.0)
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        x, obj = r.x, r.objective
        print(f"base solution: x = ({x[0]:.4f}, {x[1]:.4f}), obj = {obj:.4f}")
        lc0, uc0, lc1, uc1 = (ctypes.c_double() for _ in range(4))
        ps.PRIMAL_costsensitivity(m.task, 0, ctypes.byref(lc0), ctypes.byref(uc0))
        ps.PRIMAL_costsensitivity(m.task, 1, ctypes.byref(lc1), ctypes.byref(uc1))
    print(f"c0 in [{lc0.value:.4f}, {'+inf' if uc0.value == math.inf else '...'}]")
    print(f"c1 in [{lc1.value:.4f}, {uc1.value:.4f}] (basic: degenerate range)")

    p_in, xa, ya = solve_with_c0(0.6)               # external check
    p_out, xb, yb = solve_with_c0(0.4)
    print(f"c0=0.6 (in range):  x = ({xa:.4f}, {ya:.4f}), obj = {p_in:.4f}")
    print(f"c0=0.4 (out range): x = ({xb:.4f}, {yb:.4f}), obj = {p_out:.4f}")
    ok = (abs(x[0]) < 1e-9 and abs(x[1] - 2.0) < 1e-9 and abs(lc0.value - 0.5) < 1e-9
          and uc0.value == math.inf
          and abs(xa) < 1e-9                        # in range: x0 stays 0
          and abs(xb) > 1e-9                        # out range: x0 > 0
          and abs(p_out - 0.4 * 4.0) < 1e-6)        # all on x0: 1.6
    print(f"{'OK' if ok else 'FAIL'} (c0 range confirmed by re-solve)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
