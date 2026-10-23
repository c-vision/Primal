#!/usr/bin/env python3
"""Transportation counterexample, "Introduction to Linear Optimization"
(A. Nemirovski, 2024), Exercise 5.3.

    Opt = min { sum_ij c_ij x_ij : sum_j x_ij = a_i, sum_i x_ij = b_j, x_ij >= 0 }

Instance (p = q = 2): c11 = c22 = 1, c12 = c21 = 100.
  (a) a = b = (1, 1): trivial solution x11 = x22 = 1, optimum 2.
  (b) supplies/demands decreased to a' = (1/2, 1), b' = (1, 1/2): the book proves
      the optimum is "at least 50" (it increases!), so decreasing supply+demand
      (keeping totals equal) can increase cost.
The sample solves both LPs and checks (a) = 2 and (b) >= 50 (computed 51).
Port of examples/c/intlo_transport.c.
"""

import math

from primalsolver import Model, BK

C = [1.0, 100.0, 100.0, 1.0]                        # 11, 12, 21, 22


def solve(a0, a1, b0, b1):
    """One transportation LP: min c11 x11 + c12 x12 + c21 x21 + c22 x22."""
    with Model() as m:
        m.add_vars(4)
        m.add_cons(4)
        for j in range(4):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.cj(j, C[j])
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, a0, a0)   # row supplies
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, a1, a1)
        m.a_row(2, [0, 2], [1.0, 1.0]); m.con_bounds(2, BK.FX, b0, b0)   # column demands
        m.a_row(3, [1, 3], [1.0, 1.0]); m.con_bounds(3, BK.FX, b1, b1)
        r = m.solve()
    return (r.rc == 0), r.objective


def main() -> int:
    z1 = z2 = -1.0
    ok1, obj = solve(1.0, 1.0, 1.0, 1.0)
    if ok1:
        z1 = obj
    ok2 = False
    if ok1:
        ok2, obj = solve(0.5, 1.0, 1.0, 0.5)
        if ok2:
            z2 = obj
    ok = ok1 and ok2 and abs(z1 - 2.0) < 1e-6 and z2 >= 50.0 - 1e-6
    print(f"intlo_transport  z_initial={z1:.6g} z_new={z2:.6g} (book 2, >=50) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
