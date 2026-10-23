#!/usr/bin/env python3
"""Parametrized l1-norm approximation, "Convex Optimization" (Boyd & Vandenberghe,
2004), Exercise 5.33.

    minimize || A x + b + eps d ||_1,   x in R^3

with A = [-2 7 1; -5 -1 3; -7 3 -5; -1 4 -4; 1 5 5; 2 -5 -1],
b = (-4, 3, 9, 0, -11, 5), d = (-10, -13, -27, -10, -7, 14).
For eps = 0 the book asks to prove x* = 1 (all-ones) is optimal. The sample solves
the l1 problem as an LP (x free, z >= 0 with z_i >= +/-(Ax+b)_i, minimize sum z)
and checks x* = (1,1,1) (value 4). Port of examples/c/boyd_l1.c.
"""

import math

from primalsolver import Model, BK

A = [[-2, 7, 1], [-5, -1, 3], [-7, 3, -5], [-1, 4, -4], [1, 5, 5], [2, -5, -1]]
B = [-4, 3, 9, 0, -11, 5]


def main() -> int:
    with Model() as m:
        m.add_vars(9)                               # 3 x + 6 z
        m.add_cons(12)
        for j in range(3):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        for j in range(3, 9):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.cj(j, 1.0)
        for i in range(6):
            a = A[i]
            # z_i >= (A x + b)_i  <=>  z_i - a_i.x >= b_i
            m.a_row(2 * i, [0, 1, 2, 3 + i], [-a[0], -a[1], -a[2], 1.0])
            m.con_bounds(2 * i, BK.LO, B[i], math.inf)
            # z_i >= -(A x + b)_i  <=>  z_i + a_i.x >= -b_i
            m.a_row(2 * i + 1, [0, 1, 2, 3 + i], [a[0], a[1], a[2], 1.0])
            m.con_bounds(2 * i + 1, BK.LO, -B[i], math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"boyd_l1  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = all(abs(x[j] - 1.0) < 1e-5 for j in range(3)) and abs(z - 4.0) < 1e-5
    print(f"boyd_l1  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g}) ||Ax+b||_1={z:.6g} "
          f"(book (1,1,1), 4) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
