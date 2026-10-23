#!/usr/bin/env python3
"""LP with unique optimum, "Convex Optimization" (Boyd & Vandenberghe, 2004), Ex. 5.28.

    minimize   47 x1 + 93 x2 + 17 x3 - 93 x4
    subject to
       [ -1  -6   1   3 ] [x1]     [ -3 ]
       [ -1  -2   7   1 ] [x2]  <= [  5 ]
       [  0   3 -10  -1 ] [x3]     [ -8 ]
       [ -6 -11  -2  12 ] [x4]     [ -7 ]
       [  1   6  -1  -3 ]          [  4 ]

The book states the optimal solution is unique and equal to x* = (1, 1, 1, 1)
(objective 64). Port of examples/c/boyd_lp_unique.c.
"""

import math

from primalsolver import Model, BK

A = [[-1, -6, 1, 3], [-1, -2, 7, 1], [0, 3, -10, -1], [-6, -11, -2, 12], [1, 6, -1, -3]]
B = [-3, 5, -8, -7, 4]
C = [47, 93, 17, -93]


def main() -> int:
    with Model() as m:
        m.add_vars(4)
        m.add_cons(5)
        for j in range(4):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
            m.cj(j, C[j])
        for i in range(5):
            m.a_row(i, [0, 1, 2, 3], A[i])
            m.con_bounds(i, BK.UP, -math.inf, B[i])
        r = m.solve()
    if r.rc != 0:
        print(f"boyd_lp_unique  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = all(abs(x[j] - 1.0) < 1e-5 for j in range(4))
    print(f"boyd_lp_unique  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g},{x[3]:.6g}) obj={z:.6g} "
          f"(book (1,1,1,1)) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
