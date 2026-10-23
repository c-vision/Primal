#!/usr/bin/env python3
"""Big-M worked example, "Introduction to Linear Optimization" (A. Nemirovski, 2024),
Exercise 4.4(2) / section on the Big-M method.

    maximize   -2 x1 + x2 + x3 - 6 x4
    subject to  x1 + x2      = 2
                x3 + x4      = 2
                x1      + x3 = 2
                x1,x2,x3,x4 >= 0

The book runs the Big-M primal simplex and finds x = [0, 2, 2, 0], optimal value 4.
Port of examples/c/intlo_bigm.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(4)
        m.add_cons(3)
        for j in range(4):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.sense(SENSE.MAX)
        m.obj([0, 1, 2, 3], [-2.0, 1.0, 1.0, -6.0])
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, 2.0, 2.0)
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 2.0, 2.0)
        m.a_row(2, [0, 2], [1.0, 1.0]); m.con_bounds(2, BK.FX, 2.0, 2.0)
        r = m.solve()
    if r.rc != 0:
        print(f"intlo_bigm  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = all(abs(x[j] - v) < 1e-6 for j, v in enumerate([0.0, 2.0, 2.0, 0.0])) and abs(z - 4.0) < 1e-6
    print(f"intlo_bigm  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g},{x[3]:.6g}) obj={z:.6g} "
          f"(book [0,2,2,0], 4) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
