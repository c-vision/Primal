#!/usr/bin/env python3
"""MOSEK port "mil1" - a small pure-integer MIP.

    max 5x0 + 4x1 + 3x2
    s.t. 2x0 + 3x1 + x2 <= 5
         4x0 + x1 + 2x2 <= 11
         3x0 + 4x1 + 2x2 <= 8
         x0, x1, x2 integer >= 0

Expected solution: x = (2, 0, 1), obj = 13. Port of examples/c/mil1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA, VAR_TYPE


def main() -> int:
    with Model() as m:
        m.add_vars(3)
        m.add_cons(3)
        m.obj([0, 1, 2], [5.0, 4.0, 3.0])
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], [2.0, 3.0, 1.0])
        m.a_row(1, [0, 1, 2], [4.0, 1.0, 2.0])
        m.a_row(2, [0, 1, 2], [3.0, 4.0, 2.0])
        m.con_bounds(0, BK.UP, -math.inf, 5.0)
        m.con_bounds(1, BK.UP, -math.inf, 11.0)
        m.con_bounds(2, BK.UP, -math.inf, 8.0)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.var_type(j, VAR_TYPE.INT)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"x = ({x[0]:.0f}, {x[1]:.0f}, {x[2]:.0f}), obj = {r.objective:.4f}")
    ok = (r.solsta == SOLSTA.INTEGER_OPTIMAL and abs(x[0] - 2) < 1e-6 and abs(x[1]) < 1e-6
          and abs(x[2] - 1) < 1e-6 and abs(r.objective - 13.0) < 1e-6)
    print(f"{'OK' if ok else 'FAIL'} (expected x=(2,0,1), obj=13)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
