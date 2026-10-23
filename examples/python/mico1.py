#!/usr/bin/env python3
"""MOSEK port "mico1" - mixed-integer conic optimization (MIP + quadratic cones).

    max x0
    s.t. (t, x0, x1) in QUAD   (t >= ||(x0, x1)||),  t = 1
         x0 + x1 >= 0.5,  x0 integer >= 0, x1 >= 0

The cone gives x0^2 + x1^2 <= 1; x0 integer -> x0 in {0, 1}. With x0 = 1: x1 = 0 and
x0 + x1 = 1 >= 0.5 -> obj = 1. Optimum: x = (1, 1, 0), obj 1.
Port of examples/c/mico1.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE


def main() -> int:
    with Model() as m:
        m.add_vars(3)                               # t, x0, x1
        m.add_cons(1)
        m.var_bounds(0, BK.FX, 1.0, 1.0)            # t = 1
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.LO, 0.0, math.inf)
        m.var_type(1, VAR_TYPE.INT)
        m.cj(1, 1.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [1, 2], [1.0, 1.0])              # x0 + x1 >= 0.5
        m.con_bounds(0, BK.LO, 0.5, math.inf)
        m.cone(CT.QUAD, [0, 1, 2])                  # (t, x0, x1) in QUAD
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[1]:.4f}, {x[2]:.4f}), obj = {obj:.4f}")
    best = -1e30                                    # enumeration: x0 in {0,1}, cone + row
    for a in range(2):
        xmax = math.sqrt(1.0 - float(a) * a)        # cone: x1 <= sqrt(1-a^2)
        need = 0.5 - a                              # row: x1 >= 0.5-a
        if need > xmax + 1e-12:
            continue                                # x1 cannot cover the row
        best = max(best, float(a))
    print(f"enumeration: best = {best:.4f}")
    ok = abs(obj - best) < 1e-6 and abs(x[1] - math.floor(x[1] + 0.5)) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (MIP+cone: optimum by enumeration)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
