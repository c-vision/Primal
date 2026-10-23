#!/usr/bin/env python3
"""Exponential cone (variant of the MOSEK Optimizer API "ceo1" example).

    min x0  s.t.  (x0, x1, x2) in EXP,  x1 = 1,  -1 <= x2 <= -0.5
    i.e. x0 >= x1 * exp(x2/x1) = exp(x2)

Expected solution: x0 = exp(-1) = 0.367879 at x2 = -1.
Port of examples/c/ceo1.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(3)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.FX, 1.0, 1.0)
        m.var_bounds(2, BK.RA, -1.0, -0.5)
        m.cj(0, 1.0)                                # min x0
        m.sense(SENSE.MIN)
        m.cone(CT.PEXP, [0, 1, 2])
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}), obj = {r.objective:.6f}")
    ok = abs(r.objective - math.exp(-1.0)) < 1e-6 and abs(x[2] - (-1.0)) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (expected obj = e^-1 = 0.367879 in x2 = -1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
