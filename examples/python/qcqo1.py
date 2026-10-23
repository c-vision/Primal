#!/usr/bin/env python3
"""MOSEK port "qcqo1" - quadratic constraints.

Conic form equivalent of the MOSEK example (hand-verified optimum):
    max x0 + x1  s.t.  x0^2 + x1^2 <= 1,  x >= 0
Optimum: x* = (1,1)/sqrt(2), obj = sqrt(2) ~ 1.414214.

Implemented with qconk (quadratic coefficients on row 0, MOSEK convention:
1/2 x'Qx with Q = 2I -> x0^2 + x1^2). Port of examples/c/qcqo1.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.sense(SENSE.MAX)
        m.qconk(0, [(0, 0, 2.0), (1, 1, 2.0)])     # x0^2 + x1^2 <= 1
        m.con_bounds(0, BK.UP, -math.inf, 1.0)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}), obj = {r.objective:.6f}")
    s = 1.0 / math.sqrt(2.0)
    ok = abs(x[0] - s) < 1e-4 and abs(x[1] - s) < 1e-4 and abs(r.objective - math.sqrt(2.0)) < 1e-4
    print(f"{'OK' if ok else 'FAIL'} (expected x=(1,1)/sqrt2, obj=sqrt2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
