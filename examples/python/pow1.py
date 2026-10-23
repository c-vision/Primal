#!/usr/bin/env python3
"""Power cone (equivalent of the MOSEK docs "pgo"/pow example).

    min x + y  s.t.  x^0.5 * y^0.5 >= 1,  x, y >= 0

PPOW cone form (t,u,v): t^a * u^(1-a) >= |v|, a = 0.5. With (t,u,v) = (x, y, 1):
expected solution obj = 2 at (x, y) = (1, 1). Port of examples/c/pow1.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(3)                               # t = x, u = y, v = 1 (fixed)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.FX, 1.0, 1.0)
        m.cj(0, 1.0); m.cj(1, 1.0)                  # min x + y
        m.sense(SENSE.MIN)
        m.cone(CT.PPOW, [0, 1, 2], 0.5)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"x = {x[0]:.6f}, y = {x[1]:.6f}, obj = {r.objective:.6f}")
    ok = abs(r.objective - 2.0) < 1e-4 and abs(x[0] - 1.0) < 1e-3 and abs(x[1] - 1.0) < 1e-3
    print(f"{'OK' if ok else 'FAIL'} (expected obj = 2.0 at x = y = 1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
