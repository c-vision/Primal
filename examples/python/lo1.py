#!/usr/bin/env python3
"""Simple LP (matches the MOSEK Optimizer API "lo1" example).

    min 2x + 3y  s.t.  x + y >= 1,  x - y <= 0,  x, y >= 0

Expected solution: x = 0.5, y = 0.5, obj = 2.5 (read from the basic solution).
Port of examples/c/lo1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOL, SOLSTA


def main() -> int:
    numvar, numcon = 2, 2
    with Model() as m:
        m.add_vars(numvar)
        m.add_cons(numcon)
        for j in range(numvar):                 # x, y >= 0
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(0, 2.0)                            # min 2x + 3y
        m.cj(1, 3.0)
        m.sense(SENSE.MIN)
        m.a_row(0, [0, 1], [1.0, 1.0])          # x + y >= 1
        m.con_bounds(0, BK.LO, 1.0, math.inf)
        m.a_row(1, [0, 1], [1.0, -1.0])         # x - y <= 0
        m.con_bounds(1, BK.UP, -math.inf, 0.0)
        r = m.solve(SOL.BAS)
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    xx, obj, sta = r.x, r.objective, r.solsta
    print("solsta = %d" % sta)
    print("x = %.6f, y = %.6f, obj = %.6f" % (xx[0], xx[1], obj))
    ok = abs(xx[0] - 0.5) < 1e-6 and abs(xx[1] - 0.5) < 1e-6 and abs(obj - 2.5) < 1e-6 and sta == SOLSTA.OPTIMAL
    print("OK (expected x=0.5, y=0.5, obj=2.5)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
