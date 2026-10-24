#!/usr/bin/env python3
"""MIP (equivalent to the MOSEK Optimizer API example, A by columns).

    max x1 + 0.64 x2  s.t.  50 x1 + 31 x2 <= 250,  3 x1 - 2 x2 >= -4,  x integer >= 0

Expected solution: x1 = 5, x2 = 0, obj = 5.0000. Port of examples/c/mi1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA, VAR_TYPE


def main() -> int:
    with Model() as m:
        m.add_cons(2)
        m.add_vars(2)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0)
        m.cj(1, 0.64)
        m.a_col(0, [0, 1], [50.0, 3.0])         # A by columns: col 1 = (50, 3), col 2 = (31, -2)
        m.a_col(1, [0, 1], [31.0, -2.0])
        m.con_bounds(0, BK.UP, -math.inf, 250.0)
        m.con_bounds(1, BK.LO, -4.0, math.inf)
        m.sense(SENSE.MAX)
        m.var_type(0, VAR_TYPE.INT)
        m.var_type(1, VAR_TYPE.INT)
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    xx, obj, sta = r.x, r.objective, r.solsta
    print("solsta = %d" % sta)
    print("x1 = %.0f, x2 = %.0f, obj = %.4f" % (xx[0], xx[1], obj))
    ok = (abs(xx[0] - 5.0) < 1e-6 and abs(xx[1] - 0.0) < 1e-6 and abs(obj - 5.0) < 1e-6
          and sta == SOLSTA.INTEGER_OPTIMAL)
    print("OK (expected x1=5, x2=0, obj=5.0000)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
