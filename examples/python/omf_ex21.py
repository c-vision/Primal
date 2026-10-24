#!/usr/bin/env python3
"""Example 2.1 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007): the bond-portfolio LP.

$100,000 split between a corporate bond (4% yield, maturity 3y, rating 2) and a
government bond (3%, 4y, rating 1); average rating <= 1.5, average maturity <= 3.6y.
In thousands of dollars:

    maximize Z = 4 x1 + 3 x2
    s.t.  x1 + x2 <= 100,  2 x1 + x2 <= 150,  3 x1 + 4 x2 <= 360,  x >= 0

The book's simplex ends at x = (50, 50), Z = 350. Port of examples/c/omf_ex21.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model(3, 2) as m:
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.cj(0, -4.0); m.cj(1, -3.0)            # maximize
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.UP, -math.inf, 100.0)
        m.a_row(1, [0, 1], [2.0, 1.0]); m.con_bounds(1, BK.UP, -math.inf, 150.0)
        m.a_row(2, [0, 1], [3.0, 4.0]); m.con_bounds(2, BK.UP, -math.inf, 360.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(x[0] - 50.0) < 1e-6 and abs(x[1] - 50.0) < 1e-6 and abs(z + 350.0) < 1e-6
        print("omf_ex21  x=(%.6g,%.6g) Z=%.6g (book x=(50,50), Z=350) %s" % (x[0], x[1], -z, "OK" if ok else "FAIL"))
    else:
        print("omf_ex21  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
