#!/usr/bin/env python3
"""Exercise 2.17 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007).

    maximize 4 x1 + x2 - x3  s.t.  x1 + 3 x3 <= 6,  3 x1 + x2 + 3 x3 <= 9,  x >= 0

The book's optimum: x = (3, 0, 0), objective 12. Port of examples/c/omf_ex217.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model(2, 3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(0, -4.0); m.cj(1, -1.0); m.cj(2, 1.0)
        m.a_row(0, [0, 2], [1.0, 3.0]); m.con_bounds(0, BK.UP, -math.inf, 6.0)
        m.a_row(1, [0, 1, 2], [3.0, 1.0, 3.0]); m.con_bounds(1, BK.UP, -math.inf, 9.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(z + 12.0) < 1e-6 and abs(x[0] - 3.0) < 1e-6 and abs(x[1]) < 1e-6 and abs(x[2]) < 1e-6
        print("omf_ex217  x=(%.6g,%.6g,%.6g)  obj=%.6g (book x=(3,0,0), 12) %s"
              % (x[0], x[1], x[2], -z, "OK" if ok else "FAIL"))
    else:
        print("omf_ex217  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
