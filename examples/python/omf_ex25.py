#!/usr/bin/env python3
"""Exercise 2.5 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007).

    minimize 2 x1 + 3 x2  s.t.  x1 + x2 >= 5,  x1 >= 1,  x2 >= 2

The book asks to prove x* = (3, 2) is optimal (value 12). Port of examples/c/omf_ex25.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model(1, 2) as m:
        m.var_bounds(0, BK.LO, 1.0, math.inf)    # x1 >= 1
        m.var_bounds(1, BK.LO, 2.0, math.inf)    # x2 >= 2
        m.cj(0, 2.0); m.cj(1, 3.0)
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.LO, 5.0, math.inf)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(x[0] - 3.0) < 1e-6 and abs(x[1] - 2.0) < 1e-6 and abs(z - 12.0) < 1e-6
        print("omf_ex25  x=(%.6g,%.6g) obj=%.6g (book x=(3,2), 12) %s" % (x[0], x[1], z, "OK" if ok else "FAIL"))
    else:
        print("omf_ex25  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
