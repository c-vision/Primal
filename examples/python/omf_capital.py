#!/usr/bin/env python3
"""Capital budgeting (0-1) problem, "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 11.2.

    maximize 8 x1 + 11 x2 + 6 x3 + 4 x4
    s.t.  6.7 x1 + 10 x2 + 5.5 x3 + 3.4 x4 <= 19,  x in {0,1}^4

The book's integer optimum: x = (0, 1, 1, 1), value 21000. Port of examples/c/omf_capital.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE


def main() -> int:
    with Model(1, 4) as m:
        for j in range(4):
            m.var_bounds(j, BK.RA, 0.0, 1.0)
            m.var_type(j, VAR_TYPE.INT_BIN)
        m.sense(SENSE.MAX)
        m.cj(0, 8.0); m.cj(1, 11.0)
        m.cj(2, 6.0); m.cj(3, 4.0)
        m.a_row(0, [0, 1, 2, 3], [6.7, 10.0, 5.5, 3.4])
        m.con_bounds(0, BK.UP, -math.inf, 19.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        for j in range(4):
            ok = ok and abs(x[j] - math.floor(x[j] + 0.5)) < 1e-6
        ok = (ok and abs(x[0]) < 1e-6 and abs(x[1] - 1.0) < 1e-6 and abs(x[2] - 1.0) < 1e-6
              and abs(x[3] - 1.0) < 1e-6 and abs(z - 21.0) < 1e-6)
        print("omf_capital  x=(%.6g,%.6g,%.6g,%.6g) Z=%.6g (=%.0f) (book (0,1,1,1), 21000) %s"
              % (x[0], x[1], x[2], x[3], z, 1000.0 * z, "OK" if ok else "FAIL"))
    else:
        print("omf_capital  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
