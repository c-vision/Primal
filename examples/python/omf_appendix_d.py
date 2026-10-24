#!/usr/bin/env python3
"""Revised-simplex worked LP, "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Appendix D.

    maximize Z = x1 + 2 x2 + x3 - 2 x4
    s.t.  -2 x1 +   x2 + x3 + 2 x4 + x6           = 2
           -x1 + 2 x2 + x3      + x5 + x7         = 7
            x1       + x3 + x4 + x5          + x8 = 3,   x >= 0

The book reports x1 = 3, x2 = 5, x6 = 3, all other x = 0, Z = 13.
Port of examples/c/omf_appendix_d.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model(3, 8) as m:
        for j in range(8):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.sense(SENSE.MAX)
        m.cj(0, 1.0); m.cj(1, 2.0)
        m.cj(2, 1.0); m.cj(3, -2.0)
        m.a_row(0, [0, 1, 2, 3, 5], [-2, 1, 1, 2, 1]); m.con_bounds(0, BK.FX, 2.0, 2.0)
        m.a_row(1, [0, 1, 2, 4, 6], [-1, 2, 1, 1, 1]); m.con_bounds(1, BK.FX, 7.0, 7.0)
        m.a_row(2, [0, 2, 3, 4, 7], [1, 1, 1, 1, 1]); m.con_bounds(2, BK.FX, 3.0, 3.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(x[0] - 3.0) < 1e-6 and abs(x[1] - 5.0) < 1e-6 and abs(x[5] - 3.0) < 1e-6
        for j in range(2, 8):
            if j != 5:
                ok = ok and abs(x[j]) < 1e-6
        ok = ok and abs(z - 13.0) < 1e-6
        print("omf_appendix_d  x1=%.6g x2=%.6g x6=%.6g Z=%.6g (book 3,5,3, 13) %s"
              % (x[0], x[1], x[5], z, "OK" if ok else "FAIL"))
    else:
        print("omf_appendix_d  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
