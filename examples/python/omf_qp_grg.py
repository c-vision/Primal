#!/usr/bin/env python3
"""The QP solved by GRG in "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 5.

    minimize (x1 - 1/2)^2 + (x2 - 5/2)^2  s.t.  x1 - x2 >= 0,  x1 >= 0,  0 <= x2 <= 2

The book's optimum is x = (1.5, 1.5). Expanded, f = x1^2 + x2^2 - x1 - 5 x2 + 6.5:
Q = 2 I, c = (-1, -5), and f = solver objective + 6.5 = 2. Port of examples/c/omf_qp_grg.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model(1, 2) as m:
        m.var_bounds(0, BK.LO, 0.0, math.inf)   # x1
        m.var_bounds(1, BK.RA, 0.0, 2.0)        # 0 <= x2 <= 2
        m.cj(0, -1.0); m.cj(1, -5.0)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])      # Q = 2 I  (0.5 x'Qx = x1^2 + x2^2)
        m.a_row(0, [0, 1], [1.0, -1.0])         # x1 - x2 >= 0
        m.con_bounds(0, BK.LO, 0.0, math.inf)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        f = z + 6.5                             # add back the constant of (x1-1/2)^2+(x2-5/2)^2
        ok = abs(x[0] - 1.5) < 1e-6 and abs(x[1] - 1.5) < 1e-6 and abs(f - 2.0) < 1e-6 and x[0] - x[1] >= -1e-6
        print("omf_qp_grg  x=(%.6g,%.6g)  f=%.6g (book x=(1.5,1.5), f=2) %s" % (x[0], x[1], f, "OK" if ok else "FAIL"))
    else:
        print("omf_qp_grg  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
