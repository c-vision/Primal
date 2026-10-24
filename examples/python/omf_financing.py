#!/usr/bin/env python3
"""Short-term financing LP of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 3.1.1.

Monthly cash needs (k$) Jan..Jun: 150, 100, -200, 200, -50, -300; a credit line x_i
(1%/month, limit 100), commercial paper y_i in Jan-Mar (2% for three months), excess
funds z_i (0.5%/month). Maximize the wealth v at the end of June. The book's
sensitivity report gives shadow prices u_Jan = -1.0373, u_Feb = -1.030, u_Mar = -1.020.
Port of examples/c/omf_financing.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model(6, 14) as m:                         # 0..4 = x1..x5, 5..7 = y1..y3, 8..12 = z1..z5, 13 = v
        for i in range(5):
            m.var_bounds(i, BK.RA, 0.0, 100.0)
        for i in range(5, 13):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        m.var_bounds(13, BK.FR, -math.inf, math.inf)   # v
        m.cj(13, -1.0)                              # max v
        rhs = [150.0, 100, -200, 200, -50, -300]
        ii = [[0, 5, 8], [1, 6, 0, 8, 9], [2, 7, 1, 9, 10], [3, 5, 2, 10, 11], [4, 6, 3, 11, 12], [7, 4, 12, 13]]
        vv = [[1.0, 1.0, -1.0], [1.0, 1.0, -1.01, 1.003, -1.0], [1.0, 1.0, -1.01, 1.003, -1.0],
              [1.0, -1.02, -1.01, 1.003, -1.0], [1.0, -1.02, -1.01, 1.003, -1.0], [-1.02, -1.01, 1.003, -1.0]]
        for k in range(6):
            m.a_row(k, ii[k], vv[k])
            m.con_bounds(k, BK.FX, rhs[k], rhs[k])
        r = m.solve()
    ok = r.rc == 0
    if ok:
        obj, x, y = r.objective, r.x, r.y
        v = -obj                                    # the objective is max v, reported as min -v
        ok = abs(abs(y[0]) - 1.0373) < 1e-3 and abs(abs(y[1]) - 1.030) < 1e-3 and abs(abs(y[2]) - 1.020) < 1e-3
        print("omf_financing  v=%.4f  x=(%.2f,%.4f,%.2f,%.2f,%.2f)  uJan=%.4f uFeb=%.4f uMar=%.4f  %s"
              % (v, x[0], x[1], x[2], x[3], x[4], abs(y[0]), abs(y[1]), abs(y[2]), "OK" if ok else "FAIL"))
    else:
        print("omf_financing  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
