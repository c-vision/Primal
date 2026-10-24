#!/usr/bin/env python3
"""Workforce planning LP of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Exercise 3.15.

A restaurant needs 14, 13, 15, 16, 19, 18, 11 workers Mon..Sun; each worker works five
consecutive days. Shift i starts on day i:

    min sum_i x_i  s.t.  for each day, the five shifts covering it >= demand,  x >= 0

The book's sensitivity report: x = (4, 7, 1, 4, 3, 3, 0), 22 workers, shadow prices
(1/3, 0, 1/3, 0, 1/3, 1/3, 0), reduced cost 1/3 for Shift 7.
Port of examples/c/omf_workforce.c.
"""

import math

from primalsolver import Model, BK

D = 7
DEMAND = [14.0, 13, 15, 16, 19, 18, 11]
XBOOK = [4.0, 7, 1, 4, 3, 3, 0]
SPBOOK = [0.333333, 0, 0.333333, 0, 0.333333, 0.333333, 0]


def main() -> int:
    with Model(D, D) as m:                          # vars x1..x7 (shift Mon..Sun); rows Mon..Sun
        for i in range(D):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
            m.cj(i, 1.0)                            # min sum x_i
        for d in range(D):                          # day d is worked by the shifts starting d, d-1, ..., d-4
            m.a_row(d, [(d - k) % D for k in range(5)], [1.0] * 5)
            m.con_bounds(d, BK.LO, DEMAND[d], math.inf)
        r = m.solve()
        ok = r.rc == 0
        if ok:
            obj, x, y = r.objective, r.x, r.y
            x_ok = all(abs(x[i] - XBOOK[i]) <= 1e-6 for i in range(D))
            sp_ok = all(abs(abs(y[d]) - SPBOOK[d]) <= 1e-5 for d in range(D))   # dual sign on a >= row is negative
            rcs = m.getreducedcosts(0, D)           # Shift 7 is unused; its reduced cost is 1/3
            rc7_ok = abs(rcs[6] - 0.333333) < 1e-5
            ok = abs(obj - 22.0) < 1e-6 and x_ok and sp_ok and rc7_ok
            print("omf_workforce  workers=%.4f (book 22)  %s" % (obj, "OK" if ok else "FAIL"))
            print("  x: " + "".join("S%d=%.4g " % (i + 1, x[i]) for i in range(D))
                  + "  (book 4,7,1,4,3,3,0)  %s" % ("OK" if x_ok else "FAIL"))
            print("  shadow: " + "".join("%.6f " % abs(y[d]) for d in range(D)) + "  %s" % ("OK" if sp_ok else "FAIL"))
            print("  reduced cost Shift7 = %.6f (book 0.333333)  %s" % (rcs[6], "OK" if rc7_ok else "FAIL"))
        else:
            print("omf_workforce  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
