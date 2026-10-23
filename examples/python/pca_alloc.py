#!/usr/bin/env python3
"""Portfolio allocation LP, "Portfolio Construction and Analytics" (2016), Section 6.3.

A university invests a $10,000,000 endowment in four mutual funds:
  Fund 1 growth (ret 20.69%, risk 4), Fund 2 index (5.87%, 2),
  Fund 3 corporate bond (10.52%, 2), Fund 4 money market (2.43%, 1).
Maximize expected return subject to: total = 10,000,000; amount in Funds 1+3 <= 60%
(6,000,000); average risk <= 2 (linearized as 4x1 + 2x2 + 2x3 + x4 <= 20,000,000);
each x_i <= 4,000,000 (40%); x >= 0.

The book's spreadsheet optimum: return $931,800 with x = ($2M, $0, $4M, $4M).
Port of examples/c/pca_alloc.c.
"""

import math

from primalsolver import Model, BK

MU = [0.2069, 0.0587, 0.1052, 0.0243]
RISK = [4.0, 2.0, 2.0, 1.0]


def main() -> int:
    with Model() as m:
        m.add_vars(4)
        m.add_cons(3)
        for j in range(4):
            m.var_bounds(j, BK.RA, 0.0, 4.0e6)
            m.cj(j, -MU[j])                         # maximize return (c negated, minimize)
        m.a_row(0, [0, 1, 2, 3], [1, 1, 1, 1]); m.con_bounds(0, BK.FX, 1.0e7, 1.0e7)
        m.a_row(1, [0, 2], [1, 1]);             m.con_bounds(1, BK.UP, -math.inf, 6.0e6)
        m.a_row(2, [0, 1, 2, 3], RISK);         m.con_bounds(2, BK.UP, -math.inf, 2.0e7)
        r = m.solve()
    if r.rc != 0:
        print(f"pca_alloc  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = (abs(-z - 931800.0) < 1.0 and abs(x[0] - 2.0e6) < 1.0 and abs(x[1]) < 1.0
          and abs(x[2] - 4.0e6) < 1.0 and abs(x[3] - 4.0e6) < 1.0)
    print(f"pca_alloc  x=({x[0] / 1e6:.6g},{x[1] / 1e6:.6g},{x[2] / 1e6:.6g},{x[3] / 1e6:.6g})e6 "
          f"return={-z:.6g} (book 2,0,4,4 -> 931800) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
