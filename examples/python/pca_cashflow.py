#!/usr/bin/env python3
"""Cash-flow matching, an independent LP exercise (model taken from "Portfolio
Construction and Analytics" (2016), Section 15.2.1.4).

    minimize   sum_i p_i x_i
    subject to sum_i c_it x_i >= m_t,   t = 1..8
               x_i >= 0

Five semi-annual-coupon bonds with ask prices p = (102.36, 110.83, 96.94, 114.65,
96.63), cash-flow matrix c_it and obligations m. This sample solves the LP on its
own terms and reports the optimum it finds. (The book prints an allocation matching
ours to the cent, x = (6000, 28103.61, 3555.18, 0, 9661.84), but an objective that
exceeds the cost of those same numbers by ~$148 - an internal inconsistency of the
source, ignored here.) Port of examples/c/pca_cashflow.c.
"""

import math

from primalsolver import Model, BK

P = [102.36, 110.83, 96.94, 114.65, 96.63]
C = [[2.50, 5.00, 3.00, 4.00, 3.50],
     [2.50, 5.00, 3.00, 4.00, 3.50],
     [2.50, 5.00, 3.00, 4.00, 3.50],
     [2.50, 5.00, 3.00, 4.00, 3.50],
     [102.50, 5.00, 3.00, 4.00, 3.50],
     [0.00, 105.00, 3.00, 4.00, 3.50],
     [0.00, 0.00, 103.00, 4.00, 3.50],
     [0.00, 0.00, 0.00, 104.00, 103.50]]
M = [100000, 200000, 100000, 200000, 800000, 1200000, 400000, 1000000]


def main() -> int:
    with Model() as m:
        m.add_vars(5)
        m.add_cons(8)
        for j in range(5):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.cj(j, P[j])
        for i in range(8):
            m.a_row(i, [0, 1, 2, 3, 4], C[i])
            m.con_bounds(i, BK.LO, M[i], math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"pca_cashflow  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = (abs(x[0] - 6000.0) < 1.0 and abs(x[1] - 28103.61) < 1.0 and abs(x[2] - 3555.18) < 1.0
          and abs(x[3]) < 1.0 and abs(x[4] - 9661.84) < 1.0 and abs(z - 5007145.84) < 1.0)
    print(f"pca_cashflow  x=({x[0]:.0f},{x[1]:.2f},{x[2]:.2f},{x[3]:.0f},{x[4]:.2f}) cost={z:.2f} "
          f"(independent LP; book allocation matches) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
