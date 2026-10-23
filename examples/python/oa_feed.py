#!/usr/bin/env python3
"""Animal feed mix LP, "Optimization Algorithms: AI techniques for design, planning,
and control problems" (2024), Listing 2.6 and Appendix C.1.

    minimize   30.5 x1 + 10.0 x2 + 90 x3      (cents per kg)
    subject to 0.008 <= 0.001 x1 + 0.38 x2 + 0.002 x3 <= 0.012   (calcium)
               0.09 x1 + 0.50 x3 >= 0.22                          (protein)
               0.02 x1 + 0.08 x3 <= 0.05                          (fiber)
               x1 + x2 + x3 = 1
               0 <= x1, x2, x3 <= 1                               (fractions)

Corn = x1, Limestone = x2, Soybean meal = x3. The book's output: Corn = 65.0%,
Limestone = 3.0%, Soybean meal = 32.0%, total cost = 49.16 cents/kg.
Port of examples/c/oa_feed.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model() as m:
        m.add_vars(3)
        m.add_cons(4)
        for j in range(3):
            m.var_bounds(j, BK.RA, 0.0, 1.0)
        m.obj([0, 1, 2], [30.5, 10.0, 90.0])
        m.a_row(0, [0, 1, 2], [0.001, 0.38, 0.002]); m.con_bounds(0, BK.RA, 0.008, 0.012)
        m.a_row(1, [0, 2], [0.09, 0.50]);            m.con_bounds(1, BK.LO, 0.22, math.inf)
        m.a_row(2, [0, 2], [0.02, 0.08]);            m.con_bounds(2, BK.UP, -math.inf, 0.05)
        m.a_row(3, [0, 1, 2], [1.0, 1.0, 1.0]);      m.con_bounds(3, BK.FX, 1.0, 1.0)
        r = m.solve()
    if r.rc != 0:
        print(f"oa_feed  rc={r.rc} FAIL")
        return 1
    x, z = r.x, r.objective
    ok = (abs(x[0] - 0.65) < 0.01 and abs(x[1] - 0.03) < 0.01
          and abs(x[2] - 0.32) < 0.01 and abs(z - 49.16) < 0.05)
    print(f"oa_feed  corn={100.0 * x[0]:.4g}% lime={100.0 * x[1]:.4g}% soy={100.0 * x[2]:.4g}% "
          f"cost={z:.4g} c/kg (book 65/3/32, 49.16) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
