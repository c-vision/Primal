#!/usr/bin/env python3
"""L1 (least absolute deviations) regression as an LP, "Nonlinear Parameter
Optimization Using R Tools" (2014), Section 14.3.

    x* = argmin_x  sum_i | A x - y |_i

with A 10x3, columns (1, i, i^2) for i = 1..10, and a perturbed response y.
The book solves it via an LP with z >= +/-(Ax - y) and reports
coeffs = (3.489517, -2.174797, -0.989834), sad = 3.2951.
The sample builds the LP (x free, z >= 0) and checks both.
Port of examples/c/npo_lad.c.
"""

import math

from primalsolver import Model, BK

Y = [0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
     -45.193280, -60.690202, -77.243728, -96.406035, -117.241829]


def main() -> int:
    with Model() as m:
        m.add_vars(13)                              # 3 x (free) + 10 z (>= 0)
        m.add_cons(20)
        for j in range(3):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        for j in range(3, 13):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.cj(j, 1.0)
        for i in range(10):
            a1, a2 = 1.0, float(i + 1)              # row [1, i, i^2]
            a3 = a2 * a2
            # z_i - (A x)_i >= -y_i
            m.a_row(2 * i, [0, 1, 2, 3 + i], [-a1, -a2, -a3, 1.0])
            m.con_bounds(2 * i, BK.LO, -Y[i], math.inf)
            # z_i + (A x)_i >= y_i
            m.a_row(2 * i + 1, [0, 1, 2, 3 + i], [a1, a2, a3, 1.0])
            m.con_bounds(2 * i + 1, BK.LO, Y[i], math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"npo_lad  rc={r.rc} FAIL")
        return 1
    v, z = r.x, r.objective
    ok = (abs(v[0] - 3.489517) < 1e-4 and abs(v[1] + 2.174797) < 1e-4
          and abs(v[2] + 0.989834) < 1e-4 and abs(z - 3.2951) < 1e-4)
    print(f"npo_lad  coeffs=({v[0]:.6g},{v[1]:.6g},{v[2]:.6g}) sad={z:.6g} "
          f"(book (3.489517,-2.174797,-0.989834), 3.2951) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
