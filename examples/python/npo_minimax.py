#!/usr/bin/env python3
"""Minimax (L-infinity) regression as an LP, "Nonlinear Parameter Optimization Using
R Tools" (2014), Section 14.4.

    x* = argmin_x  max_i | A x - y |_i

with A 10x3 (columns 1, i, i^2 for i = 1..10) and the perturbed response y of the
L1 example (Section 14.3). The book reports coeffs = (2.73966, -1.92954, -1.00881),
mad = 0.726338. LP form: x free, t free; -t <= (Ax - y)_i <= t.
Port of examples/c/npo_minimax.c.
"""

import math

from primalsolver import Model, BK

Y = [0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
     -45.193280, -60.690202, -77.243728, -96.406035, -117.241829]


def main() -> int:
    with Model() as m:
        m.add_vars(4)                               # 3 x + t
        m.add_cons(20)
        for j in range(4):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.cj(3, 1.0)                                # min t
        for i in range(10):
            a1, a2 = 1.0, float(i + 1)
            a3 = a2 * a2
            # (A x)_i - y_i <= t  =>  a_i.x - t <= y_i
            m.a_row(2 * i, [0, 1, 2, 3], [a1, a2, a3, -1.0])
            m.con_bounds(2 * i, BK.UP, -math.inf, Y[i])
            # -(A x)_i + y_i <= t  =>  -a_i.x - t <= -y_i
            m.a_row(2 * i + 1, [0, 1, 2, 3], [-a1, -a2, -a3, -1.0])
            m.con_bounds(2 * i + 1, BK.UP, -math.inf, -Y[i])
        r = m.solve()
    if r.rc != 0:
        print(f"npo_minimax  rc={r.rc} FAIL")
        return 1
    v, z = r.x, r.objective
    ok = (abs(v[0] - 2.73966) < 1e-4 and abs(v[1] + 1.92954) < 1e-4
          and abs(v[2] + 1.00881) < 1e-4 and abs(z - 0.726338) < 1e-5)
    print(f"npo_minimax  coeffs=({v[0]:.6g},{v[1]:.6g},{v[2]:.6g}) mad={z:.6g} "
          f"(book (2.73966,-1.92954,-1.00881), 0.726338) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
