#!/usr/bin/env python3
"""Equilibrium of masses hanging on strings (SOCP).

Source: MOSEK/Tutorials, equilibrium/equilibrium.py. Masses at positions x_i in R^2
are joined by strings of given maximum lengths; the system settles in the
minimum-energy (lowest) configuration:

    minimize    g * sum_i w_i x_i,y
    subject to  || x_i - x_j ||_2 <= l_ij   for every string (i,j)
                x_i = f_i                   for the pinned masses

The string constraint is the quadratic cone (l_ij, x_i - x_j) in Q^3. This sample
pins two masses at (-1,0) and (1,0) and hangs one free mass from two strings of
length l = 2; by symmetry the mass settles directly below the midpoint at
y = -sqrt(l^2 - 1) = -sqrt(3). Port of examples/c/equilibrium.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

LSTR = 2.0
X0, X1, D0, D1, L, NV = 0, 1, 2, 4, 6, 7


def solve():
    """One free mass at (x0, x1) joined to two fixed anchors; minimize x1."""
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(4)
        for k in range(6):
            m.var_bounds(k, BK.FR, -math.inf, math.inf)
        m.var_bounds(L, BK.FX, LSTR, LSTR)
        # D0 = (x0,x1) - (-1,0) ; D1 = (x0,x1) - (1,0)
        m.a_row(0, [D0, X0], [1.0, -1.0]);     m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [D0 + 1, X1], [1.0, -1.0]); m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [D1, X0], [1.0, -1.0]);     m.con_bounds(2, BK.FX, -1.0, -1.0)
        m.a_row(3, [D1 + 1, X1], [1.0, -1.0]); m.con_bounds(3, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [L, D0, D0 + 1])
        m.cone(CT.QUAD, [L, D1, D1 + 1])
        m.cj(X1, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, 0.0
    return True, r.x[X0], r.x[X1]


def main() -> int:
    ok, x0, y = solve()
    want = -math.sqrt(LSTR * LSTR - 1.0)           # -sqrt(3)
    okall = ok and abs(y - want) < 1e-7 and abs(x0) < 1e-6
    print(f"equilibrium  l={LSTR:.1f}")
    print(f"  free mass = ({x0:.8f}, {y:.8f})   (expected (0, -sqrt(3)={want:.8f}))")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
