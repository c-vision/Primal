#!/usr/bin/env python3
"""OMF Exercise 20.7 - relative robust portfolio (SOCP).

    min t  s.t. 5.662-(6x1+4x2)<=t, 5.662-(4x1+6x2)<=t, 5.0-(5x1+5x2)<=t,
              TE(x) <= 0.10, e'x=1, x>=0.
Book x=(0.5,0.5,0), t=0.662.   Port of c_examples/finance/omf_ex207.c.
"""

import math

from primalsolver import Model, BK, CT, SOLSTA

L11, L21, L22 = 0.42, 0.231, 0.235678147


def main() -> int:
    with Model(maxcon=7, maxvar=7) as m:   # x1,x2,x3,t,s=0.10,y1,y2
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.var_bounds(3, BK.FR)
        m.var_bounds(4, BK.FX, 0.10)
        m.var_bounds(5, BK.FR); m.var_bounds(6, BK.FR)
        m.cj(3, 1.0)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 3], [6.0, 4.0, 1.0]); m.con_bounds(1, BK.LO, 5.662)
        m.a_row(2, [0, 1, 3], [4.0, 6.0, 1.0]); m.con_bounds(2, BK.LO, 5.662)
        m.a_row(3, [0, 1, 3], [5.0, 5.0, 1.0]); m.con_bounds(3, BK.LO, 5.0)
        m.a_row(4, [0, 5], [-L11, 1.0]); m.con_bounds(4, BK.FX, -0.5 * L11, -0.5 * L11)
        m.a_row(5, [0, 1, 6], [-L21, -L22, 1.0]); m.con_bounds(5, BK.FX, -0.5 * (L21 + L22), -0.5 * (L21 + L22))
        m.cone(CT.QUAD, [4, 5, 6])
        r = m.solve()
    x = r.x
    te = math.hypot(x[5], x[6])
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(x[0] - 0.5) < 1e-4 and abs(x[1] - 0.5) < 1e-4
          and abs(x[2]) < 1e-4 and abs(r.objective - 0.662) < 1e-3 and te <= 0.10 + 1e-6)
    print(f"omf_ex207  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g}) t={r.objective:.6g} TE={te:.6g} "
          f"(book (0.5,0.5,0), 0.662) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
