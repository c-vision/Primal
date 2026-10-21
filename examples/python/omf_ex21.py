#!/usr/bin/env python3
"""OMF Example 2.1 (Cornuejols & Tutuncu) - the running bond-portfolio LP.

    max  Z = 4 x1 + 3 x2
    s.t. x1 + x2 <= 100;  2 x1 + x2 <= 150;  3 x1 + 4 x2 <= 360;  x >= 0.
Book optimum x=(50,50), Z=350.   Port of c_examples/finance/omf_ex21.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA


def main() -> int:
    with Model(maxcon=3, maxvar=2) as m:
        m.var_bounds(0, BK.LO, 0.0)
        m.var_bounds(1, BK.LO, 0.0)
        m.cj(0, 4.0); m.cj(1, 3.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.UP, up=100.0)
        m.a_row(1, [0, 1], [2.0, 1.0]); m.con_bounds(1, BK.UP, up=150.0)
        m.a_row(2, [0, 1], [3.0, 4.0]); m.con_bounds(2, BK.UP, up=360.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 50) < 1e-6
          and abs(r.x[1] - 50) < 1e-6 and abs(r.objective - 350) < 1e-6)
    print(f"omf_ex21  x=({r.x[0]:.6g},{r.x[1]:.6g}) Z={r.objective:.6g} "
          f"(book (50,50), 350) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
