#!/usr/bin/env python3
"""OMF Exercise 2.17 (simplex).

    max 4 x1 + x2 - x3
    s.t. x1 + 3 x3 <= 6;  3 x1 + x2 + 3 x3 <= 9;  x >= 0.
Book optimum x=(3,0,0), objective 12.   Port of c_examples/finance/omf_ex217.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA


def main() -> int:
    with Model(maxcon=2, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.cj(0, 4.0); m.cj(1, 1.0); m.cj(2, -1.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 2], [1.0, 3.0]); m.con_bounds(0, BK.UP, up=6.0)
        m.a_row(1, [0, 1, 2], [3.0, 1.0, 3.0]); m.con_bounds(1, BK.UP, up=9.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 3) < 1e-6
          and abs(r.x[1]) < 1e-6 and abs(r.x[2]) < 1e-6 and abs(r.objective - 12) < 1e-6)
    print(f"omf_ex217  x=({r.x[0]:.6g},{r.x[1]:.6g},{r.x[2]:.6g}) obj={r.objective:.6g} "
          f"(book (3,0,0), 12) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
