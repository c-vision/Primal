#!/usr/bin/env python3
"""IntroLO (Nemirovski) Section 4.3.2 - worked simplex LP.

    max 5 x1 + 3 x2 + 6 x3  s.t. 3 rows, x >= 0.
Book optimum (6,12,6), value 102.   Port of c_examples/books/intlo_simplex.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA


def main() -> int:
    with Model(maxcon=3, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.cj(0, 5.0); m.cj(1, 3.0); m.cj(2, 6.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 2.0]); m.con_bounds(0, BK.UP, up=30.0)
        m.a_row(1, [0, 1, 2], [4.0, 1.0, 4.0]); m.con_bounds(1, BK.UP, up=60.0)
        m.a_row(2, [0, 1, 2], [2.0, 1.0, 1.0]); m.con_bounds(2, BK.UP, up=30.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 6) < 1e-6 and abs(r.x[1] - 12) < 1e-6
          and abs(r.x[2] - 6) < 1e-6 and abs(r.objective - 102) < 1e-6)
    print(f"intlo_simplex  x=({r.x[0]:.6g},{r.x[1]:.6g},{r.x[2]:.6g}) obj={r.objective:.6g} "
          f"(book (6,12,6), 102) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
