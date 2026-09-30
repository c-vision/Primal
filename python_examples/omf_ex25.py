#!/usr/bin/env python3
"""OMF Exercise 2.5 (duality).

    min 2 x1 + 3 x2  s.t. x1 + x2 >= 5, x1 >= 1, x2 >= 2.
Book optimum x=(3,2), value 12.   Port of c_examples/finance/omf_ex25.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=1, maxvar=2) as m:
        m.var_bounds(0, BK.LO, 1.0)
        m.var_bounds(1, BK.LO, 2.0)
        m.cj(0, 2.0); m.cj(1, 3.0)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.LO, 5.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 3) < 1e-6
          and abs(r.x[1] - 2) < 1e-6 and abs(r.objective - 12) < 1e-6)
    print(f"omf_ex25  x=({r.x[0]:.6g},{r.x[1]:.6g}) obj={r.objective:.6g} "
          f"(book (3,2), 12) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
