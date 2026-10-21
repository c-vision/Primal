#!/usr/bin/env python3
"""OMF Section 2.2 - first duality example.

    min -x1 - x2  s.t. 2 x1 + x2 <= 12, x1 + 2 x2 <= 9, x >= 0.
Book optimum x=(5,2), objective -7.   Port of c_examples/finance/omf_duality.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=2, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
        m.cj(0, -1.0); m.cj(1, -1.0)
        m.a_row(0, [0, 1], [2.0, 1.0]); m.con_bounds(0, BK.UP, up=12.0)
        m.a_row(1, [0, 1], [1.0, 2.0]); m.con_bounds(1, BK.UP, up=9.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 5) < 1e-6
          and abs(r.x[1] - 2) < 1e-6 and abs(r.objective + 7) < 1e-6)
    print(f"omf_duality  x=({r.x[0]:.6g},{r.x[1]:.6g}) obj={r.objective:.6g} "
          f"(book (5,2), -7) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
