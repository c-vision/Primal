#!/usr/bin/env python3
"""OMF Section 5 (GRG) - quadratic program.

    min (x1-1/2)^2 + (x2-5/2)^2  s.t. x1 - x2 >= 0, x1 >= 0, 0 <= x2 <= 2.
Expanding: 0.5 x'Qx + c'x with Q=2I, c=(-1,-5). Book optimum x=(1.5,1.5), f=2.
Port of c_examples/finance/omf_qp_grg.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=1, maxvar=2) as m:
        m.var_bounds(0, BK.LO, 0.0)
        m.var_bounds(1, BK.RA, 0.0, 2.0)
        m.cj(0, -1.0); m.cj(1, -5.0)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])
        m.a_row(0, [0, 1], [1.0, -1.0]); m.con_bounds(0, BK.LO, 0.0)
        r = m.solve()
    f = r.objective + 6.5
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 1.5) < 1e-6
          and abs(r.x[1] - 1.5) < 1e-6 and abs(f - 2.0) < 1e-6
          and r.x[0] - r.x[1] >= -1e-6)
    print(f"omf_qp_grg  x=({r.x[0]:.6g},{r.x[1]:.6g}) f={f:.6g} "
          f"(book (1.5,1.5), 2) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
