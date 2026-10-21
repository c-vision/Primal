#!/usr/bin/env python3
"""OMF Appendix D - revised-simplex worked LP.

    max x1 + 2x2 + x3 - 2x4  s.t. (3 equality rows), x1..x8 >= 0.
Book optimum x1=3, x2=5, x6=3, Z=13.   Port of c_examples/finance/omf_appendix_d.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA


def main() -> int:
    with Model(maxcon=3, maxvar=8) as m:
        for j in range(8):
            m.var_bounds(j, BK.LO, 0.0)
        m.sense(SENSE.MAX)
        m.cj(0, 1.0); m.cj(1, 2.0); m.cj(2, 1.0); m.cj(3, -2.0)
        m.a_row(0, [0, 1, 2, 3, 5], [-2.0, 1.0, 1.0, 2.0, 1.0]); m.con_bounds(0, BK.FX, 2.0, 2.0)
        m.a_row(1, [0, 1, 2, 4, 6], [-1.0, 2.0, 1.0, 1.0, 1.0]); m.con_bounds(1, BK.FX, 7.0, 7.0)
        m.a_row(2, [0, 2, 3, 4, 7], [1.0, 1.0, 1.0, 1.0, 1.0]); m.con_bounds(2, BK.FX, 3.0, 3.0)
        r = m.solve()
    x = r.x
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(x[0] - 3) < 1e-6 and abs(x[1] - 5) < 1e-6
          and abs(x[5] - 3) < 1e-6 and abs(r.objective - 13) < 1e-6
          and all(abs(x[j]) < 1e-6 for j in range(2, 8) if j != 5))
    print(f"omf_appendix_d  x1={x[0]:.6g} x2={x[1]:.6g} x6={x[5]:.6g} Z={r.objective:.6g} "
          f"(book 3,5,3,13) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
