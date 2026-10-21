#!/usr/bin/env python3
"""MOSEK port "mi1" - two-variable MIP.

    max x1 + 0.64 x2  s.t. 50 x1 + 31 x2 <= 250, 3 x1 - 2 x2 >= -4, int >= 0.
Expected x=(5,0), obj=5.   Port of c_examples/mosek_comparison/mi1.c.
"""

from primalsolver import Model, BK, SENSE, VAR_TYPE, SOLSTA


def main() -> int:
    with Model(maxcon=2, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
            m.var_type(j, VAR_TYPE.INT)
        m.cj(0, 1.0); m.cj(1, 0.64)
        m.a_col(0, [0, 1], [50.0, 3.0])
        m.a_col(1, [0, 1], [31.0, -2.0])
        m.con_bounds(0, BK.UP, up=250.0)
        m.con_bounds(1, BK.LO, -4.0)
        m.sense(SENSE.MAX)
        r = m.solve()
    ok = (r.solsta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL)
          and abs(r.x[0] - 5) < 1e-6 and abs(r.x[1]) < 1e-6 and abs(r.objective - 5.0) < 1e-6)
    print(f"mi1  x1={r.x[0]:.0f}, x2={r.x[1]:.0f}, obj={r.objective:.4f} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
