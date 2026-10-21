#!/usr/bin/env python3
"""MOSEK port "lo1" - simple LP.

    min 2x + 3y  s.t. x + y >= 1, x - y <= 0, x,y >= 0.
Expected x=y=0.5, obj=2.5.   Port of c_examples/mosek_comparison/lo1.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=2, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
        m.cj(0, 2.0); m.cj(1, 3.0)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.LO, 1.0)
        m.a_row(1, [0, 1], [1.0, -1.0]); m.con_bounds(1, BK.UP, up=0.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 0.5) < 1e-6
          and abs(r.x[1] - 0.5) < 1e-6 and abs(r.objective - 2.5) < 1e-6)
    print(f"lo1  x={r.x[0]:.6f}, y={r.x[1]:.6f}, obj={r.objective:.6f} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
