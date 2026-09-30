#!/usr/bin/env python3
"""PrimalSolver - Python quickstart.

    min  -3 x0 - 2 x1
    s.t.  x0 +  x1 <= 4
          x0 + 3x1 <= 6
          0 <= x0, x1 <= 3
    optimum: x = (3, 1), objective = -11

Run (after `pip install primalsolver`):

    python python_examples/quickstart.py
"""

from primalsolver import BK, Model


def main() -> None:
    with Model(maxcon=2, maxvar=2) as m:
        m.obj([0, 1], [-3.0, -2.0])            # objective
        m.a_ij(0, 0, 1.0); m.a_ij(0, 1, 1.0)   # x0 +  x1 <= 4
        m.a_ij(1, 0, 1.0); m.a_ij(1, 1, 3.0)   # x0 + 3x1 <= 6
        m.con_bounds(0, BK.UP, up=4.0)
        m.con_bounds(1, BK.UP, up=6.0)
        m.var_bounds(0, BK.RA, 0.0, 3.0)       # 0 <= x0 <= 3 (RA = ranged)
        m.var_bounds(1, BK.RA, 0.0, 3.0)
        r = m.solve()

    print("x =", [round(v, 6) for v in r.x], " objective =", round(r.objective, 6))


if __name__ == "__main__":
    main()
