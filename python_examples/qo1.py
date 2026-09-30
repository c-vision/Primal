#!/usr/bin/env python3
"""MOSEK port "qo1" - QP.

    min x^2 + y^2  s.t. x + y >= 1, x,y >= 0.  Expected x=y=0.5, obj=0.5.
Port of c_examples/mosek_comparison/qo1.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=1, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])   # Q = diag(2): 0.5 x'Qx = x^2+y^2
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.LO, 1.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0] - 0.5) < 1e-6
          and abs(r.x[1] - 0.5) < 1e-6 and abs(r.objective - 0.5) < 1e-6)
    print(f"qo1  x={r.x[0]:.6f}, y={r.x[1]:.6f}, obj={r.objective:.6f} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
