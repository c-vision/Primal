#!/usr/bin/env python3
"""MOSEK port "simple" - the simplest possible LP, checking the whole API chain.

    max x0 + x1  s.t.  x0 + x1 = 1,  0 <= x0, x1 <= 1

Optimum: any point on the row with obj = 1 (degenerate solution: check the primal
objective, strong duality and feasibility, not the point).
Port of examples/c/simple.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.var_bounds(0, BK.RA, 0.0, 1.0)
        m.var_bounds(1, BK.RA, 0.0, 1.0)
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, y, po, dobj = r.x, r.y, r.objective, r.dual
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}), y = ({y[0]:.4f})")
    print(f"primal obj = {po:.4f}, dual obj = {dobj:.4f}, sta = {r.solsta}")
    ok = (r.solsta == SOLSTA.OPTIMAL
          and abs(po - 1.0) < 1e-9
          and abs(dobj - 1.0) < 1e-6                # strong duality
          and abs(x[0] + x[1] - 1.0) < 1e-6         # feasibility
          and abs(x[0] - (1.0 - x[1])) < 1e-6)
    print(f"{'OK' if ok else 'FAIL'} (expected obj=1, pobj=dobj, feasible)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
