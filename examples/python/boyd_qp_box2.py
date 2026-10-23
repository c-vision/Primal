#!/usr/bin/env python3
"""Barrier-method example, "Convex Optimization" (Boyd & Vandenberghe, 2004), Ex. 11.1.

    minimize x^2 + 1  subject to  2 <= x <= 4

The book states the feasible set is [2, 4] and the optimal point x* = 2 (objective
x^2 = 4 plus the constant 1 gives 5). Port of examples/c/boyd_qp_box2.c.
"""

from primalsolver import Model, BK


def main() -> int:
    with Model() as m:
        m.add_vars(1)
        m.var_bounds(0, BK.RA, 2.0, 4.0)
        m.cj(0, 0.0)
        m.qobj([(0, 0, 2.0)])                      # 0.5*2*x^2 = x^2
        r = m.solve()
    if r.rc != 0:
        print(f"boyd_qp_box2  rc={r.rc} FAIL")
        return 1
    ok = abs(r.x[0] - 2.0) < 1e-5
    print(f"boyd_qp_box2  x={r.x[0]:.6g} f={r.objective + 1.0:.6g} (book x*=2, f=5) "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
