#!/usr/bin/env python3
"""MOSEK port "reoptimization" - solve, change a bound, re-optimize.

Sequence (verified by hand):
  1. max x0 + x1 s.t. x0 + x1 <= 4, x >= 0     -> obj 4
  2. the row bound becomes 10                   -> obj 10
  3. the row becomes 2                          -> obj 2
Port of examples/c/reoptimization.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    steps = [4.0, 10.0, 2.0]
    expect = [4.0, 10.0, 2.0]
    ok = True
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.a_row(0, [0, 1], [1.0, 1.0])
        for k in range(3):
            m.con_bounds(0, BK.UP, -math.inf, steps[k])
            r = m.solve()
            if r.rc != 0:
                print(f"optimize {k} rc={r.rc}")
                return 1
            print(f"step {k + 1} (rhs={steps[k]:g}): obj = {r.objective:.4f} (expected {expect[k]:.4f})")
            if abs(r.objective - expect[k]) > 1e-6:
                ok = False
    print(f"{'OK' if ok else 'FAIL'} (reoptimization: 3 successive solves)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
