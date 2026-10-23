#!/usr/bin/env python3
"""MOSEK port "parallel" - several independent tasks.

The tasks are solved in sequence (documented deviation from the reference, which
runs them in parallel threads: the example checks task isolation). Three
independent tasks with known optima (verified by hand):
  A: max x s.t. x <= 4         -> 4
  B: min x s.t. x >= 2         -> 2
  C: max x0+x1 s.t. x0+x1 <= 7 -> 7
Port of examples/c/parallel.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as a, Model() as b, Model() as c:
        a.add_vars(1); a.add_cons(1)                # task A: max x s.t. x <= 4
        a.cj(0, 1.0)
        a.sense(SENSE.MAX)
        a.a_row(0, [0], [1.0]); a.con_bounds(0, BK.UP, -math.inf, 4.0)

        b.add_vars(1); b.add_cons(1)                # task B: min x s.t. x >= 2
        b.cj(0, 1.0)
        b.a_row(0, [0], [1.0]); b.con_bounds(0, BK.LO, 2.0, math.inf)

        c.add_vars(2); c.add_cons(1)                # task C: max x0+x1 s.t. x0+x1 <= 7
        c.sense(SENSE.MAX)
        c.cj(0, 1.0); c.cj(1, 1.0)
        c.a_row(0, [0, 1], [1.0, 1.0]); c.con_bounds(0, BK.UP, -math.inf, 7.0)

        ra, rb, rc = a.solve(), b.solve(), c.solve()
    print(f"A: rc={ra.rc} pobj={ra.objective:.4f} (expected 4)")
    print(f"B: rc={rb.rc} pobj={rb.objective:.4f} (expected 2)")
    print(f"C: rc={rc.rc} pobj={rc.objective:.4f} (expected 7)")
    ok = ((ra.rc == 0 and abs(ra.objective - 4.0) < 1e-9)
          and (rb.rc == 0 and abs(rb.objective - 2.0) < 1e-9)
          and (rc.rc == 0 and abs(rc.objective - 7.0) < 1e-9))
    print(f"{'OK' if ok else 'FAIL'} (3 independent tasks, isolation verified)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
