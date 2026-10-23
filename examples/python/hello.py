#!/usr/bin/env python3
"""MOSEK port "hello" - the simplest possible task.

A free variable, min x, x in [2, 4]  ->  x = 2, obj = 2.
Port of examples/c/hello.c.
"""

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(1)
        m.cj(0, 1.0)
        m.var_bounds(0, BK.RA, 2.0, 4.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    print(f"x = {r.x[0]:.6f}, obj = {r.objective:.6f}")
    ok = abs(r.x[0] - 2.0) < 1e-6 and abs(r.objective - 2.0) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (expected x = 2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
