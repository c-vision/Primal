#!/usr/bin/env python3
"""MOSEK port "solvelinear" - solve a linear system A x = b as an optimization problem.

min ||Ax - b||_inf via an LP: min t s.t. -t <= (Ax-b)_i <= t. 3x3 system (verified
by hand): A = [2 1 -1; -3 -1 2; -2 1 2], b = (8, -11, -3), solution x = (2, 3, -1);
the LP optimum is t = 0 at that x. Port of examples/c/solvelinear.c.
"""

import math

from primalsolver import Model, BK

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
B = [8, -11, -3]


def main() -> int:
    with Model() as m:
        m.add_vars(4)                               # x0, x1, x2, t
        for j in range(3):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.var_bounds(3, BK.LO, 0.0, math.inf)
        m.cj(3, 1.0)                                # min t
        m.add_cons(6)                               # (Ax)_i - t <= b_i, -(Ax)_i - t <= -b_i
        sub = [0, 1, 2, 3]
        for i in range(3):
            a = A[i]
            m.a_row(i, sub, [a[0], a[1], a[2], -1.0])
            m.con_bounds(i, BK.UP, -math.inf, B[i])
            m.a_row(3 + i, sub, [-a[0], -a[1], -a[2], -1.0])
            m.con_bounds(3 + i, BK.UP, -math.inf, -B[i])
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), t = {x[3]:.6f} (expected (2, 3, -1), t=0)")
    ok = (abs(x[0] - 2.0) < 1e-6 and abs(x[1] - 3.0) < 1e-6 and abs(x[2] + 1.0) < 1e-6
          and abs(r.objective) < 1e-6)
    print(f"{'OK' if ok else 'FAIL'} (linear system solved: residual ~ 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
