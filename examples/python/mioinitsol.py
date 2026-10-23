#!/usr/bin/env python3
"""MOSEK port "mioinitsol" - provide an initial solution to the MIP.

    max 3x0 + 2x1 + 2x2
    s.t. x0 + x1 + x2 <= 10, x0 + 2x1 <= 9, -x0 + 3x1 + x2 <= 10
         x0, x1 integer >= 0, x2 >= 0

The initial solution x = (3, 3, 1) is feasible and integer (rows 7 <= 10, 9 <= 9,
10 <= 10; objective 17). The optimum the solver returns is confirmed by complete
enumeration. Port of examples/c/mioinitsol.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

C = [3.0, 2.0, 2.0]


def main() -> int:
    x0 = [3.0, 3.0, 1.0]                            # initial solution (feasible and integer)
    with Model() as m:
        m.add_vars(3)
        m.add_cons(3)
        m.sense(SENSE.MAX)
        for j in range(3):
            m.cj(j, C[j])
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]);  m.con_bounds(0, BK.UP, -math.inf, 10.0)
        m.a_row(1, [0, 1], [1.0, 2.0]);          m.con_bounds(1, BK.UP, -math.inf, 9.0)
        m.a_row(2, [0, 1, 2], [-1.0, 3.0, 1.0]); m.con_bounds(2, BK.UP, -math.inf, 10.0)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_type(0, VAR_TYPE.INT)
        m.var_type(1, VAR_TYPE.INT)
        m.warm_start(x=x0)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), obj = {obj:.4f} "
          f"(initial obj = {3.0 * x0[0] + 2.0 * x0[1] + 2.0 * x0[2]:.4f})")
    best = -1e30                                    # complete enumeration (x2 continuous)
    for a in range(11):
        for b in range(6):
            if a + 2 * b > 9.0 + 1e-9:
                continue
            v = min(10.0 - a - b, 10.0 + a - 3.0 * b)
            if v > 1e-9:
                best = max(best, 3.0 * a + 2.0 * b + 2.0 * v)
    print(f"enumeration: best = {best:.4f}")
    ok = (abs(obj - best) < 1e-6 and abs(x[0] - math.floor(x[0] + 0.5)) < 1e-6
          and abs(x[1] - math.floor(x[1] + 0.5)) < 1e-6)
    print(f"{'OK' if ok else 'FAIL'} (optimum confirmed, initsol accepted)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
