#!/usr/bin/env python3
"""MOSEK port "milo1" - mixed MIP (some variables integer, some continuous).

    max 3x0 + 2x1 + 2x2
    s.t.  x0 + x1 + x2 <= 10
          x0 + 2x1      <=  9
         -x0 + 3x1 + x2 <= 10
         x0, x1 integer >= 0, x2 continuous >= 0

Check: solsta INTEGER_OPTIMAL, integrality of x0 and x1, objective consistent with
x, and optimality against the complete enumeration of (x0, x1) with x2 optimized.
Port of examples/c/milo1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA, VAR_TYPE

C = [3.0, 2.0, 2.0]


def main() -> int:
    with Model() as m:
        m.add_vars(3)
        m.add_cons(3)
        for j in range(3):
            m.cj(j, C[j])
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])
        m.a_row(1, [0, 1], [1.0, 2.0])
        m.a_row(2, [0, 1, 2], [-1.0, 3.0, 1.0])
        m.con_bounds(0, BK.UP, -math.inf, 10.0)
        m.con_bounds(1, BK.UP, -math.inf, 9.0)
        m.con_bounds(2, BK.UP, -math.inf, 10.0)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_type(0, VAR_TYPE.INT)
        m.var_type(1, VAR_TYPE.INT)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), obj = {obj:.4f}")
    ok = r.solsta == SOLSTA.INTEGER_OPTIMAL
    best = -1e30
    if ok:
        for a in range(21):
            for b in range(21):
                if a + 2 * b > 9.0 + 1e-9:          # row 1
                    continue
                v = min(10.0 - a - b, 10.0 + a - 3.0 * b)   # x2 from rows 0 and 2
                if v > 0 and 3.0 * a + 2.0 * b + 2.0 * v > best:
                    best = 3.0 * a + 2.0 * b + 2.0 * v
    print(f"enumeration: best = {best:.4f}, clone = {obj:.4f}")
    if abs(obj - best) > 1e-6:
        ok = False
    if abs(x[0] - math.floor(x[0] + 0.5)) > 1e-6 or abs(x[1] - math.floor(x[1] + 0.5)) > 1e-6:
        ok = False
    if abs(obj - (3 * x[0] + 2 * x[1] + 2 * x[2])) > 1e-6:
        ok = False
    print(f"{'OK' if ok else 'FAIL'} (optimum confirmed by enumeration)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
