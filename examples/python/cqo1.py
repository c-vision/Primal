#!/usr/bin/env python3
"""MOSEK port "cqo1" - small conic problem with one quadratic and one rotated cone.

The Julia example uses the ACC API; here the ACCs are cones over direct variables:

    min  x3 + x4 + x5
    s.t. x0 + x1 + 2*x2 = 1
         (x3, x0, x1) in QUAD_3      -> x3 >= sqrt(x0^2 + x1^2)
         (x4, x5, x2) in RQUAD_3     -> 2*x4*x5 >= x2^2
         x0,x1,x2 >= 0; x3,x4,x5 free

Known optimum (from the MOSEK docs): obj = 1/sqrt(2) ~ 0.7071; both the objective
and the cone membership are checked. Port of examples/c/cqo1.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE


def main() -> int:
    with Model() as m:
        m.add_vars(6)
        m.add_cons(1)
        m.sense(SENSE.MIN)
        m.obj([3, 4, 5], [1.0, 1.0, 1.0])
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        for j in range(3, 6):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 2.0])      # x0 + x1 + 2*x2 = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.cone(CT.QUAD, [3, 0, 1])                  # QUAD_3: (x3, x0, x1)
        m.cone(CT.RQUAD, [4, 5, 2])                 # RQUAD_3: (x4, x5, x2)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, po = r.x, r.objective
    print("x = (" + ", ".join(f"{v:.6f}" for v in x[:6]) + ")")
    print(f"obj = {po:.6f} (expected 1/sqrt(2) = {1.0 / math.sqrt(2.0):.6f})")
    ok = (abs(x[0] + x[1] + 2 * x[2] - 1.0) < 1e-6 and abs(po - 1.0 / math.sqrt(2.0)) < 1e-6
          and x[3] * x[3] >= x[0] * x[0] + x[1] * x[1] - 1e-8
          and 2 * x[4] * x[5] >= x[2] * x[2] - 1e-8
          and abs(x[3] + x[4] + x[5] - po) < 1e-9)
    print(f"{'OK' if ok else 'FAIL'} (equality + QUAD/RQUAD cones + optimum)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
