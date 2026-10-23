#!/usr/bin/env python3
"""MOSEK port "acc1" - affine conic constraints (ACC).

    min  x1
    s.t. x0 = 3 (FX row)
         ACC: (v0, v1) in QUAD with v0 = 2*x0 + x1, v1 = x0 - 1
    => v0 = 6 + x1 >= |v1| = |2| = 2  =>  x1 >= -4

Optimum: x = (3, -4), obj = -4. Port of examples/c/acc1.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        m.var_bounds(0, BK.FX, 3.0, 3.0)
        m.var_bounds(1, BK.FR, -math.inf, math.inf)
        m.cj(1, 1.0)
        m.a_row(0, [0], [1.0])
        m.con_bounds(0, BK.FX, 3.0, 3.0)
        # ACC: (v0, v1) in QUAD, v0 = 2*x0 + x1, v1 = x0 - 1 (via AFE + domain)
        m.afes(2)
        m.afe_entry(0, 0, 2.0)
        m.afe_entry(0, 1, 1.0)
        m.afe_entry(1, 0, 1.0)
        m.afe_g(1, -1.0)
        dom = ctypes.c_longlong()
        if ps.PRIMAL_appendquadraticconedomain(m.task, 2, ctypes.byref(dom)) != 0:
            print("domain rc")
            return 1
        m.acc(dom.value, [0, 1], [0.0, 0.0])
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        nv = min(m.numvar(), 16)                    # the ACC auxiliaries add variables
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}), obj = {obj:.4f} (numvar={nv} incl. ACC auxiliaries)")
    v0 = 2.0 * x[0] + x[1]                          # external check: v0 >= |v1|
    v1 = x[0] - 1.0
    ok = (abs(x[0] - 3.0) < 1e-6 and abs(x[1] + 4.0) < 1e-4 and abs(obj + 4.0) < 1e-4
          and v0 >= abs(v1) - 1e-6)
    print(f"v0 = {v0:.4f}, |v1| = {abs(v1):.4f} (ACC satisfied)")
    print(f"{'OK' if ok else 'FAIL'} (expected x=(3,-4), obj=-4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
