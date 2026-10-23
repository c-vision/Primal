#!/usr/bin/env python3
"""MOSEK port "gp1" - geometric programming via log-log transform and exponential cones.

GP:  min x0 * x1^-1   s.t.  x0 + x1 <= 1,  x0 >= 0.5,  x1 >= 0.5
With x_i = e^{y_i}: min e^{y0 - y1} s.t. log(e^{y0} + e^{y1}) <= 0, y_i >= log(0.5).
The only feasible point is x0 = x1 = 0.5, so the GP objective is 1.
Conic model: t >= e^{y0-y1} as (t, 1, y0-y1) in PEXP; the log-sum-exp as
(u_k, 1, y_k) in PEXP with sum u_k <= 1. Port of examples/c/gp1.c.
"""

import math

from primalsolver import Model, BK, CT

Y0, Y1, T, ONE, U0, U1 = range(6)
IPAR_INTPNT_MAX_ITERATIONS = 11


def main() -> int:
    with Model() as m:
        m.add_vars(6)                               # y0, y1 (log of the x), t, one, u0, u1
        m.var_bounds(Y0, BK.LO, math.log(0.5), math.inf)
        m.var_bounds(Y1, BK.LO, math.log(0.5), math.inf)
        m.var_bounds(T, BK.LO, 0.0, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        m.var_bounds(U0, BK.LO, 0.0, math.inf)
        m.var_bounds(U1, BK.LO, 0.0, math.inf)
        m.cj(T, 1.0)                                # min t
        m.add_cons(2)
        m.a_row(0, [U0, U1], [1.0, 1.0])            # u0 + u1 <= 1 (log-sum-exp <= log 1)
        m.con_bounds(0, BK.UP, -math.inf, 1.0)
        m.add_vars(1)                               # v = y0 - y1, an extra variable
        v = m.numvar() - 1
        m.var_bounds(v, BK.FR, -math.inf, math.inf)
        m.a_row(1, [v, Y0, Y1], [1.0, -1.0, 1.0])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.cone(CT.PEXP, [T, ONE, v])                # (t, one, v) in PEXP
        m.cone(CT.PEXP, [U0, ONE, Y0])              # (u_k, one, y_k) in PEXP
        m.cone(CT.PEXP, [U1, ONE, Y1])
        m.int_param(IPAR_INTPNT_MAX_ITERATIONS, 5000)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    xx, po = r.x, r.objective
    x0, x1 = math.exp(xx[Y0]), math.exp(xx[Y1])
    print(f"y = ({xx[Y0]:.4f}, {xx[Y1]:.4f}) -> x = ({x0:.4f}, {x1:.4f}), t = {xx[T]:.6f}")
    print(f"GP objective = x0/x1 = {x0 / x1:.6f} (expected 1), conic pobj = {po:.6f}")
    ok = (abs(x0 + x1 - 1.0) < 1e-3 and x0 >= 0.5 - 1e-6 and x1 >= 0.5 - 1e-6
          and abs(po - 1.0) < 1e-3 and abs(x0 / x1 - 1.0) < 1e-3)
    print(f"{'OK' if ok else 'FAIL'} (GP log-log: x=(0.5,0.5), obj=1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
