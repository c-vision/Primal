#!/usr/bin/env python3
"""Risk budgeting: target risk contributions.

Find w > 0 with w_i (Sigma w)_i = b_i (w'Sigma w). The convex formulation (Spinu 2013)
is  min (1/2) w'Sigma w - sum_i b_i ln(w_i),  normalised after; the logarithms use the
exponential cone s_i >= -ln(w_i) <=> (w_i, 1, -s_i) in K_exp. Hand case: Sigma = I,
b = (0.8, 0.2): w ~ (sqrt(0.8), sqrt(0.2)), normalised (2/3, 1/3), risk contributions
exactly (0.8, 0.2). Port of examples/c/risk_budgeting.c.
"""

import math

from primalsolver import Model, BK, CT


def main() -> int:
    b = [0.8, 0.2]
    with Model() as m:
        m.add_vars(7)                               # w0, w1, s0, s1, v0, v1, one
        m.add_cons(2)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        for j in range(2, 6):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.var_bounds(6, BK.FX, 1.0, 1.0)
        # (w_i, one, v_i) in K_exp  =>  w_i >= exp(v_i) = exp(-s_i)
        m.cone(CT.PEXP, [0, 6, 4])
        m.cone(CT.PEXP, [1, 6, 5])
        m.a_row(0, [4, 2], [1.0, 1.0]); m.con_bounds(0, BK.FX, 0.0, 0.0)   # v_i + s_i = 0
        m.a_row(1, [5, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 0.0, 0.0)
        # objective: (1/2)(w0^2 + w1^2) + b0 s0 + b1 s1
        m.qobj([(0, 0, 2.0)])
        m.qobj([(1, 1, 2.0)])
        m.cj(2, b[0])
        m.cj(3, b[1])
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    x = r.x
    w0 = x[0] / (x[0] + x[1])
    w1 = x[1] / (x[0] + x[1])
    rc0 = (w0 * w0) / (w0 * w0 + w1 * w1)
    rc1 = (w1 * w1) / (w0 * w0 + w1 * w1)
    # the conic IPM delivers ~1e-5 here; the budgets are the meaningful check
    ok = abs(w0 - 2.0 / 3.0) < 1e-3 and abs(w1 - 1.0 / 3.0) < 1e-3 and abs(rc0 - b[0]) < 1e-4 and abs(rc1 - b[1]) < 1e-4
    print("w = (%.6f, %.6f) (expected 0.6667, 0.3333)" % (w0, w1))
    print("risk contributions = (%.6f, %.6f) (expected 0.8, 0.2)" % (rc0, rc1))
    print("%s (risk budgeting, log via exponential cone)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
