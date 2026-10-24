#!/usr/bin/env python3
"""Mean-CVaR portfolio.

    max mu'w - lambda CVaR_beta(w)  s.t.  e'w = 1,  w >= 0

with the Rockafellar-Uryasev CVaR linearised: one LP in (w, alpha, z) with
z_s >= -r_s'w - alpha, z_s >= 0. Hand case: 2 assets, 2 equiprobable scenarios,
beta = 1/2, lambda = 1, mu = (0.2, 0.1), r0 = (0.1, 0.3), r1 = (0.4, -0.1): the optimum
sits at the kink r0'w = r1'w, w = (4/7, 3/7), value 12/35. Port of examples/c/mean_cvar.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    mu = [0.2, 0.1]
    R0 = [0.1, 0.3]
    R1 = [0.4, -0.1]
    with Model() as m:
        m.add_vars(5)                           # w0, w1, alpha, z0, z1
        m.add_cons(3)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.FR, -math.inf, math.inf)
        m.var_bounds(3, BK.LO, 0.0, math.inf)
        m.var_bounds(4, BK.LO, 0.0, math.inf)
        m.cj(0, mu[0])                          # objective: max mu'w - alpha - z0 - z1
        m.cj(1, mu[1])
        m.cj(2, -1.0)
        m.cj(3, -1.0)
        m.cj(4, -1.0)
        m.sense(SENSE.MAX)
        # z_s + r_s'w + alpha >= 0
        m.a_row(0, [3, 0, 1, 2], [1.0, R0[0], R0[1], 1.0]); m.con_bounds(0, BK.LO, 0.0, math.inf)
        m.a_row(1, [4, 0, 1, 2], [1.0, R1[0], R1[1], 1.0]); m.con_bounds(1, BK.LO, 0.0, math.inf)
        m.a_row(2, [0, 1], [1.0, 1.0]); m.con_bounds(2, BK.FX, 1.0, 1.0)   # e'w = 1
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    obj, x = r.objective, r.x
    w0, w1 = x[0], x[1]
    cvar = max(-(R0[0] * w0 + R0[1] * w1), -(R1[0] * w0 + R1[1] * w1))
    ok = (abs(w0 - 4.0 / 7.0) < 1e-6 and abs(w1 - 3.0 / 7.0) < 1e-6 and abs(obj - 12.0 / 35.0) < 1e-6
          and abs((mu[0] * w0 + mu[1] * w1) - cvar - obj) < 1e-6)
    print("w = (%.6f, %.6f) (expected 0.5714, 0.4286)" % (w0, w1))
    print("obj = %.6f (expected 12/35 = %.6f), CVaR = %.6f" % (obj, 12.0 / 35.0, cvar))
    print("%s (mean-CVaR, optimum at the kink r0'w = r1'w)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
