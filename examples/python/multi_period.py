#!/usr/bin/env python3
"""Two-period portfolio with transaction costs.

    max sum_{t=0,1} mu_t'w_t - c (|w_0 - w_init|_1 + |w_1 - w_0|_1)
    s.t.  e'w_t = 1,  w_t >= 0

with the absolute values linearised by z_t >= +-(w_t - w_{t-1}), z_t >= 0. Hand case:
mu = (0.5, 0.1), w_init = (0.5, 0.5), c = 0.1: move to (1, 0) in the first period
(cost 0.1, gain 0.2), stay there: value 0.9. Port of examples/c/multi_period.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    mu = [0.5, 0.1]
    w0init = [0.5, 0.5]
    cost = 0.1
    with Model() as m:
        m.add_vars(8)                           # w0(0,1), w1(2,3), z0(4,5), z1(6,7)
        m.add_cons(10)
        for j in range(8):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.sense(SENSE.MAX)
        for j in range(2):
            m.cj(j, mu[j])
            m.cj(2 + j, mu[j])
            m.cj(4 + j, -cost)
            m.cj(6 + j, -cost)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)   # e'w_0 = 1
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 1.0, 1.0)   # e'w_1 = 1
        for j in range(2):                      # z0 >= +-(w0 - w_init)
            m.a_row(2 + j, [j, 4 + j], [1.0, -1.0])
            m.con_bounds(2 + j, BK.UP, -math.inf, w0init[j])
            m.a_row(4 + j, [j, 4 + j], [-1.0, -1.0])
            m.con_bounds(4 + j, BK.UP, -math.inf, -w0init[j])
        for j in range(2):                      # z1 >= +-(w1 - w0)
            m.a_row(6 + j, [2 + j, j, 6 + j], [1.0, -1.0, -1.0])
            m.con_bounds(6 + j, BK.UP, -math.inf, 0.0)
            m.a_row(8 + j, [2 + j, j, 6 + j], [-1.0, 1.0, -1.0])
            m.con_bounds(8 + j, BK.UP, -math.inf, 0.0)
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    obj, x = r.objective, r.x
    turnover0 = abs(x[0] - w0init[0]) + abs(x[1] - w0init[1])
    turnover1 = abs(x[2] - x[0]) + abs(x[3] - x[1])
    ok = (abs(obj - 0.9) < 1e-7 and abs(x[0] - 1.0) < 1e-6 and abs(x[1]) < 1e-6 and abs(x[2] - 1.0) < 1e-6
          and abs(x[3]) < 1e-6 and abs(turnover0 - 1.0) < 1e-6 and abs(turnover1) < 1e-6)
    print("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (expected 0.9)" % (x[0], x[1], x[2], x[3], obj))
    print("turnover = %.4f then %.4f (expected 1.0, 0.0)" % (turnover0, turnover1))
    print("%s (multi-period: the second period is already on target)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
