#!/usr/bin/env python3
"""The max-Sharpe case (A3) of arXiv:2508.03704, Section 4.4.

    max (mu'w - rf) / sqrt(w'Sigma w)  s.t.  e'w = 1

is a ratio, not a QP. Charnes-Cooper: kappa = 1/sqrt(w'Sigma w), y = kappa w,

    max mu'y - rf kappa  s.t.  y'Sigma y <= 1,  e'y = kappa,  kappa >= 0,

and w = y/kappa. Hand case: mu = (3, 1), rf = 0, Sigma = I: max Sharpe ||mu|| = sqrt(10)
at w = (3/4, 1/4). Port of examples/c/sharpe_ratio.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    mu = [3.0, 1.0]
    with Model() as m:
        m.add_vars(3)                           # y0, y1, kappa
        m.add_cons(2)                           # 0: y'Sigma y <= 1, 1: e'y - kappa = 0
        m.var_bounds(0, BK.FR, -math.inf, math.inf)
        m.var_bounds(1, BK.FR, -math.inf, math.inf)
        m.var_bounds(2, BK.LO, 0.0, math.inf)
        m.qconk(0, [(0, 0, 2.0), (1, 1, 2.0)])  # Sigma = I: 1/2 x'Qx with Q = 2I
        m.con_bounds(0, BK.UP, -math.inf, 1.0)
        m.a_row(1, [0, 1, 2], [1.0, 1.0, -1.0])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.cj(0, mu[0])
        m.cj(1, mu[1])
        m.sense(SENSE.MAX)
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    sharpe, x = r.objective, r.x
    w0, w1 = x[0] / x[2], x[1] / x[2]
    norm = math.sqrt(w0 * w0 + w1 * w1)
    ratio = (mu[0] * w0 + mu[1] * w1) / norm
    ok = (abs(sharpe - math.sqrt(10.0)) < 1e-6 and abs(w0 - 0.75) < 1e-5 and abs(w1 - 0.25) < 1e-5
          and abs(ratio - sharpe) < 1e-5 and abs(w0 + w1 - 1.0) < 1e-6)
    print("Sharpe = %.6f (expected sqrt(10) = %.6f), w = (%.4f, %.4f) (expected .75,.25)"
          % (sharpe, math.sqrt(10.0), w0, w1))
    print("reconstructed (mu'w)/||w|| = %.6f, e'w = %.2e" % (ratio, w0 + w1))
    print("%s (A3 is not quadratic: solved with Charnes-Cooper)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
