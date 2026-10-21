#!/usr/bin/env python3
"""PrimalSolver - max-Sharpe ratio via Charnes-Cooper (quadratic constraint).

Port of c_examples/finance/sharpe_ratio.c (arXiv:2508.03704, case A3):

    max (mu'w - rf)/sqrt(w'Sigma w)  s.t. e'w = 1
    -> max mu'y - rf*kappa  s.t. y'Sigma y <= 1, e'y = kappa, kappa >= 0.

Hand case: mu=(3,1), rf=0, Sigma=I -> Sharpe sqrt(10) at w = (3/4, 1/4).

    python python_examples/sharpe_ratio.py
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA

MU = [3.0, 1.0]
RF = 0.0


def main() -> int:
    with Model(maxcon=2, maxvar=3) as m:      # y0, y1, kappa
        m.var_bounds(0, BK.FR)
        m.var_bounds(1, BK.FR)
        m.var_bounds(2, BK.LO, 0.0)
        m.qconk(0, [(0, 0, 2.0), (1, 1, 2.0)])   # Sigma = I: 1/2 y'Qy, Q=2I
        m.con_bounds(0, BK.UP, up=1.0)
        m.a_row(1, [0, 1, 2], [1.0, 1.0, -1.0])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.cj(0, MU[0]); m.cj(1, MU[1])
        m.sense(SENSE.MAX)
        r = m.solve()

    w0, w1 = r.x[0] / r.x[2], r.x[1] / r.x[2]
    ratio = (MU[0] * w0 + MU[1] * w1) / math.hypot(w0, w1)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - math.sqrt(10.0)) < 1e-6
          and abs(w0 - 0.75) < 1e-5 and abs(w1 - 0.25) < 1e-5
          and abs(ratio - r.objective) < 1e-5 and abs(w0 + w1 - 1.0) < 1e-6)
    print(f"Sharpe = {r.objective:.6f} (sqrt(10)={math.sqrt(10.0):.6f}) w=({w0:.4f},{w1:.4f}) "
          f"reconstructed={ratio:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
