#!/usr/bin/env python3
"""PrimalSolver - mean-CVaR portfolio (one LP).

Port of c_examples/finance/mean_cvar.c:

    max  mu'w - lambda * CVaR_beta(w)   s.t. e'w = 1, w >= 0,
    CVaR linearised (Rockafellar-Uryasev): z_s >= -r_s'w - alpha, z_s >= 0.

Hand case: 2 assets, 2 equiprobable scenarios, beta = 1/2, lambda = 1:
mu=(0.2,0.1), r0=(0.1,0.3), r1=(0.4,-0.1). Optimum at the kink r0'w = r1'w:
w = (4/7, 3/7), value 12/35.

    python python_examples/mean_cvar.py
"""

from primalsolver import Model, BK, SENSE, SOLSTA

MU = [0.2, 0.1]
R0 = [0.1, 0.3]
R1 = [0.4, -0.1]


def main() -> int:
    with Model(maxcon=3, maxvar=5) as m:     # w0, w1, alpha, z0, z1
        for j in (0, 1, 3, 4):
            m.var_bounds(j, BK.LO, 0.0)
        m.var_bounds(2, BK.FR)
        m.cj(0, MU[0]); m.cj(1, MU[1])
        m.cj(2, -1.0); m.cj(3, -1.0); m.cj(4, -1.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [3, 0, 1, 2], [1.0, R0[0], R0[1], 1.0])
        m.con_bounds(0, BK.LO, 0.0)
        m.a_row(1, [4, 0, 1, 2], [1.0, R1[0], R1[1], 1.0])
        m.con_bounds(1, BK.LO, 0.0)
        m.a_row(2, [0, 1], [1.0, 1.0])
        m.con_bounds(2, BK.FX, 1.0, 1.0)
        r = m.solve()

    w0, w1 = r.x[0], r.x[1]
    cvar = max(-(R0[0] * w0 + R0[1] * w1), -(R1[0] * w0 + R1[1] * w1))
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(w0 - 4.0 / 7.0) < 1e-6
          and abs(w1 - 3.0 / 7.0) < 1e-6 and abs(r.objective - 12.0 / 35.0) < 1e-6
          and abs((MU[0] * w0 + MU[1] * w1) - cvar - r.objective) < 1e-6)
    print(f"w = ({w0:.6f}, {w1:.6f}) obj = {r.objective:.6f} (12/35={12.0 / 35.0:.6f}) "
          f"CVaR = {cvar:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
