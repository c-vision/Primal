#!/usr/bin/env python3
"""Max-Sharpe portfolio, long-only, with a sector diversification cap
(matimacazaga/portfolio_optimization_with_mosek). As a minimum-norm problem with the
normalisation mu'y = 1 (rf = 0):

    minimize s  s.t.  s >= ||G y||,  mu'y = 1,  e'y = z,  y, z >= 0,  sector caps

and w = y/z, Sharpe = 1/s. Case: Sigma = I, mu = (3,1,3,1), sectors {0,1}, {2,3}. Cap
0.5 does not bind (Sharpe = sqrt(20)); cap 0.4 binds and the Sharpe drops.
Port of examples/c/sharpe_ratio_sectors.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

NA = 4


def solve_case(cap):
    """Solve with the sector-{0,1} cap; returns (ok, Sharpe or -1, w)."""
    mu = [3.0, 1.0, 3.0, 1.0]
    Y0, Z, S = 0, NA, NA + 1                        # y(4), z, s
    with Model() as m:
        m.add_vars(S + 1)
        m.sense(SENSE.MIN)
        for i in range(NA):
            m.var_bounds(Y0 + i, BK.LO, 0.0, math.inf)
        m.var_bounds(Z, BK.LO, 0.0, math.inf)
        m.var_bounds(S, BK.FR, -math.inf, math.inf)
        m.add_cons(3)                               # mu'y = 1 ; e'y - z = 0 ; cap
        m.a_row(0, [Y0, Y0 + 1, Y0 + 2, Y0 + 3], mu)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [Y0, Y0 + 1, Y0 + 2, Y0 + 3, Z], [1, 1, 1, 1, -1])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [Y0, Y0 + 1, Z], [1.0, 1.0, -cap])   # y0 + y1 - cap z <= 0
        m.con_bounds(2, BK.UP, -math.inf, 0.0)
        m.cone(CT.QUAD, [S, Y0, Y0 + 1, Y0 + 2, Y0 + 3])   # s >= ||y|| (G = I)
        m.cj(S, 1.0)
        r = m.solve()
    w = [0.0] * NA
    obj = -1.0
    if r.rc == 0:
        z = r.x[Z]
        w = [r.x[Y0 + i] / z if z > 0 else 0.0 for i in range(NA)]
        obj = r.x[S]                                # ||y|| = 1/Sharpe
    return r.rc == 0, (1.0 / obj if obj > 0 else -1.0), w


def main() -> int:
    print("sharpe_ratio_sectors (4 assets, 2 sectors, Sigma=I)")
    allok = True
    ok, sh, w = solve_case(0.5)                     # case A: the cap does not bind
    sw = w[0] + w[1] + w[2] + w[3]
    good = (ok and abs(sh - math.sqrt(20.0)) < 1e-4 and abs(w[0] + w[1] - 0.5) < 1e-4
            and abs(w[0] - 0.375) < 1e-4 and abs(w[1] - 0.125) < 1e-4
            and abs(w[2] - 0.375) < 1e-4 and abs(w[3] - 0.125) < 1e-4 and abs(sw - 1.0) < 1e-6)
    print("  A cap=0.50: Sharpe=%.6f (expected %.6f)  w=(%.4f,%.4f,%.4f,%.4f)  %s"
          % (sh, math.sqrt(20.0), w[0], w[1], w[2], w[3], "OK" if good else "FAIL"))
    allok &= good
    ok, sh2, w = solve_case(0.4)                    # case B: the cap binds
    s, s1 = w[0] + w[1] + w[2] + w[3], w[0] + w[1]
    good = (ok and sh2 > 0 and sh2 < math.sqrt(20.0) - 1e-6 and all(v >= -1e-9 for v in w)
            and abs(s - 1.0) < 1e-6 and s1 <= 0.4 + 1e-6)
    print("  B cap=0.40: Sharpe=%.6f (< %.6f)  w=(%.4f,%.4f,%.4f,%.4f) sum=%.6f s1=%.6f  %s"
          % (sh2, math.sqrt(20.0), w[0], w[1], w[2], w[3], s, s1, "OK" if good else "FAIL"))
    allok &= good
    print("OK" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
