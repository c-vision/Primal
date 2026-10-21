#!/usr/bin/env python3
"""PrimalSolver - max-Sharpe, long-only, with a sector diversification cap.

Port of c_examples/finance/sharpe_ratio_sectors.c. Minimum-norm form with the
normalisation mu'y - rf z = 1:

    min s  s.t. s >= ||G y||, mu'y - rf z = 1, e'y = z, y >= 0, z >= 0,
           y_{0,1} <= cap * z.

4 assets, Sigma = I, mu = (3,1,3,1), rf = 0: uncapped Sharpe = sqrt(20). At cap
0.5 the optimum already satisfies it (value stays sqrt(20)); at cap 0.4 it binds
and the value drops. w = y/z, Sharpe = 1/s.

    python python_examples/sharpe_ratio_sectors.py
"""

import math

from primalsolver import Model, BK, CT, SOLSTA

NA = 4
MU = [3.0, 1.0, 3.0, 1.0]


def solve_case(cap):
    y0, z, s = 0, NA, NA + 1
    with Model(maxcon=3, maxvar=NA + 2) as m:
        for i in range(NA):
            m.var_bounds(y0 + i, BK.LO, 0.0)
        m.var_bounds(z, BK.LO, 0.0)
        m.var_bounds(s, BK.FR)
        m.a_row(0, [y0 + i for i in range(NA)], MU); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [y0 + i for i in range(NA)] + [z], [1.0] * NA + [-1.0])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [y0, y0 + 1, z], [1.0, 1.0, -cap]); m.con_bounds(2, BK.UP, up=0.0)
        m.cone(CT.QUAD, [s, y0, y0 + 1, y0 + 2, y0 + 3])
        m.cj(s, 1.0)
        r = m.solve()
    zval = r.x[z]
    w = [r.x[y0 + i] / zval if zval > 0 else 0.0 for i in range(NA)]
    return (1.0 / r.x[s] if r.x[s] > 0 else -1.0), w, r.solsta


def main() -> int:
    ok = True
    sh, w, _ = solve_case(0.5)
    good = (abs(sh - math.sqrt(20.0)) < 1e-4 and abs(w[0] + w[1] - 0.5) < 1e-4
            and abs(w[0] - 0.375) < 1e-4 and abs(w[1] - 0.125) < 1e-4
            and abs(w[2] - 0.375) < 1e-4 and abs(w[3] - 0.125) < 1e-4
            and abs(sum(w) - 1.0) < 1e-6)
    print(f"  A cap=0.50: Sharpe={sh:.6f} (expected {math.sqrt(20.0):.6f}) "
          f"w=({w[0]:.4f},{w[1]:.4f},{w[2]:.4f},{w[3]:.4f}) {'OK' if good else 'FAIL'}")
    ok &= good

    sh2, w2, sta = solve_case(0.4)
    good = (sta == SOLSTA.OPTIMAL and sh2 > 0 and sh2 < math.sqrt(20.0) - 1e-6
            and all(v >= -1e-9 for v in w2) and abs(sum(w2) - 1.0) < 1e-6
            and w2[0] + w2[1] <= 0.4 + 1e-6)
    print(f"  B cap=0.40: Sharpe={sh2:.6f} (< {math.sqrt(20.0):.6f}) "
          f"w=({w2[0]:.4f},{w2[1]:.4f},{w2[2]:.4f},{w2[3]:.4f}) {'OK' if good else 'FAIL'}")
    ok &= good
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
