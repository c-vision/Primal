#!/usr/bin/env python3
"""Worst-case CVaR over a box of return uncertainty.

CVaR (Rockafellar-Uryasev) is an LP in (w, alpha, z). If each scenario return may
deviate in a box r_s + delta, |delta| <= d_s, and w >= 0, the worst case is the low
corner r_s - d_s: the robust CVaR is the same LP on shifted returns. Hand case: 2
equiprobable scenarios, beta = 1/2, r0 = (0.10, 0.20), r1 = (0.30, 0.00),
d = (0.05, 0.05): at w = (1/2, 1/2) the robust CVaR is -0.10, the nominal -0.15.
Port of examples/c/robust_cvar.c.
"""

import math

from primalsolver import Model, BK


def solve_cvar(robust):
    """Robust: subtract the box radius from each scenario return."""
    R0, R1, D = [0.10, 0.20], [0.30, 0.00], [0.05, 0.05]
    r0 = [R0[j] - (D[j] if robust else 0.0) for j in range(2)]
    r1 = [R1[j] - (D[j] if robust else 0.0) for j in range(2)]
    with Model() as m:
        m.add_vars(5)                           # w0, w1, alpha, z0, z1
        m.add_cons(3)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.FR, -math.inf, math.inf)
        m.var_bounds(3, BK.LO, 0.0, math.inf)
        m.var_bounds(4, BK.LO, 0.0, math.inf)
        m.cj(2, 1.0); m.cj(3, 1.0); m.cj(4, 1.0)
        # z_s + r_s'w + alpha >= 0
        m.a_row(0, [3, 0, 1, 2], [1.0, r0[0], r0[1], 1.0]); m.con_bounds(0, BK.LO, 0.0, math.inf)
        m.a_row(1, [4, 0, 1, 2], [1.0, r1[0], r1[1], 1.0]); m.con_bounds(1, BK.LO, 0.0, math.inf)
        m.a_row(2, [0, 1], [1.0, 1.0]); m.con_bounds(2, BK.FX, 1.0, 1.0)   # e'w = 1
        r = m.solve()
    if r.rc != 0:
        return 1e30, [float("nan"), float("nan")]
    return r.objective, r.x[:2]


def main() -> int:
    nom, wn = solve_cvar(False)
    rob, wr = solve_cvar(True)
    ok = (abs(rob - (-0.10)) < 1e-7 and abs(wr[0] - 0.5) < 1e-6 and abs(wr[1] - 0.5) < 1e-6
          and abs(nom - (-0.15)) < 1e-7 and rob >= nom - 1e-9)
    print("nominal CVaR = %.6f (expected -0.15), w = (%.4f, %.4f)" % (nom, wn[0], wn[1]))
    print("robust  CVaR = %.6f (expected -0.10), w = (%.4f, %.4f)" % (rob, wr[0], wr[1]))
    print("%s (robustness costs: robust CVaR >= nominal)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
