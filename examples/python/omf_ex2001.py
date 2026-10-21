#!/usr/bin/env python3
"""OMF Example 20.1 - mean-variance with a tracking-error constraint (SOCP).

    max mu'x  s.t. TE(x) <= 0.10, e'x = 1, x >= 0,
    TE = ||L' (x1-0.5, x2-0.5)||,  Sigma2 = L L'  (Cholesky of the 2x2 block).
Book: mu=(6,4,0) -> (0.831,0.169,0), 5.662;  mu=(5,5,0) -> 5.0.
Port of c_examples/finance/omf_ex2001.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, SOLSTA

S11, S12, S22 = 0.1764, 0.09702, 0.1089
L11 = math.sqrt(S11)
L21 = S12 / L11
L22 = math.sqrt(S22 - L21 * L21)


def solve(mu1, mu2, want, wx1, wx2, name):
    with Model(maxcon=3, maxvar=6) as m:   # x1,x2,x3,r1,r2,t(=0.10)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.var_bounds(3, BK.FR); m.var_bounds(4, BK.FR)
        m.var_bounds(5, BK.FX, 0.10)
        m.cj(0, mu1); m.cj(1, mu2)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 3], [-L11, -L21, 1.0]); m.con_bounds(1, BK.FX, -0.5 * (L11 + L21), -0.5 * (L11 + L21))
        m.a_row(2, [1, 4], [-L22, 1.0]); m.con_bounds(2, BK.FX, -0.5 * L22, -0.5 * L22)
        m.cone(CT.QUAD, [5, 3, 4])
        r = m.solve()

    x = r.x
    d1, d2 = x[0] - 0.5, x[1] - 0.5
    te = math.sqrt(S11 * d1 * d1 + 2.0 * S12 * d1 * d2 + S22 * d2 * d2)
    xcheck = wx1 < 0.0 or (abs(x[0] - wx1) < 2e-3 and abs(x[1] - wx2) < 2e-3)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - want) < 1e-3 and xcheck
          and te <= 0.10 + 1e-6 and abs(sum(x[:3]) - 1.0) < 1e-6)
    print(f"omf_ex2001 {name} x=({x[0]:.4f},{x[1]:.4f},{x[2]:.4f}) TE={te:.4f} "
          f"obj={r.objective:.4f} (book {want:.4f}) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    a = solve(6.0, 4.0, 5.662, 0.831, 0.169, "mu=(6,4,0)")
    b = solve(5.0, 5.0, 5.000, -1.0, -1.0, "mu=(5,5,0)")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
