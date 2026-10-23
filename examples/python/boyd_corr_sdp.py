#!/usr/bin/env python3
"""Bounding a correlation coefficient, "Convex Optimization" (Boyd & Vandenberghe,
2004), Example 8.3.

Find the minimum and maximum possible value of rho_14 given
  0.6 <= rho_12 <= 0.9,  0.8 <= rho_13 <= 0.9,
  0.5 <= rho_24 <= 0.7, -0.8 <= rho_34 <= -0.4,
and the symmetric 4x4 matrix with unit diagonal and off-diagonal rho_ij positive
semidefinite (an SDP). The book reports min rho_14 = -0.39, max rho_14 = 0.23.

Model: bar variable X (4x4, PSD) with rows <E_ij,X> = 2 rho_ij (off-diagonal) and
<E_ii,X> = 1; rho_ij are the scalar variables. Port of examples/c/boyd_corr_sdp.c.
"""

import math

from primalsolver import Model, BK

RIDX = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}
LO = [0.6, 0.8, -math.inf, -math.inf, 0.5, -0.8]
UP = [0.9, 0.9, math.inf, math.inf, 0.7, -0.4]


def solve(sgn):
    """Solve the correlation SDP for one objective sign; returns (ok, rho_14)."""
    with Model() as m:
        m.add_vars(6)                               # rho_12, rho_13, rho_14, rho_23, rho_24, rho_34
        for k in range(6):
            m.var_bounds(k, BK.RA, LO[k], UP[k])
        m.cj(2, sgn)                                # minimize (sgn=+1) or maximize (-1) rho_14
        msym = [m.sparsesymmat(4, [(i, j, 1.0)]) for i in range(4) for j in range(i, 4)]
        m.barvars([4])
        m.add_cons(10)
        row = 0
        for i in range(4):
            for j in range(i, 4):
                m.baraij(row, 0, [(msym[row], 1.0)])
                if i == j:
                    m.con_bounds(row, BK.FX, 1.0, 1.0)
                else:
                    m.a_row(row, [RIDX[(i, j)]], [-2.0])
                    m.con_bounds(row, BK.FX, 0.0, 0.0)
                row += 1
        r = m.solve()
    return (r.rc == 0), (r.x[2] if r.rc == 0 else 0.0)


def main() -> int:
    ok1, lo14 = solve(+1.0)
    ok2, hi14 = solve(-1.0) if ok1 else (False, 0.0)
    ok = ok1 and ok2 and abs(lo14 + 0.39) < 0.01 and abs(hi14 - 0.23) < 0.01
    print(f"boyd_corr_sdp  min rho14={lo14:.4g} max rho14={hi14:.4g} (book -0.39, 0.23) "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
