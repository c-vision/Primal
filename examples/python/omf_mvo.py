#!/usr/bin/env python3
"""Markowitz mean-variance optimization of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 8.1.1.

    minimize x'Sigma x  s.t.  mu'x >= R,  xS + xB + xM = 1,  x >= 0

with the geometric means and covariance of Tables 8.2-8.3 (US stocks, bonds, money
market), solved for R = 6.5% .. 8.5% and checked against Table 8.3 (2 decimals).
Port of examples/c/omf_mvo.c.
"""

import math

from primalsolver import Model, BK

MU = [0.1073, 0.0737, 0.0627]
SIG = [[0.02778, 0.00387, 0.00021], [0.00387, 0.01112, -0.00020], [0.00021, -0.00020, 0.00115]]
R = [0.065, 0.070, 0.075, 0.080, 0.085]
VAR = [0.0010, 0.0014, 0.0026, 0.0044, 0.0070]
XS = [0.03, 0.13, 0.24, 0.35, 0.45]
XB = [0.10, 0.12, 0.14, 0.16, 0.18]
XM = [0.87, 0.75, 0.62, 0.49, 0.37]


def solve_mvo(k):
    """Solve the MVO QP for target R[k] and check the book portfolio."""
    with Model(2, 3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        # objective x'Sigma x = 0.5 x'Qx with Q = 2 Sigma (upper triangle)
        m.qobj([(0, 0, 2 * SIG[0][0]), (0, 1, 2 * SIG[0][1]), (0, 2, 2 * SIG[0][2]),
                (1, 1, 2 * SIG[1][1]), (1, 2, 2 * SIG[1][2]), (2, 2, 2 * SIG[2][2])])
        m.cj(0, 0.0); m.cj(1, 0.0); m.cj(2, 0.0)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # budget: xS+xB+xM = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], MU)                   # return: mu'x >= R
        m.con_bounds(1, BK.LO, R[k], math.inf)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        var, x = r.objective, r.x
        ok = (abs(var - VAR[k]) < 1e-4 and abs(x[0] - XS[k]) < 0.02 and abs(x[1] - XB[k]) < 0.02
              and abs(x[2] - XM[k]) < 0.02)
        print("omf_mvo R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) (book %.2f,%.2f,%.2f) %s"
              % (R[k], var, VAR[k], x[0], x[1], x[2], XS[k], XB[k], XM[k], "OK" if ok else "FAIL"))
    else:
        print("omf_mvo R=%.3f rc=%d FAIL" % (R[k], r.rc))
    return ok


def main() -> int:
    ok = True
    for k in range(5):
        ok &= solve_mvo(k)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
