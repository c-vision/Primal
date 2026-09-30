#!/usr/bin/env python3
"""OMF Section 8.1.1 - Markowitz mean-variance frontier (QP).

    min x'Sigma x  s.t. mu'x >= R, sum x = 1, x >= 0.
Five target returns from Table 8.3.   Port of c_examples/finance/omf_mvo.c.
"""

from primalsolver import Model, BK, SOLSTA

MU = [0.1073, 0.0737, 0.0627]
SIG = [[0.02778, 0.00387, 0.00021],
       [0.00387, 0.01112, -0.00020],
       [0.00021, -0.00020, 0.00115]]
R = [0.065, 0.070, 0.075, 0.080, 0.085]
VAR = [0.0010, 0.0014, 0.0026, 0.0044, 0.0070]
XS = [0.03, 0.13, 0.24, 0.35, 0.45]
XB = [0.10, 0.12, 0.14, 0.16, 0.18]
XM = [0.87, 0.75, 0.62, 0.49, 0.37]


def solve(k):
    with Model(maxcon=2, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.qobj([(0, 0, 2 * SIG[0][0]), (0, 1, 2 * SIG[0][1]), (0, 2, 2 * SIG[0][2]),
                (1, 1, 2 * SIG[1][1]), (1, 2, 2 * SIG[1][2]), (2, 2, 2 * SIG[2][2])])
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], MU); m.con_bounds(1, BK.LO, R[k])
        r = m.solve()
    x = r.x
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - VAR[k]) < 1e-4
          and abs(x[0] - XS[k]) < 0.02 and abs(x[1] - XB[k]) < 0.02 and abs(x[2] - XM[k]) < 0.02)
    print(f"omf_mvo R={R[k]:.3f} var={r.objective:.4f} (book {VAR[k]:.4f}) "
          f"x=({x[0]:.2f},{x[1]:.2f},{x[2]:.2f}) (book {XS[k]:.2f},{XB[k]:.2f},{XM[k]:.2f}) "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = True
    for k in range(5):
        ok &= solve(k)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
