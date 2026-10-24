#!/usr/bin/env python3
"""Black-Litterman portfolio optimization of "Optimization Methods in Finance"
(Cornuejols & Tutuncu, 2007), Section 8.2. Equilibrium returns and covariance of
Section 8.1.1; two views (tau = 0.1): mu_M = 0.02 (omega 1e-5) and mu_S - mu_B = 0.05
(omega 1e-3). The posterior mean

    mu_bar = [(tau Sigma)^-1 + P'Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P'Omega^-1 q]

is checked against the book (0.1177, 0.0751, 0.0234), then the Markowitz QP is solved
for five targets and checked against Table 8.5. Port of examples/c/omf_bl.c.
"""

import math

from primalsolver import Model, BK

PI_0 = [0.1073, 0.0737, 0.0627]
SIG = [[0.02778, 0.00387, 0.00021], [0.00387, 0.01112, -0.00020], [0.00021, -0.00020, 0.00115]]
MU_BOOK = [0.1177, 0.0751, 0.0234]
R = [0.040, 0.045, 0.050, 0.055, 0.060]
VAR = [0.0012, 0.0015, 0.0020, 0.0025, 0.0032]
XS = [0.08, 0.11, 0.15, 0.18, 0.22]
XB = [0.17, 0.21, 0.24, 0.28, 0.31]


def invert3(a):
    """Invert a 3x3 matrix by cofactors."""
    d = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
         - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
         + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    return [[(a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d, (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d,
             (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d],
            [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d, (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d,
             (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d],
            [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d, (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d,
             (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d]]


def solve_bl(k, mu):
    """Solve the Markowitz QP for target R[k] and check the book portfolio."""
    with Model(2, 3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.qobj([(0, 0, 2 * SIG[0][0]), (0, 1, 2 * SIG[0][1]), (0, 2, 2 * SIG[0][2]),
                (1, 1, 2 * SIG[1][1]), (1, 2, 2 * SIG[1][2]), (2, 2, 2 * SIG[2][2])])
        m.cj(0, 0.0); m.cj(1, 0.0); m.cj(2, 0.0)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], mu)
        m.con_bounds(1, BK.LO, R[k], math.inf)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        var, x = r.objective, r.x
        ok = abs(var - VAR[k]) < 1e-4 and abs(x[0] - XS[k]) < 0.02 and abs(x[1] - XB[k]) < 0.02
        print("omf_bl R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) (book %.2f,%.2f,%.2f) %s"
              % (R[k], var, VAR[k], x[0], x[1], x[2], XS[k], XB[k], 1.0 - XS[k] - XB[k], "OK" if ok else "FAIL"))
    else:
        print("omf_bl R=%.3f rc=%d FAIL" % (R[k], r.rc))
    return ok


def main() -> int:
    tau = 0.1
    om = [0.00001, 0.001]
    q = [0.02, 0.05]
    P = [[0, 0, 1], [1, -1, 0]]
    sig = [[tau * SIG[i][j] for j in range(3)] for i in range(3)]
    sinv = invert3(sig)
    # A = (tau Sigma)^-1 + P'Omega^-1 P ; b = (tau Sigma)^-1 pi + P'Omega^-1 q
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(2):
                s += P[k][i] * (1.0 / om[k]) * P[k][j]
            A[i][j] = sinv[i][j] + s
        s = 0.0
        for k in range(2):
            s += P[k][i] * (1.0 / om[k]) * q[k]
        t2 = 0.0
        for j in range(3):
            t2 += sinv[i][j] * PI_0[j]
        b[i] = t2 + s
    Ai = invert3(A)
    mu = [0.0] * 3
    for i in range(3):
        for j in range(3):
            mu[i] += Ai[i][j] * b[j]
    print("omf_bl mu_bar=(%.4f,%.4f,%.4f) (book %.4f,%.4f,%.4f) %s"
          % (mu[0], mu[1], mu[2], MU_BOOK[0], MU_BOOK[1], MU_BOOK[2],
             "OK" if all(abs(mu[i] - MU_BOOK[i]) < 1e-3 for i in range(3)) else "FAIL"))
    ok = True
    for k in range(5):
        ok &= solve_bl(k, mu)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
