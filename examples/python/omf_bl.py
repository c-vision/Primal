#!/usr/bin/env python3
"""OMF Section 8.2 / Example 8.1 - Black-Litterman portfolio (QP).

Computes the posterior mean mu_bar (eq. 8.7) then the Markowitz frontier for
five targets (Table 8.5).   Port of c_examples/finance/omf_bl.c.
"""

from primalsolver import Model, BK, SOLSTA

PI_0 = [0.1073, 0.0737, 0.0627]
SIG = [[0.02778, 0.00387, 0.00021],
       [0.00387, 0.01112, -0.00020],
       [0.00021, -0.00020, 0.00115]]
MU_BOOK = [0.1177, 0.0751, 0.0234]
R = [0.040, 0.045, 0.050, 0.055, 0.060]
VAR = [0.0012, 0.0015, 0.0020, 0.0025, 0.0032]
XS = [0.08, 0.11, 0.15, 0.18, 0.22]
XB = [0.17, 0.21, 0.24, 0.28, 0.31]


def invert3(a):
    d = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
         - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
         + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    inv = [[0.0] * 3 for _ in range(3)]
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d
    return inv


def bl_posterior(tau=0.1, om=(1e-5, 1e-3), q=(0.02, 0.05), P=((0, 0, 1), (1, -1, 0))):
    sig = [[tau * SIG[i][j] for j in range(3)] for i in range(3)]
    sinv = invert3(sig)
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for i in range(3):
        for j in range(3):
            A[i][j] = sinv[i][j] + sum(P[k][i] * (1.0 / om[k]) * P[k][j] for k in range(2))
        b[i] = sum(sinv[i][j] * PI_0[j] for j in range(3)) + sum(P[k][i] * (1.0 / om[k]) * q[k] for k in range(2))
    Ai = invert3(A)
    return [sum(Ai[i][j] * b[j] for j in range(3)) for i in range(3)]


def solve(k, mu):
    with Model(maxcon=2, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.qobj([(0, 0, 2 * SIG[0][0]), (0, 1, 2 * SIG[0][1]), (0, 2, 2 * SIG[0][2]),
                (1, 1, 2 * SIG[1][1]), (1, 2, 2 * SIG[1][2]), (2, 2, 2 * SIG[2][2])])
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], mu); m.con_bounds(1, BK.LO, R[k])
        r = m.solve()
    x = r.x
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - VAR[k]) < 1e-4
          and abs(x[0] - XS[k]) < 0.02 and abs(x[1] - XB[k]) < 0.02)
    print(f"omf_bl R={R[k]:.3f} var={r.objective:.4f} (book {VAR[k]:.4f}) "
          f"x=({x[0]:.2f},{x[1]:.2f},{x[2]:.2f}) (book {XS[k]:.2f},{XB[k]:.2f},"
          f"{1 - XS[k] - XB[k]:.2f}) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    mu = bl_posterior()
    mu_ok = all(abs(mu[i] - MU_BOOK[i]) < 1e-3 for i in range(3))
    print(f"omf_bl mu_bar=({mu[0]:.4f},{mu[1]:.4f},{mu[2]:.4f}) "
          f"(book {MU_BOOK[0]:.4f},{MU_BOOK[1]:.4f},{MU_BOOK[2]:.4f}) {'OK' if mu_ok else 'FAIL'}")
    ok = mu_ok
    for k in range(5):
        ok &= solve(k, mu)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
