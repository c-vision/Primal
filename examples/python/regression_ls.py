#!/usr/bin/env python3
"""PrimalSolver - least-squares regression as an SOCP.

Port of c_examples/finance/regression_ls.c (Schmelzer et al., arXiv:1310.3397,
Section 2):  min ||X w - y||_2  <=>  min v  s.t. (v, Xw - y) in QUAD.

Two variants: unconstrained (checked against the normal equations X'Xw = X'y)
and the constrained portfolio form sum(w)=1, w>=0 (Section 2.6).

    python python_examples/regression_ls.py [n] [m]     (default 60 6)
"""

import sys

from primalsolver import Model, BK, CT, SOLSTA

N, M = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (60, 6)


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def build(constrained, X, y):
    m = Model(maxcon=N + (1 if constrained else 0), maxvar=M + N + 1)
    w0, r0, v = 0, M, M + N
    for j in range(M):
        if constrained:
            m.var_bounds(w0 + j, BK.LO, 0.0)
        else:
            m.var_bounds(w0 + j, BK.FR)
    for i in range(N):
        m.var_bounds(r0 + i, BK.FR)
    m.var_bounds(v, BK.FR)
    m.cj(v, 1.0)
    for i in range(N):
        cols, vals = [], []
        for j in range(M):
            if X[i][j] != 0.0:
                cols.append(w0 + j); vals.append(-X[i][j])
        cols.append(r0 + i); vals.append(1.0)
        m.a_row(i, cols, vals)
        m.con_bounds(i, BK.FX, -y[i], -y[i])
    m.cone(CT.QUAD, [v] + [r0 + i for i in range(N)])
    if constrained:
        m.a_row(N, [w0 + j for j in range(M)], [1.0] * M)
        m.con_bounds(N, BK.FX, 1.0, 1.0)
    r = m.solve()
    m.close()
    return r


def solve_dense(A, b):
    n = len(b)
    aug = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= f * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def main() -> int:
    rng = RNG(13103397)
    X = [[(rng.next() - 0.5) * 2.0 for _ in range(M)] for _ in range(N)]
    y = [(rng.next() - 0.5) * 2.0 for _ in range(N)]

    un = build(False, X, y)
    w1 = un.x[:M]
    G = [[sum(X[i][j] * X[i][k] for i in range(N)) for k in range(M)] for j in range(M)]
    rhs = [sum(X[i][j] * y[i] for i in range(N)) for j in range(M)]
    wref = solve_dense(G, rhs)
    wdiff = max(abs(w1[j] - wref[j]) for j in range(M))

    con = build(True, X, y)
    budget = sum(con.x[:M])
    minw = min(con.x[:M])
    ok = (un.solsta == SOLSTA.OPTIMAL and con.solsta == SOLSTA.OPTIMAL
          and wdiff < 1e-6 and abs(budget - 1.0) < 1e-6 and minw > -1e-6
          and con.objective >= un.objective - 1e-6)
    print(f"regr_ls n={N} m={M} uncons={un.objective:.6f} cons={con.objective:.6f} "
          f"max|w-wref|={wdiff:.2e} budget={budget:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
