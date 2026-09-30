#!/usr/bin/env python3
"""PrimalSolver - regularised regression (Schmelzer et al., arXiv:1310.3397 §3).

Port of c_examples/finance/regression_regularized.c. Three variants on the
trade p = w - w0:

    ridge  (L2):  min ||Xw-y||^2 + lambda ||w-w0||^2
    sparse (L1):  min ||Xw-y||^2 + lambda sum_i |p_i|        (LASSO)
    3/2:          min ||Xw-y||^2 + lambda sum_i |p_i|^{3/2}

The squared residual is a quadratic objective (Q=2X'X ...); the penalties use
the conic forms (t_i, p_i) in QUAD and (t_i, 1, p_i) in POW^{2/3,1/3}. Ridge is
checked against (X'X + lambda I)w = X'y + lambda w0; sparse/3-2 against their
cone feasibility.

    python python_examples/regression_regularized.py [n] [m] [lambda]
"""

import sys

from primalsolver import Model, BK, CT, SOLSTA

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


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
    rng = RNG(27182818)
    X = [[(rng.next() - 0.5) * 2.0 for _ in range(M)] for _ in range(N)]
    y = [(rng.next() - 0.5) * 2.0 for _ in range(N)]
    w0 = [0.0] * M
    G = [[sum(X[i][a] * X[i][b] for i in range(N)) for b in range(M)] for a in range(M)]
    Xy = [sum(X[i][a] * y[i] for i in range(N)) for a in range(M)]
    yy = sum(v * v for v in y)
    ok = True

    # ridge
    with Model(maxcon=0, maxvar=M) as m:
        for j in range(M):
            m.var_bounds(j, BK.FR)
        terms = [(a, b, 2.0 * G[a][b]) for a in range(M) for b in range(a, M) if G[a][b] != 0.0]
        terms += [(j, j, 2.0 * LAM) for j in range(M)]
        m.qobj(terms)
        for j in range(M):
            m.cj(j, -2.0 * Xy[j] - 2.0 * LAM * w0[j])
        m.cfix(yy + LAM * sum(v * v for v in w0))
        r = m.solve()
    rw = r.x[:M]
    A = [[G[j][k] + (LAM if j == k else 0.0) for k in range(M)] for j in range(M)]
    rhs = [Xy[j] + LAM * w0[j] for j in range(M)]
    wref = solve_dense(A, rhs)
    wdiff = max(abs(rw[j] - wref[j]) for j in range(M))
    ok = ok and r.solsta == SOLSTA.OPTIMAL and wdiff <= 1e-5

    def penalty(kind):
        w_, p_, t_ = 0, M, 2 * M
        one = 3 * M
        nv = 3 * M if kind == "sparse" else 3 * M + 1
        with Model(maxcon=M, maxvar=nv) as m:
            for j in range(M):
                m.var_bounds(w_ + j, BK.FR)
                m.var_bounds(p_ + j, BK.FR)
                m.var_bounds(t_ + j, BK.LO, 0.0); m.cj(t_ + j, LAM)
            m.qobj([(a, b, 2.0 * G[a][b]) for a in range(M) for b in range(a, M) if G[a][b] != 0.0])
            for j in range(M):
                m.cj(j, -2.0 * Xy[j])
            m.cfix(yy)
            for j in range(M):              # p_j - w_j = -w0_j
                m.a_row(j, [w_ + j, p_ + j], [-1.0, 1.0])
                m.con_bounds(j, BK.FX, -w0[j], -w0[j])
            if kind == "sparse":
                for j in range(M):
                    m.cone(CT.QUAD, [t_ + j, p_ + j])
            else:
                m.var_bounds(one, BK.FX, 1.0, 1.0)
                for j in range(M):
                    m.cone(CT.PPOW, [t_ + j, one, p_ + j], param=2.0 / 3.0)
            r = m.solve()
        return r

    sp = penalty("sparse")
    worst_sp = max(abs(sp.x[M + j]) - sp.x[2 * M + j] for j in range(M))
    ok = ok and sp.solsta == SOLSTA.OPTIMAL and worst_sp <= 1e-5

    p32 = penalty("pow32")
    worst32 = max(abs(p32.x[M + j]) ** 1.5 - p32.x[2 * M + j] for j in range(M))
    ok = ok and p32.solsta == SOLSTA.OPTIMAL and worst32 <= 1e-4

    print(f"regr_reg n={N} m={M} lam={LAM:.2f} ridge wdiff={wdiff:.2e} "
          f"sparse worst={worst_sp:.2e} pow3_2 worst={worst32:.2e} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
