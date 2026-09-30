#!/usr/bin/env python3
"""PrimalSolver - portfolio management as conic programs (arXiv:1310.3397 §4).

Port of c_examples/finance/portfolio_mgmt.c: five models, each verified
(budget, bounds, cone feasibility). N, M default 40, 6.

    python python_examples/portfolio_mgmt.py [n] [m]
"""

import sys

from primalsolver import Model, BK, CT, SENSE, SOLSTA

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
M = int(sys.argv[2]) if len(sys.argv) > 2 else 6


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def build_ls(X, rM, tracking):
    with Model(maxcon=1, maxvar=M) as m:
        for j in range(M):
            m.var_bounds(j, BK.LO, 0.0, 1.0)
        m.qobj([(a, b, 2.0 * sum(X[i][a] * X[i][b] for i in range(N)))
                for a in range(M) for b in range(a, M)])
        for a in range(M):
            c = sum(X[i][a] * (rM[i] if tracking else 0.0) for i in range(N))
            m.cj(a, -2.0 * c)
        if tracking:
            m.cfix(sum(v * v for v in rM))
        m.a_row(0, list(range(M)), [1.0] * M); m.con_bounds(0, BK.FX, 1.0, 1.0)
        r = m.solve()
    return sum(r.x[:M]), r.solsta


def main() -> int:
    rng = RNG(314159)
    X = [[(rng.next() - 0.5) * 0.1 for _ in range(M)] for _ in range(N)]
    rM = [(rng.next() - 0.5) * 0.08 for _ in range(N)]
    mu = [0.001 + rng.next() * 0.02 for _ in range(M)]
    ok = True

    bud, sta = build_ls(X, rM, False)
    ok &= sta == SOLSTA.OPTIMAL and abs(bud - 1.0) < 1e-6
    print(f"minvar     n={N} m={M} budget={bud:.6f}")

    bud, sta = build_ls(X, rM, True)
    ok &= sta == SOLSTA.OPTIMAL and abs(bud - 1.0) < 1e-6
    print(f"tracking   n={N} m={M} budget={bud:.6f}")

    # max return with risk cone (sigma, Xw) in Q
    sigma = 0.5
    w0, r0, tc = 0, M, M + N
    with Model(maxcon=N + 1, maxvar=M + N + 1) as m:
        m.sense(SENSE.MAX)
        for j in range(M):
            m.cj(w0 + j, mu[j]); m.var_bounds(w0 + j, BK.LO, 0.0, 1.0)
        for i in range(N):
            m.var_bounds(r0 + i, BK.FR)
        m.var_bounds(tc, BK.FX, sigma)
        for i in range(N):
            cols = [w0 + j for j in range(M) if X[i][j] != 0.0] + [r0 + i]
            vals = [-X[i][j] for j in range(M) if X[i][j] != 0.0] + [1.0]
            m.a_row(i, cols, vals); m.con_bounds(i, BK.FX, 0.0, 0.0)
        m.a_row(N, [w0 + j for j in range(M)], [1.0] * M); m.con_bounds(N, BK.FX, 1.0, 1.0)
        m.cone(CT.QUAD, [tc] + [r0 + i for i in range(N)])
        r = m.solve()
    risk = sum(r.x[r0 + i] ** 2 for i in range(N)) ** 0.5
    bud = sum(r.x[w0 + j] for j in range(M))
    ok &= r.solsta == SOLSTA.OPTIMAL and risk <= sigma * (1 + 1e-4) and abs(bud - 1.0) < 1e-6
    print(f"maxreturn  n={N} m={M} risk={risk:.6f} budget={bud:.6f}")

    # 130/30 leverage: (t_i, w_i) in Q^2, sum t_i <= 1.6, sum w = 1
    w0, t0 = 0, M
    with Model(maxcon=2, maxvar=2 * M) as m:
        m.sense(SENSE.MAX)
        for j in range(M):
            m.cj(w0 + j, mu[j]); m.var_bounds(w0 + j, BK.FR); m.var_bounds(t0 + j, BK.LO, 0.0)
            m.cone(CT.QUAD, [t0 + j, w0 + j])
        m.a_row(0, [t0 + j for j in range(M)], [1.0] * M); m.con_bounds(0, BK.UP, up=1.6)
        m.a_row(1, [w0 + j for j in range(M)], [1.0] * M); m.con_bounds(1, BK.FX, 1.0, 1.0)
        r = m.solve()
    lev = sum(r.x[t0 + j] for j in range(M))
    bud = sum(r.x[w0 + j] for j in range(M))
    ok &= r.solsta == SOLSTA.OPTIMAL and lev <= 1.6 + 1e-4 and abs(bud - 1.0) < 1e-6
    print(f"130_30     n={N} m={M} leverage={lev:.6f} budget={bud:.6f}")

    # robust: max mu'w - u, (u, A w) in Q, A tridiagonal
    w0, z0, uv = 0, M, 2 * M
    with Model(maxcon=M + 1, maxvar=2 * M + 1) as m:
        m.sense(SENSE.MAX)
        for j in range(M):
            m.cj(w0 + j, mu[j]); m.var_bounds(w0 + j, BK.LO, 0.0, 1.0)
        for j in range(M):
            m.var_bounds(z0 + j, BK.FR)
        m.var_bounds(uv, BK.LO, 0.0); m.cj(uv, -1.0)
        for i in range(M):
            cols, vals = [], []
            for j in range(M):
                a = 0.01 if i == j else (0.002 if j == (i + 1) % M else 0.0)
                if a != 0.0:
                    cols.append(w0 + j); vals.append(-a)
            cols.append(z0 + i); vals.append(1.0)
            m.a_row(i, cols, vals); m.con_bounds(i, BK.FX, 0.0, 0.0)
        m.a_row(M, [w0 + j for j in range(M)], [1.0] * M); m.con_bounds(M, BK.FX, 1.0, 1.0)
        m.cone(CT.QUAD, [uv] + [z0 + j for j in range(M)])
        r = m.solve()
    u = r.x[uv]
    normz = sum(r.x[z0 + j] ** 2 for j in range(M)) ** 0.5
    bud = sum(r.x[w0 + j] for j in range(M))
    ok &= r.solsta == SOLSTA.OPTIMAL and u >= normz - 1e-5 and abs(bud - 1.0) < 1e-6
    print(f"robust     n={N} m={M} u={u:.6f} |Aw|={normz:.6f} budget={bud:.6f}")

    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
