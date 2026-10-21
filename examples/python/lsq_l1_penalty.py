#!/usr/bin/env python3
"""PrimalSolver - positivity least squares with a weighted L1 penalty.

Port of c_examples/finance/lsq_l1_penalty.c (MosekRegression
`lsq_pos_l1_penalty`):

    min ||X w - rhs||_2^2 + sum_i gamma_i |w_i - w0_i|   s.t. e'w = 1, w >= 0.
Conic: (1/2, v, Xw-rhs) in Qr for the squared residual and (t_i, p_i) in Q^2
per asset; objective min v + sum t_i. Case A reduces to a QP + gamma; case B is
checked by a brute-force scan.

    python python_examples/lsq_l1_penalty.py
"""

import math

from primalsolver import Model, BK, CT, SOLSTA


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def bits(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x3FF) / 1024.0


def l1_solve(N, M, X, b, gam, w0):
    w_, r_, v_, p_, t_, half = 0, M, M + N, M + N + 1, M + N + 1 + M, M + N + 1 + 2 * M
    nv = half + 1
    with Model(maxcon=1 + N + M, maxvar=nv) as m:
        for i in range(M):
            m.var_bounds(w_ + i, BK.LO, 0.0)
        for i in range(N):
            m.var_bounds(r_ + i, BK.FR)
        m.var_bounds(v_, BK.FR)
        for i in range(M):
            m.var_bounds(p_ + i, BK.FR)
            m.var_bounds(t_ + i, BK.LO, 0.0)
        m.var_bounds(half, BK.FX, 0.5)
        m.a_row(0, [w_ + i for i in range(M)], [1.0] * M); m.con_bounds(0, BK.FX, 1.0, 1.0)
        for i in range(N):
            cols, vals = [r_ + i], [1.0]
            for j in range(M):
                if X[i][j] != 0.0:
                    cols.append(w_ + j); vals.append(-X[i][j])
            m.a_row(1 + i, cols, vals); m.con_bounds(1 + i, BK.FX, -b[i], -b[i])
        for i in range(M):
            m.a_row(1 + N + i, [p_ + i, w_ + i], [1.0, -gam[i]])
            m.con_bounds(1 + N + i, BK.FX, -gam[i] * w0[i], -gam[i] * w0[i])
        m.cone(CT.RQUAD, [half, v_] + [r_ + i for i in range(N)])
        for i in range(M):
            m.cone(CT.QUAD, [t_ + i, p_ + i])
        m.cj(v_, 1.0)
        for i in range(M):
            m.cj(t_ + i, 1.0)
        r = m.solve()
    return r.objective, r.x[w_:w_ + M], r.solsta


def qp_solve(N, M, X, b):
    with Model(maxcon=1, maxvar=M) as m:
        for i in range(M):
            m.var_bounds(i, BK.LO, 0.0)
        m.a_row(0, list(range(M)), [1.0] * M); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.qobj([(i, j, 2.0 * sum(X[k][i] * X[k][j] for k in range(N)))
                for i in range(M) for j in range(i, M)])
        cf = sum(v * v for v in b)
        m.cfix(cf)
        for i in range(M):
            m.cj(i, -2.0 * sum(b[k] * X[k][i] for k in range(N)))
        r = m.solve()
    return r.objective, r.x[:M], r.solsta


def main() -> int:
    allok = True
    print("lsq_l1_penalty")

    # Case A: w0=0, uniform gamma -> conic obj == QP obj + gamma
    N, M, gamma = 8, 3, 0.7
    rng = RNG(987654321)
    b = []
    X = []
    for _ in range(N):
        b.append(rng.bits())
        X.append([0.2 + rng.bits() for _ in range(M)])
    gam = [gamma] * M
    w0 = [0.0] * M
    ol, wl, _ = l1_solve(N, M, X, b, gam, w0)
    oq, wq, _ = qp_solve(N, M, X, b)
    dw = max(abs(wl[i] - wq[i]) for i in range(M))
    good = abs(ol - (oq + gamma)) < 1e-6 and dw < 1e-6
    print(f"  A (w0=0, gamma={gamma:.2f}): L1 obj={ol:.8f} QP+gamma={oq + gamma:.8f} "
          f"|dw|={dw:.2e} {'OK' if good else 'FAIL'}")
    allok &= good

    # Case B: M=2, general gamma/w0 -> brute force over w1
    N, M = 5, 2
    X = [[0.5, 1.0], [1.0, 0.4], [0.2, 0.9], [0.8, 0.6], [0.4, 0.4]]
    b = [0.7, 0.5, 0.3, 0.9, 0.4]
    gam = [0.3, 1.1]
    w0 = [0.6, 0.4]
    ol, wl, _ = l1_solve(N, M, X, b, gam, w0)
    best, bestw1 = math.inf, 0.0
    for k in range(2000001):
        w1 = k / 2000000.0
        w2 = 1.0 - w1
        f = sum((X[i][0] * w1 + X[i][1] * w2 - b[i]) ** 2 for i in range(N))
        f += gam[0] * abs(w1 - w0[0]) + gam[1] * abs(w2 - w0[1])
        if f < best:
            best, bestw1 = f, w1
    good = abs(ol - best) < 1e-5 and abs(wl[0] - bestw1) < 1e-4
    print(f"  B (M=2, gamma=({gam[0]},{gam[1]})): conic obj={ol:.8f} brute={best:.8f} "
          f"w1={wl[0]:.6f} (bf {bestw1:.6f}) {'OK' if good else 'FAIL'}")
    allok &= good

    print("OK" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
