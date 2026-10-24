#!/usr/bin/env python3
"""Positivity least squares with a weighted L1 penalty (tschm/MosekRegression,
lsq_pos_l1_penalty):

    min ||X w - rhs||_2^2 + sum_i gamma_i |w_i - w0_i|  s.t.  e'w = 1,  w >= 0

Conic form: (1/2, v, Xw - rhs) in Qr, (t_i, p_i) in Q^2, min v + sum t_i. Checks:
A) w0 = 0 and uniform gamma reduce to the QP + gamma (same w); B) M = 2 with general
gamma and w0, against a fine brute-force scan. Port of examples/c/lsq_l1_penalty.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE


def l1_solve(N, M, X, b, gam, w0):
    """Conic model; returns (objective or -1, w)."""
    W0, R0, V, P0, T0 = 0, M, M + N, M + N + 1, M + N + 1 + M
    HALF = M + N + 1 + 2 * M
    with Model() as m:
        m.add_vars(HALF + 1)
        m.add_cons(1 + N + M)
        m.sense(SENSE.MIN)
        for i in range(M):
            m.var_bounds(W0 + i, BK.LO, 0.0, math.inf)
        for i in range(N):
            m.var_bounds(R0 + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(V, BK.FR, -math.inf, math.inf)
        for i in range(M):
            m.var_bounds(P0 + i, BK.FR, -math.inf, math.inf)
        for i in range(M):
            m.var_bounds(T0 + i, BK.LO, 0.0, math.inf)
        m.var_bounds(HALF, BK.FX, 0.5, 0.5)
        m.a_row(0, list(range(W0, W0 + M)), [1.0] * M)   # e'w = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for i in range(N):                          # r - X w = -b
            sub, val = [R0 + i], [1.0]
            for j in range(M):
                if X[i][j] != 0.0:
                    sub.append(W0 + j); val.append(-X[i][j])
            m.a_row(1 + i, sub, val)
            m.con_bounds(1 + i, BK.FX, -b[i], -b[i])
        for i in range(M):                          # p_i - gamma_i w_i = -gamma_i w0_i
            m.a_row(1 + N + i, [P0 + i, W0 + i], [1.0, -gam[i]])
            m.con_bounds(1 + N + i, BK.FX, -gam[i] * w0[i], -gam[i] * w0[i])
        m.cone(CT.RQUAD, [HALF, V] + list(range(R0, R0 + N)))   # v >= ||r||^2
        for i in range(M):                          # t_i >= |p_i|
            m.cone(CT.QUAD, [T0 + i, P0 + i])
        m.cj(V, 1.0)
        for i in range(M):
            m.cj(T0 + i, 1.0)
        r = m.solve()
    if r.rc != 0:
        return -1.0, [0.0] * M
    return r.objective, r.x[W0:W0 + M]


def qp_solve(N, M, X, b):
    """QP reference: min ||Xw - b||^2 s.t. e'w = 1, w >= 0; returns (objective, w)."""
    with Model() as m:
        m.add_vars(M)
        m.add_cons(1)
        m.sense(SENSE.MIN)
        for i in range(M):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        m.a_row(0, list(range(M)), [1.0] * M)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        # ||Xw-b||^2 = 0.5 w'(2X'X)w - 2(X'b)'w + b'b, lower triangle, full coefficient
        for i in range(M):
            for j in range(i + 1):
                q = 0.0
                for k in range(N):
                    q += 2.0 * X[k][i] * X[k][j]
                m.qobj_ij(i, j, q)
        cf = 0.0
        for k in range(N):
            cf += b[k] * b[k]
        m.cfix(cf)
        for i in range(M):
            c = 0.0
            for k in range(N):
                c += -2.0 * b[k] * X[k][i]
            m.cj(i, c)
        r = m.solve()
    if r.rc != 0:
        return -1.0, [0.0] * M
    return r.objective, r.x[:M]


def main() -> int:
    allok = True
    print("lsq_l1_penalty")
    # case A: w0 = 0, uniform gamma -> reduce to the QP + gamma
    N, M = 8, 3
    X, b = [[0.0] * M for _ in range(N)], [0.0] * N
    st = 987654321
    for i in range(N):
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        b[i] = ((st >> 16) & 0x3FF) / 1024.0
        for j in range(M):
            st = (st * 1103515245 + 12345) & 0xFFFFFFFF
            X[i][j] = 0.2 + ((st >> 16) & 0x3FF) / 1024.0
    gamma = 0.7
    ol, wl = l1_solve(N, M, X, b, [gamma] * M, [0.0] * M)
    oq, wq = qp_solve(N, M, X, b)
    dw = max(abs(wl[i] - wq[i]) for i in range(M))
    good = ol > 0 and oq > 0 and dw < 1e-6 and abs(ol - (oq + gamma)) < 1e-6
    print("  A (w0=0, gamma=%.2f): L1 obj=%.8f  QP obj+gamma=%.8f  |dw|=%.2e  %s"
          % (gamma, ol, oq + gamma, dw, "OK" if good else "FAIL"))
    allok &= good

    # case B: M = 2, general gamma and w0 -> brute-force scan of w1
    N, M = 5, 2
    X = [[0.5, 1.0], [1.0, 0.4], [0.2, 0.9], [0.8, 0.6], [0.4, 0.4]]
    b = [0.7, 0.5, 0.3, 0.9, 0.4]
    gam, w0 = [0.3, 1.1], [0.6, 0.4]
    ol, wl = l1_solve(N, M, X, b, gam, w0)
    best, bestw1 = math.inf, 0.0
    for k in range(2000001):
        w1 = k / 2000000.0
        w2 = 1.0 - w1
        f = 0.0
        for i in range(N):
            rr = X[i][0] * w1 + X[i][1] * w2 - b[i]
            f += rr * rr
        f += gam[0] * abs(w1 - w0[0]) + gam[1] * abs(w2 - w0[1])
        if f < best:
            best, bestw1 = f, w1
    good = ol > 0 and abs(ol - best) < 1e-5 and abs(wl[0] - bestw1) < 1e-4
    print("  B (M=2, gamma=(%.1f,%.1f)): conic obj=%.8f  brute obj=%.8f  w1=%.6f (bf %.6f)  %s"
          % (gam[0], gam[1], ol, best, wl[0], bestw1, "OK" if good else "FAIL"))
    allok &= good
    print("OK" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
