#!/usr/bin/env python3
"""Nearest correlation matrix, "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Example 10.1.

Given an invalid correlation estimate Sigma_hat (smallest eigenvalue -0.1337), solve

    min ||Sigma - Sigma_hat||_F  s.t.  Sigma PSD,  diag(Sigma) = 1.

Model: bar variable X (4x4 PSD), an auxiliary d_ij per i <= j with
sum d_ij^2 = ||Sigma_hat - X||_F^2, and the cone (t, d). Checked against Higham's
alternating projections and the book's 2-digit values. Port of examples/c/omf_nearestcorr.c.
"""

import math

from primalsolver import Model, BK, CT

N = 4
A = [[1.0, 0.8, 0.5, 0.2], [0.8, 1.0, 0.9, 0.1], [0.5, 0.9, 1.0, 0.7], [0.2, 0.1, 0.7, 1.0]]
REF = [[1.00, 0.76, 0.53, 0.18], [0.76, 1.00, 0.82, 0.15], [0.53, 0.82, 1.00, 0.65], [0.18, 0.15, 0.65, 1.00]]
HIGHAM = [[1.0000, 0.7698, 0.5301, 0.1823], [0.7698, 1.0000, 0.8169, 0.1488],
          [0.5301, 0.8169, 1.0000, 0.6513], [0.1823, 0.1488, 0.6513, 1.0000]]


def main() -> int:
    nd = N * (N + 1) // 2
    with Model() as m:
        m.add_vars(1 + nd)                          # 0 = t (Frobenius norm), 1..nd = d_ij
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0)
        for k in range(nd):
            m.var_bounds(1 + k, BK.FR, -math.inf, math.inf)
        msym = [m.sparsesymmat(N, [(i, j, 1.0)]) for i in range(N) for j in range(i, N)]
        m.barvars([N])
        m.add_cons(nd + N)
        row = 0
        for i in range(N):
            for j in range(i, N):
                w = 1.0 if i == j else 2.0
                rhs = A[i][j] if i == j else 2.0 * A[i][j]
                m.baraij(row, 0, [(msym[row], 1.0)])
                m.a_row(row, [1 + row], [math.sqrt(w)])
                m.con_bounds(row, BK.FX, rhs, rhs)
                row += 1
        for k in range(N):                          # X_kk = 1
            m.baraij(row, 0, [(msym[k * (2 * N - k + 1) // 2], 1.0)])
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        m.cone(CT.QUAD, list(range(nd + 1)))        # t >= ||d|| = ||Sigma_hat - X||_F
        r = m.solve()
        ok = r.rc == 0
        if ok:
            Xr, po = m.getbarxj(0), r.objective
    if ok:
        fro, dev_book, dev_high = 0.0, 0.0, 0.0
        for i in range(N):
            for j in range(N):
                xij = Xr[i * N + j]
                d = xij - A[i][j]
                fro += d * d
                if abs(xij - REF[i][j]) > dev_book:
                    dev_book = abs(xij - REF[i][j])
                if abs(xij - HIGHAM[i][j]) > dev_high:
                    dev_high = abs(xij - HIGHAM[i][j])
        fro = math.sqrt(fro)
        ok = dev_high < 5e-3 and abs(Xr[0] - 1.0) < 1e-5 and abs(po - fro) < 1e-4
        print("omf_nearestcorr  ||Sig-X||_F=%.6g  max|X-Higham|=%.4g  max|X-book|=%.4g (book 2 sig figs) %s"
              % (fro, dev_high, dev_book, "OK" if ok else "FAIL"))
    else:
        print("omf_nearestcorr  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
