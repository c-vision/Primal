#!/usr/bin/env python3
"""OMF Example 10.1 - nearest correlation matrix (SDP).

Port of c_examples/finance/omf_nearestcorr.c:

    min ||Sigma - Sigma_hat||_F  s.t. Sigma >= 0, diag(Sigma) = 1.
Model: bar variable X (4x4 PSD); auxiliaries d_ij make sum d_ij^2 = ||Sigma_hat
- X||_F^2, epigraph t with (t, d) in QUAD. Checked against Higham's projection.

    python python_examples/omf_nearestcorr.py
"""

import math

from primalsolver import Model, BK, CT, SOLSTA

N = 4
A = [[1.0, 0.8, 0.5, 0.2],
     [0.8, 1.0, 0.9, 0.1],
     [0.5, 0.9, 1.0, 0.7],
     [0.2, 0.1, 0.7, 1.0]]
HIGHAM = [[1.0000, 0.7698, 0.5301, 0.1823],
          [0.7698, 1.0000, 0.8169, 0.1488],
          [0.5301, 0.8169, 1.0000, 0.6513],
          [0.1823, 0.1488, 0.6513, 1.0000]]


def main() -> int:
    nd = N * (N + 1) // 2
    msym = {}
    with Model(maxcon=nd + N, maxvar=1 + nd) as m:
        m.var_bounds(0, BK.LO, 0.0)          # t
        m.cj(0, 1.0)
        for k in range(nd):
            m.var_bounds(1 + k, BK.FR)       # d_ij
        # one sparse symmetric matrix per (i,j), i<=j
        for i in range(N):
            for j in range(i, N):
                msym[(i, j)] = m.sparsesymmat(N, [(i, j, 1.0)])
        m.barvars([N])
        row = 0
        for i in range(N):
            for j in range(i, N):
                w = 1.0 if i == j else 2.0
                rhs = A[i][j] if i == j else 2.0 * A[i][j]
                m.baraij(row, 0, [(msym[(i, j)], 1.0)])
                m.a_row(row, [1 + row], [math.sqrt(w)])   # d index = row (i<=j order)
                m.con_bounds(row, BK.FX, rhs, rhs)
                row += 1
        for k in range(N):                   # X_kk = 1
            m.baraij(row, 0, [(msym[(k, k)], 1.0)])
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        m.cone(CT.QUAD, [0] + [1 + k for k in range(nd)])
        r = m.solve()
        X = m.getbarxj(0)

    fro = math.sqrt(sum((X[i * N + j] - A[i][j]) ** 2 for i in range(N) for j in range(N)))
    dev_high = max(abs(X[i * N + j] - HIGHAM[i][j]) for i in range(N) for j in range(N))
    ok = (r.solsta == SOLSTA.OPTIMAL and dev_high < 5e-3
          and abs(X[0] - 1.0) < 1e-5 and abs(r.objective - fro) < 1e-4)
    print(f"omf_nearestcorr ||Sig-X||_F={fro:.6g} max|X-Higham|={dev_high:.4g} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
