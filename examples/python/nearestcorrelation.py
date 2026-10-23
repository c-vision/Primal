#!/usr/bin/env python3
"""Nearest correlation matrix (Higham), port of the "nearestcorrelation" example of the
MOSEK C API docs.

    min  ||A - X||_F  s.t.  X PSD, diag(X) = 1

modelled as min t s.t. (t, d) in QUAD with d_ij = sqrt(w_ij)(A_ij - X_ij) over the
upper triangle (w = 1 on the diagonal, 2 off it), X a PSD bar variable and FX rows
X_ii = 1. A = tridiag(-1, 2, -1). Higham's alternating projections (PSD cone and
unit diagonal) provide the independent check. Port of examples/c/nearestcorrelation.c.
"""

import math

from primalsolver import Model, BK, CT

N = 3
A = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]


def fro(x):
    """Frobenius norm of A - X."""
    s = 0.0
    for i in range(N):
        for j in range(N):
            d = x[i][j] - A[i][j]
            s += d * d
    return math.sqrt(s)


def eig3(mat):
    """Jacobi eigen-decomposition of a symmetric 3x3 matrix: (eigenvalues, V)."""
    w = [row[:] for row in mat]
    v = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]
    for _ in range(100):
        off = 0.0
        for p in range(N):
            for q in range(p + 1, N):
                off += w[p][q] * w[p][q]
        if off < 1e-30:
            break
        for p in range(N):
            for q in range(p + 1, N):
                apq = w[p][q]
                if abs(apq) < 1e-32:
                    continue
                theta = (w[q][q] - w[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                sn = t * c
                for k in range(N):
                    wp, wq = w[p][k], w[q][k]
                    w[p][k], w[q][k] = c * wp - sn * wq, sn * wp + c * wq
                for k in range(N):
                    wp, wq = w[k][p], w[k][q]
                    w[k][p], w[k][q] = c * wp - sn * wq, sn * wp + c * wq
                for k in range(N):
                    vp, vq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vp - sn * vq, sn * vp + c * vq
    return [w[i][i] for i in range(N)], v


def higham():
    """Higham alternating projection: nearest correlation matrix (reference)."""
    x = [row[:] for row in A]
    for _ in range(500):
        ev, v = eig3(x)                             # PSD projection: clip negative eigenvalues
        s = [[0.0] * N for _ in range(N)]
        for k in range(N):
            lk = ev[k] if ev[k] > 0.0 else 0.0
            for i in range(N):
                for j in range(N):
                    s[i][j] += lk * v[i][k] * v[j][k]
        for i in range(N):                          # diag = 1
            s[i][i] = 1.0
        diff = max(abs(s[i][j] - x[i][j]) for i in range(N) for j in range(N))
        x = s
        if diff < 1e-12:
            break
    return x


def print_matrix(x):
    for i in range(N):
        print("   " + "".join(f" {x[i][j]:8.4f}" for j in range(N)))


def main() -> int:
    ntri = N * (N + 1) // 2
    with Model() as m:
        m.add_vars(1)                               # t
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0)                                # min t
        msym = [m.sparsesymmat(N, [(i, j, 1.0)]) for i in range(N) for j in range(i, N)]
        m.barvars([N])
        m.add_vars(ntri)                            # d_ij, upper triangle row-major
        for k in range(ntri):
            m.var_bounds(1 + k, BK.FR, -math.inf, math.inf)
        m.add_cons(ntri + N)
        row = cnt = 0
        for i in range(N):                          # w <E_ij,X>/w... : <E_ij,X> + sqrt(w) d_ij = rhs
            for j in range(i, N):
                w = 1.0 if i == j else 2.0
                rhs = A[i][j] if i == j else 2.0 * A[i][j]
                m.baraij(row, 0, [(msym[cnt], 1.0)])
                m.a_row(row, [1 + cnt], [math.sqrt(w)])
                m.con_bounds(row, BK.FX, rhs, rhs)
                row += 1
                cnt += 1
        diag_ids = [msym[k] for k, (i, j) in enumerate((i, j) for i in range(N) for j in range(i, N)) if i == j]
        for k in range(N):                          # diag rows: X_kk = 1
            m.baraij(row, 0, [(diag_ids[k], 1.0)])
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        m.cone(CT.QUAD, [0] + [1 + k for k in range(ntri)])   # t >= ||d||
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        xr = m.getbarxj(0)
    x = [[xr[i * N + j] for j in range(N)] for i in range(N)]
    print("X =")
    print_matrix(x)
    print(f"obj (Frobenius norm) = {r.objective:.6f}")
    xh = higham()                                   # independent check: Higham
    print("Higham:")
    print_matrix(xh)
    fh, fx = fro(xh), fro(x)
    print(f"||A-X_higham||_F = {fh:.6f}, ||A-X_clone||_F = {fx:.6f}")
    ev, _ = eig3(x)                                 # feasible: diag = 1, PSD
    psd_ok = all(e >= -1e-6 for e in ev)
    diag_ok = all(abs(x[i][i] - 1.0) <= 1e-5 for i in range(N))
    ok = psd_ok and diag_ok and fx <= fh + 1e-3
    print(f"{'OK' if ok else 'FAIL'} (PSD={'yes' if psd_ok else 'no'}, diag=1={'yes' if diag_ok else 'no'}, "
          f"norm ~ Higham)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
