#!/usr/bin/env python3
"""MOSEK port "sparsecholesky" - a Cholesky factor exists iff the matrix is PSD.

The linear SDP version: fix the banded matrix A (3x3 tridiagonal, the 1-D Laplacian)
as a bar variable X with FX rows and solve min 0 with X PSD: the problem is feasible
iff A is PSD, i.e. iff the Cholesky factorization exists. The independent check is
an explicit 3x3 Cholesky factorization with ||L L' - A||_1.
Port of examples/c/sparsecholesky.c.
"""

import math

from primalsolver import Model, BK, SOLSTA

D = 3
A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]          # SPD tridiagonal: the 1-D Laplacian


def chol3(mat):
    """Explicit 3x3 Cholesky (independent check); returns L or None."""
    low = [[0.0] * D for _ in range(D)]
    for i in range(D):
        for j in range(i + 1):
            s = mat[i][j]
            for k in range(j):
                s -= low[i][k] * low[j][k]
            if i == j:
                if s <= 0:
                    return None
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def main() -> int:
    with Model() as m:
        m.add_vars(1)
        m.var_bounds(0, BK.FX, 0.0, 0.0)            # dummy
        m.barvars([D])
        me = [m.sparsesymmat(D, [(i, i, 1.0)]) for i in range(D)]
        mo = [m.sparsesymmat(D, [(i, j, 1.0)]) for i in range(D) for j in range(i + 1, D)]
        m.add_cons(6)                               # fix X = A: FX rows on every entry
        r = 0
        for i in range(D):
            m.baraij(r, 0, [(me[i], 1.0)])
            m.con_bounds(r, BK.FX, A[i][i], A[i][i])
            r += 1
        oc = 0
        for i in range(D):
            for j in range(i + 1, D):               # <E_ij, X> = 2 X_ij = 2 A_ij
                m.baraij(r, 0, [(mo[oc], 1.0)])
                m.con_bounds(r, BK.FX, 2.0 * A[i][j], 2.0 * A[i][j])
                r += 1
                oc += 1
        res = m.solve()
    if res.rc != 0:
        print(f"optimize rc={res.rc}")
        return 1
    sta = res.solsta
    print(f"SDP feasibility of A: solsta = {sta} ({'PSD' if sta == SOLSTA.OPTIMAL else 'NOT PSD'})")
    low = chol3(A)                                  # independent check: explicit Cholesky
    ok_chol = low is not None
    print(f"Direct Cholesky: {'exists (A SPD)' if ok_chol else 'does not exist'}")
    if ok_chol:
        print("L =")
        for i in range(D):
            print("   " + "".join(f" {low[i][j]:8.4f}" for j in range(D)))
        err = 0.0                                   # A = L L' check
        for i in range(D):
            for j in range(D):
                s = 0.0
                for k in range(D):
                    s += low[i][k] * low[j][k]
                err += abs(s - A[i][j])
        print(f"||L L' - A||_1 = {err:.3e}")
        if err > 1e-9:
            ok_chol = False
    ok = sta == SOLSTA.OPTIMAL and ok_chol
    print(f"{'OK' if ok else 'FAIL'} (A PSD via SDP = Cholesky exists)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
