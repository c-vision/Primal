#!/usr/bin/env python3
"""Lovasz capacity number of C5, "Lectures on Modern Convex Optimization 2020-2023"
(Ben-Tal & Nemirovski).

With the 5-node cycle C5, associate the affine matrix L(x): entry 1 between
non-adjacent nodes (and on the diagonal), the arc variable between adjacent nodes.
The Lovasz capacity number is
    theta(C5) = min_{lambda,x} { lambda : lambda I - L(x) >= 0 },
a semidefinite program. The book states theta(C5) = sqrt(5) exactly.

Model: bar variable X (5x5, PSD) with one equality row per entry (i,j), i <= j:
<E_ij,X> = (lambda I - L(x))_ij. Port of examples/c/lmco_lovasz.c.
"""

import math

from primalsolver import Model, BK

ARC = {(0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4, (0, 4): 5}   # arc variable per adjacent pair


def main() -> int:
    with Model() as m:
        m.add_vars(6)                               # 0 = lambda, 1..5 = xAB, xBC, xCD, xDE, xEA
        m.var_bounds(0, BK.LO, 1.0, math.inf)
        for k in range(1, 6):
            m.var_bounds(k, BK.RA, -10.0, 10.0)
        m.cj(0, 1.0)                                # min lambda
        msym = [m.sparsesymmat(5, [(i, j, 1.0)]) for i in range(5) for j in range(i, 5)]
        m.barvars([5])
        m.add_cons(15)
        row = 0
        for i in range(5):
            for j in range(i, 5):
                m.baraij(row, 0, [(msym[row], 1.0)])
                if i == j:                          # <E_ii,X> = lambda - 1
                    m.a_row(row, [0], [-1.0])
                    m.con_bounds(row, BK.FX, -1.0, -1.0)
                elif (i, j) in ARC:                 # <E_ij,X> = 2X_ij = -2 x_arc
                    m.a_row(row, [ARC[(i, j)]], [2.0])
                    m.con_bounds(row, BK.FX, 0.0, 0.0)
                else:                               # non-arc: <E_ij,X> = -2
                    m.con_bounds(row, BK.FX, -2.0, -2.0)
                row += 1
        r = m.solve()
    if r.rc != 0:
        print(f"lmco_lovasz  rc={r.rc} FAIL")
        return 1
    ok = abs(r.x[0] - math.sqrt(5.0)) < 1e-6
    print(f"lmco_lovasz  theta(C5)={r.x[0]:.9g} (book sqrt(5)={math.sqrt(5.0):.9g}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
