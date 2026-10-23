#!/usr/bin/env python3
"""Lowner-John inner ellipsoidal approximation (MOSEK Cookbook 11.6,
lownerjohn_ellipsoid.cc).

The max-volume ellipsoid { C u + d : ||u|| <= 1 } inscribed in { x : A x <= b } solves
    maximize   t
    s.t.       t <= det(C)^(1/n)            (det_rootn)
               || C a_i || <= b_i - a_i'd    (one QUAD cone per face)
               C >= 0.
det_rootn for n = 2 uses the PSD block Y = [[C, Z],[Z', diag(Z)]] (4x4), Z lower
triangular, and t <= sqrt(Z00*Z11) as the rotated cone 2*Z00*(Z11/2) >= t^2.

Hand instance: the unit square [0,1]^2 -> the circle C = (1/2) I, d = (1/2,1/2),
t = 1/2. Port of examples/c/lownerjohn_ellipsoid.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

T, D0, D1, X00, X01, X11, Z00, Z11, UU, AA, BB, CC, EE, NV = range(14)
R = 14


def main() -> int:
    with Model(maxcon=R, maxvar=NV) as m:
        m.sense(SENSE.MAX)
        m.barvars([4])
        m.var_bounds(T, BK.LO, 0.0, math.inf)
        m.cj(T, 1.0)
        m.var_bounds(D0, BK.RA, 0.0, 1.0)
        m.var_bounds(D1, BK.RA, 0.0, 1.0)
        m.var_bounds(X00, BK.LO, 0.0, math.inf)
        m.var_bounds(X01, BK.FR, -math.inf, math.inf)
        for v in (X11, Z00, Z11, UU, AA, BB, CC, EE):
            m.var_bounds(v, BK.LO, 0.0, math.inf)
        e = {ij: m.sparsesymmat(4, [(ij[0], ij[1], 1.0)])
             for ij in [(0, 0), (0, 1), (1, 1), (0, 2), (1, 3), (0, 3), (2, 2), (3, 3), (2, 3)]}
        # Y entries = scalar variables (off-diagonal counted twice: coefficient 1/2)
        for row, (var, ij, coef) in enumerate([(X00, (0, 0), 1.0), (X01, (0, 1), 0.5), (X11, (1, 1), 1.0),
                                               (Z00, (0, 2), 0.5), (Z11, (1, 3), 0.5)]):
            m.a_row(row, [var], [-1.0])
            m.baraij(row, 0, [(e[ij], coef)])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
        m.baraij(5, 0, [(e[(0, 3)], 0.5)])
        m.con_bounds(5, BK.FX, 0.0, 0.0)            # Z01 = 0
        m.baraij(6, 0, [(e[(2, 2)], 1.0)])
        m.baraij(6, 0, [(e[(0, 2)], -0.5)])
        m.con_bounds(6, BK.FX, 0.0, 0.0)            # Y22 = Z00
        m.baraij(7, 0, [(e[(3, 3)], 1.0)])
        m.baraij(7, 0, [(e[(1, 3)], -0.5)])
        m.con_bounds(7, BK.FX, 0.0, 0.0)            # Y33 = Z11
        m.baraij(8, 0, [(e[(2, 3)], 0.5)])
        m.con_bounds(8, BK.FX, 0.0, 0.0)            # Y23 = 0
        m.a_row(9, [UU], [1.0])
        m.baraij(9, 0, [(e[(1, 3)], -0.25)])
        m.con_bounds(9, BK.FX, 0.0, 0.0)            # U = Z11/2
        m.a_row(10, [AA, D0], [1.0, 1.0]);  m.con_bounds(10, BK.FX, 1.0, 1.0)   # A = 1-d0
        m.a_row(11, [BB, D0], [1.0, -1.0]); m.con_bounds(11, BK.FX, 0.0, 0.0)   # B = d0
        m.a_row(12, [CC, D1], [1.0, 1.0]);  m.con_bounds(12, BK.FX, 1.0, 1.0)   # C = 1-d1
        m.a_row(13, [EE, D1], [1.0, -1.0]); m.con_bounds(13, BK.FX, 0.0, 0.0)   # E = d1
        m.cone(CT.RQUAD, [Z00, UU, T])              # Z00*Z11 >= T^2
        m.cone(CT.QUAD, [AA, X00, X01])
        m.cone(CT.QUAD, [BB, X00, X01])
        m.cone(CT.QUAD, [CC, X01, X11])
        m.cone(CT.QUAD, [EE, X01, X11])
        r = m.solve()
    ok = r.rc == 0
    obj, x = (r.objective, r.x) if ok else (0.0, [0.0] * NV)
    good = (ok and abs(obj - 0.5) < 1e-5 and abs(x[X00] - 0.5) < 1e-4 and abs(x[X11] - 0.5) < 1e-4
            and abs(x[X01]) < 1e-4 and abs(x[D0] - 0.5) < 1e-4 and abs(x[D1] - 0.5) < 1e-4)
    print(f"lownerjohn: rc={r.rc} t*={obj:.8f}  C=({x[X00]:.5f},{x[X01]:.5f},{x[X11]:.5f}) "
          f"d=({x[D0]:.5f},{x[D1]:.5f})  (expected 0.5, C=0.5I, d=(0.5,0.5))  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
