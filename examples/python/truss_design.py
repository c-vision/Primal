#!/usr/bin/env python3
"""Minimum-volume truss by semidefinite programming (MOSEK Modeling Cookbook / MOSEK
Tutorials, "truss topology design").

Two symmetric bars A-C and B-C (A=(-1,1), B=(1,1) fixed, C=(0,0) free and loaded with
F=(0,-1)); cross-sections t_1, t_2 >= 0, Young modulus E=1, length L=sqrt(2).
Minimize the volume L(t_1+t_2) subject to a compliance bound f'K(t)^{-1}f <= 1, which
the Schur complement turns into the LMI

    X = [[K(t), f], [f', 1]] >= 0,   K(t) = sum_i t_i (E/L_i) d_i d_i'

By symmetry t1 = t2 = sqrt(2) and the volume is 4; the sample cross-checks the
optimum and the compliance recomputed from the published t.
Port of examples/c/truss_design.c.
"""

import math

from primalsolver import Model, BK

NDOF = 2
DIM = NDOF + 1                                      # [[K, f], [f', 1]]
L = 1.4142135623730951                              # sqrt(2)
K0 = 1.0 / (2.0 * 1.4142135623730951)               # 1/(2 sqrt2)


def barentry(m, row, i, j, coef):
    """Add the LMI entry coef * X[i][j] to row `row` of the (DIM x DIM) bar block."""
    mid = m.sparsesymmat(DIM, [(i, j, 1.0)])
    m.baraij(row, 0, [(mid, coef)])


def main() -> int:
    with Model() as m:
        m.add_vars(2)                               # 0 = t1, 1 = t2 (bar cross-sections)
        m.add_cons(6)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.cj(0, L)                                  # min L (t1 + t2)
        m.cj(1, L)
        m.barvars([DIM])
        barentry(m, 0, 0, 0, 1.0)                   # row 0: X[0][0] - K0 (t1 + t2) = 0
        m.a_row(0, [0, 1], [-K0, -K0]); m.con_bounds(0, BK.FX, 0.0, 0.0)
        barentry(m, 1, 1, 1, 1.0)                   # row 1: X[1][1] - K0 (t1 + t2) = 0
        m.a_row(1, [0, 1], [-K0, -K0]); m.con_bounds(1, BK.FX, 0.0, 0.0)
        barentry(m, 2, 0, 1, 0.5)                   # row 2: X[0][1] - K0 (t2 - t1) = 0
        m.a_row(2, [0, 1], [K0, -K0]); m.con_bounds(2, BK.FX, 0.0, 0.0)
        barentry(m, 3, 0, 2, 0.5)                   # row 3: X[0][2] = 0   (f_x = 0)
        m.con_bounds(3, BK.FX, 0.0, 0.0)
        barentry(m, 4, 1, 2, 0.5)                   # row 4: X[1][2] = -1  (f_y = -1)
        m.con_bounds(4, BK.FX, -1.0, -1.0)
        barentry(m, 5, 2, 2, 1.0)                   # row 5: X[2][2] = 1
        m.con_bounds(5, BK.FX, 1.0, 1.0)
        r = m.solve()
    if r.rc != 0:
        print(f"truss_design  rc={r.rc} FAIL")
        return 1
    obj, t1, t2 = r.objective, r.x[0], r.x[1]
    # compliance recomputed from t: sqrt(2)/(2 t1 t2) * (t1+t2)
    comp = (L / (2.0 * t1 * t2)) * (t1 + t2) if t1 > 0 and t2 > 0 else 1e30
    ok = abs(obj - 4.0) < 1e-5 and comp <= 1.0 + 1e-6
    print(f"truss_design  t=({t1:.6f},{t2:.6f}) volume={obj:.6f} (expected 4) compliance={comp:.6f} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
