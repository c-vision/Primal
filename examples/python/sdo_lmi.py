#!/usr/bin/env python3
"""MOSEK port "sdo_lmi" - linear matrix inequality (LMI) with scalar variables.

    max x0 + x1  s.t.  F(x) = F0 + x0*F1 + x1*F2 PSD (3x3 LMI),  x0 + x1 <= 2.5

with F0 = I3, F1 = -E00, F2 = -E11, i.e. F(x) = diag(1 - x0, 1 - x1, 1). The LMI
binds at x0 = x1 = 1, obj = 2, X = diag(0, 0, 1). Model: a 3x3 PSD bar X equal to
F(x) entry by entry (FX rows on the diagonal, off-diagonals of X set to 0).
Port of examples/c/sdo_lmi.c.
"""

import math

from primalsolver import Model, BK, SENSE

D = 3


def main() -> int:
    with Model() as m:
        m.add_vars(2)                               # scalar variables x0, x1
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.sense(SENSE.MAX)
        m.barvars([D])                              # bar X 3x3 PSD
        me = [m.sparsesymmat(D, [(i, i, 1.0)]) for i in range(D)]
        m.add_cons(7)                               # 3 diagonal + 3 off-diagonal + 1 linear
        r = 0
        m.baraij(r, 0, [(me[0], 1.0)]); m.a_row(r, [0], [1.0])   # X_00 = 1 - x0
        m.con_bounds(r, BK.FX, 1.0, 1.0); r += 1
        m.baraij(r, 0, [(me[1], 1.0)]); m.a_row(r, [1], [1.0])   # X_11 = 1 - x1
        m.con_bounds(r, BK.FX, 1.0, 1.0); r += 1
        m.baraij(r, 0, [(me[2], 1.0)])                           # X_22 = 1
        m.con_bounds(r, BK.FX, 1.0, 1.0); r += 1
        # off-diagonals to 0 (the symmetric matrix with entry (i,j) counts 2X_ij)
        mo = [m.sparsesymmat(D, [(i, j, 1.0)]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        for k in range(3):
            m.baraij(r, 0, [(mo[k], 1.0)])
            m.con_bounds(r, BK.FX, 0.0, 0.0)
            r += 1
        m.a_row(r, [0, 1], [1.0, 1.0])              # x0 + x1 <= 2.5
        m.con_bounds(r, BK.UP, -math.inf, 2.5)
        res = m.solve()
        if res.rc != 0:
            print(f"optimize rc={res.rc}")
            return 1
        xb = m.getbarxj(0)
    x, po = res.x, res.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}), obj = {po:.4f} (expected (1,1), 2)")
    print(f"X = diag({xb[0]:.4f}, {xb[4]:.4f}, {xb[8]:.4f}) (expected (0,0,1))")
    ok = (abs(x[0] - 1.0) < 1e-4 and abs(x[1] - 1.0) < 1e-4 and abs(po - 2.0) < 1e-4
          and abs(xb[0]) < 1e-4 and abs(xb[4]) < 1e-4 and abs(xb[8] - 1.0) < 1e-4)
    print(f"{'OK' if ok else 'FAIL'} (LMI active: x=(1,1), X=diag(0,0,1) PSD at the boundary)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
