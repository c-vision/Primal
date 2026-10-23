#!/usr/bin/env python3
"""SDP duality (port of the "dual_sdo_l1" example of the MOSEK C API docs).

The dual of  min <C, X> s.t. <A_i, X> = b_i, X PSD  is  max b'y s.t.
S = C - sum_i y_i A_i PSD. The example builds the dual as a SEPARATE problem and
checks strong duality <C, X*> = b'y*.

Primal: min X00 s.t. X11 = 1, X PSD (2x2)      -> X* = diag(0, 1), pobj = 0.
Dual:   max y s.t. S = C - y A_1 = diag(0, -y) PSD -> y* = 0, dobj = 0.
Port of examples/c/dual_sdo_l1.c.
"""

import math

from primalsolver import Model, BK, SENSE


def primal_sdp():
    """Solve min X00 s.t. X11 = 1; returns (pobj, X)."""
    po, xm = -1e30, [0.0] * 4
    with Model() as m:
        m.add_vars(1)
        m.var_bounds(0, BK.FX, 0.0, 0.0)            # dummy scalar
        mc = m.sparsesymmat(2, [(0, 0, 1.0)])
        ma = m.sparsesymmat(2, [(1, 1, 1.0)])
        m.barvars([2])
        m.barcj(0, [(mc, 1.0)])                     # min X00
        m.add_cons(1)
        m.baraij(0, 0, [(ma, 1.0)])
        m.con_bounds(0, BK.FX, 1.0, 1.0)            # X11 = 1
        r = m.solve()
        if r.rc == 0:
            xm, po = m.getbarxj(0), r.objective
    return po, xm


def dual_sdp():
    """Solve max y s.t. S = C - y A_1 PSD; returns (dobj, y, S)."""
    dobj, y, s = -1e30, 0.0, [0.0] * 4
    with Model() as m:
        m.add_vars(1)                               # y
        m.var_bounds(0, BK.FR, -math.inf, math.inf)
        m.cj(0, 1.0)                                # max y (b_1 = 1)
        m.sense(SENSE.MAX)
        m.barvars([2])
        mc = m.sparsesymmat(2, [(0, 0, 1.0)])
        ma = m.sparsesymmat(2, [(1, 1, 1.0)])
        mo = m.sparsesymmat(2, [(0, 1, 1.0)])
        m.add_cons(3)
        m.baraij(0, 0, [(mc, 1.0)]); m.con_bounds(0, BK.FX, 0.0, 0.0)   # S_00 = 0
        m.baraij(1, 0, [(mo, 1.0)]); m.con_bounds(1, BK.FX, 0.0, 0.0)   # S_01 = 0
        m.baraij(2, 0, [(ma, 1.0)])                                     # S_11 = -y
        m.a_row(2, [0], [1.0])
        m.con_bounds(2, BK.FX, 0.0, 0.0)
        r = m.solve()
        if r.rc == 0:
            s, dobj, y = m.getbarxj(0), r.objective, r.x[0]
    return dobj, y, s


def main() -> int:
    pobj, xm = primal_sdp()
    dobj, y, s = dual_sdp()
    print(f"primal: pobj = {pobj:.6f} (expected 0), X = diag({xm[0]:.4f}, {xm[3]:.4f})")
    print(f"dual:   dobj = {dobj:.6f} (expected 0), y = {y:.4f}, S = diag({s[0]:.4f}, {s[3]:.4f})")
    ok = (abs(pobj) < 1e-6 and abs(dobj) < 1e-6 and abs(pobj - dobj) < 1e-6      # strong duality
          and abs(xm[0]) < 1e-4 and abs(xm[3] - 1.0) < 1e-4
          and abs(s[0]) < 1e-4 and abs(s[3]) < 1e-4)                            # S = diag(0,0) PSD
    print(f"{'OK' if ok else 'FAIL'} (strong duality: <C,X*> = b'y* = 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
