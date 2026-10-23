#!/usr/bin/env python3
"""The SDO relaxation of a mixed-integer QCQP (MOSEK Cookbook 11.10,
qcqp_sdo_relaxation.cc).

The relaxation drops x in {integer} and relaxes X = x x' to the PSD block
    Z = [[X, x],[x', 1]] >= 0        with   diag(X) >= x,
minimizing <P, X> + 2 q'x. The Cookbook compares its rounded x with the integer
least-squares min |A x - b| over integer x (a QUAD cone). The Cookbook's random data
are not reproducible, so both models are solved on a hand instance: A = I,
c = (2/5, 3/5). The relaxation optimum is the integral point x = (0,1) with value
-1/5; the integer least-squares is |(0,1) - c| = sqrt(8)/5 = 0.565685...
Port of examples/c/qcqp_sdo_relaxation.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE

X00, X01, X11, X0, X1, NV = 0, 1, 2, 3, 4, 5


def sdo_relax(c):
    """The SDO relaxation; returns (rc, value)."""
    with Model(maxcon=8, maxvar=NV) as m:
        m.sense(SENSE.MIN)
        m.barvars([3])                              # Z : 3x3 PSD
        for v in range(NV):
            m.var_bounds(v, BK.FR, -math.inf, math.inf)
        m.cj(X00, 1.0); m.cj(X11, 1.0)              # <I,X>
        m.cj(X0, -2.0 * c[0]); m.cj(X1, -2.0 * c[1])   # 2 q'x, q = -c
        e00, e01, e11, e02, e12, e22 = (m.sparsesymmat(3, [(i, j, 1.0)])
                                        for i, j in [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)])
        # Z entries <- scalar variables (off-diagonal counted twice)
        for r, (var, e, coef) in enumerate([(X00, e00, 1.0), (X01, e01, 0.5), (X11, e11, 1.0),
                                            (X0, e02, 0.5), (X1, e12, 0.5)]):
            m.a_row(r, [var], [-1.0])
            m.baraij(r, 0, [(e, coef)])
            m.con_bounds(r, BK.FX, 0, 0)
        m.baraij(5, 0, [(e22, 1.0)])                # Z22 = 1
        m.con_bounds(5, BK.FX, 1.0, 1.0)
        m.a_row(6, [X00, X0], [1.0, -1.0])          # diag(X) >= x
        m.con_bounds(6, BK.LO, 0.0, math.inf)
        m.a_row(7, [X11, X1], [1.0, -1.0])
        m.con_bounds(7, BK.LO, 0.0, math.inf)
        r = m.solve()
    return r.rc, (r.objective if r.rc == 0 else 0.0)


def int_ls(c):
    """The integer least-squares min |x - c| over integer x, via (t, x-c) in QUAD."""
    xv0, xv1, tt, e0, e1 = 0, 1, 2, 3, 4
    with Model(maxcon=1, maxvar=3) as m:
        m.sense(SENSE.MIN)
        for j in range(2):
            m.var_bounds(j, BK.LO, -100.0, math.inf)
            m.var_type(j, VAR_TYPE.INT)
        m.var_bounds(tt, BK.LO, 0.0, math.inf)
        m.cj(tt, 1.0)
        m.add_vars(2)                               # the cone members are auxiliary
        m.var_bounds(e0, BK.FR, -math.inf, math.inf)
        m.var_bounds(e1, BK.FR, -math.inf, math.inf)
        m.a_row(0, [e0, xv0], [1.0, -1.0])
        m.con_bounds(0, BK.FX, -c[0], -c[0])
        m.add_cons(1)
        m.a_row(1, [e1, xv1], [1.0, -1.0])
        m.con_bounds(1, BK.FX, -c[1], -c[1])
        m.cone(CT.QUAD, [tt, e0, e1])               # t >= ||x - c||
        r = m.solve()
    return r.rc, (r.objective if r.rc == 0 else 0.0)


def main() -> int:
    c = [2.0 / 5.0, 3.0 / 5.0]
    r1, relax = sdo_relax(c)
    r2, ils = int_ls(c)
    expect_relax = -1.0 / 5.0                       # diag(X) >= x cuts x = c
    expect_ils = math.sqrt(8.0) / 5.0               # sqrt(8/25)
    good = r1 == 0 and r2 == 0 and abs(relax - expect_relax) < 1e-6 and abs(ils - expect_ils) < 1e-6
    print(f"qcqp_sdo_relaxation: relax={relax:.8f} (expected -1/5={expect_relax:.8f})  "
          f"int_ls={ils:.8f} (expected sqrt8/5={expect_ils:.8f})  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
