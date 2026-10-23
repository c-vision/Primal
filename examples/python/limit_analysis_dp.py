#!/usr/bin/env python3
"""Drucker-Prager yield cone of a rigid-plastic limit analysis, as a second-order cone
program (a0082273/limit-analysis-with-node-based-element-and-MOSEK).

The yield function is the Drucker-Prager surface
    yf(sigma) = sqrt(J2) + 3 alpha m - k <= 0,   m = tr(sigma)/3,
i.e. the second-order cone || s || <= sqrt(2) (k - alpha tr(sigma)) on the deviatoric
stress s, with alpha = sin(phi)/sqrt(9 + 3 sin^2 phi) and
k = sqrt(3) c cos(phi)/sqrt(3 + sin^2 phi). A proportional stress sigma = t * sigma0
is admissible up to t* = k / (sqrt(J2(sigma0)) + alpha tr(sigma0)).
This ports the CONE (the model's core), not the full node-based FEM mesh.

Case: uniaxial sigma0 = (1,0,0,0,0,0), c = 1, phi = 30 deg -> t* = 1.128237.
Port of examples/c/limit_analysis_dp.c.
"""

import math
import sys

from primalsolver import Model, BK, CT, SENSE

T, SIG, SDEV, RHS, NV = 0, 1, 7, 13, 14


def main() -> int:
    c, phi = 1.0, 30.0 * math.pi / 180.0            # material
    alpha = math.sin(phi) / math.sqrt(9.0 + 3.0 * math.sin(phi) * math.sin(phi))
    kk = math.sqrt(3.0) * c * math.cos(phi) / math.sqrt(3.0 + math.sin(phi) * math.sin(phi))
    s0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]             # proportional reference stress: uniaxial
    with Model() as m:
        m.add_vars(NV)
        m.sense(SENSE.MAX)
        for j in range(NV):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.add_cons(6 + 6 + 1)
        r = 0
        for i in range(6):                          # sigma_i - s0_i * t = 0
            m.a_row(r, [SIG + i, T], [1.0, -s0[i]])
            m.con_bounds(r, BK.FX, 0.0, 0.0)
            r += 1
        for i in range(6):                          # sdev_i - sigma_i + (sigma0+sigma1+sigma2)/3 = 0
            if i < 3:
                sub = [SDEV + i, SIG + i, SIG + (i + 1) % 3, SIG + (i + 2) % 3]
                val = [1.0, -2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            else:                                   # shear: the mean stress does not change it
                sub, val = [SDEV + i, SIG + i], [1.0, -1.0]
            m.a_row(r, sub, val)
            m.con_bounds(r, BK.FX, 0.0, 0.0)
            r += 1
        # RHS - sqrt2*k + sqrt2*alpha*(sigma0+sigma1+sigma2) = 0
        sa = math.sqrt(2.0) * alpha
        m.a_row(r, [RHS, SIG + 0, SIG + 1, SIG + 2], [1.0, sa, sa, sa])
        m.con_bounds(r, BK.FX, math.sqrt(2.0) * kk, math.sqrt(2.0) * kk)
        m.cone(CT.QUAD, [RHS] + [SDEV + i for i in range(6)])   # RHS >= ||s||
        m.cj(T, 1.0)
        res = m.solve()
    print(f"rc={res.rc} prosta={res.prosta} solsta={res.solsta}", file=sys.stderr)
    ok = res.rc == 0
    v = res.x if ok else [0.0] * NV
    tstar = v[T] if ok else 0.0

    want = kk / (1.0 / math.sqrt(3.0) + alpha * 1.0)   # hand value + cone check at the stress
    s0v, s1v, s2v = v[SIG + 0], v[SIG + 1], v[SIG + 2]
    mm = (s0v + s1v + s2v) / 3.0
    d = [s0v - mm, s1v - mm, s2v - mm, v[SIG + 3], v[SIG + 4], v[SIG + 5]]
    sq = sum(x * x for x in d)
    yf = math.sqrt(sq / 2.0) + 3.0 * alpha * mm - kk   # must be ~ 0 at yield
    good = ok and abs(tstar - want) < 1e-7 and abs(tstar - 1.128237) < 1e-5 and abs(yf) < 1e-7
    print(f"limit_analysis_dp  alpha={alpha:.8f} k={kk:.8f}  t*={tstar:.8f} (expected {want:.8f})")
    print(f"  sigma_x={s0v:.6f}  sqrt(J2)={math.sqrt(sq / 2.0):.6f}  m={mm:.6f}  yf={yf:.2e}  "
          f"{'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
