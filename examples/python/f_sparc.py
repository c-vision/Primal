#!/usr/bin/env python3
"""Subcarrier and power allocation as a mixed-integer exp-cone program
(MOSEK Tutorials, f-sparc).

    maximize    sum_{i,j} z_ij
    subject to  sum_j x_ij <= 1,  p_tilde_ij <= x_ij,  x binary
                SIGMA * t + sum p_tilde = 1,   t in [1/P, 1/SIGMA]
                sum_i z_ij >= t * d_j
                (t + p_tilde/n, t, z * log(2)/BW) in K_exp   per (i,j)

The cone is PRIMAL_CT_PEXP: x1 >= x2 exp(x3/x2), i.e. t + p/n >= t * 2^(z/(BW t)).
Instance I=1, J=1, n=1, d=0.2 (BW=1.25, P=36, SIGMA=10): the model reduces to
maximize BW t log2(1 + (1 - SIGMA t)/(n t)) over t in [1/P, 1/SIGMA], checked
against a dense scan of t (optimum z ~ 0.2206 at t ~ 0.058).
Port of examples/c/f_sparc.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE

SIGMA = 10.0
PMAX = 36.0
BW = 1.25
NOISE = 1.0
DEMAND = 0.2
Z, X, PP, TT, TP, ZC, NV = 0, 1, 2, 3, 4, 5, 6


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(5)
        m.var_bounds(Z, BK.LO, 0.0, math.inf)
        m.var_bounds(X, BK.RA, 0.0, 1.0)
        m.var_type(X, VAR_TYPE.INT_BIN)
        m.var_bounds(PP, BK.LO, 0.0, math.inf)
        m.var_bounds(TT, BK.RA, 1.0 / PMAX, 1.0 / SIGMA)
        m.var_bounds(TP, BK.FR, -math.inf, math.inf)
        m.var_bounds(ZC, BK.FR, -math.inf, math.inf)
        m.a_row(0, [TT, PP], [SIGMA, 1.0]);                 m.con_bounds(0, BK.FX, 1.0, 1.0)      # SIGMA*t + p = 1
        m.a_row(1, [PP, X], [1.0, -1.0]);                   m.con_bounds(1, BK.UP, -math.inf, 0.0)  # p - x <= 0
        m.a_row(2, [TT, Z], [DEMAND, -1.0]);                m.con_bounds(2, BK.UP, -math.inf, 0.0)  # t*d - z <= 0
        m.a_row(3, [TP, PP, TT], [1.0, -1.0 / NOISE, -1.0]); m.con_bounds(3, BK.FX, 0.0, 0.0)     # tp = p/n + t
        m.a_row(4, [ZC, Z], [1.0, -math.log(2.0) / BW]);    m.con_bounds(4, BK.FX, 0.0, 0.0)      # zc = z log2/BW
        m.cone(CT.PEXP, [TP, TT, ZC])
        m.cj(Z, 1.0)
        m.sense(SENSE.MAX)
        r = m.solve()
    ok = r.rc == 0
    z, tv = (r.x[Z], r.x[TT]) if ok else (0.0, 0.0)

    best, bt = -1e300, 0.0                          # reference: dense scan of t
    for q in range(400001):
        tt = 1.0 / PMAX + (1.0 / SIGMA - 1.0 / PMAX) * q / 400000.0
        p = 1.0 - SIGMA * tt
        if p < 0.0:
            break
        zz = BW * tt * math.log2(1.0 + p / (NOISE * tt))
        if zz < DEMAND * tt:
            continue                                # demand constraint
        if zz > best:
            best, bt = zz, tt
    okall = ok and abs(z - best) < 1e-4 * (1.0 + abs(best))
    print(f"f_sparc  I=1 J=1  n={NOISE:.1f} d={DEMAND:.1f}")
    print(f"  z* = {z:.8f}  t = {tv:.6f}   [scan {best:.8f} at t={bt:.6f}]")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
