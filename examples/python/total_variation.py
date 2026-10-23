#!/usr/bin/env python3
"""Total-variation denoising (MOSEK Cookbook 11.3, total_variation.cc).

    minimize    sum over the grid of the TV terms t
    subject to  (t, du) in QUAD            (the total variation)
                (sigma, f - u) in QUAD     (the fidelity ball ||f-u|| <= sigma)
                u in [0,1]

The MOSEK example uses a random 100x200 image with no known value. The same
technique is exercised here on the smallest non-trivial signal, a 1-D two-point
signal, which has a closed form: f = (0,1), sigma = 1/2, u in [0,1]^2,
    minimize |u1-u0|  s.t.  u0^2 + (1-u1)^2 <= 1/4
whose optimum is d = 1 - 1/sqrt(2) = 0.29289322 at u0 = 1/(2 sqrt2) = 0.35355339.
Port of examples/c/total_variation.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

F = [0.0, 1.0]
SIGMA = 0.5
U, TT, DX, E0, E1, SG, NV = 0, 2, 3, 4, 5, 6, 7


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(3)
        m.sense(SENSE.MIN)
        for j in range(2):
            m.var_bounds(U + j, BK.RA, 0.0, 1.0)
        m.var_bounds(TT, BK.LO, 0.0, math.inf)
        for j in (DX, E0, E1):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.var_bounds(SG, BK.FX, SIGMA, SIGMA)
        m.cj(TT, 1.0)
        # dx = u1 - u0 ; e_j = f_j - u_j
        m.a_row(0, [DX, U, U + 1], [1.0, 1.0, -1.0]); m.con_bounds(0, BK.FX, 0.0, 0.0)
        m.a_row(1, [E0, U], [1.0, 1.0]);              m.con_bounds(1, BK.FX, F[0], F[0])
        m.a_row(2, [E1, U + 1], [1.0, 1.0]);          m.con_bounds(2, BK.FX, F[1], F[1])
        m.cone(CT.QUAD, [TT, DX])                    # TV cone (t, dx)
        m.cone(CT.QUAD, [SG, E0, E1])                # fidelity cone (sigma, e0, e1)
        r = m.solve()
    ok = r.rc == 0
    obj, u = (r.objective, r.x) if ok else (0.0, [0.0] * NV)
    tv = 1.0 - 1.0 / math.sqrt(2.0)
    a0 = 1.0 / (2.0 * math.sqrt(2.0))
    good = (ok and abs(obj - tv) < 1e-7 and abs(u[0] - a0) < 1e-6 and abs(u[1] - (1.0 - a0)) < 1e-6
            and u[0] * u[0] + (1.0 - u[1]) * (1.0 - u[1]) <= SIGMA * SIGMA + 1e-9)
    print(f"total_variation: TV*={obj:.8f}  u=({u[0]:.6f}, {u[1]:.6f})  "
          f"(expected 1-1/sqrt2, u=(1/(2sqrt2),1-1/(2sqrt2)))  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
