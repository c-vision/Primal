#!/usr/bin/env python3
"""The convex core of the SCA subproblem of vkumar-ucd/secure_ISAC_GC23
(IRS-secure-ISAC beamforming, GLOBECOM 2023).

The repository is an iterative nonconvex algorithm (successive convex
approximation): each step maximizes a beampattern gain subject to
communication-SINR and power constraints, a second-order cone program. This sample
solves that convex subproblem at one step:

    maximize   a'w                        (linearized beampattern gain)
    subject to h'w >= sqrt(gamma) ||w||   (communication SINR, phase-aligned)
               ||w|| <= sqrt(P)            (transmit power)

Hand instance: Nb = 2, h = (1,0), a = (1,1)/sqrt(2), gamma = 1/2, P = 2
-> w = (1,1), a'w = sqrt(2), gain = 2, with both constraints active.
Port of examples/c/secure_beamforming.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

H = [1.0, 0.0]
A = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
GAMMA, P = 0.5, 2.0
W, PB, U, Q, NV, NROW = 0, 2, 3, 5, 6, 3


def main() -> int:
    sg, sp = math.sqrt(GAMMA), math.sqrt(P)
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(NROW)
        for v in range(NV):
            m.var_bounds(v, BK.FR, -math.inf, math.inf)
        m.var_bounds(Q, BK.FX, sp, sp)              # sqrt(P)
        m.sense(SENSE.MAX)
        for j in range(2):
            m.cj(W + j, A[j])                       # maximize a'w
        m.a_row(0, [PB, W, W + 1], [1.0, -H[0], -H[1]])   # p = h'w
        m.con_bounds(0, BK.FX, 0.0, 0.0)
        for j in range(2):                          # u_j = sqrt(gamma) w_j
            m.a_row(1 + j, [U + j, W + j], [1.0, -sg])
            m.con_bounds(1 + j, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [PB, U, U + 1])             # SINR: (h'w, sqrt(gamma) w) in QUAD
        m.cone(CT.QUAD, [Q, W, W + 1])              # power: (sqrt(P), w) in QUAD
        r = m.solve()
    ok = r.rc == 0
    obj, w = (r.objective, r.x) if ok else (0.0, [0.0] * NV)
    # the optimum is degenerate (the objective gradient is radial on the power
    # circle), so w is accurate to ~1e-4 while the objective is to 1e-7.
    good = (ok and abs(obj - math.sqrt(2.0)) < 1e-6 and abs(w[0] - 1.0) < 1e-3
            and abs(w[1] - 1.0) < 1e-3 and abs(w[0] * w[0] + w[1] * w[1] - P) < 1e-6)
    print(f"secure_beamforming: a'w={obj:.8f}  w=({w[0]:.6f},{w[1]:.6f})  gain={obj * obj:.8f}  "
          f"(expected sqrt2, w=(1,1), gain=2)  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
