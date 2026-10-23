#!/usr/bin/env python3
"""The Bounded Real Lemma (H-infinity norm) as an SDP
(S-Salavati/Convex-Optimization-with-LMIs-in-{C-plus-plus,Python}).

For a stable LTI system (A,B,C,D), ||G||_inf < gamma iff there is P > 0 with
    [[A'P + PA + C'C,  PB + C'D],
     [B'P + D'C,       D'D - gamma^2 I]] <= 0.
The model minimises gamma^2; the LMI is written as a PSD bar variable R = -M with
entrywise equalities, entries addressed through symmetric matrices.

Case: SISO first order A = -1, B = 1, C = 1, D = 0, whose H-infinity norm is
||G||_inf = |C B| / |A| = 1: the Schur condition g >= P^2/(2P-1) is minimised at
P = 1, g = 1. Port of examples/c/bounded_real_lmi.c.
"""

import math

from primalsolver import Model, BK, SENSE

B, C, D = 1.0, 1.0, 0.0                             # SISO, A = -1, |CB|/|A| = 1
G = 0


def main() -> int:
    with Model() as m:
        r00 = m.sparsesymmat(2, [(0, 0, 1.0)])
        r01 = m.sparsesymmat(2, [(0, 1, 1.0)])
        r11 = m.sparsesymmat(2, [(1, 1, 1.0)])
        p00 = m.sparsesymmat(1, [(0, 0, 1.0)])
        m.barvars([1])                              # barvar 0 : P (1x1 PSD)
        m.barvars([2])                              # barvar 1 : R (2x2 PSD) = -M
        m.add_vars(1)
        m.var_bounds(G, BK.LO, 0.0, math.inf)       # g = gamma^2 >= 0
        m.sense(SENSE.MIN)
        m.cj(G, 1.0)
        m.add_cons(3)
        # M = [[-2P + C^2, P*B + C*D],[P*B + D*C, D^2 - g]]  (A = -1)
        # row 0 : R00 = -M00 = 2P - C^2  ->  R00 - 2P = -C^2
        m.baraij(0, 1, [(r00, 1.0)])
        m.baraij(0, 0, [(p00, -2.0)])
        m.con_bounds(0, BK.FX, -C * C, -C * C)
        # row 1 : R01 = -M01 = -(P*B + C*D)  ->  R01 + P*B = -C*D
        m.baraij(1, 1, [(r01, 0.5)])                # <E01,R>/2 = R01
        m.baraij(1, 0, [(p00, B)])
        m.con_bounds(1, BK.FX, -C * D, -C * D)
        # row 2 : R11 = -M11 = g - D^2  ->  R11 - g = -D^2
        m.baraij(2, 1, [(r11, 1.0)])
        m.a_row(2, [G], [-1.0])
        m.con_bounds(2, BK.FX, -D * D, -D * D)
        r = m.solve()
        ok = r.rc == 0
        g2, p = (r.x[G], m.getbarxj(0)) if ok else (0.0, [0.0])
    # independent check: the LMI at the returned point must be <= 0 (2x2), and
    # gamma = sqrt(g2) must equal |CB|/|A| = 1
    m00, m01, m11 = -2 * p[0] + C * C, p[0] * B + C * D, D * D - g2
    tr, det = m00 + m11, m00 * m11 - m01 * m01
    lmax = tr / 2 + math.sqrt(max(0.0, tr * tr / 4 - det))
    gamma = math.sqrt(g2)
    good = ok and abs(gamma - 1.0) < 1e-6 and lmax <= 1e-6
    print(f"bounded_real_lmi  gamma^2*={g2:.8f}  gamma*={gamma:.8f} (expected 1)  P={p[0]:.6f}  "
          f"lmax={lmax:.2e}  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
