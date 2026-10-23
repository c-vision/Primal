#!/usr/bin/env python3
"""Lyapunov LMI for the stability of a linear system (andel-vitorio/linear-matrix-
inequality, "LMIs & Linear System Analysis and Control", Problem 1).

    minimize   trace(P)
    subject to P >= 0,     A' P + P A + Q <= 0

with Q = I. The minimum-trace solution saturates the inequality, i.e. solves
A'P + PA + Q = 0. Case A = [[0,-2],[1,-3]]: P* = [[1,-1/2],[-1/2,1/2]] (PSD,
eigenvalues 0.691, 0.809) and trace(P*) = 3/2.

The LMI A'P+PA+I <= 0 is written with a second PSD bar variable S and the three
entrywise equalities S = -(A'P+PA+I); <M,X> counts the off-diagonal of M twice, so a
unit E_ij with i != j brings 2*X_ij. Port of examples/c/lmi_lyapunov.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m00 = m.sparsesymmat(2, [(0, 0, 1.0)])
        m01 = m.sparsesymmat(2, [(0, 1, 1.0)])
        m11 = m.sparsesymmat(2, [(1, 1, 1.0)])
        m.barvars([2])                              # barvar 0 : P (2x2 PSD)
        m.barvars([2])                              # barvar 1 : S (2x2 PSD) = -(A'P+PA+I)
        m.add_cons(3)
        m.sense(SENSE.MIN)
        # Row 0: S00 + 2 P01 = -1  ->  <E00,S> + <E01,P> = -1
        m.baraij(0, 1, [(m00, 1.0)])
        m.baraij(0, 0, [(m01, 1.0)])
        m.con_bounds(0, BK.FX, -1.0, -1.0)
        # Row 1: S01 + P11 - 2 P00 - 3 P01 = 0
        #   -> 0.5<E01,S> + <E11,P> - 2<E00,P> - 1.5<E01,P> = 0
        m.baraij(1, 1, [(m01, 0.5)])
        m.baraij(1, 0, [(m11, 1.0)])
        m.baraij(1, 0, [(m00, -2.0)])
        m.baraij(1, 0, [(m01, -1.5)])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        # Row 2: S11 - 4 P01 - 6 P11 = -1  ->  <E11,S> - 2<E01,P> - 6<E11,P> = -1
        m.baraij(2, 1, [(m11, 1.0)])
        m.baraij(2, 0, [(m01, -2.0)])
        m.baraij(2, 0, [(m11, -6.0)])
        m.con_bounds(2, BK.FX, -1.0, -1.0)
        # objective: minimize trace(P) = <E00,P> + <E11,P>
        m.barcj(0, [(m00, 1.0)])
        m.barcj(0, [(m11, 1.0)])
        r = m.solve()
        ok = r.rc == 0
        obj, p, s = (r.objective, m.getbarxj(0), m.getbarxj(1)) if ok else (0.0, [0.0] * 4, [0.0] * 4)
    tr_p, det_p = p[0] + p[3], p[0] * p[3] - p[1] * p[2]
    lmin = tr_p / 2 - math.sqrt(max(0.0, tr_p * tr_p / 4 - det_p))
    good = (ok and abs(obj - 1.5) < 1e-6 and abs(p[0] - 1.0) < 1e-6 and abs(p[1] + 0.5) < 1e-6
            and abs(p[2] + 0.5) < 1e-6 and abs(p[3] - 0.5) < 1e-6 and lmin > -1e-9)
    print(f"lmi_lyapunov  obj={obj:.8f} (expected 1.5 = 3/2)  "
          f"P=[[{p[0]:.6f},{p[1]:.6f}],[{p[2]:.6f},{p[3]:.6f}]]  lmin={lmin:.2e}")
    print(f"  trace(P)={tr_p:.8f}  S (=-(A'P+PA+I)) = [[{s[0]:.2e},{s[1]:.2e}],[.,{s[2]:.2e}]]  "
          f"{'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
