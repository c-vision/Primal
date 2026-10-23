#!/usr/bin/env python3
"""MOSEK port "sdo2" - SDP with TWO bar variables (PSD matrices) and a constant.

    min  <I, X1> + <I, X2> + 1.0            (cfix = 1)
    s.t. <J, X1> + <I, X2> = 1              (J = all-ones matrix)
         X1, X2 >= 0 (PSD)

Analysis: X1 pays <J,X1> with tr(X1) >= <J,X1>/2 = 1/2 (rank-1, X1 = J/4), while
X2 = 0. Expected: obj = 0.5 + 1 = 1.5, X1 = J/4, X2 = 0. Port of examples/c/sdo2.c.
"""

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m.add_cons(1)
        mi = m.sparsesymmat(2, [(0, 0, 1.0), (1, 1, 1.0)])
        mj = m.sparsesymmat(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0)])
        m.barvars([2, 2])
        m.cfix(1.0)
        m.sense(SENSE.MIN)
        m.barcj(0, [(mi, 1.0)])                     # <I, X1> + <I, X2> in the objective
        m.barcj(1, [(mi, 1.0)])
        m.baraij(0, 0, [(mj, 1.0)])                 # <J, X1> + <I, X2> = 1
        m.baraij(0, 1, [(mi, 1.0)])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        x1 = m.getbarxj(0)
        x2 = m.getbarxj(1)
    po = r.objective
    print(f"X1 = [{x1[0]:.6f} {x1[1]:.6f}; {x1[2]:.6f} {x1[3]:.6f}]")
    print(f"X2 = [{x2[0]:.6f} {x2[1]:.6f}; {x2[2]:.6f} {x2[3]:.6f}]")
    print(f"obj = {po:.6f}")
    ok = (abs(po - 1.5) < 1e-4 and all(abs(v - 0.25) <= 1e-4 for v in x1)
          and all(abs(v) <= 1e-4 for v in x2))
    print(f"{'OK' if ok else 'FAIL'} (expected obj = 1.5, X1 = J/4, X2 = 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
