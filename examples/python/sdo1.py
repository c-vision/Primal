#!/usr/bin/env python3
"""SDP (variant of the MOSEK docs "sdo1" example).

    min tr(X)  s.t.  <J,X> = 1 (J = all-ones 2x2 matrix),  X >= 0

Analysis: X = [[a,b],[b,c]] PSD, a+c+2b = 1  =>  tr = s = a+c, ac >= (1-s)^2/4 and
ac <= s^2/4  =>  s >= 1/2. Expected solution: tr(X) = 0.5 with
X = [[0.25,0.25],[0.25,0.25]]. Port of examples/c/sdo1.c.
"""

from primalsolver import Model, BK, SENSE


def main() -> int:
    with Model() as m:
        m.barvars([2])                              # one 2x2 X
        # J = [[1,1],[1,1]] = E00 + E11 + 2*E01: entry (0,1) counts twice in the
        # inner product, only one triangle is given
        mj = m.sparsesymmat(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0)])
        m.add_cons(1)                               # <J,X> = 1
        m.baraij(0, 0, [(mj, 1.0)])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        mi = m.sparsesymmat(2, [(0, 0, 1.0), (1, 1, 1.0)])
        m.barcj(0, [(mi, 1.0)])                     # min tr(X) = <I,X>
        m.sense(SENSE.MIN)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    print(f"tr(X) = {r.objective:.6f}")
    ok = abs(r.objective - 0.5) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (expected tr(X) = 0.5, X = J/2 rank-1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
