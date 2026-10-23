#!/usr/bin/env python3
"""MOSEK port "djc1" - disjunctive constraints (DJC) on AFEs and domains.

    min x0   s.t. 0 <= x0 <= 10 and the disjunction
         [ x0 <= 2 ]  OR  [ x0 >= 6 AND x0 <= 7 ]
The first disjunct gives min x0 = 0; max x0 with the same disjunction is 7.

DJC encoding: a single AFE f = x0; term 0 is the RMINUS domain (f - 2 <= 0); term 1
is RMINUS (f - 7 <= 0) AND RPLUS (f - 6 >= 0). The reference convention is
F x + g - b, so the b values are the constants 2, 7, 6.
Port of examples/c/djc1.c.
"""

from primalsolver import Model, APPROX, BK, SENSE


def solve(maximize):
    """Solve the disjunction for min or max of x0; returns (objective, x0)."""
    po, x0 = -1e30, 0.0
    with Model() as m:
        m.add_vars(1)
        m.var_bounds(0, BK.RA, 0.0, 10.0)
        m.cj(0, 1.0)
        if maximize:
            m.sense(SENSE.MAX)
        m.afes(1)                                   # one AFE f = x0
        m.afe_entry(0, 0, 1.0)
        dminus = m.domain(APPROX.RMINUS, 1)         # two one-dimensional linear domains
        dplus = m.domain(APPROX.RPLUS, 1)
        m.djcs(1)
        m.djc(0, [dminus, dminus, dplus], [0, 0, 0], [2.0, 7.0, 6.0], [1, 2])
        r = m.solve()
        if r.rc == 0:                               # the DJC binaries add variables
            po, x0 = r.objective, r.x[0]
    return po, x0


def main() -> int:
    pmin, xmin = solve(False)
    pmax, xmax = solve(True)
    print(f"min: x0 = {xmin:.4f} (expected 0), obj = {pmin:.4f}")
    print(f"max: x0 = {xmax:.4f} (expected 7), obj = {pmax:.4f}")
    ok = abs(xmin) < 1e-6 and abs(pmin) < 1e-6 and abs(xmax - 7.0) < 1e-6 and abs(pmax - 7.0) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (disjunction: x<=2 OR 6<=x<=7)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
