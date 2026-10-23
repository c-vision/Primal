#!/usr/bin/env python3
"""MOSEK port "solutionquality" - measuring solution quality.

Primal/dual violations via PRIMAL_getprimalinfeas / PRIMAL_getdualinfeas:
  1. optimal LP: min x0+x1 s.t. x0+x1 >= 1, x >= 0 -> both violations ~ 0;
  2. the getters refuse a task that has no solution (ERR_ARG).
Port of examples/c/solutionquality.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SOL

RES_ERR_ARG = 1001


def main() -> int:
    with Model() as m:                              # 1. optimal LP
        m.add_vars(2)
        m.add_cons(1)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.LO, 1.0, math.inf)
        r = m.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        pinf = r.pinf
        dinf = m.getdualinfeas()
    print(f"primal violation = {pinf:.3e}, dual = {dinf:.3e} (expected ~ 0)")

    with Model() as t2:                             # 2. without a solution: refused
        t2.add_vars(1)
        dummy = ctypes.c_double()
        rejected = ps.PRIMAL_getprimalinfeas(t2.task, SOL.ITR, ctypes.byref(dummy)) == RES_ERR_ARG

    ok = pinf < 1e-6 and dinf < 1e-6 and rejected
    print(f"{'OK' if ok else 'FAIL'} (quality measured + protected getters)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
