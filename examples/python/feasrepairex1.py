#!/usr/bin/env python3
"""MOSEK port "feasrepairex1" - feasibility repair.

An infeasible LP is "repaired" by finding the point with minimum total violation
(elastic relaxation, PRIMAL_feasrepair). Infeasible problem (verified by hand):
    min x0  s.t.  x0 >= 2,  x0 <= 1   (no feasible x)
Minimum repair: the minimum total violation is 1 (moving one bound by 1).
Port of examples/c/feasrepairex1.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SOL

RES_OK = 0
RES_ERR_INFEASIBLE = 1002


def main() -> int:
    with Model() as m:
        m.add_vars(1)
        m.add_cons(2)
        m.var_bounds(0, BK.FR, -math.inf, math.inf)
        m.cj(0, 1.0)
        m.a_row(0, [0], [1.0]); m.con_bounds(0, BK.LO, 2.0, math.inf)    # x0 >= 2
        m.a_row(1, [0], [1.0]); m.con_bounds(1, BK.UP, -math.inf, 1.0)   # x0 <= 1

        # 1. optimize: infeasible
        r = m.solve()
        print(f"optimize: rc={r.rc} (expected 1002 infeasible)")
        infeas = r.rc == RES_ERR_INFEASIBLE

        # 2. feasrepair: point of minimum violation
        rc = ps.PRIMAL_feasrepair(m.task)
        if rc != RES_OK:
            print(f"feasrepair rc={rc}")
            return 1
        xbuf = (ctypes.c_double * 1)()
        ps.PRIMAL_getxx(m.task, SOL.ITR, xbuf)
    x0 = xbuf[0]
    # total violation of the repaired point: (2 - x0)+ + (x0 - 1)+ = 1
    viol = 0.0
    if x0 < 2.0:
        viol += 2.0 - x0
    if x0 > 1.0:
        viol += x0 - 1.0
    print(f"feasrepair: x0 = {x0:.4f}, total violation = {viol:.4f} (expected = 1)")
    ok = infeas and abs(viol - 1.0) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (repair to minimum violation)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
