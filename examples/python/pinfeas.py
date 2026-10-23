#!/usr/bin/env python3
"""MOSEK port "pinfeas" - a primal-infeasible LP.

    min 0  s.t.  x0 + x1 >= 4,  x0 + x1 <= 2,  x >= 0

The two rows are mutually exclusive: the solver must detect the infeasibility
(rc = ERR_INFEASIBLE) and publish the certificate (solsta = PRIM_INFEAS_CER).
Port of examples/c/pinfeas.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA

RES_ERR_INFEASIBLE = 1002


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(2)
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.a_row(1, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.LO, 4.0, math.inf)
        m.con_bounds(1, BK.UP, -math.inf, 2.0)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.sense(SENSE.MIN)
        r = m.solve()
    print(f"rc = {r.rc} (ERR_INFEASIBLE = {RES_ERR_INFEASIBLE}), solsta = {r.solsta} "
          f"(PRIM_INFEAS_CER = {SOLSTA.PRIM_INFEAS_CER})")
    ok = r.rc == RES_ERR_INFEASIBLE and r.solsta == SOLSTA.PRIM_INFEAS_CER
    print(f"{'OK' if ok else 'FAIL'} (expected infeasible with certificate)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
