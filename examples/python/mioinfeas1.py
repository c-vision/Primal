#!/usr/bin/env python3
"""MOSEK port "mioinfeas1" - an infeasible MIP.

    min 0  s.t.  x >= 5,  x <= 2,  x integer >= 0

Expected (reference table 7.3, integer problems): prosta = PRIM_INFEAS,
solsta = UNKNOWN. The reference never publishes an infeasibility certificate as
solsta on a MIP: that number is reserved for a Farkas vector, which a
branch-and-bound does not produce. rc stays ERR_INFEASIBLE.
Port of examples/c/mioinfeas1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA, VAR_TYPE

RES_ERR_INFEASIBLE = 1002
PRO_STA_PRIM_INFEAS = 4


def main() -> int:
    with Model() as m:
        m.add_vars(1)
        m.add_cons(2)
        m.a_row(0, [0], [1.0])
        m.a_row(1, [0], [1.0])
        m.con_bounds(0, BK.LO, 5.0, math.inf)
        m.con_bounds(1, BK.UP, -math.inf, 2.0)
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_type(0, VAR_TYPE.INT)
        m.sense(SENSE.MIN)
        r = m.solve()
    print(f"rc = {r.rc}, prosta = {r.prosta}, solsta = {r.solsta}")
    ok = (r.rc == RES_ERR_INFEASIBLE and r.prosta == PRO_STA_PRIM_INFEAS
          and r.solsta == SOLSTA.UNKNOWN)
    print(f"{'OK' if ok else 'FAIL'} (expected MIP infeasible: prosta PRIM_INFEAS + solsta UNKNOWN)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
