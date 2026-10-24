#!/usr/bin/env python3
"""Branch-and-bound examples from "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 11.3.2.

Two pure integer LPs on the same region -x1 + x2 <= 2, 8 x1 + 2 x2 <= 19, x >= 0 integer:
    (A) max x1 + x2    -> the book's optimum x = (1, 3), objective 4
    (B) max 3 x1 + x2  -> the book's optimum x = (2, 1), objective 7
Port of examples/c/omf_bb.c.
"""

import math

from primalsolver import Model, BK, VAR_TYPE


def solve_case(name, c1, c2, want_obj, wx1, wx2):
    """Solve one branch-and-bound MILP case and check the book optimum."""
    with Model(2, 2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.var_type(j, VAR_TYPE.INT)
        m.cj(0, -c1); m.cj(1, -c2)
        m.a_row(0, [0, 1], [-1.0, 1.0]); m.con_bounds(0, BK.UP, -math.inf, 2.0)
        m.a_row(1, [0, 1], [8.0, 2.0]); m.con_bounds(1, BK.UP, -math.inf, 19.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(z + want_obj) < 1e-6 and abs(x[0] - wx1) < 1e-6 and abs(x[1] - wx2) < 1e-6
        print("omf_bb %-3s x=(%.6g,%.6g) obj=%.6g (book x=(%.4g,%.4g), %.4g) %s"
              % (name, x[0], x[1], -z, wx1, wx2, want_obj, "OK" if ok else "FAIL"))
    else:
        print("omf_bb %-3s rc=%d FAIL" % (name, r.rc))
    return ok


def main() -> int:
    a = solve_case("(A)", 1.0, 1.0, 4.0, 1.0, 3.0)
    b = solve_case("(B)", 3.0, 1.0, 7.0, 2.0, 1.0)
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
