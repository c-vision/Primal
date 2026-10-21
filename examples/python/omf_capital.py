#!/usr/bin/env python3
"""OMF Section 11.2 - capital budgeting (0-1) problem.

    max 8 x1 + 11 x2 + 6 x3 + 4 x4
    s.t. 6.7 x1 + 10 x2 + 5.5 x3 + 3.4 x4 <= 19,  x in {0,1}.
Book integer optimum x=(0,1,1,1), value 21000 (21).   Port of omf_capital.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE, SOLSTA


def main() -> int:
    with Model(maxcon=1, maxvar=4) as m:
        for j in range(4):
            m.var_bounds(j, BK.RA, 0.0, 1.0)
            m.var_type(j, VAR_TYPE.INT_BIN)
        m.sense(SENSE.MAX)
        m.cj(0, 8.0); m.cj(1, 11.0); m.cj(2, 6.0); m.cj(3, 4.0)
        m.a_row(0, [0, 1, 2, 3], [6.7, 10.0, 5.5, 3.4]); m.con_bounds(0, BK.UP, up=19.0)
        r = m.solve()
    x = r.x
    ok = (r.solsta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL)
          and all(abs(x[j] - round(x[j])) < 1e-6 for j in range(4))
          and abs(x[0]) < 1e-6 and abs(x[1] - 1) < 1e-6 and abs(x[2] - 1) < 1e-6
          and abs(x[3] - 1) < 1e-6 and abs(r.objective - 21.0) < 1e-6)
    print(f"omf_capital  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g},{x[3]:.6g}) "
          f"Z={1000.0 * r.objective:.0f} (book (0,1,1,1), 21000) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
