#!/usr/bin/env python3
"""QP (matches the MOSEK Optimizer API "qo1" example).

    min x^2 + y^2  s.t.  x + y >= 1,  x, y >= 0

Expected solution: x = 0.5, y = 0.5, obj = 0.5. The quadratic term is 0.5 x'Qx with
Q = diag(2). Port of examples/c/qo1.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    numvar, numcon = 2, 1
    with Model() as m:
        m.add_vars(numvar)
        m.add_cons(numcon)
        for j in range(numvar):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])      # Q = diag(2): 0.5 x'Qx = x^2 + y^2
        m.sense(SENSE.MIN)
        m.a_row(0, [0, 1], [1.0, 1.0])          # x + y >= 1
        m.con_bounds(0, BK.LO, 1.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    xx, obj = r.x, r.objective
    print("x = %.6f, y = %.6f, obj = %.6f" % (xx[0], xx[1], obj))
    ok = abs(xx[0] - 0.5) < 1e-6 and abs(xx[1] - 0.5) < 1e-6 and abs(obj - 0.5) < 1e-6
    print("OK (expected x=0.5, y=0.5, obj=0.5)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
