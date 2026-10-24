#!/usr/bin/env python3
"""Minimum variance with a cardinality constraint (a mixed-integer QP).

    min w'Sigma w  s.t.  e'w = 1,  w >= 0,  sum_j z_j <= K,  w_j <= z_j,  z binary

Hand case: Sigma = I, N = 3, K = 2. Unconstrained the minimum is 1/3 at
(1/3, 1/3, 1/3); with K = 2 it is 1/2 at (1/2, 1/2, 0) (any pair).
Port of examples/c/cardinality_portfolio.c.
"""

import math

from primalsolver import Model, BK, VAR_TYPE


def main() -> int:
    with Model() as m:
        m.add_vars(6)                           # 0..2 w, 3..5 z
        m.add_cons(5)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, 1.0)
            m.var_bounds(3 + j, BK.RA, 0.0, 1.0)
            m.var_type(3 + j, VAR_TYPE.INT_BIN)
            m.a_row(j, [j, 3 + j], [1.0, -1.0])     # w_j - z_j <= 0
            m.con_bounds(j, BK.UP, -math.inf, 0.0)
        m.a_row(3, [0, 1, 2], [1.0, 1.0, 1.0])  # e'w = 1
        m.con_bounds(3, BK.FX, 1.0, 1.0)
        m.a_row(4, [3, 4, 5], [1.0, 1.0, 1.0])  # sum z <= 2
        m.con_bounds(4, BK.UP, -math.inf, 2.0)
        for j in range(3):                      # 1/2 w'Qw with Q = 2I -> w0^2+w1^2+w2^2
            m.qobj([(j, j, 2.0)])
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    obj, x = r.objective, r.x
    used = sum(1 for j in range(3) if x[j] > 1e-6)
    quad = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    ok = abs(obj - 0.5) < 1e-6 and used <= 2 and abs(x[0] + x[1] + x[2] - 1.0) < 1e-6 and abs(quad - obj) < 1e-6
    print("w = (%.4f, %.4f, %.4f), active = %d, obj = %.6f (expected 0.5)" % (x[0], x[1], x[2], used, obj))
    print("%s (MIQP: cardinality K=2 over 3 active)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
