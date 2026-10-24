#!/usr/bin/env python3
"""Transaction-cost LP with free variables and a degenerate (cost-free) column.

StackOverflow 37586543 (MOSEK Fusion): n = 3, initial holdings x0 = (-20, -50, -10),
transaction costs t = (.01, .01, .01):

    min t'z  s.t.  l - x0 >= 0,  z - (x - x0) >= 0,  z - (x0 - x) >= 0,  e'x = 0
    x free, z free, l >= 0

z >= |x - x0| and e'(x - x0) = 80 give the value 0.8 on a whole FACE (every
x_i >= x0_i); the asker's vector (60, -50, -10) is one of its points, so the check is
the value plus face membership. Port of examples/c/transaction_cost.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    x0 = [-20.0, -50.0, -10.0]
    with Model() as m:
        m.add_vars(9)                           # 0..2 x, 3..5 z, 6..8 l
        m.add_cons(10)                          # 0..2 long1, 3..5 buy, 6..8 sell, 9 longeqshort
        for i in range(3):
            m.var_bounds(i, BK.FR, -math.inf, math.inf)
            m.var_bounds(3 + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(6 + i, BK.LO, 0.0, math.inf)
            m.cj(3 + i, 0.01)
        for i in range(3):
            m.a_row(i, [6 + i], [1.0])
            m.con_bounds(i, BK.LO, x0[i], math.inf)
            m.a_row(3 + i, [3 + i, i], [1.0, -1.0])
            m.con_bounds(3 + i, BK.LO, -x0[i], math.inf)
            m.a_row(6 + i, [3 + i, i], [1.0, 1.0])
            m.con_bounds(6 + i, BK.LO, x0[i], math.inf)
        m.a_row(9, [0, 1, 2], [1.0, 1.0, 1.0])
        m.con_bounds(9, BK.FX, 0.0, 0.0)
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    obj, x, pinf = r.objective, r.x, r.pinf
    sx, viol = 0.0, 0.0
    for i in range(3):
        sx += x[i]
        if x0[i] - x[i] > viol:
            viol = x0[i] - x[i]
        need = abs(x[i] - x0[i])
        if need - x[3 + i] > viol:
            viol = need - x[3 + i]
    on_face = viol <= 1e-6 and abs(sx) <= 1e-6
    ok = abs(obj - 0.8) < 1e-6 and on_face and pinf <= 1e-6
    print("x = (%.4f, %.4f, %.4f), obj = %.6f (expected 0.8)" % (x[0], x[1], x[2], obj))
    print("e'x = %.2e, x_i >= x0_i: %s, getprimalinfeas = %.2e" % (sx, "yes" if on_face else "NO", pinf))
    print("%s (optimum on a face: value 0.8, not the post's vector)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
