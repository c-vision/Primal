#!/usr/bin/env python3
"""PrimalSolver - transaction-cost LP (port of c_examples/finance/transaction_cost.c).

Source: StackOverflow 37586543 ("MOSEK Markowitz portfolio transaction costs").
n = 3, initial holdings x0 = (-20,-50,-10), costs t = (.01,.01,.01):

    min  t'z
    s.t. long1:       l - x0 >= 0
         buy:         z - (x - x0) >= 0
         sell:        z - (x0 - x) >= 0
         longeqshort: e'x = 0
    x free, z free, l >= 0.

The optimum is not a point but a FACE; the value is 0.01*80 = 0.8 and x must
satisfy x_i >= x0_i and e'x = 0 (not the specific vector from the post).

    python python_examples/transaction_cost.py
"""

from primalsolver import Model, BK, SOLSTA

X0 = [-20.0, -50.0, -10.0]
COST = [0.01, 0.01, 0.01]


def main() -> int:
    with Model(maxcon=10, maxvar=9) as m:      # vars 0..2 x, 3..5 z, 6..8 l
        for i in range(3):
            m.var_bounds(i, BK.FR)
            m.var_bounds(3 + i, BK.FR)
            m.var_bounds(6 + i, BK.LO, 0.0)
            m.cj(3 + i, COST[i])
            m.a_row(i, [6 + i], [1.0])
            m.con_bounds(i, BK.LO, X0[i])
            m.a_row(3 + i, [3 + i, i], [1.0, -1.0])
            m.con_bounds(3 + i, BK.LO, -X0[i])
            m.a_row(6 + i, [3 + i, i], [1.0, 1.0])
            m.con_bounds(6 + i, BK.LO, X0[i])
        m.a_row(9, [0, 1, 2], [1.0, 1.0, 1.0])
        m.con_bounds(9, BK.FX, 0.0, 0.0)
        r = m.solve()

    sx = sum(r.x[:3])
    viol = 0.0
    for i in range(3):
        viol = max(viol, X0[i] - r.x[i], abs(r.x[i] - X0[i]) - r.x[3 + i])
    on_face = viol <= 1e-6 and abs(sx) <= 1e-6
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 0.8) < 1e-6
          and on_face and r.pinf <= 1e-6)
    print(f"x = ({r.x[0]:.4f}, {r.x[1]:.4f}, {r.x[2]:.4f}), obj = {r.objective:.6f} (expected 0.8)")
    print(f"e'x = {sx:.2e}, x_i >= x0_i: {'yes' if on_face else 'NO'}, pinf = {r.pinf:.2e}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
