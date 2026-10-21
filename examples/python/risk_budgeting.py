#!/usr/bin/env python3
"""PrimalSolver - risk budgeting via exponential cones.

Port of c_examples/finance/risk_budgeting.c (Spinu 2013):

    min (1/2) w'Sigma w - sum_i b_i ln(w_i),   (w_i, 1, -s_i) in K_exp.
Hand case Sigma=I, b=(0.8,0.2): normalised w=(2/3,1/3), risk contributions
(0.8,0.2).

    python python_examples/risk_budgeting.py
"""

from primalsolver import Model, BK, CT, SOLSTA

B = [0.8, 0.2]


def main() -> int:
    with Model(maxcon=2, maxvar=7) as m:   # w0,w1,s0,s1,v0,v1,one
        m.var_bounds(0, BK.LO, 0.0)
        m.var_bounds(1, BK.LO, 0.0)
        m.var_bounds(2, BK.FR); m.var_bounds(3, BK.FR)
        m.var_bounds(4, BK.FR); m.var_bounds(5, BK.FR)
        m.var_bounds(6, BK.FX, 1.0)
        m.cone(CT.PEXP, [0, 6, 4])       # w0 >= exp(v0) = exp(-s0)
        m.cone(CT.PEXP, [1, 6, 5])
        m.a_row(0, [4, 2], [1.0, 1.0]); m.con_bounds(0, BK.FX, 0.0, 0.0)
        m.a_row(1, [5, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])
        m.cj(2, B[0]); m.cj(3, B[1])
        r = m.solve()

    s = r.x[0] + r.x[1]
    w0, w1 = r.x[0] / s, r.x[1] / s
    rc0 = w0 * w0 / (w0 * w0 + w1 * w1)
    rc1 = w1 * w1 / (w0 * w0 + w1 * w1)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(w0 - 2.0 / 3.0) < 1e-3
          and abs(w1 - 1.0 / 3.0) < 1e-3 and abs(rc0 - B[0]) < 1e-4 and abs(rc1 - B[1]) < 1e-4)
    print(f"risk_budgeting w=({w0:.6f},{w1:.6f}) (2/3,1/3) risk=({rc0:.6f},{rc1:.6f}) "
          f"(0.8,0.2) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
