#!/usr/bin/env python3
"""Soft-margin primal SVM (MOSEK Cookbook 11.2, primal_svm.cc).

    minimize   t + c * sum_i xi_i
    subject to y_i * ( <X_i, w> - b ) + xi_i >= 1     (each sample)
               xi_i >= 0
               (t, 1, w) in RQUAD   =>  t >= ||w||^2 / 2

The MOSEK example draws X at random (not portable), so the technique is exercised
on two hand-derived instances:
  A) separable diamond X = (2,0),(0,2),(-2,0),(0,-2), y = +1,+1,-1,-1, c = 1:
     optimum 1/4, w = (1/2,1/2), b = 0.
  B) soft margin: X = (0),(0), y = +1,-1, c = 1: optimum 2, w = 0, xi = (1,1).
Port of examples/c/primal_svm.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE


def svm(nd, xs, y, c):
    """Solve one SVM; returns (ok, obj, w, b)."""
    nim = len(xs)
    t_, b_, w_, xi_ = 0, 1, 2, 2 + nd
    one = 2 + nd + nim
    with Model() as m:
        m.add_vars(3 + nd + nim)
        m.add_cons(nim)
        m.sense(SENSE.MIN)
        m.var_bounds(t_, BK.FR, -math.inf, math.inf)
        m.var_bounds(b_, BK.FR, -math.inf, math.inf)
        for j in range(nd):
            m.var_bounds(w_ + j, BK.FR, -math.inf, math.inf)
        for i in range(nim):
            m.var_bounds(xi_ + i, BK.LO, 0.0, math.inf)
        m.var_bounds(one, BK.FX, 1.0, 1.0)
        m.cj(t_, 1.0)
        for i in range(nim):
            m.cj(xi_ + i, c)
        m.cone(CT.RQUAD, [t_, one] + [w_ + j for j in range(nd)])   # t >= ||w||^2 / 2
        for i in range(nim):                        # y_i(<X_i,w>-b) + xi_i >= 1
            sub = [w_ + j for j in range(nd) if xs[i][j] != 0.0]
            val = [y[i] * xs[i][j] for j in range(nd) if xs[i][j] != 0.0]
            m.a_row(i, sub + [b_, xi_ + i], val + [-y[i], 1.0])
            m.con_bounds(i, BK.LO, 1.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, [0.0] * nd, 0.0
    return True, r.objective, [r.x[w_ + j] for j in range(nd)], r.x[b_]


def main() -> int:
    bad = 0
    ok, obj, w, b = svm(2, [(2, 0), (0, 2), (-2, 0), (0, -2)], [1, 1, -1, -1], 1.0)   # A) separable
    good = ok and abs(obj - 0.25) < 1e-7 and abs(w[0] - 0.5) < 1e-6 and abs(w[1] - 0.5) < 1e-6 and abs(b) < 1e-6
    print(f"SVM A (separable, c=1): obj={obj:.8f} w=({w[0]:.4f},{w[1]:.4f}) b={b:.2e}  "
          f"(expected 0.25, w=(1/2,1/2), b=0)  {'OK' if good else 'FAIL'}")
    bad |= not good
    ok, obj, w, b = svm(1, [(0, 0), (0, 0)], [1, -1], 1.0)                            # B) soft margin
    good = ok and abs(obj - 2.0) < 1e-7 and abs(w[0]) < 1e-6 and abs(b) < 1e-6
    print(f"SVM B (soft, c=1):      obj={obj:.8f} w=({w[0]:.4f}) b={b:.2e}      "
          f"(expected 2, w=0, b=0)          {'OK' if good else 'FAIL'}")
    bad |= not good
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
