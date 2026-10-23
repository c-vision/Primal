#!/usr/bin/env python3
"""Least-squares fit by a convex piecewise-linear function (MOSEK/Tutorials,
pwl-convex-approximation).

Given data points X_i in R^d and values Y_i, fit convex PWL function values t_i at
the knots together with subgradients s_i:

    minimize    m
    subject to  m >= sum_i (t_i - Y_i)^2          (rotated cone)
                t_i >= t_j + < s_j, X_i - X_j >    for all i,j  (convexity)

The squared error is the rotated cone (1/2, m, t - Y). Hand-derived instance (d=1):
X = (0,1,2), Y = (0,1,0) -> t = (1/3,1/3,1/3), m = 2/3.
Port of examples/c/pwl_convex.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

NP = 3
XD = [0.0, 1.0, 2.0]
YD = [0.0, 1.0, 0.0]
M, T, S, AD, HALF = 0, 1, 1 + NP, 1 + 2 * NP, 1 + 3 * NP
NV = 1 + 3 * NP + 1


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(NP + NP * (NP - 1))              # auxDiff + off-diagonal convexity
        m.var_bounds(M, BK.LO, 0.0, math.inf)
        for i in range(NP):
            m.var_bounds(T + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(S + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(AD + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(HALF, BK.FX, 0.5, 0.5)
        row = 0
        for i in range(NP):                         # auxDiff_i = t_i - Y_i
            m.a_row(row, [AD + i, T + i], [1.0, -1.0])
            m.con_bounds(row, BK.FX, -YD[i], -YD[i])
            row += 1
        for i in range(NP):
            for j in range(NP):                     # t_j - t_i + s_j(X_i - X_j) <= 0
                if i == j:
                    continue                        # the i=j row is the trivial 0<=0
                m.a_row(row, [T + j, T + i, S + j], [1.0, -1.0, XD[i] - XD[j]])
                m.con_bounds(row, BK.UP, -math.inf, 0.0)
                row += 1
        m.cone(CT.RQUAD, [HALF, M, AD, AD + 1, AD + 2])   # 2*HALF*M >= sum AD_i^2
        m.cj(M, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    mv, tv = (r.x[M], [r.x[T + i] for i in range(NP)]) if ok else (0.0, [0.0] * NP)
    sse = sum((tv[i] - YD[i]) ** 2 for i in range(NP))
    okall = (ok and abs(mv - 2.0 / 3.0) < 1e-7 and all(abs(tv[i] - 1.0 / 3.0) < 1e-6 for i in range(NP))
             and abs(mv - sse) < 1e-7)
    print(f"pwl_convex  n={NP}")
    print(f"  t = ({tv[0]:.8f}, {tv[1]:.8f}, {tv[2]:.8f})   (expected 1/3,1/3,1/3)")
    print(f"  m = SSE = {mv:.8f}   (expected 2/3={2.0 / 3.0:.8f})   direct sse={sse:.8f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
