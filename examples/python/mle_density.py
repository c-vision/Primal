#!/usr/bin/env python3
"""MLE of a log-concave density via exponential cones (MOSEK/Tutorials,
mle-convex-density-function).

On a grid y_0 < .. < y_{n-1} with spacings dy_i, estimate x_i = g(y_i) of a concave
g = log f by maximizing the weighted log-likelihood:

    minimize    sum_i weight_i * u_i
    subject to  u_i >= -log(x_i)                         (PEXP cone per i)
                x concave: -dy_{i+1} x_i + (dy_i+dy_{i+1}) x_{i+1} - dy_i x_{i+2} <= 0
                sum_i ((x_i+x_{i+1})/2) dy_i = 1         (density integrates to 1)

The cone (x_i, 1, -u_i) in K_exp means x_i >= exp(-u_i). Hand-derived instance:
n=3, dy = (1/2,1/2,1/2), weights (1,2,3) -> x = (2/3, 2/3, 2) and
objective = -3 log(4/3) = -0.8630462174. Port of examples/c/mle_density.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

ND = 3
DY = [0.5, 0.5, 0.5]
W = [1.0, 2.0, 3.0]
X, U, NNEG, ONE, NV = 0, ND, 2 * ND, 3 * ND, 3 * ND + 1


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons((ND - 2) + 1 + ND)               # concavity + normalization + links
        for i in range(ND):
            m.var_bounds(X + i, BK.FR, -math.inf, math.inf)
        for i in range(ND):
            m.var_bounds(U + i, BK.FR, -math.inf, math.inf)
        for i in range(ND):
            m.var_bounds(NNEG + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        row = 0
        for i in range(ND - 2):                     # concavity of x
            m.a_row(row, [X + i, X + i + 1, X + i + 2], [-DY[i + 1], DY[i] + DY[i + 1], -DY[i]])
            m.con_bounds(row, BK.UP, -math.inf, 0.0)
            row += 1
        val = [0.0] * ND                            # normalization: sum ((x_i+x_{i+1})/2) dy_i = 1
        for i in range(ND - 1):
            val[i] += 0.5 * DY[i]
            val[i + 1] += 0.5 * DY[i]
        m.a_row(row, [X + i for i in range(ND)], val)
        m.con_bounds(row, BK.FX, 1.0, 1.0)
        row += 1
        for i in range(ND):                         # -u_i - N_i = 0
            m.a_row(row, [NNEG + i, U + i], [1.0, 1.0])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
        for i in range(ND):
            m.cone(CT.PEXP, [X + i, ONE, NNEG + i])
        for i in range(ND):
            m.cj(U + i, W[i])
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    obj, xv = 0.0, [0.0] * ND
    if ok:
        xv = [r.x[X + i] for i in range(ND)]
        for i in range(ND):
            obj += W[i] * r.x[U + i]
    want = -3.0 * math.log(4.0 / 3.0)
    okall = (ok and abs(xv[0] - 2.0 / 3.0) < 2e-3 and abs(xv[1] - 2.0 / 3.0) < 2e-3
             and abs(xv[2] - 2.0) < 2e-3 and abs(obj - want) < 1e-6)
    print(f"mle_density  n={ND}")
    print(f"  x = ({xv[0]:.8f}, {xv[1]:.8f}, {xv[2]:.8f})   (expected 2/3, 2/3, 2)")
    print(f"  objective = {obj:.10f}   (expected {want:.10f})")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
