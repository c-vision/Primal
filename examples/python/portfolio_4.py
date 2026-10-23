#!/usr/bin/env python3
"""MOSEK port "portfolio_4_transcost" - portfolio optimization with transaction costs,
as a MIQP (MIP + quadratic objective): a cardinality-limited budget with binary
variables plus linear transaction costs.

    max  r'x - gamma*x'Sigma x - f' |x - x0|
    s.t. sum(x) = 1,  0 <= x_j <= y_j,  y binary,  sum(y) <= k

Data (MOSEK tutorial, n=3): r, GT, gamma = 0.05, f = 0.01, x0 uniform, k = 3 (no
effective limit with n=3, but it exercises the binaries). |x - x0| is linear with
auxiliary variables u_j (2 rows per j). Check: budget, binary logic, objective not
above portfolio_1. Port of examples/c/portfolio_4.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

N = 3
K = 3
R = [0.10717, 0.07502, 0.11902]
GT = [0.1667, 0.0, 0.0,
      0.0247, 0.1592, 0.0,
      0.0197, 0.0158, 0.1327]
GAMMA = 0.05
F = 0.01
X0 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def main() -> int:
    s = [0.0] * 9
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for q in range(3):
                acc += GT[i * 3 + q] * GT[j * 3 + q]
            s[i * 3 + j] = acc
    with Model() as m:
        m.add_vars(3 * N)                           # x (0..2), u (3..5), y (6..8)
        m.add_cons(2 + N + 2 * N)                   # budget, cardinality, x <= y, |x-x0| <= u
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, R[j])
            m.cj(N + j, -F)
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.var_bounds(N + j, BK.LO, 0.0, math.inf)
            m.var_bounds(2 * N + j, BK.RA, 0.0, 1.0)
            m.var_type(2 * N + j, VAR_TYPE.INT_BIN)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # budget: sum x = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [6, 7, 8], [1.0, 1.0, 1.0])      # cardinality: sum y <= k
        m.con_bounds(1, BK.UP, -math.inf, float(K))
        for j in range(N):                          # x_j - y_j <= 0
            m.a_row(2 + j, [j, 2 * N + j], [1.0, -1.0])
            m.con_bounds(2 + j, BK.UP, -math.inf, 0.0)
        for j in range(N):                          # |x_j - x0_j| <= u_j
            m.a_row(2 + N + j, [j, N + j], [1.0, -1.0])
            m.con_bounds(2 + N + j, BK.UP, -math.inf, X0[j])
            m.a_row(2 + 2 * N + j, [j, N + j], [-1.0, -1.0])
            m.con_bounds(2 + 2 * N + j, BK.UP, -math.inf, -X0[j])
        g = -2.0 * GAMMA                            # obj = r'x - f'u - gamma x'Sx
        m.qobj([(0, 0, g * s[0]), (0, 1, g * s[1]), (0, 2, g * s[2]),
                (1, 1, g * s[4]), (1, 2, g * s[5]), (2, 2, g * s[8])])
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})")
    print(f"u = ({x[3]:.4f}, {x[4]:.4f}, {x[5]:.4f})")
    print(f"y = ({x[6]:.0f}, {x[7]:.0f}, {x[8]:.0f})")
    print(f"obj = {obj:.6f}")
    ok = abs(x[0] + x[1] + x[2] - 1.0) < 1e-5
    nact = 0
    for j in range(N):
        if x[j] > 1e-6:
            nact += 1
        if abs(x[2 * N + j] - math.floor(x[2 * N + j] + 0.5)) > 1e-6 or x[j] > x[2 * N + j] + 1e-6:
            ok = False
    if nact > K or obj > 0.119066 + 1e-6:
        ok = False
    print(f"{'OK' if ok else 'FAIL'} (MIQP: budget, binaries, |x-x0|<=u, obj <= 0.119066)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
