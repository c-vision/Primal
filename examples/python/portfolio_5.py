#!/usr/bin/env python3
"""MOSEK port "portfolio_5_card" - portfolio optimization with a CARDINALITY constraint
(at most k assets with nonzero weight), a MIP with binary variables.

    max  r'x - gamma*x'Sigma x
    s.t. sum(x) = 1,  0 <= x_j <= y_j,  y_j binary,  sum(y) <= k

Data (MOSEK tutorial, n=3): r, GT, gamma = 0.05, k = 2. Check: cardinality respected,
budget, and the objective is not above the version without cardinality (0.119066).
Port of examples/c/portfolio_5.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

N = 3
K = 2
R = [0.10717, 0.07502, 0.11902]
GT = [0.1667, 0.0, 0.0,
      0.0247, 0.1592, 0.0,
      0.0197, 0.0158, 0.1327]
GAMMA = 0.05


def main() -> int:
    s = [0.0] * 9
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += GT[i * 3 + k] * GT[j * 3 + k]
            s[i * 3 + j] = acc
    with Model() as m:
        m.add_vars(2 * N)                           # x (3), y (3 binaries)
        m.add_cons(2 + N)                           # budget, cardinality, x_j <= y_j
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, R[j])
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            m.var_bounds(N + j, BK.RA, 0.0, 1.0)
            m.var_type(N + j, VAR_TYPE.INT_BIN)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # budget: sum x = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [3, 4, 5], [1.0, 1.0, 1.0])      # cardinality: sum y <= k
        m.con_bounds(1, BK.UP, -math.inf, float(K))
        for j in range(N):                          # x_j - y_j <= 0
            m.a_row(2 + j, [j, N + j], [1.0, -1.0])
            m.con_bounds(2 + j, BK.UP, -math.inf, 0.0)
        g = -2.0 * GAMMA                            # obj = r'x - gamma x'Sx (Q = -2*gamma*S)
        m.qobj([(0, 0, g * s[0]), (0, 1, g * s[1]), (0, 2, g * s[2]),
                (1, 1, g * s[4]), (1, 2, g * s[5]), (2, 2, g * s[8])])
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})")
    print(f"y = ({x[3]:.0f}, {x[4]:.0f}, {x[5]:.0f})")
    print(f"obj = {obj:.6f}")
    nact = 0                                        # active assets; 99 flags a broken link/binary
    for j in range(N):
        if x[j] > 1e-6:
            nact += 1
        if x[N + j] > 1e-6 and x[j] > x[N + j] + 1e-6:
            nact = 99
        if abs(x[N + j] - math.floor(x[N + j] + 0.5)) > 1e-6:
            nact = 99
    ok = abs(x[0] + x[1] + x[2] - 1.0) < 1e-5 and nact <= K and obj <= 0.119066 + 1e-6
    print(f"{'OK' if ok else 'FAIL'} (budget, cardinality <= {K}, obj <= 0.119066)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
