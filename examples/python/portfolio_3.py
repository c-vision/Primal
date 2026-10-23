#!/usr/bin/env python3
"""MOSEK port "portfolio_3_impact" - portfolio optimization with linear transaction
costs, as a MIP: the cost f_j u_j is paid only if the asset is traded, which makes
u_j semi-continuous.

    max  r'x - gamma*x'Sx - f' u
    s.t. sum(x) = 1,  0 <= x <= 1
         u_j >= |x_j - x0_j|  (2 linear rows per j)
         u_j semi-continuous: u_j = 0 or 0.0001 <= u_j <= 1

Data: r, GT, gamma from the MOSEK tutorial, f = 0.01 uniform, x0 = (1/3, 1/3, 1/3).
Check: budget, semi-continuity, and the objective is not above the version without
costs (0.1191). Port of examples/c/portfolio_3.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

N = 3
R = [0.10717, 0.07502, 0.11902]
GT = [0.1667, 0.0, 0.0,
      0.0247, 0.1592, 0.0,
      0.0197, 0.0158, 0.1327]
GAMMA = 0.05
F = 0.01                                            # linear transaction cost
X0 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def main() -> int:
    s = [0.0] * 9
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += GT[i * 3 + k] * GT[j * 3 + k]
            s[i * 3 + j] = acc
    with Model() as m:
        m.add_vars(2 * N)                           # x (3), u (3)
        m.add_cons(1 + 2 * N)                       # budget, u_j >= +/-(x_j - x0_j)
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, R[j])
            m.cj(N + j, -F)
            m.var_bounds(j, BK.RA, 0.0, 1.0)
            m.var_bounds(N + j, BK.RA, 0.0001, 1.0)   # u_j semi-continuous: 0 or [0.0001, 1]
            m.var_type(N + j, VAR_TYPE.SEMI_CONT)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # budget: sum x = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for j in range(N):                          # |x_j - x0_j| <= u_j
            m.a_row(1 + j, [j, N + j], [1.0, -1.0])
            m.con_bounds(1 + j, BK.UP, -math.inf, X0[j])
            m.a_row(1 + N + j, [j, N + j], [-1.0, -1.0])
            m.con_bounds(1 + N + j, BK.UP, -math.inf, -X0[j])
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
    print(f"obj = {obj:.6f}")
    ok = abs(x[0] + x[1] + x[2] - 1.0) < 1e-5      # budget, |x-x0| <= u, semi-continuity
    for j in range(N):
        if abs(x[j] - X0[j]) > x[N + j] + 1e-6:
            ok = False
        if 1e-6 < x[N + j] < 0.0001 - 1e-9:
            ok = False
    print(f"{'OK' if ok else 'FAIL'} (budget+semi-continuity verified, obj <= 0.1191: "
          f"{'yes' if obj <= 0.1191 else 'no'})")
    if obj > 0.1191:                                # without costs: pobj ~ 0.119066
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
