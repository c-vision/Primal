#!/usr/bin/env python3
"""MOSEK port "portfolio_2_frontier" - mean-variance frontier.

For a grid of gamma values it solves the Markowitz QP and prints the pair
(risk, expected return): risk must decrease as gamma grows.

    max   r'x - gamma * x' Sigma x
    s.t.  sum(x) = 1,  x >= 0

Port of examples/c/portfolio_2.c.
"""

import math

from primalsolver import Model, BK, SENSE

R = [0.10717, 0.07502, 0.11902]
GT = [0.1667, 0.0, 0.0,
      0.0247, 0.1592, 0.0,
      0.0197, 0.0158, 0.1327]


def sigma():
    s = [0.0] * 9
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += GT[i * 3 + k] * GT[j * 3 + k]
            s[i * 3 + j] = acc
    return s


def solve_gamma(gamma, s):
    """Solve the Markowitz QP for one gamma; returns (x, risk, return) or None."""
    n = 3
    with Model() as m:
        m.add_vars(n)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        for j in range(n):
            m.cj(j, R[j])
        m.qobj([(0, 0, -2.0 * gamma * s[0]), (0, 1, -2.0 * gamma * s[1]), (0, 2, -2.0 * gamma * s[2]),
                (1, 1, -2.0 * gamma * s[4]), (1, 2, -2.0 * gamma * s[5]), (2, 2, -2.0 * gamma * s[8])])
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for j in range(n):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        return None
    x = r.x[:n]
    risk = 0.0
    for i in range(n):
        for j in range(n):
            risk += x[i] * s[i * 3 + j] * x[j]
    ret = R[0] * x[0] + R[1] * x[1] + R[2] * x[2]
    return x, risk, ret


def main() -> int:
    s = sigma()
    gammas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    print(f"{'gamma':>8} {'risk':>12} {'return':>12}   x")
    prev_risk = math.inf
    ok = True
    for g in gammas:
        sol = solve_gamma(g, s)
        if sol is None:
            ok = False
            break
        x, risk, ret = sol
        print(f"{g:8.2f} {risk:12.6f} {ret:12.6f}   ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})")
        if abs(x[0] + x[1] + x[2] - 1.0) > 1e-6 or any(v < -1e-9 for v in x):   # feasibility
            ok = False
        # monotonicity: risk does not grow as gamma grows (flat up to solver
        # precision while the optimum is the same vertex)
        if risk > prev_risk + 1e-6:
            ok = False
        prev_risk = risk
    print(f"{'OK' if ok else 'FAIL'} (monotone risk frontier)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
