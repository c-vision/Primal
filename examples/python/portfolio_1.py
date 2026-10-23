#!/usr/bin/env python3
"""MOSEK port "portfolio_1_basic" - Markowitz portfolio optimization, basic version.

    max   r'x - gamma * x' Sigma x
    s.t.  sum(x) = 1,  x >= 0

Data (3 assets, from the MOSEK tutorial): r = (0.10717, 0.07502, 0.11902),
Sigma = GT GT' with the tutorial's lower-triangular factor, gamma = 0.05.
Check: pobj = dobj (QP optimality certificate) and feasibility.
Port of examples/c/portfolio_1.c.
"""

import math

from primalsolver import Model, BK, SENSE

N = 3
R = [0.10717, 0.07502, 0.11902]
GT = [0.1667, 0.0, 0.0,
      0.0247, 0.1592, 0.0,
      0.0197, 0.0158, 0.1327]
GAMMA = 0.05


def main() -> int:
    s = [0.0] * 9                                   # Sigma = GT * GT'
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += GT[i * 3 + k] * GT[j * 3 + k]
            s[i * 3 + j] = acc
    with Model() as m:
        m.add_vars(N)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, R[j])
        # obj = r'x - gamma*x'Sx = 0.5 x'Qx with Q = -2*gamma*S (upper triangle given)
        g = -2.0 * GAMMA
        m.qobj([(0, 0, g * s[0]), (0, 1, g * s[1]), (0, 2, g * s[2]),
                (1, 1, g * s[4]), (1, 2, g * s[5]), (2, 2, g * s[8])])
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # sum(x) = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for j in range(N):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, po, dobj = r.x, r.objective, r.dual
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f})")
    print(f"pobj = {po:.6f}, dobj = {dobj:.6f}")
    ok = abs(x[0] + x[1] + x[2] - 1.0) < 1e-6 and all(x[j] >= -1e-9 for j in range(N))
    q = 0.0
    for i in range(3):
        for j in range(3):
            q += x[i] * s[i * 3 + j] * x[j]
    lin = R[0] * x[0] + R[1] * x[1] + R[2] * x[2]
    expect = lin - GAMMA * q
    print(f"obj check: {po:.6f} (expected {expect:.6f})")
    if abs(po - expect) > 1e-6 or abs(po - dobj) > 1e-5 * (1 + abs(po)):
        ok = False
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
