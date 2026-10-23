#!/usr/bin/env python3
"""Stigler's nutrition model (the DIET LP), the classic MOSEK example
(AlexHahnPublic/PortfolioOptimizationMosek, Examples/diet.py; GAMS DIET / Dantzig 1963).

    minimize    c' x   subject to  N x >= b,   x >= 0

Its LP dual is the "shadow price" problem max b'y s.t. N'y <= c, y >= 0, and strong
duality gives the same optimum - an independent cross-check.

Hand case: N = [[2,1,0],[1,2,0],[0,0,1]], b = (4,5,1), c = (1,1,2): total cost 5 at
x = (1,2,1); the dual optimum is y = (1/3,1/3,2) with value 5.
Port of examples/c/diet.c.
"""

import math

from primalsolver import Model, BK, SENSE

NF = 3
NN = 3
N = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
BB = [4.0, 5.0, 1.0]
CC = [1.0, 1.0, 2.0]


def main() -> int:
    print("diet (Stigler, 3 foods x 3 nutrients)")
    all_ok = True

    xp, cost, ok = [0.0] * NF, 0.0, False          # primal: min c'x s.t. N x >= b, x >= 0
    with Model() as m:
        m.add_vars(NF)
        m.add_cons(NN)
        m.sense(SENSE.MIN)
        for j in range(NF):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        for i in range(NN):
            m.a_row(i, list(range(NF)), N[i])
            m.con_bounds(i, BK.LO, BB[i], math.inf)
        for j in range(NF):
            m.cj(j, CC[j])
        r = m.solve()
        if r.rc == 0:
            xp, cost, ok = r.x[:NF], r.objective, True
    mindef = 0.0                                    # nutrient deficit at xp
    for i in range(NN):
        s = sum(N[i][j] * xp[j] for j in range(NF))
        mindef = max(mindef, BB[i] - s)
    good = (ok and abs(cost - 5.0) < 1e-7 and abs(xp[0] - 1.0) < 1e-7 and abs(xp[1] - 2.0) < 1e-7
            and abs(xp[2] - 1.0) < 1e-7 and mindef < 1e-9)
    print(f"  primal: cost={cost:.6f} (expected 5)  x=({xp[0]:.4f},{xp[1]:.4f},{xp[2]:.4f})  "
          f"deficit={mindef:.2e}  {'OK' if good else 'FAIL'}")
    all_ok &= good

    yd, value, ok = [0.0] * NN, 0.0, False         # dual: max b'y s.t. N'y <= c, y >= 0
    with Model() as m:
        m.add_vars(NN)
        m.add_cons(NF)
        m.sense(SENSE.MAX)
        for i in range(NN):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        for j in range(NF):                         # N'y : row j = sum_i N[i][j] y_i
            m.a_row(j, list(range(NN)), [N[i][j] for i in range(NN)])
            m.con_bounds(j, BK.UP, -math.inf, CC[j])
        for i in range(NN):
            m.cj(i, BB[i])
        r = m.solve()
        if r.rc == 0:
            yd, value, ok = r.x[:NN], r.objective, True
    vmax = 0.0
    for j in range(NF):
        s = sum(N[i][j] * yd[i] for i in range(NN))
        vmax = max(vmax, s - CC[j])
    good = ok and abs(value - 5.0) < 1e-7 and vmax < 1e-9 and abs(value - cost) < 1e-7
    print(f"  dual:    value={value:.6f} (expected 5)  y=({yd[0]:.4f},{yd[1]:.4f},{yd[2]:.4f})  "
          f"viol={vmax:.2e}  strong duality {'OK' if good else 'FAIL'}")
    all_ok &= good

    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
