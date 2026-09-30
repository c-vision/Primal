#!/usr/bin/env python3
"""PrimalSolver - two-period portfolio with turnover costs (LP).

Port of c_examples/finance/multi_period.c:

    max  sum_t mu_t'w_t - c*(|w_0-w_init|_1 + |w_1-w_0|_1),  e'w_t = 1, w_t >= 0.

Hand case: mu=(0.5,0.1), w_init=(0.5,0.5), c=0.1 -> the first move to (1,0)
costs 0.1 and gains 0.2, so it is taken; the second period pays nothing.
Value = (0.5+0.5) - 0.1 = 0.9.

    python python_examples/multi_period.py
"""

from primalsolver import Model, BK, SENSE, SOLSTA

MU = [0.5, 0.1]
W_INIT = [0.5, 0.5]
COST = 0.1


def main() -> int:
    with Model(maxcon=10, maxvar=8) as m:    # w0(0,1), w1(2,3), z0(4,5), z1(6,7)
        for j in range(8):
            m.var_bounds(j, BK.LO, 0.0)
        m.sense(SENSE.MAX)
        for j in range(2):
            m.cj(j, MU[j]); m.cj(2 + j, MU[j])
            m.cj(4 + j, -COST); m.cj(6 + j, -COST)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 1.0, 1.0)
        for j in range(2):
            m.a_row(2 + j, [j, 4 + j], [1.0, -1.0]); m.con_bounds(2 + j, BK.UP, up=W_INIT[j])
            m.a_row(4 + j, [j, 4 + j], [-1.0, -1.0]); m.con_bounds(4 + j, BK.UP, up=-W_INIT[j])
        for j in range(2):
            m.a_row(6 + j, [2 + j, j, 6 + j], [1.0, -1.0, -1.0]); m.con_bounds(6 + j, BK.UP, up=0.0)
            m.a_row(8 + j, [2 + j, j, 6 + j], [-1.0, 1.0, -1.0]); m.con_bounds(8 + j, BK.UP, up=0.0)
        r = m.solve()

    x = r.x
    turn0 = abs(x[0] - W_INIT[0]) + abs(x[1] - W_INIT[1])
    turn1 = abs(x[2] - x[0]) + abs(x[3] - x[1])
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 0.9) < 1e-7
          and abs(x[0] - 1.0) < 1e-6 and abs(x[1]) < 1e-6
          and abs(x[2] - 1.0) < 1e-6 and abs(x[3]) < 1e-6
          and abs(turn0 - 1.0) < 1e-6 and abs(turn1) < 1e-6)
    print(f"w0=({x[0]:.4f},{x[1]:.4f}) w1=({x[2]:.4f},{x[3]:.4f}) obj={r.objective:.6f} "
          f"(0.9) turnover=({turn0:.4f},{turn1:.4f}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
