#!/usr/bin/env python3
"""PrimalSolver - two-period index tracking with turnover (LP).

Port of c_examples/finance/index_tracking.c:

    min sum_t |w_t - b_t|_1 + c*(|w_0 - w_init|_1 + |w_1 - w_0|_1),
    e'w_t = 1, w_t >= 0, all abs values linearised by z/v.

Hand case: b0=(0.6,0.4), b1=(0.3,0.7), w_init=(0.5,0.5), c=0.1 -> value 0.08,
w0=(0.6,0.4), w1=(0.3,0.7).

    python python_examples/index_tracking.py
"""

from primalsolver import Model, BK, SOLSTA

B0 = [0.6, 0.4]
B1 = [0.3, 0.7]
W_INIT = [0.5, 0.5]
C = 0.1


def main() -> int:
    # vars: w0(0,1), w1(2,3), z0(4,5), z1(6,7), v0(8,9), v1(10,11)
    with Model(maxcon=18, maxvar=12) as m:
        for j in range(12):
            m.var_bounds(j, BK.LO, 0.0)
        for j in range(2):
            m.cj(4 + j, 1.0); m.cj(6 + j, 1.0)
            m.cj(8 + j, C); m.cj(10 + j, C)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 1.0, 1.0)
        r = 2
        for j in range(2):                      # z0 >= +/- (w0 - b0)
            m.a_row(r, [j, 4 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, up=B0[j]); r += 1
            m.a_row(r, [j, 4 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, up=-B0[j]); r += 1
        for j in range(2):                      # z1 >= +/- (w1 - b1)
            m.a_row(r, [2 + j, 6 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, up=B1[j]); r += 1
            m.a_row(r, [2 + j, 6 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, up=-B1[j]); r += 1
        for j in range(2):                      # v0 >= +/- (w0 - w_init)
            m.a_row(r, [j, 8 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, up=W_INIT[j]); r += 1
            m.a_row(r, [j, 8 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, up=-W_INIT[j]); r += 1
        for j in range(2):                      # v1 >= +/- (w1 - w0)
            m.a_row(r, [2 + j, j, 10 + j], [1.0, -1.0, -1.0]); m.con_bounds(r, BK.UP, up=0.0); r += 1
            m.a_row(r, [2 + j, j, 10 + j], [-1.0, 1.0, -1.0]); m.con_bounds(r, BK.UP, up=0.0); r += 1
        sol = m.solve()

    x = sol.x
    track0 = abs(x[0] - B0[0]) + abs(x[1] - B0[1])
    track1 = abs(x[2] - B1[0]) + abs(x[3] - B1[1])
    ok = (sol.solsta == SOLSTA.OPTIMAL and abs(sol.objective - 0.08) < 1e-7
          and abs(x[0] - 0.6) < 1e-6 and abs(x[1] - 0.4) < 1e-6
          and abs(x[2] - 0.3) < 1e-6 and abs(x[3] - 0.7) < 1e-6
          and abs(track0) < 1e-6 and abs(track1) < 1e-6)
    print(f"w0=({x[0]:.4f},{x[1]:.4f}) w1=({x[2]:.4f},{x[3]:.4f}) obj={sol.objective:.6f} "
          f"(0.08) track=({track0:.4f},{track1:.4f}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
