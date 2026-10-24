#!/usr/bin/env python3
"""Two-period index tracking with turnover.

    min sum_t |w_t - b_t|_1 + c (|w_0 - w_init|_1 + |w_1 - w_0|_1)
    s.t.  e'w_t = 1,  w_t >= 0

with the absolute values linearised by z_t >= +-(w_t - b_t), v_0 >= +-(w_0 - w_init),
v_1 >= +-(w_1 - w_0). Hand case: b_0 = (0.6, 0.4), b_1 = (0.3, 0.7),
w_init = (0.5, 0.5), c = 0.1: w_t = b_t, value 0.08. Port of examples/c/index_tracking.c.
"""

import math

from primalsolver import Model, BK, SENSE


def main() -> int:
    b0, b1, winit = [0.6, 0.4], [0.3, 0.7], [0.5, 0.5]
    c = 0.1
    with Model() as m:                          # w0(0,1), w1(2,3), z0(4,5), z1(6,7), v0(8,9), v1(10,11)
        m.add_vars(12)
        m.add_cons(18)
        for j in range(12):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.sense(SENSE.MIN)
        for j in range(2):
            m.cj(4 + j, 1.0)                    # z0
            m.cj(6 + j, 1.0)                    # z1
            m.cj(8 + j, c)                      # v0
            m.cj(10 + j, c)                     # v1
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)   # e'w_0 = 1
        m.a_row(1, [2, 3], [1.0, 1.0]); m.con_bounds(1, BK.FX, 1.0, 1.0)   # e'w_1 = 1
        r = 2
        for j in range(2):                      # z0 >= +-(w0 - b0)
            m.a_row(r, [j, 4 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, b0[j]); r += 1
            m.a_row(r, [j, 4 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, -b0[j]); r += 1
        for j in range(2):                      # z1 >= +-(w1 - b1)
            m.a_row(r, [2 + j, 6 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, b1[j]); r += 1
            m.a_row(r, [2 + j, 6 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, -b1[j]); r += 1
        for j in range(2):                      # v0 >= +-(w0 - w_init)
            m.a_row(r, [j, 8 + j], [1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, winit[j]); r += 1
            m.a_row(r, [j, 8 + j], [-1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, -winit[j]); r += 1
        for j in range(2):                      # v1 >= +-(w1 - w0)
            m.a_row(r, [2 + j, j, 10 + j], [1.0, -1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, 0.0); r += 1
            m.a_row(r, [2 + j, j, 10 + j], [-1.0, 1.0, -1.0]); m.con_bounds(r, BK.UP, -math.inf, 0.0); r += 1
        res = m.solve()
    if res.rc != 0:
        print("optimize rc=%d" % res.rc)
        return 1
    obj, x = res.objective, res.x
    track0 = abs(x[0] - b0[0]) + abs(x[1] - b0[1])
    track1 = abs(x[2] - b1[0]) + abs(x[3] - b1[1])
    ok = (abs(obj - 0.08) < 1e-7 and abs(x[0] - 0.6) < 1e-6 and abs(x[1] - 0.4) < 1e-6
          and abs(x[2] - 0.3) < 1e-6 and abs(x[3] - 0.7) < 1e-6 and abs(track0) < 1e-6 and abs(track1) < 1e-6)
    print("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (expected 0.08)" % (x[0], x[1], x[2], x[3], obj))
    print("tracking = %.4f then %.4f (expected 0.0, 0.0)" % (track0, track1))
    print("%s (index tracking over two periods with turnover cost)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
