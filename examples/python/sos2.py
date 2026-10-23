#!/usr/bin/env python3
"""SOS type-2 constraint (port of the "sos2" example of the MOSEK C API docs).

At most two nonzero members ADJACENT in the weight order - the basis of
piecewise-linear approximation. Approximate f(t) on breakpoints t = (0,1,2) with
values f = (0,1,0.5) via SOS2 weights:
    t = sum_i w_i t_i,  y = sum_i w_i f_i,  sum w = 1,  w >= 0,  SOS2{w0,w1,w2}
With t fixed at 1.5 the SOS2 solution must interpolate the segment [1,2], where f
goes from 1 to 0.5: y = 1 - 0.5*(1.5-1) = 0.75. Port of examples/c/sos2.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model() as m:
        m.add_vars(4)                               # w0, w1, w2, y
        m.add_cons(3)
        for j in range(4):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_bounds(3, BK.FR, -math.inf, math.inf)                     # y free
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(0, BK.FX, 1.0, 1.0)   # sum w = 1
        m.a_row(1, [0, 1, 2], [0.0, 1.0, 2.0]); m.con_bounds(1, BK.FX, 1.5, 1.5)   # t = 1.5
        m.a_row(2, [0, 1, 2, 3], [0.0, 1.0, 0.5, -1.0])                 # y = w1 + 0.5 w2
        m.con_bounds(2, BK.FX, 0.0, 0.0)
        m.sos(2, [0, 1, 2], [0.0, 1.0, 2.0])         # SOS2 on the weights, breakpoint order
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x = r.x
    print(f"w = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), y = {x[3]:.4f} (expected 0.75)")
    ok = (abs(x[3] - 0.75) < 1e-6 and abs(x[0] + x[1] + x[2] - 1.0) < 1e-6
          and abs(x[1] + 2.0 * x[2] - 1.5) < 1e-6)
    if x[0] > 1e-6 and x[2] > 1e-6:                 # SOS2: w0 and w2 not both positive
        ok = False
    print(f"{'OK' if ok else 'FAIL'} (piecewise-linear SOS2: y(1.5)=0.75, adjacency respected)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
