#!/usr/bin/env python3
"""Smallest ball enclosing a finite point set (MOSEK/Tutorials,
minimum-ellipsoid/minimum-ellipsoid.py, the primal model of the notebook).

    minimize    r0
    subject to  r0 >= || p0 - p_i ||_2 ,   i = 1..k
                r0 >= 0,  p0 free

i.e. one quadratic cone [r0, p0 - p_i] in Q^(n+1) per point. Hand-derived instances:
2 points at distance d -> r* = d/2; equilateral triangle, side a -> r* = a/sqrt(3).
Port of examples/c/min_enclosing_ball.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

PX, PY, R, D = 0, 1, 2, 3


def min_ball(points):
    """Smallest enclosing ball of the points; returns (ok, radius, center)."""
    k = len(points)
    with Model() as m:
        m.add_vars(3 + 2 * k)                       # px, py, r, then (dx_i, dy_i) per point
        m.add_cons(2 * k)
        m.var_bounds(PX, BK.FR, -math.inf, math.inf)
        m.var_bounds(PY, BK.FR, -math.inf, math.inf)
        m.var_bounds(R, BK.LO, 0.0, math.inf)
        for i, (p0, p1) in enumerate(points):
            m.var_bounds(D + 2 * i, BK.FR, -math.inf, math.inf)
            m.var_bounds(D + 2 * i + 1, BK.FR, -math.inf, math.inf)
            # dx_i - px = -P[i][0] ; dy_i - py = -P[i][1]
            m.a_row(2 * i, [D + 2 * i, PX], [1.0, -1.0])
            m.con_bounds(2 * i, BK.FX, -p0, -p0)
            m.a_row(2 * i + 1, [D + 2 * i + 1, PY], [1.0, -1.0])
            m.con_bounds(2 * i + 1, BK.FX, -p1, -p1)
        for i in range(k):                          # r >= || p0 - p_i ||
            m.cone(CT.QUAD, [R, D + 2 * i, D + 2 * i + 1])
        m.cj(R, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, (0.0, 0.0)
    return True, r.x[R], (r.x[PX], r.x[PY])


def main() -> int:
    ok2, r2, _ = min_ball([(0.0, 0.0), (2.0, 0.0)])
    ok3, r3, _ = min_ball([(0.0, 0.0), (2.0, 0.0), (1.0, 1.7320508075688772)])   # side 2
    want2 = 1.0                                     # d/2, d = 2
    want3 = 2.0 / math.sqrt(3.0)                    # a/sqrt(3), a = 2
    ok = ok2 and ok3 and abs(r2 - want2) < 1e-7 and abs(r3 - want3) < 1e-7
    print("min_enclosing_ball")
    print(f"  2 points (d=2):    r*={r2:.10f}   expected d/2={want2:.10f}")
    print(f"  triangle (a=2):    r*={r3:.10f}   expected a/sqrt(3)={want3:.10f}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
