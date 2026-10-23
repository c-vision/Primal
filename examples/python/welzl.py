#!/usr/bin/env python3
"""Welzl's exact minimum enclosing circle (tschm/min_circle).

Welzl's algorithm is expected-linear-time and exact: it processes the points in
random order, keeping the (up to 3) points known to lie on the boundary:

    welzl(P, R):
      if P empty or |R| = 3:  return circle(R)
      p = first(P)
      D = welzl(P without p, R)
      if p inside D: return D
      return welzl(P without p, R + {p})

It is cross-checked against this library's SOCP (min r s.t. ||x-p_i|| <= r) on
several deterministic pseudo-random point sets: two independent methods must agree.
Port of examples/c/welzl.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

KP = 10
MASK64 = (1 << 64) - 1                              # the C generator uses a 64-bit unsigned long


def circ_of(r_pts):
    """Circle (center, radius) through 0, 1, 2 or 3 boundary points."""
    m = len(r_pts)
    if m == 0:
        return (0.0, 0.0), -1.0                     # empty: contains nothing
    if m == 1:
        return (r_pts[0][0], r_pts[0][1]), 0.0
    if m == 2:
        c = (0.5 * (r_pts[0][0] + r_pts[1][0]), 0.5 * (r_pts[0][1] + r_pts[1][1]))
        s = sum((r_pts[0][d] - r_pts[1][d]) ** 2 for d in range(2))
        return c, 0.5 * math.sqrt(s)
    (ax, ay), (bx, by), (cx, cy) = r_pts
    dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(dd) < 1e-14:
        return (0.0, 0.0), math.inf                 # collinear
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    c0 = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd
    c1 = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd
    return (c0, c1), math.sqrt((ax - c0) * (ax - c0) + (ay - c1) * (ay - c1))


def inside(circ, p):
    """True when point p lies inside the circle (within tolerance)."""
    (c0, c1), r = circ
    if r < 0:
        return False
    return math.sqrt((p[0] - c0) ** 2 + (p[1] - c1) ** 2) <= r + 1e-9


def welzl(pts, start, rbuf, m):
    """Welzl's recursive routine on pts[start:], with boundary rbuf[:m]."""
    if start == len(pts) or m == 3:
        return circ_of(rbuf[:m])
    d = welzl(pts, start + 1, rbuf, m)
    if inside(d, pts[start]):
        return d
    rbuf[m] = pts[start]
    return welzl(pts, start + 1, rbuf, m + 1)


def socp_circle(pts):
    """SOCP oracle (same formulation as min_circle): returns (ok, radius)."""
    k = len(pts)
    px, py, rr, d0 = 0, 1, 2, 3
    with Model() as m:
        m.add_vars(3 + 2 * k)
        m.add_cons(2 * k)
        m.var_bounds(px, BK.FR, -math.inf, math.inf)
        m.var_bounds(py, BK.FR, -math.inf, math.inf)
        m.var_bounds(rr, BK.LO, 0.0, math.inf)
        for i, (p0, p1) in enumerate(pts):
            m.var_bounds(d0 + 2 * i, BK.FR, -math.inf, math.inf)
            m.var_bounds(d0 + 2 * i + 1, BK.FR, -math.inf, math.inf)
            m.a_row(2 * i, [d0 + 2 * i, px], [1.0, -1.0])
            m.con_bounds(2 * i, BK.FX, -p0, -p0)
            m.a_row(2 * i + 1, [d0 + 2 * i + 1, py], [1.0, -1.0])
            m.con_bounds(2 * i + 1, BK.FX, -p1, -p1)
        for i in range(k):
            m.cone(CT.QUAD, [rr, d0 + 2 * i, d0 + 2 * i + 1])
        m.cj(rr, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    return (r.rc == 0), (r.x[rr] if r.rc == 0 else 0.0)


def main() -> int:
    all_ok, nsets = True, 20
    print("welzl")
    for s in range(nsets):
        seed = 1000 + 7 * s

        def rnd():
            """Deterministic pseudo-random draw in [0,1) from the per-set seed."""
            nonlocal seed
            seed = (seed * 1103515245 + 12345) & MASK64
            return ((seed >> 16) % 1000) / 1000.0

        pts = []
        for _ in range(KP):
            x = rnd()
            pts.append((x, rnd()))
        for i in range(KP - 1, 0, -1):              # Fisher-Yates shuffle, same PRNG
            j = min(int(rnd() * (i + 1)), i)
            pts[i], pts[j] = pts[j], pts[i]
        w = welzl(pts, 0, [None] * 3, 0)
        ok, rs = socp_circle(pts)
        good = ok and abs(w[1] - rs) < 1e-6
        if not good:
            print(f"  set {s:2d}  welzl={w[1]:.8f}  socp={rs:.8f}  FAIL")
            all_ok = False
        elif s == 0:
            print(f"  set  0  welzl={w[1]:.8f}  socp={rs:.8f}  OK")
    print(f"  {nsets} point sets, Welzl == SOCP: {'OK' if all_ok else 'FAIL'}")
    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
