#!/usr/bin/env python3
"""Minimum enclosing circle, SOCP vs exact (tschm/min_circle).

Model (identical to the MOSEK minimum-ellipsoid tutorial):
    minimize r  s.t.  || x - p_i ||_2 <= r  for every point p_i,
one quadratic cone [r, x - p_i] in Q^3 per point. The SOCP is cross-checked against
an EXACT oracle: the minimum enclosing circle is the circumcircle of two or three of
the points, so brute-forcing all pairs and triples gives the true optimum.
Cases: the repo's tests ([[0,0],[4,0]] -> r=2; [[0,0],[4,0],[2,4]] -> r=2.5) plus a
deterministic pseudo-random point set. Port of examples/c/min_circle.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

MASK64 = (1 << 64) - 1                              # the C generator uses a 64-bit unsigned long


def socp_circle(pts):
    """Minimum enclosing circle SOCP; returns (ok, radius)."""
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


def circum(sel):
    """Circumcircle (center, radius) of 2 (pair) or 3 (triple) points."""
    if len(sel) == 2:
        (ax, ay), (bx, by) = sel
        return (0.5 * (ax + bx), 0.5 * (ay + by)), 0.5 * math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    (ax, ay), (bx, by), (cx, cy) = sel
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return (0.0, 0.0), math.inf
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    c0 = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    c1 = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    dx, dy = ax - c0, ay - c1
    return (c0, c1), math.sqrt(dx * dx + dy * dy)


def contains_all(pts, c, r):
    """True when the circle (c, r) contains all the points."""
    return all(math.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) <= r + 1e-9 for p in pts)


def exact_circle(pts):
    """Exact minimum enclosing circle by brute force over pairs and triples."""
    k, best = len(pts), math.inf
    for i in range(k):
        for j in range(i + 1, k):
            c, r = circum([pts[i], pts[j]])
            if r < best and contains_all(pts, c, r):
                best = r
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                c, r = circum([pts[i], pts[j], pts[l]])
                if r < best and contains_all(pts, c, r):
                    best = r
    return best


def run(name, pts):
    """Run one point set through SOCP and exact oracle and compare."""
    ok, rs = socp_circle(pts)
    re = exact_circle(pts)
    good = ok and abs(rs - re) < 1e-6
    print(f"  {name:<14} k={len(pts)}  SOCP r={rs:.8f}  exact r={re:.8f}  {'OK' if good else 'FAIL'}")
    return good


def main() -> int:
    print("min_circle")
    all_ok = run("two points", [(0.0, 0.0), (4.0, 0.0)])
    all_ok &= run("three points", [(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)])
    s0, pr = 12345, []                              # deterministic pseudo-random 8 points (LCG)
    for _ in range(8):
        p = []
        for _ in range(2):
            s0 = (s0 * 1103515245 + 12345) & MASK64
            p.append(((s0 >> 16) % 1000) / 1000.0)
        pr.append(tuple(p))
    all_ok &= run("random (seed)", pr)
    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
