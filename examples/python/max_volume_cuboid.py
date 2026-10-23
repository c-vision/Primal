#!/usr/bin/env python3
"""The largest axis-aligned cuboid inscribed in a polyhedron {x : a_i'x <= b_i}.

Source: MOSEK Modeling Cookbook, "Geometric programming" / max-volume cuboid.

    max  prod_j (2 r_j)
    s.t. a_i' c + |a_i|' r <= b_i   (cuboid [c-r, c+r] inside the polyhedron)
         r >= 0

With n = 2 the product is a geometric mean, carried by the 3D power cone
(r_0, r_1, t) in POW^{1/2} <=> sqrt(r_0 r_1) >= t; the volume is (2 t)^2.
Hand-derived optima: unit square [0,1]^2 -> volume 1.00; right triangle
{x>=0, y>=0, x+y<=1} -> volume 0.25 (c = r = (1/4, 1/4)).

    python max_volume_cuboid.py [square|triangle]   (default square)

Port of examples/c/max_volume_cuboid.c.
"""

import math
import sys

from primalsolver import Model, BK, CT

SQ = [(-1, 0, 0), (0, -1, 0), (1, 0, 1), (0, 1, 1)]   # each face a0*x0 + a1*x1 <= b
TRI = [(-1, 0, 0), (0, -1, 0), (1, 1, 1)]


def solve(name, faces, want):
    """Solve the max-volume cuboid GP on one polyhedron and check its volume."""
    with Model() as m:
        m.add_vars(5)                               # 0=c0  1=c1  2=r0  3=r1  4=t
        m.add_cons(len(faces))
        m.var_bounds(0, BK.FR, -math.inf, math.inf)
        m.var_bounds(1, BK.FR, -math.inf, math.inf)
        m.var_bounds(2, BK.LO, 0.0, math.inf)
        m.var_bounds(3, BK.LO, 0.0, math.inf)
        m.var_bounds(4, BK.FR, -math.inf, math.inf)
        m.cj(4, -1.0)                               # maximize t
        for i, (a0, a1, b) in enumerate(faces):
            sub, val = [], []
            if a0 != 0.0:
                sub.append(0); val.append(a0)
            if a1 != 0.0:
                sub.append(1); val.append(a1)
            if a0 != 0.0:
                sub.append(2); val.append(abs(a0))
            if a1 != 0.0:
                sub.append(3); val.append(abs(a1))
            m.a_row(i, sub, val)
            m.con_bounds(i, BK.UP, -math.inf, b)
        m.cone(CT.PPOW, [2, 3, 4], 0.5)             # sqrt(r0*r1) >= |t|
        r = m.solve()
    if r.rc != 0:
        print(f"{name:<8} rc={r.rc} FAIL")
        return False
    x = r.x
    vol = 4.0 * x[4] * x[4]
    ok = abs(vol - want) < 1e-5
    print(f"{name:<8} c=({x[0]:.4f},{x[1]:.4f}) r=({x[2]:.4f},{x[3]:.4f}) volume={vol:.6f} "
          f"(expected {want:.4f}) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "square"
    if which.startswith("t"):
        ok = solve("triangle", TRI, 0.25)
    else:
        ok = solve("square", SQ, 1.00)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
