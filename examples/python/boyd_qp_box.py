#!/usr/bin/env python3
"""Box-constrained QP, "Convex Optimization" (Boyd & Vandenberghe, 2004), Exercise 4.3.

    minimize (1/2) x'Px + q'x + r   subject to  -1 <= x_i <= 1

with P = [13 12 -2; 12 17 6; -2 6 12], q = (-22, -14.5, 13), r = 1. The book asks to
prove x* = (1, 1/2, -1) is optimal (objective -21.625).
Port of examples/c/boyd_qp_box.c.
"""

from primalsolver import Model, BK


def main() -> int:
    with Model(0, 3) as m:
        for j in range(3):
            m.var_bounds(j, BK.RA, -1.0, 1.0)
        m.cj(0, -22.0); m.cj(1, -14.5); m.cj(2, 13.0)
        # upper triangle of P with the full coefficient (mirrored internally)
        m.qobj([(0, 0, 13.0), (0, 1, 12.0), (0, 2, -2.0), (1, 1, 17.0), (1, 2, 6.0), (2, 2, 12.0)])
        r = m.solve()
    ok = r.rc == 0
    if ok:
        x, z = r.x, r.objective
        ok = abs(x[0] - 1.0) < 1e-5 and abs(x[1] - 0.5) < 1e-5 and abs(x[2] + 1.0) < 1e-5
        print("boyd_qp_box  x=(%.6g,%.6g,%.6g) f=%.6g (book (1,1/2,-1), f=-21.625) %s"
              % (x[0], x[1], x[2], z + 1.0, "OK" if ok else "FAIL"))
    else:
        print("boyd_qp_box  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
