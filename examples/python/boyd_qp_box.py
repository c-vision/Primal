#!/usr/bin/env python3
"""Boyd & Vandenberghe, Exercise 4.3 - box-constrained QP.

    min 0.5 x'Px + q'x + r  s.t. -1 <= x_i <= 1,  P=[[13,12,-2],[12,17,6],[-2,6,12]],
    q=(-22,-14.5,13), r=1.  Book x*=(1,1/2,-1), f=-21.625.
Port of c_examples/books/boyd_qp_box.c.
"""

from primalsolver import Model, BK, SOLSTA

P = [[13.0, 12.0, -2.0], [12.0, 17.0, 6.0], [-2.0, 6.0, 12.0]]
Q = [-22.0, -14.5, 13.0]


def main() -> int:
    with Model(maxcon=0, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.RA, -1.0, 1.0)
        m.cj(0, Q[0]); m.cj(1, Q[1]); m.cj(2, Q[2])
        m.qobj([(0, 0, P[0][0]), (0, 1, P[0][1]), (0, 2, P[0][2]),
                (1, 1, P[1][1]), (1, 2, P[1][2]), (2, 2, P[2][2])])
        r = m.solve()
    x = r.x
    f = r.objective + 1.0   # add r = 1
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(x[0] - 1) < 1e-5 and abs(x[1] - 0.5) < 1e-5
          and abs(x[2] + 1) < 1e-5 and abs(f + 21.625) < 1e-3)
    print(f"boyd_qp_box  x=({x[0]:.6g},{x[1]:.6g},{x[2]:.6g}) f={f:.6g} "
          f"(book (1,1/2,-1), -21.625) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
