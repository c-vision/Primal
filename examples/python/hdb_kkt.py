#!/usr/bin/env python3
"""Handbook (2009) Section 6.3 - Lagrangian-duality LP.

    min x1 + x2  s.t. x1 + 2 x2 = 2, x >= 0.
Book x*=(0,1), p*=1, equality multiplier |nu|=1/2.
Port of c_examples/books/hdb_kkt.c.
"""

from primalsolver import Model, BK, SOLSTA


def main() -> int:
    with Model(maxcon=1, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
        m.cj(0, 1.0); m.cj(1, 1.0)
        m.a_row(0, [0, 1], [1.0, 2.0]); m.con_bounds(0, BK.FX, 2.0, 2.0)
        r = m.solve()
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.x[0]) < 1e-6 and abs(r.x[1] - 1) < 1e-6
          and abs(r.objective - 1) < 1e-6 and abs(abs(r.y[0]) - 0.5) < 1e-6)
    print(f"hdb_kkt  x=({r.x[0]:.6g},{r.x[1]:.6g}) p*={r.objective:.6g} |y|={abs(r.y[0]):.6g} "
          f"(book (0,1), 1, 1/2) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
