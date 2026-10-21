#!/usr/bin/env python3
"""OMF Section 3.1.1 - short-term financing LP.

Port of c_examples/finance/omf_financing.c. Checks the book's shadow prices
u_Jan=-1.0373, u_Feb=-1.030, u_Mar=-1.020 (primal not unique).

    python python_examples/omf_financing.py
"""

from primalsolver import Model, BK, SENSE, SOLSTA

RHS = [150, 100, -200, 200, -50, -300]


def main() -> int:
    # vars: x1..x5 (0..4), y1..y3 (5..7), z1..z5 (8..12), v (13); rows 0..5
    with Model(maxcon=6, maxvar=14) as m:
        for i in range(5):
            m.var_bounds(i, BK.RA, 0.0, 100.0)
        for i in range(5, 13):
            m.var_bounds(i, BK.LO, 0.0)
        m.var_bounds(13, BK.FR)
        m.cj(13, 1.0)
        m.sense(SENSE.MAX)
        rows = [
            ([0, 5, 8], [1.0, 1.0, -1.0]),
            ([1, 6, 0, 8, 9], [1.0, 1.0, -1.01, 1.003, -1.0]),
            ([2, 7, 1, 9, 10], [1.0, 1.0, -1.01, 1.003, -1.0]),
            ([3, 5, 2, 10, 11], [1.0, -1.02, -1.01, 1.003, -1.0]),
            ([4, 6, 3, 11, 12], [1.0, -1.02, -1.01, 1.003, -1.0]),
            ([7, 4, 12, 13], [-1.02, -1.01, 1.003, -1.0]),
        ]
        for k, (idx, val) in enumerate(rows):
            m.a_row(k, idx, val)
            m.con_bounds(k, BK.FX, float(RHS[k]), float(RHS[k]))
        r = m.solve()

    u = [abs(r.y[k]) for k in range(3)]
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(u[0] - 1.0373) < 1e-3
          and abs(u[1] - 1.030) < 1e-3 and abs(u[2] - 1.020) < 1e-3)
    print(f"omf_financing  v={r.objective:.4f}  uJan={u[0]:.4f} uFeb={u[1]:.4f} "
          f"uMar={u[2]:.4f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
