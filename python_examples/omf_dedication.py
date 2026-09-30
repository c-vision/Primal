#!/usr/bin/env python3
"""OMF Section 3.2 / Exercise 3.11 - dedication (cash-flow matching) LP.

Port of c_examples/finance/omf_dedication.c. 8-year liabilities, 10 bonds, 0%
reinvestment. Checks the book cost 93944, holdings, shadow prices and reduced
costs (Bond 2 = 0.830612, Bond 7 = 8.786840, z0 = 0.028571, z8 = 0.524289).

    python python_examples/omf_dedication.py
"""

from primalsolver import Model, BK, SOLSTA

T, N = 8, 10
L = [0, 12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000]
P = [102, 99, 101, 98, 98, 104, 100, 101, 102, 94]
C = [5.0, 3.5, 5.0, 3.5, 4.0, 9.0, 6.0, 8.0, 9.0, 7.0]
MAT = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8]
XBOOK = [62, 0, 125, 152, 157, 123, 0, 124, 104, 93]


def main() -> int:
    with Model(maxcon=T, maxvar=N + T + 1) as m:   # x[0..9], z[0..8]
        for i in range(N):
            m.var_bounds(i, BK.LO, 0.0)
        for j in range(T + 1):
            m.var_bounds(N + j, BK.LO, 0.0)
        m.cj(N, 1.0)                                # z(0)
        for i in range(N):
            m.cj(i, float(P[i]))
        for yr in range(1, T + 1):
            idx, val = [], []
            for i in range(N):
                a = (C[i] if MAT[i] >= yr else 0.0) + (100.0 if MAT[i] == yr else 0.0)
                if a != 0.0:
                    idx.append(i); val.append(a)
            idx.append(N + (yr - 1)); val.append(1.0)   # + z(t-1)
            idx.append(N + yr); val.append(-1.0)         # - z(t)
            m.a_row(yr - 1, idx, val)
            m.con_bounds(yr - 1, BK.FX, float(L[yr]), float(L[yr]))
        r = m.solve()
        rc = m.getreducedcosts(0, N + T + 1)

    x = r.x
    hold_err = max(abs(x[i] - XBOOK[i]) for i in range(N))
    sp_book = [0.971428571, 0.915646259, 0.883045779]
    sp_ok = all(abs(abs(r.y[k]) - sp_book[k]) < 1e-5 for k in range(3))
    rc_book = [0.830612245, 8.786840002, 0.028571429, 0.524288903]
    rc_ok = (abs(rc[1] - rc_book[0]) < 1e-5 and abs(rc[6] - rc_book[1]) < 1e-5
             and abs(rc[N] - rc_book[2]) < 1e-5 and abs(rc[N + 8] - rc_book[3]) < 1e-5)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 93944.0) < 1.0
          and hold_err < 1.0 and sp_ok and rc_ok)
    print(f"omf_dedication cost={r.objective:.4f} (93944) max|x-xbook|={hold_err:.4f} "
          f"shadows={'OK' if sp_ok else 'FAIL'} reduced={'OK' if rc_ok else 'FAIL'} "
          f"(B2={rc[1]:.6f} B7={rc[6]:.6f} z0={rc[N]:.6f} z8={rc[N + 8]:.6f}) "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
