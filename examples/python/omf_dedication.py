#!/usr/bin/env python3
"""Dedication (cash-flow matching) bond portfolio of "Optimization Methods in Finance"
(Cornuejols & Tutuncu, 2007), Section 3.2 and Exercise 3.11: fund an 8-year liability
stream with a least-cost portfolio of 10 bonds (0% reinvestment).

    min z(0) + sum_i P(i) x(i)
    s.t.  sum_{M(i) >= t} C(i) x(i) + sum_{M(i) = t} 100 x(i) + z(t-1) - z(t) = L(t)

The book: cost $93,944, its holdings, shadow prices 0.971428571, 0.915646259,
0.883045779 and reduced costs (Bond 2, Bond 7, z0, z8).
Port of examples/c/omf_dedication.c.
"""

import math

from primalsolver import Model, BK

T = 8
N = 10
L = [0.0, 12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000]
P = [102.0, 99, 101, 98, 98, 104, 100, 101, 102, 94]
C = [5.0, 3.5, 5.0, 3.5, 4.0, 9.0, 6.0, 8.0, 9.0, 7.0]
M = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8]
XBOOK = [62.0, 0, 125, 152, 157, 123, 0, 124, 104, 93]   # book optimum holdings


def main() -> int:
    with Model(T, N + T + 1) as m:                  # vars x(i), z(0..8); rows year 1..8
        for i in range(N):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        for j in range(T + 1):
            m.var_bounds(N + j, BK.LO, 0.0, math.inf)
        m.cj(N, 1.0)                                # z(0) in the objective
        for i in range(N):
            m.cj(i, P[i])
        for yr in range(1, T + 1):
            idx, val = [], []
            for i in range(N):
                a = 0.0
                if M[i] >= yr:
                    a += C[i]
                if M[i] == yr:
                    a += 100.0
                if a != 0.0:
                    idx.append(i); val.append(a)
            idx.append(N + (yr - 1)); val.append(1.0)   # + z(t-1)
            idx.append(N + yr); val.append(-1.0)        # - z(t)
            m.a_row(yr - 1, idx, val)
            m.con_bounds(yr - 1, BK.FX, L[yr], L[yr])
        r = m.solve()
        ok = r.rc == 0
        if ok:
            z, x, sp = r.objective, r.x, r.y
            cost_book, hold_err = 0.0, 0.0
            for i in range(N):
                cost_book += P[i] * XBOOK[i]
                d = abs(x[i] - XBOOK[i])
                if d > hold_err:
                    hold_err = d
            # shadow prices: the book prints them positive (same magnitude)
            sp_book = [0.971428571, 0.915646259, 0.883045779]
            sp_ok = all(abs(abs(sp[k]) - sp_book[k]) < 1e-5 for k in range(3))
            # reduced costs from the Section 3.3.2 sensitivity table
            rc = m.getreducedcosts(0, N + T + 1)
            rc_book = [0.830612245, 8.786840002, 0.028571429, 0.524288903]
            rc_ok = (abs(rc[1] - rc_book[0]) < 1e-5 and abs(rc[6] - rc_book[1]) < 1e-5
                     and abs(rc[N] - rc_book[2]) < 1e-5 and abs(rc[N + 8] - rc_book[3]) < 1e-5)
            ok = abs(z - 93944.0) < 1.0 and hold_err < 1.0 and sp_ok and rc_ok
            print("omf_dedication  cost=%.4f (book 93944)  max|x-xbook|=%.4f  %s" % (z, hold_err, "OK" if ok else "FAIL"))
            print("  shadow:  sp1=%.6f sp2=%.6f sp3=%.6f  %s" % (abs(sp[0]), abs(sp[1]), abs(sp[2]), "OK" if sp_ok else "FAIL"))
            print("  reduced: B2=%.6f B7=%.6f z0=%.6f z8=%.6f  %s" % (rc[1], rc[6], rc[N], rc[N + 8], "OK" if rc_ok else "FAIL"))
            print("  holdings: " + "".join("B%d=%.4g " % (i + 1, x[i]) for i in range(N)))
            print("  (bond-only cost = %.2f, z0 = %.4f, book bond-only = %.2f)" % (z - x[N], x[N], cost_book))
        else:
            print("omf_dedication  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
