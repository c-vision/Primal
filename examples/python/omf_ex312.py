#!/usr/bin/env python3
"""Exercise 3.12 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007): dedication.

A 9-year liability stream (24, 26, 28, 28, 26, 29, 32, 33, 34 million $) is funded by a
least-cost portfolio of 16 bonds, cash carried at r = 2%:

    min z(0) + sum_i P(i) x(i)
    s.t.  sum_{M(i) >= t} C(i) x(i) + sum_{M(i) = t} 100 x(i) + (1+r) z(t-1) - z(t) = L(t)

The book gives no optimum, so the sample recomputes the cash flows and checks
feasibility and the objective. Port of examples/c/omf_ex312.c.
"""

import math

from primalsolver import Model, BK

T = 9
N = 16
R = 0.02
L = [0.0, 24, 26, 28, 28, 26, 29, 32, 33, 34]
P = [102.44, 99.95, 100.02, 102.66, 87.90, 85.43, 83.42, 103.82, 110.29, 108.85,
     109.95, 107.36, 104.62, 99.07, 103.78, 64.66]
C = [5.625, 4.75, 4.25, 5.25, 0.0, 0.0, 0.0, 5.75, 6.875, 6.5, 6.625, 6.125, 5.625, 4.75, 5.5, 0.0]
M = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]


def main() -> int:
    with Model(T, N + T + 1) as m:                  # vars x(i), z(0..9); rows year 1..9
        for i in range(N):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        for j in range(T + 1):
            m.var_bounds(N + j, BK.LO, 0.0, math.inf)
        m.cj(N, 1.0)
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
            idx.append(N + (yr - 1)); val.append(1.0 + R)   # +(1+r) z(t-1)
            idx.append(N + yr); val.append(-1.0)            # - z(t)
            m.a_row(yr - 1, idx, val)
            m.con_bounds(yr - 1, BK.FX, L[yr], L[yr])
        r = m.solve()
    ok = r.rc == 0
    if ok:
        z, x = r.objective, r.x
        # independent feasibility recomputation of the cash-flow recurrence
        cost, minz, maxviol, carry = x[N], 0.0, 0.0, 0.0
        for i in range(N):
            cost += P[i] * x[i]
        for yr in range(1, T + 1):
            inflow = 0.0
            for i in range(N):
                if M[i] >= yr:
                    inflow += C[i] * x[i]
                if M[i] == yr:
                    inflow += 100.0 * x[i]
            carry = (1.0 + R) * carry + inflow - L[yr]
            if carry < minz:
                minz = carry
            zt = x[N + yr]
            if abs(zt - carry) > 1e-6:
                d = abs(zt - carry)
                if d > maxviol:
                    maxviol = d
        ok = abs(cost - z) < 1e-6 and minz > -1e-6 and maxviol < 1e-6
        print("omf_ex312  cost=%.4f  min surplus=%.3e  max |z-recurrence|=%.2e  %s"
              % (z, minz, maxviol, "OK" if ok else "FAIL"))
        print("  x = " + "".join("%.4g " % x[i] for i in range(N)))
    else:
        print("omf_ex312  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
