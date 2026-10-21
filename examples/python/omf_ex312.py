#!/usr/bin/env python3
"""OMF Exercise 3.12 - dedication / cash-flow matching (LP).

Port of c_examples/finance/omf_ex312.c. 9-year liability stream, 16 bonds,
2% reinvestment. The book gives no optimum, so feasibility is recomputed.

    python python_examples/omf_ex312.py
"""

from primalsolver import Model, BK, SOLSTA

T, N, RB = 9, 16, 0.02
L = [0, 24, 26, 28, 28, 26, 29, 32, 33, 34]
P = [102.44, 99.95, 100.02, 102.66, 87.90, 85.43, 83.42, 103.82, 110.29,
     108.85, 109.95, 107.36, 104.62, 99.07, 103.78, 64.66]
C = [5.625, 4.75, 4.25, 5.25, 0.0, 0.0, 0.0, 5.75, 6.875, 6.5, 6.625,
     6.125, 5.625, 4.75, 5.5, 0.0]
MAT = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]


def main() -> int:
    with Model(maxcon=T, maxvar=N + T + 1) as m:   # x[0..15], z[0..9]
        for i in range(N):
            m.var_bounds(i, BK.LO, 0.0)
        for j in range(T + 1):
            m.var_bounds(N + j, BK.LO, 0.0)
        m.cj(N, 1.0)                                # z(0)
        for i in range(N):
            m.cj(i, P[i])
        for yr in range(1, T + 1):
            idx, val = [], []
            for i in range(N):
                a = (C[i] if MAT[i] >= yr else 0.0) + (100.0 if MAT[i] == yr else 0.0)
                if a != 0.0:
                    idx.append(i); val.append(a)
            idx.append(N + (yr - 1)); val.append(1.0 + RB)   # +(1+r) z(t-1)
            idx.append(N + yr); val.append(-1.0)             # - z(t)
            m.a_row(yr - 1, idx, val)
            m.con_bounds(yr - 1, BK.FX, float(L[yr]), float(L[yr]))
        r = m.solve()

    x = r.x
    cost = x[N] + sum(P[i] * x[i] for i in range(N))
    carry = 0.0
    minz = 0.0
    maxviol = 0.0
    for yr in range(1, T + 1):
        inflow = sum((C[i] if MAT[i] >= yr else 0.0) * x[i] for i in range(N))
        inflow += sum(100.0 * x[i] for i in range(N) if MAT[i] == yr)
        carry = (1.0 + RB) * carry + inflow - L[yr]
        minz = min(minz, carry)
        maxviol = max(maxviol, abs(x[N + yr] - carry))
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(cost - r.objective) < 1e-6
          and minz > -1e-6 and maxviol < 1e-6)
    print(f"omf_ex312  cost={r.objective:.4f}  min surplus={minz:.3e}  "
          f"max |z-recurrence|={maxviol:.2e}  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
