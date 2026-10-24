#!/usr/bin/env python3
"""Exercise 4.10 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007).

$20,000 to invest in bonds B, shares S of XYZ and calls C (bought > 0, sold < 0,
|C| <= 50), with three equally likely scenarios for XYZ ($20, $40, $12):
    (i)   max 10 B + 4 S, budget 90 B + 20 S + 1000 C <= 20000 -> profit 14000
    (iii) max mean scenario profit with every P_i >= 2000      -> 2800 shares, sell 36 calls, 11200
    (iv)  max the riskless profit Z (P_i >= Z)                  -> Z = 7272.73, expected 9090.91
Port of examples/c/omf_ex410.c.
"""

import math

from primalsolver import Model, BK


def scenario_rows(m):
    """Budget and P1 = 10B + 20S + 1500C, P2 = 10B - 500C, P3 = 10B - 8S - 1000C."""
    m.a_row(0, [0, 1, 2], [90.0, 20.0, 1000.0])
    m.con_bounds(0, BK.UP, -math.inf, 20000.0)
    m.a_row(1, [0, 1, 2, 3], [10.0, 20.0, 1500.0, -1.0]); m.con_bounds(1, BK.FX, 0.0, 0.0)
    m.a_row(2, [0, 2, 4], [10.0, -500.0, -1.0]); m.con_bounds(2, BK.FX, 0.0, 0.0)
    m.a_row(3, [0, 1, 2, 5], [10.0, -8.0, -1000.0, -1.0]); m.con_bounds(3, BK.FX, 0.0, 0.0)


def budget(x):
    return 90.0 * x[0] + 20.0 * x[1] + 1000.0 * x[2]


def part_i():
    with Model(1, 3) as m:
        m.var_bounds(0, BK.LO, 0.0, math.inf)       # B
        m.var_bounds(1, BK.LO, 0.0, math.inf)       # S
        m.var_bounds(2, BK.RA, -50.0, 50.0)         # C
        m.cj(0, -10.0); m.cj(1, -4.0)               # max 10B + 4S
        m.a_row(0, [0, 1, 2], [90.0, 20.0, 1000.0])
        m.con_bounds(0, BK.UP, -math.inf, 20000.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        z, x = r.objective, r.x
        ok = (abs(z + 14000.0) < 1e-6 and abs(x[0] - 0.0) < 1e-6 and abs(x[1] - 3500.0) < 1e-6
              and abs(x[2] + 50.0) < 1e-6 and budget(x) <= 20000.0 + 1e-6)
        print("omf_ex410 (i)   B=%.4g S=%.4g C=%.4g  profit=%.6g (book 14000) %s" % (x[0], x[1], x[2], -z, "OK" if ok else "FAIL"))
    else:
        print("omf_ex410 (i)   rc=%d FAIL" % r.rc)
    return ok


def part_iii():
    with Model(4, 6) as m:                          # B S C P1 P2 P3 ; rows budget, P definitions
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.RA, -50.0, 50.0)
        for k in range(3, 6):
            m.var_bounds(k, BK.LO, 2000.0, math.inf)
        m.cj(3, -1.0 / 3.0); m.cj(4, -1.0 / 3.0); m.cj(5, -1.0 / 3.0)
        scenario_rows(m)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        z, x = r.objective, r.x
        ok = (abs(z + 11200.0) < 1e-6 and abs(x[1] - 2800.0) < 1e-6 and abs(x[2] + 36.0) < 1e-6
              and budget(x) <= 20000.0 + 1e-6 and all(x[k] >= 2000.0 - 1e-6 for k in range(3, 6)))
        print("omf_ex410 (iii) B=%.4g S=%.4g C=%.4g  P=(%.6g,%.6g,%.6g)  profit=%.6g (book 11200) %s"
              % (x[0], x[1], x[2], x[3], x[4], x[5], -z, "OK" if ok else "FAIL"))
    else:
        print("omf_ex410 (iii) rc=%d FAIL" % r.rc)
    return ok


def part_iv():
    with Model(7, 7) as m:                          # B S C P1 P2 P3 Z ; rows budget, P, P >= Z
        m.var_bounds(0, BK.LO, 0.0, math.inf)
        m.var_bounds(1, BK.LO, 0.0, math.inf)
        m.var_bounds(2, BK.RA, -50.0, 50.0)
        for k in range(3, 7):
            m.var_bounds(k, BK.FR, -math.inf, math.inf)
        m.cj(6, -1.0)                               # max Z
        scenario_rows(m)
        for k in range(3):                          # P_i - Z >= 0
            m.a_row(4 + k, [3 + k, 6], [1.0, -1.0])
            m.con_bounds(4 + k, BK.LO, 0.0, math.inf)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        z, x = r.objective, r.x
        exp_profit = (x[3] + x[4] + x[5]) / 3.0
        # the book rounds; the exact vertex is S=2272.73, C=-25.4545, Z=7272.73
        ok = (abs(z + 7272.727) < 1e-2 and abs(exp_profit - 9090.909) < 1e-2 and abs(budget(x) - 20000.0) < 1e-6
              and all(x[k] >= x[6] - 1e-6 for k in range(3, 6)))
        print("omf_ex410 (iv)  B=%.4g S=%.4g C=%.4g  riskless=%.6g (book 7272)  expected=%.6g (book 9091) %s"
              % (x[0], x[1], x[2], -z, exp_profit, "OK" if ok else "FAIL"))
    else:
        print("omf_ex410 (iv)  rc=%d FAIL" % r.rc)
    return ok


def main() -> int:
    a = part_i()
    b = part_iii()
    c = part_iv()
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
