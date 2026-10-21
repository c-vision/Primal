#!/usr/bin/env python3
"""OMF Exercise 4.10 (asset/liability). Three LPs over bonds B, shares S, calls C.

    (i)   max 10B + 4S                          -> book B=0, S=3500, C=-50, 14000
    (iii) max (P1+P2+P3)/3, Pi >= 2000          -> book 2800 shares, -36 calls, 11200
    (iv)  max Z, Pi >= Z (riskless)             -> book Z=7272.73, expected 9090.91
Port of c_examples/finance/omf_ex410.c.
"""

from primalsolver import Model, BK, SENSE, SOLSTA

BUDGET = [90.0, 20.0, 1000.0]


def part_i():
    with Model(maxcon=1, maxvar=3) as m:     # B, S, C
        m.var_bounds(0, BK.LO, 0.0)
        m.var_bounds(1, BK.LO, 0.0)
        m.var_bounds(2, BK.RA, -50.0, 50.0)
        m.cj(0, 10.0); m.cj(1, 4.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], BUDGET); m.con_bounds(0, BK.UP, up=20000.0)
        r = m.solve()
    x = r.x
    budget = sum(BUDGET[k] * x[k] for k in range(3))
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 14000) < 1e-6
          and abs(x[0]) < 1e-6 and abs(x[1] - 3500) < 1e-6 and abs(x[2] + 50) < 1e-6
          and budget <= 20000.0 + 1e-6)
    print(f"omf_ex410 (i)   B={x[0]:.4g} S={x[1]:.4g} C={x[2]:.4g} profit={r.objective:.6g} "
          f"(book 14000) {'OK' if ok else 'FAIL'}")
    return ok


def part_iii():
    with Model(maxcon=4, maxvar=6) as m:     # B,S,C,P1,P2,P3
        m.var_bounds(0, BK.LO, 0.0); m.var_bounds(1, BK.LO, 0.0); m.var_bounds(2, BK.RA, -50, 50)
        for k in (3, 4, 5):
            m.var_bounds(k, BK.LO, 2000.0)
        m.cj(3, 1.0 / 3.0); m.cj(4, 1.0 / 3.0); m.cj(5, 1.0 / 3.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], BUDGET); m.con_bounds(0, BK.UP, up=20000.0)
        m.a_row(1, [0, 1, 2, 3], [10.0, 20.0, 1500.0, -1.0]); m.con_bounds(1, BK.FX, 0, 0)
        m.a_row(2, [0, 2, 4], [10.0, -500.0, -1.0]); m.con_bounds(2, BK.FX, 0, 0)
        m.a_row(3, [0, 1, 2, 5], [10.0, -8.0, -1000.0, -1.0]); m.con_bounds(3, BK.FX, 0, 0)
        r = m.solve()
    x = r.x
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 11200) < 1e-6
          and abs(x[1] - 2800) < 1e-6 and abs(x[2] + 36) < 1e-6
          and all(x[k] >= 2000.0 - 1e-6 for k in (3, 4, 5)))
    print(f"omf_ex410 (iii) B={x[0]:.4g} S={x[1]:.4g} C={x[2]:.4g} "
          f"P=({x[3]:.6g},{x[4]:.6g},{x[5]:.6g}) profit={r.objective:.6g} "
          f"(book 11200) {'OK' if ok else 'FAIL'}")
    return ok


def part_iv():
    with Model(maxcon=7, maxvar=7) as m:     # B,S,C,P1,P2,P3,Z
        m.var_bounds(0, BK.LO, 0.0); m.var_bounds(1, BK.LO, 0.0); m.var_bounds(2, BK.RA, -50, 50)
        for k in range(3, 7):
            m.var_bounds(k, BK.FR)
        m.cj(6, 1.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1, 2], BUDGET); m.con_bounds(0, BK.UP, up=20000.0)
        m.a_row(1, [0, 1, 2, 3], [10.0, 20.0, 1500.0, -1.0]); m.con_bounds(1, BK.FX, 0, 0)
        m.a_row(2, [0, 2, 4], [10.0, -500.0, -1.0]); m.con_bounds(2, BK.FX, 0, 0)
        m.a_row(3, [0, 1, 2, 5], [10.0, -8.0, -1000.0, -1.0]); m.con_bounds(3, BK.FX, 0, 0)
        for k in range(3):
            m.a_row(4 + k, [3 + k, 6], [1.0, -1.0]); m.con_bounds(4 + k, BK.LO, 0.0)
        r = m.solve()
    x = r.x
    budget = sum(BUDGET[k] * x[k] for k in range(3))
    exp_profit = (x[3] + x[4] + x[5]) / 3.0
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 7272.727) < 1e-2
          and abs(exp_profit - 9090.909) < 1e-2 and abs(budget - 20000) < 1e-6
          and all(x[3 + k] >= x[6] - 1e-6 for k in range(3)))
    print(f"omf_ex410 (iv)  B={x[0]:.4g} S={x[1]:.4g} C={x[2]:.4g} riskless={r.objective:.6g} "
          f"(book 7272) expected={exp_profit:.6g} (book 9091) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = part_i() and part_iii() and part_iv()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
