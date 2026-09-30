#!/usr/bin/env python3
"""OMF Exercise 11.8 - Gomory mixed-integer cut.

    max 10 x1 + 13 x2  s.t. 10 x1 + 14 x2 <= 43, x integer >= 0.
LP relaxation 43 at x1=4.3; the GMI cut (k=3) is 3 x1 + 4 x2 <= 12; the
cut-augmented LP and the MIP both give 40 at (4,0).  Port of omf_ex118.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE, SOLSTA


def gmi_cut(a, b, k):
    kb = k * b
    f0 = kb - math.floor(kb)
    cut = []
    for aj in a:
        ka = k * aj
        fj = ka - math.floor(ka)
        t = fj - f0
        cut.append(math.floor(ka) + (t / (1.0 - f0) if t > 0.0 else 0.0))
    return cut, math.floor(kb)


def solve(with_cut, integer):
    with Model(maxcon=2 if with_cut else 1, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
            if integer:
                m.var_type(j, VAR_TYPE.INT)
        m.cj(0, 10.0); m.cj(1, 13.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1], [10.0, 14.0]); m.con_bounds(0, BK.UP, up=43.0)
        if with_cut:
            m.a_row(1, [0, 1], [3.0, 4.0]); m.con_bounds(1, BK.UP, up=12.0)
        r = m.solve()
    return r


def main() -> int:
    ok = True
    r = solve(False, False)
    relax_ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 43) < 1e-6
                and abs(r.x[0] - 4.3) < 1e-6 and abs(r.x[1]) < 1e-6)
    print(f"omf_ex118 (i)  LP relaxation: x1={r.x[0]:.4f} x2={r.x[1]:.4f} z={r.objective:.6g} "
          f"(book 4.3/0/43) {'OK' if relax_ok else 'FAIL'}")
    ok &= relax_ok

    cut, rhs = gmi_cut([1.0, 1.4, 0.1], 4.3, 3)
    cut_ok = (abs(cut[0] - 3.0) < 1e-9 and abs(cut[1] - 4.0) < 1e-9
              and abs(cut[2]) < 1e-9 and abs(rhs - 12.0) < 1e-9)
    print(f"omf_ex118 (ii) GMI k=3: {cut[0]:.4g} x1 + {cut[1]:.4g} x2 + {cut[2]:.4g} x3 "
          f"<= {rhs:.4g} {'OK' if cut_ok else 'FAIL'}")
    ok &= cut_ok

    lp = solve(True, False)
    mip = solve(True, True)
    cut_lp_ok = (lp.solsta == SOLSTA.OPTIMAL and abs(lp.objective - 40) < 1e-6
                 and abs(lp.x[0] - 4) < 1e-6 and abs(lp.x[1]) < 1e-6)
    mip_ok = (mip.solsta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL)
              and abs(mip.objective - 40) < 1e-6 and abs(mip.x[0] - 4) < 1e-6
              and abs(mip.x[1]) < 1e-6)
    print(f"omf_ex118 (iv) LP+cut: x=({lp.x[0]:.4g},{lp.x[1]:.4g}) z={lp.objective:.6g}   "
          f"MIP: x=({mip.x[0]:.4g},{mip.x[1]:.4g}) z={mip.objective:.6g}  (book 40 at (4,0)) "
          f"{'OK' if (cut_lp_ok and mip_ok) else 'FAIL'}")
    ok &= cut_lp_ok and mip_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
