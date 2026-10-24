#!/usr/bin/env python3
"""Exercise 11.8 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007): Gomory mixed integer cuts.

    max 10 x1 + 13 x2  s.t.  10 x1 + 14 x2 <= 43,  x >= 0 integer

(i) The LP relaxation's tableau row is x1 + 1.4 x2 + 0.1 x3 = 4.3 (z = 43).
(ii)/(iii) The GMI cut of that row times k = 3 is 3 x1 + 4 x2 <= 12.
(iv) With the cut the LP optimum is 40 at (4, 0), the integer optimum.
Port of examples/c/omf_ex118.c.
"""

import math

from primalsolver import Model, BK, VAR_TYPE


def gmi_cut(a, b, k):
    """GMI cut for the row sum_j a_j x_j = b multiplied by k: (cut_a, cut_rhs)."""
    kb = float(k) * b
    f0 = kb - math.floor(kb)
    cut = []
    for aj in a:
        ka = float(k) * aj
        fj = ka - math.floor(ka)
        t = fj - f0
        cut.append(math.floor(ka) + (t / (1.0 - f0) if t > 0.0 else 0.0))
    return cut, math.floor(kb)


def solve_ip(with_cut, int_flag):
    """LP relaxation of the IP (optionally with the GMI cut); int_flag = MIP."""
    with Model(2 if with_cut else 1, 2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
            if int_flag:
                m.var_type(j, VAR_TYPE.INT)
        m.cj(0, -10.0); m.cj(1, -13.0)              # max
        m.a_row(0, [0, 1], [10.0, 14.0])
        m.con_bounds(0, BK.UP, -math.inf, 43.0)
        if with_cut:
            m.a_row(1, [0, 1], [3.0, 4.0])
            m.con_bounds(1, BK.UP, -math.inf, 12.0)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, 0.0, 0.0
    return True, r.objective, r.x[0], r.x[1]


def main() -> int:
    ok = True
    good, z, x1, x2 = solve_ip(False, False)        # (i) LP relaxation
    if not good:
        print("omf_ex118 LP FAIL")
        return 1
    relax_ok = abs(-z - 43.0) < 1e-6 and abs(x1 - 4.3) < 1e-6 and abs(x2) < 1e-6
    print("omf_ex118 (i)  LP relaxation: x1=%.4f x2=%.4f z=%.6g (book 4.3/0/43) %s"
          % (x1, x2, -z, "OK" if relax_ok else "FAIL"))
    ok &= relax_ok

    cut, rhs = gmi_cut([1.0, 1.4, 0.1], 4.3, 3)     # (ii)/(iii) GMI k = 3 -> 3 x1 + 4 x2 <= 12
    cut_ok = (abs(cut[0] - 3.0) < 1e-9 and abs(cut[1] - 4.0) < 1e-9 and abs(cut[2] - 0.0) < 1e-9
              and abs(rhs - 12.0) < 1e-9 and 3.0 * 4.3 > 12.0)   # cuts off the LP point
    print("omf_ex118 (ii) GMI k=3: %.4g x1 + %.4g x2 + %.4g x3 <= %.4g (3*4.3=%.1f>12) %s"
          % (cut[0], cut[1], cut[2], rhs, 3.0 * 4.3, "OK" if cut_ok else "FAIL"))
    ok &= cut_ok

    lp_ok, zc, xc1, xc2 = solve_ip(True, False)     # (iv) LP with the cut, and the MIP
    mip_ok, zi, xi1, xi2 = solve_ip(True, True)
    cut_lp_ok = lp_ok and abs(-zc - 40.0) < 1e-6 and abs(xc1 - 4.0) < 1e-6 and abs(xc2) < 1e-6
    mip_good = mip_ok and abs(-zi - 40.0) < 1e-6 and abs(xi1 - 4.0) < 1e-6 and abs(xi2) < 1e-6
    print("omf_ex118 (iv) LP+cut: x=(%.4g,%.4g) z=%.6g   MIP: x=(%.4g,%.4g) z=%.6g  (book 40 at (4,0)) %s"
          % (xc1, xc2, -zc, xi1, xi2, -zi, "OK" if (cut_lp_ok and mip_good) else "FAIL"))
    ok &= cut_lp_ok and mip_good
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
