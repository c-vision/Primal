#!/usr/bin/env python3
"""Exercise 4.11 of "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007): currency arbitrage.

With the exchange rates of February 14, 2002 (Dollar, Euro, Pound, Yen):

    max D  s.t.  one balance row per currency over the 12 conversions,  0 <= vars,  D <= 10000

The dollar -> euro -> yen -> dollar cycle makes ~$0.0003 per dollar
(1.1486 * 116.12 * 0.00750 = 1.000316), the reverse loses ~$0.0002; the LP drives D
to its cap (several cycles are optimal). Port of examples/c/omf_ex411.c.
"""

import math

from primalsolver import Model, BK

NV = 13
DE, DP, DY, ED, EP, EY, PD, PE, PY, YD, YE, YP, D_ = range(13)


def main() -> int:
    c_dey = 1.1486 * 116.12 * 0.00750               # > 1 : arbitrage
    c_dye = 133.38 * 0.00861 * 0.8706               # < 1 : loss
    cycles_ok = (c_dey > 1.0 and c_dye < 1.0 and abs((c_dey - 1.0) - 0.0003) < 5e-5
                 and abs((1.0 - c_dye) - 0.0002) < 5e-5)
    with Model(4, NV) as m:
        for j in range(NV):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_bounds(D_, BK.UP, 0.0, 10000.0)
        m.cj(D_, -1.0)                              # max D
        m.a_row(0, [D_, DE, DP, DY, ED, PD, YD], [1.0, 1.0, 1.0, 1.0, -0.8706, -1.4279, -0.00750])   # Dollar
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [ED, EP, EY, DE, PE, YE], [1.0, 1.0, 1.0, -1.1486, -1.6401, -0.00861])            # Euro
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [PD, PE, PY, DP, EP, YP], [1.0, 1.0, 1.0, -0.7003, -0.6097, -0.00525])            # Pound
        m.con_bounds(2, BK.FX, 0.0, 0.0)
        m.a_row(3, [YD, YE, YP, DY, EY, PY], [1.0, 1.0, 1.0, -133.38, -116.12, -190.45])             # Yen
        m.con_bounds(3, BK.FX, 0.0, 0.0)
        r = m.solve()
    ok = r.rc == 0
    if ok:
        z, x = r.objective, r.x
        d = -z
        ok = cycles_ok and abs(d - 10000.0) < 1.0
        print("omf_ex411  arbitrage D=%.4f (cap 10000)  %s" % (d, "OK" if ok else "FAIL"))
        print("  cycle $->E->Yen->$ = %.6f (>1, +%.4f/dollar)  $->Yen->E->$ = %.6f (<1, %.4f/dollar)  %s"
              % (c_dey, c_dey - 1.0, c_dye, c_dye - 1.0, "OK" if cycles_ok else "FAIL"))
        print("  flows (one of several optimal cycles): DE=%.6g DP=%.4g DY=%.4g ED=%.4g YD=%.4g"
              % (x[DE], x[DP], x[DY], x[ED], x[YD]))
    else:
        print("omf_ex411  rc=%d FAIL" % r.rc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
