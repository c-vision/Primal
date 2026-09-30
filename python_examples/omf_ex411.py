#!/usr/bin/env python3
"""OMF Exercise 4.11 - arbitrage in the currency market (LP).

Port of c_examples/finance/omf_ex411.c. 12 conversion quantities + the
arbitrage amount D, one balance equation per currency; max D (capped at 10000).
The book's $->E->Yen->$ cycle earns ~0.03%, the reverse loses ~0.02%.

    python python_examples/omf_ex411.py
"""

from primalsolver import Model, BK, SENSE, SOLSTA

DE, DP, DY, ED, EP, EY, PD, PE, PY, YD, YE, YP, D_ = range(13)
NV = 13


def main() -> int:
    c_dey = 1.1486 * 116.12 * 0.00750
    c_dye = 133.38 * 0.00861 * 0.8706
    cycles_ok = (c_dey > 1.0 and c_dye < 1.0 and abs((c_dey - 1.0) - 0.0003) < 5e-5
                 and abs((1.0 - c_dye) - 0.0002) < 5e-5)

    with Model(maxcon=4, maxvar=NV) as m:
        for j in range(NV):
            m.var_bounds(j, BK.LO, 0.0)
        m.var_bounds(D_, BK.UP, up=10000.0)
        m.cj(D_, 1.0)
        m.sense(SENSE.MAX)
        m.a_row(0, [D_, DE, DP, DY, ED, PD, YD],
                   [1.0, 1.0, 1.0, 1.0, -0.8706, -1.4279, -0.00750])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [ED, EP, EY, DE, PE, YE], [1.0, 1.0, 1.0, -1.1486, -1.6401, -0.00861])
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [PD, PE, PY, DP, EP, YP], [1.0, 1.0, 1.0, -0.7003, -0.6097, -0.00525])
        m.con_bounds(2, BK.FX, 0.0, 0.0)
        m.a_row(3, [YD, YE, YP, DY, EY, PY], [1.0, 1.0, 1.0, -133.38, -116.12, -190.45])
        m.con_bounds(3, BK.FX, 0.0, 0.0)
        r = m.solve()

    d = r.objective
    ok = r.solsta == SOLSTA.OPTIMAL and cycles_ok and abs(d - 10000.0) < 1.0
    print(f"omf_ex411  arbitrage D={d:.4f} (cap 10000)  "
          f"$->E->Yen->$={c_dey:.6f}(>1)  reverse={c_dye:.6f}(<1)  "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
