#!/usr/bin/env python3
"""OMF Exercise 3.15 - workforce planning LP.

Port of c_examples/finance/omf_workforce.c. Each worker works 5 consecutive
days; minimise the number of workers. Book optimum x=(4,7,1,4,3,3,0) = 22,
shadow prices (1/3,0,1/3,0,1/3,1/3,0), reduced cost Shift7 = 1/3.

    python python_examples/omf_workforce.py
"""

from primalsolver import Model, BK, SOLSTA

D = 7
DEMAND = [14, 13, 15, 16, 19, 18, 11]
XBOOK = [4, 7, 1, 4, 3, 3, 0]
SPBOOK = [0.333333, 0, 0.333333, 0, 0.333333, 0.333333, 0]


def main() -> int:
    with Model(maxcon=D, maxvar=D) as m:
        for i in range(D):
            m.var_bounds(i, BK.LO, 0.0)
            m.cj(i, 1.0)
        for d in range(D):
            idx = [((d - k) % D + D) % D for k in range(5)]
            m.a_row(d, idx, [1.0] * 5)
            m.con_bounds(d, BK.LO, float(DEMAND[d]))
        r = m.solve()

    x = r.x
    sp = [abs(r.y[d]) for d in range(D)]
    rcs = m_reduced = None
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 22.0) < 1e-6
          and all(abs(x[i] - XBOOK[i]) < 1e-6 for i in range(D))
          and all(abs(sp[d] - SPBOOK[d]) < 1e-5 for d in range(D)))
    print(f"omf_workforce  workers={r.objective:.4f} (book 22)  "
          f"x={[round(v, 4) for v in x]}  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
