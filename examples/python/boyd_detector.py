#!/usr/bin/env python3
"""Minimax detector, "Convex Optimization" (Boyd & Vandenberghe, 2004), Example 7.4.

Binary hypothesis testing with n = 4 and
    P = [ 0.70 0.10 ; 0.20 0.10 ; 0.05 0.70 ; 0.05 0.10 ]
(column j = distribution of the observation under hypothesis j+1). A randomized
detector T (2x4, each column a probability distribution) minimizes the worst-case
error; the minimax detector T = [1 2/3 0 0; 0 1/3 1 1] yields P_fp = P_fn = 1/6.

LP form: T0_i, T1_i >= 0 with T0_i + T1_i = 1, and t with
t >= sum_i P[i][1] T1_i (false negative), t >= sum_i P[i][0] T0_i (false positive);
minimize t. Port of examples/c/boyd_detector.c.
"""

import math

from primalsolver import Model, BK

P0 = [0.70, 0.20, 0.05, 0.05]                       # hypothesis 1 column
P1 = [0.10, 0.10, 0.70, 0.10]                       # hypothesis 2 column


def main() -> int:
    with Model() as m:
        m.add_vars(9)                               # 8 T + t
        m.add_cons(6)                               # 4 column balances + 2 error rows
        for j in range(8):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_bounds(8, BK.FR, -math.inf, math.inf)
        m.cj(8, 1.0)                                # minimize t
        for i in range(4):                          # T0_i + T1_i = 1
            m.a_row(i, [i, 4 + i], [1.0, 1.0])
            m.con_bounds(i, BK.FX, 1.0, 1.0)
        # false negative: t - sum_i P0[i] T1_i >= 0
        m.a_row(4, [4, 5, 6, 7, 8], [-P0[0], -P0[1], -P0[2], -P0[3], 1.0])
        m.con_bounds(4, BK.LO, 0.0, math.inf)
        # false positive: t - sum_i P1[i] T0_i >= 0
        m.a_row(5, [0, 1, 2, 3, 8], [-P1[0], -P1[1], -P1[2], -P1[3], 1.0])
        m.con_bounds(5, BK.LO, 0.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"boyd_detector  rc={r.rc} FAIL")
        return 1
    v, z = r.x, r.objective
    fn = sum(P0[i] * v[4 + i] for i in range(4))    # recomputed error probabilities
    fp = sum(P1[i] * v[i] for i in range(4))
    ok = abs(z - 1.0 / 6.0) < 1e-6 and abs(fn - 1.0 / 6.0) < 1e-6 and abs(fp - 1.0 / 6.0) < 1e-6
    print(f"boyd_detector  t={z:.6g} Pfp={fp:.6g} Pfn={fn:.6g} (book 1/6 all) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
