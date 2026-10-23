#!/usr/bin/env python3
"""Minimum s-t cut as a linear program, and the CORRELATED min-cut (maxdan94/lp_mincut).

    minimize    sum_e w_e d_e
    subject to  d_e - p_{s(e)} + p_{t(e)} >= 0     for every arc e
                p_s - p_t = 1                       (separate s from t)
                d_e, p_v >= 0

The optimum is the min-cut capacity, equal to the max flow. The repo's research part
adds, for a pair of CORRELATED arcs, the constraint d_ij = d_kl ("ij is cut iff kl is
cut"). Instance: the repo's graph.txt, s=0, t=3, arcs 0->1 (0.5), 0->2 (0.3),
1->3 (0.4), 2->3 (0.5); the min cut is 0.7. Port of examples/c/lp_mincut.c.
"""

import math

from primalsolver import Model, BK, SENSE

NE = 4                                              # arcs
NN = 4                                              # nodes
SRC = [0, 0, 1, 2]
DST = [1, 2, 3, 3]
W = [0.5, 0.3, 0.4, 0.5]


def mincut(cor):
    """Min cut with a list of correlated arc pairs forced equal; returns (value, d)."""
    d0, p0, nv = 0, NE, NE + NN
    with Model() as m:
        m.add_vars(nv)
        m.sense(SENSE.MIN)
        for i in range(nv):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        for e in range(NE):
            m.cj(d0 + e, W[e])
        m.add_cons(NE + 1 + len(cor))
        r = 0
        for e in range(NE):                         # d_e - p_s + p_t >= 0
            m.a_row(r, [d0 + e, p0 + SRC[e], p0 + DST[e]], [1.0, -1.0, 1.0])
            m.con_bounds(r, BK.LO, 0.0, math.inf)
            r += 1
        m.a_row(r, [p0 + 0, p0 + 3], [1.0, -1.0])   # p_s - p_t = 1
        m.con_bounds(r, BK.FX, 1.0, 1.0)
        r += 1
        for a, b in cor:                            # d_ij = d_kl
            m.a_row(r, [d0 + a, d0 + b], [1.0, -1.0])
            m.con_bounds(r, BK.FX, 0.0, 0.0)
            r += 1
        res = m.solve()
    if res.rc != 0:
        return -1.0, [0.0] * NE
    return res.objective, [res.x[d0 + e] for e in range(NE)]


def main() -> int:
    print("lp_mincut (min-cut LP; graph.txt of maxdan94/lp_mincut)")
    cut, d = mincut([])
    # the value must equal the hand max flow 0.4 + 0.3 = 0.7
    good = cut > 0 and abs(cut - 0.7) < 1e-7
    print(f"  min cut = {cut:.8f} (expected 0.7)  d=({d[0]:.4f},{d[1]:.4f},{d[2]:.4f},{d[3]:.4f})  "
          f"{'OK' if good else 'FAIL'}")
    # correlated variant: force d(0->1) = d(2->3) (arcs 0 and 3); the cut may rise
    # (a correlation only restricts the feasible set) but must be >= the plain cut
    cc, d2 = mincut([(0, 3)])
    good2 = cc > 0 and cc >= cut - 1e-9 and abs(d2[0] - d2[3]) < 1e-7
    print(f"  correlated d01=d23: cut = {cc:.8f} (d01={d2[0]:.4f}=d23={d2[3]:.4f})  >= 0.7  "
          f"{'OK' if good2 else 'FAIL'}")
    ok = good and good2
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
