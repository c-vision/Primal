#!/usr/bin/env python3
"""Nesting of convex polygons on a dotted board, a MILP (eellookkaa/mddm, the
"Dotted-Board Model").

Each polygon i chooses exactly one dot d of a grid, and the board size is minimised:

    minimise  z_length + z_width
    s.t.      sum_d delta[i,d] = 1                 (each polygon placed once)
              z_length >= (right edge of i at d) * delta[i,d]
              z_width  >= (top   edge of i at d) * delta[i,d]
              delta[i,d] + delta[j,e] <= 1          for e in NFP(i,j,d)
              delta[i,d] in {0,1}

The no-overlap family uses the No-Fit Polygon, written by hand for this tiny case:
two unit squares on a 2 x 2 board sampled by a dot grid at spacing 0.5; two squares
overlap iff |i1-i2| <= 1 and |j1-j2| <= 1. The optimum is EXACTLY 3.
Port of examples/c/nesting_dotted_board.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

NG = 3                                              # dots per axis (indices 0..2)
NDOT = NG * NG                                      # 9 dots
NP = 2                                              # two squares
X0, ZL, ZW = 0, NP * NDOT, NP * NDOT + 1
NV = NP * NDOT + 2


def overlap(d1, d2):
    return abs(d1 // NG - d2 // NG) <= 1 and abs(d1 % NG - d2 % NG) <= 1


def main() -> int:
    pairs = [(i, j, d1, d2) for i in range(NP) for j in range(NP) if i != j
             for d1 in range(NDOT) for d2 in range(NDOT) if overlap(d1, d2)]
    with Model() as m:
        m.add_vars(NV)
        m.sense(SENSE.MIN)
        for i in range(NP):
            for d in range(NDOT):
                m.var_type(X0 + i * NDOT + d, VAR_TYPE.INT)
                m.var_bounds(X0 + i * NDOT + d, BK.RA, 0.0, 1.0)
        m.var_bounds(ZL, BK.LO, 0.0, math.inf)
        m.var_bounds(ZW, BK.LO, 0.0, math.inf)
        m.cj(ZL, 1.0)
        m.cj(ZW, 1.0)
        m.add_cons(2 * NP * NDOT + NP + len(pairs))
        r = 0
        for i in range(NP):                         # (coef)*delta - z <= 0
            for d in range(NDOT):
                m.a_row(r, [X0 + i * NDOT + d, ZL], [0.5 * (d // NG) + 1.0, -1.0])
                m.con_bounds(r, BK.UP, -math.inf, 0.0)
                r += 1
                m.a_row(r, [X0 + i * NDOT + d, ZW], [0.5 * (d % NG) + 1.0, -1.0])
                m.con_bounds(r, BK.UP, -math.inf, 0.0)
                r += 1
        for i in range(NP):                         # sum_d delta[i,d] = 1
            m.a_row(r, [X0 + i * NDOT + d for d in range(NDOT)], [1.0] * NDOT)
            m.con_bounds(r, BK.FX, 1.0, 1.0)
            r += 1
        for i, j, d1, d2 in pairs:                  # no-overlap
            m.a_row(r, [X0 + i * NDOT + d1, X0 + j * NDOT + d2], [1.0, 1.0])
            m.con_bounds(r, BK.UP, -math.inf, 1.0)
            r += 1
        res = m.solve()
    ok = res.rc == 0
    v, obj = (res.x, res.objective) if ok else ([0.0] * NV, 0.0)
    chosen, legal = [], 1                           # recover the dots, check no overlap
    for i in range(NP):
        c = -1
        for d in range(NDOT):
            if v[X0 + i * NDOT + d] > 0.5:
                c = d
        chosen.append(c)
        if c < 0:
            legal = 0
    if legal:
        for i in range(NP):
            for j in range(NP):
                if i != j and overlap(chosen[i], chosen[j]):
                    legal = 0
    good = ok and abs(obj - 3.0) < 1e-6 and legal
    rc_ = [(c // NG, c % NG) if c >= 0 else (-1, -1) for c in chosen]
    print("nesting_dotted_board (2 squares, board 2x2, fine grid)")
    print(f"  z = {obj:.6f} (expected 3)  chosen dots = ({rc_[0][0]},{rc_[0][1]}) / ({rc_[1][0]},{rc_[1][1]})  "
          f"legal={legal}  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
