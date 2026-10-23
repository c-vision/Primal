#!/usr/bin/env python3
"""Exact planar cover with rectangular bricks, by binary MIP (MOSEK/Tutorials,
exact-planar-cover/exactcover.py).

Given a board of R x C unit cells and a set of brick shapes, choose placements so
that every cell is covered exactly once:

    x_{s,r,c} in {0,1}   (shape s anchored at row r, column c)
    sum_{placements covering (i,j)} x = 1     for every cell (i,j)
    minimize  sum x                            (fewest bricks)

Hand-derived instances:
  A) 2 x 3 board, dominoes 1x2 / 2x1  -> exactly 3 bricks (area/2)
  B) 3 x 3 board, 2 x 2 squares       -> infeasible (area 9 not divisible by 4)
Port of examples/c/exact_cover.c.
"""

from primalsolver import Model, BK, VAR_TYPE

RES_ERR_INFEASIBLE = 1002
RES_ERR_UNBOUNDED = 1003


def cover(rows, cols, shapes):
    """Returns (1, bricks) if optimal, (0, None) if infeasible, (-1, None) otherwise."""
    var = {}                                        # (s, r, c) -> variable index
    for s, (h, w) in enumerate(shapes):
        for r in range(rows):
            for c in range(cols):
                if r + h <= rows and c + w <= cols:
                    var[(s, r, c)] = len(var)
    nv = len(var)
    with Model() as m:
        m.add_vars(nv)
        m.add_cons(rows * cols)
        for k in range(nv):
            m.var_bounds(k, BK.RA, 0.0, 1.0)
            m.var_type(k, VAR_TYPE.INT_BIN)
            m.cj(k, 1.0)
        for i in range(rows):
            for j in range(cols):
                sub = [k for (s, r, c), k in var.items()
                       if r <= i < r + shapes[s][0] and c <= j < c + shapes[s][1]]
                m.a_row(i * cols + j, sub, [1.0] * len(sub))
                m.con_bounds(i * cols + j, BK.FX, 1.0, 1.0)
        res = m.solve()
    if res.rc == 0:
        return 1, int(res.objective + 0.5)
    if res.rc in (RES_ERR_INFEASIBLE, RES_ERR_UNBOUNDED):
        return 0, None                              # infeasible (no exact cover)
    return -1, None


def status(v):
    return "optimal" if v == 1 else ("infeasible" if v == 0 else "error")


def main() -> int:
    a, na = cover(2, 3, [(1, 2), (2, 1)])           # dominoes
    b, _ = cover(3, 3, [(2, 2)])                    # 2x2 squares
    na = na if na is not None else 0
    ok = (a == 1 and na == 3) and b == 0
    print("exact_cover")
    print(f"  A) 2x3, domino: {status(a)}  bricks={na} (expected 3)")
    print(f"  B) 3x3, 2x2:    {status(b)} (expected infeasible)")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
