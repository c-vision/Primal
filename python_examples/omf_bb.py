#!/usr/bin/env python3
"""OMF Section 11.3.2 - branch-and-bound examples (two MILPs).

Region: -x1 + x2 <= 2, 8 x1 + 2 x2 <= 19, x integer >= 0.
    (A) max x1 + x2    -> book (1,3), value 4.
    (B) max 3 x1 + x2  -> book (2,1), value 7.
Port of c_examples/finance/omf_bb.c.
"""

from primalsolver import Model, BK, SENSE, VAR_TYPE, SOLSTA


def solve(c1, c2, want, wx1, wx2, name):
    with Model(maxcon=2, maxvar=2) as m:
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0)
            m.var_type(j, VAR_TYPE.INT)
        m.cj(0, c1); m.cj(1, c2)
        m.sense(SENSE.MAX)
        m.a_row(0, [0, 1], [-1.0, 1.0]); m.con_bounds(0, BK.UP, up=2.0)
        m.a_row(1, [0, 1], [8.0, 2.0]); m.con_bounds(1, BK.UP, up=19.0)
        r = m.solve()
    ok = (r.solsta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL)
          and abs(r.objective - want) < 1e-6
          and abs(r.x[0] - wx1) < 1e-6 and abs(r.x[1] - wx2) < 1e-6)
    print(f"omf_bb {name} x=({r.x[0]:.6g},{r.x[1]:.6g}) obj={r.objective:.6g} "
          f"(book ({wx1:.4g},{wx2:.4g}), {want:.4g}) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    a = solve(1.0, 1.0, 4.0, 1.0, 3.0, "(A)")
    b = solve(3.0, 1.0, 7.0, 2.0, 1.0, "(B)")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
