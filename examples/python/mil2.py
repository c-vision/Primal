#!/usr/bin/env python3
"""Semi-continuous and semi-integer variables (port of the "mil2" example of the
MOSEK C API docs).

    min x0 + 2*x1  s.t.  x0 + x1 >= 3  with
      x0 semi-continuous [2,10]   (x0 = 0 or 2 <= x0 <= 10)
      x1 semi-integer    [2.5,10] (x1 = 0 or 2.5 <= x1 <= 10, integer)

Enumeration: x0=0 -> x1 >= 3 integer -> cost 6; x1=0 -> x0 = 3, cost 3 (best);
both active -> cost >= 2+6 = 8. Optimum: x=(3,0), obj=3. Port of examples/c/mil2.c.
"""

import math

from primalsolver import Model, BK, VAR_TYPE


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        m.var_bounds(0, BK.RA, 2.0, 10.0)
        m.var_bounds(1, BK.RA, 2.5, 10.0)
        m.var_type(0, VAR_TYPE.SEMI_CONT)
        m.var_type(1, VAR_TYPE.SEMI_INT)
        m.cj(0, 1.0); m.cj(1, 2.0)
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.LO, 3.0, math.inf)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}), obj = {obj:.4f}")

    best = math.inf                                 # exhaustive enumeration of the semi cases
    for a in range(2):                              # x0 active?
        for b in range(2):                          # x1 active?
            if not a and not b:
                continue                            # 0 + 0 < 3
            v1 = 3.0 if b else 0.0
            x0 = max(2.0, 3.0 - v1) if a else 0.0
            x1 = math.ceil(max(2.5, 3.0 - (2.0 if a else 0.0))) if b else 0.0
            if not a and x1 < 3.0 - 1e-9 and b:
                continue
            cov = (x0 if a else 0.0) + (x1 if b else 0.0)
            if cov < 3.0 - 1e-9:
                # the active member must cover alone or with its companion
                if a and not b:
                    x0 = 3.0
                if not a and b:
                    x1 = 3.0
                if a and b:
                    continue                        # both active: minimum coverage 2+3=5
                cov = (x0 if a else 0.0) + (x1 if b else 0.0)
            c = (x0 if a else 0.0) + 2.0 * (x1 if b else 0.0)
            if cov >= 3.0 - 1e-9 and c < best:
                best = c
    print(f"enumeration: best = {best:.4f}")
    ok = abs(obj - 3.0) < 1e-6 and abs(obj - best) < 1e-6 and abs(x[0] - 3.0) < 1e-6 and abs(x[1]) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (expected x=(3,0), obj=3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
