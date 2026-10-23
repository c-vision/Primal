#!/usr/bin/env python3
"""SOS type-1 constraint (port of the "sos1" example of the MOSEK C API docs).

    min x0 + x1 + x2  s.t.  x0 + 2*x1 + 3*x2 >= 2.5,  SOS1 {x0, x1, x2} (weights 1,2,3),
                            0 <= x <= 1

SOS1: at most ONE nonzero variable. The most "efficient" one to cover the row is x2
(3 units of coverage per unit of cost): x2 = 2.5/3 = 0.8333, obj = 5/6.
Port of examples/c/sos1.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    with Model() as m:
        m.add_vars(3)
        m.add_cons(1)
        for j in range(3):
            m.var_bounds(j, BK.RA, 0.0, 1.0)
            m.cj(j, 1.0)
        m.a_row(0, [0, 1, 2], [1.0, 2.0, 3.0])
        m.con_bounds(0, BK.LO, 2.5, math.inf)
        m.sos(1, [0, 1, 2], [1.0, 2.0, 3.0])
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}), obj = {obj:.4f}")
    # SOS1 check (at most one nonzero) by enumerating the 3 members: only member k
    # active with x_k = 2.5/coverage_k if <= 1
    best = math.inf
    for cov in (1.0, 2.0, 3.0):
        need = 2.5 / cov
        if need <= 1.0 + 1e-9 and need < best:
            best = need
    nact = sum(1 for j in range(3) if x[j] > 1e-6)
    ok = abs(obj - best) < 1e-6 and nact <= 1 and abs(obj - (x[0] + x[1] + x[2])) < 1e-6
    print(f"enumeration: best = {best:.4f}, nonzeros = {nact}")
    print(f"{'OK' if ok else 'FAIL'} (expected obj=5/6, a single nonzero member)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
