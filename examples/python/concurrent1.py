#!/usr/bin/env python3
"""MOSEK port "concurrent1" - concurrent optimizer.

Several algorithms compete on the same problem and the best one is taken. Here the
four optimizers (FREE, INTPNT, PRIMAL_SIMPLEX, DUAL_SIMPLEX) run in SEQUENCE on the
same problem and the primal objectives are compared (documented deviation: no real
parallelism). Problem: max 3x0+x1+5x2+x3 with EQ/LO/UP rows, known pobj 250/3.
Port of examples/c/concurrent1.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA

IPAR_OPTIMIZER = 0
OPTIMIZERS = [("FREE", 2), ("INTPNT", 4), ("PRIMAL_SIMPLEX", 8), ("DUAL_SIMPLEX", 1)]


def build(m):
    """Build the reference LP used by every optimizer."""
    m.add_vars(4)
    m.add_cons(3)
    m.sense(SENSE.MAX)
    for j, c in enumerate([3, 1, 5, 1]):
        m.cj(j, c)
    m.con_bounds(0, BK.FX, 30, 30)
    m.con_bounds(1, BK.LO, 15, 0)
    m.con_bounds(2, BK.UP, 0, 25)
    m.var_bounds(0, BK.LO, 0, 0)
    m.var_bounds(1, BK.RA, 0, 10)
    m.var_bounds(2, BK.LO, 0, 0)
    m.var_bounds(3, BK.LO, 0, 0)
    m.a_col(0, [0, 1], [3, 2])
    m.a_col(1, [0, 1, 2], [1, 1, 2])
    m.a_col(2, [0, 1], [2, 3])
    m.a_col(3, [1, 2], [1, 3])


def main() -> int:
    best, bopt = -math.inf, -1
    vals = []
    all_ok = True
    for k, (name, opt) in enumerate(OPTIMIZERS):
        po = -math.inf
        with Model() as m:
            build(m)
            m.int_param(IPAR_OPTIMIZER, opt)
            r = m.solve()
        if r.rc == 0 and r.solsta == SOLSTA.OPTIMAL:
            po = r.objective
            print(f"{name:<16} pobj = {po:.6f}")
        else:
            all_ok = False
        vals.append(po)
        if po > best:
            best, bopt = po, k
    print(f"best: {OPTIMIZERS[bopt][0] if bopt >= 0 else '?'} with pobj = {best:.6f} "
          f"(expected 83.333333 = 250/3)")
    ok = all_ok and abs(best - 250.0 / 3.0) < 1e-6 and all(abs(v - best) <= 1e-6 for v in vals)
    print(f"{'OK' if ok else 'FAIL'} (4 optimizers, same optimum — simulated concurrency)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
