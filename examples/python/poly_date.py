#!/usr/bin/env python3
""""poly-date": the shortest homogeneous three-variable polynomial equation with all
coefficients 1 satisfied by a date (d, m, y) (aszek/poly-date, Python + MOSEK Fusion).

For each degree k, enumerate the monomials d^a m^b y^c with a+b+c = k and their
values val_i. Two DISJOINT subsets s1, s2 with equal sums give a valid equation
P_s1 = P_s2; the model finds the shortest:

    minimize   sum_i (s1_i + s2_i)
    subject to sum_i (s1_i - s2_i) val_i = 0
               s1_i + s2_i <= 1          (disjoint)
               sum_i s1_i >= 1           (an equation, not 0 = 0)
               s1, s2 in {0,1}

The smallest degree with a solution is reported; the coefficients are verified in
exact integer arithmetic. Known case: 8/11/15 has its first equation at degree 3,
both sides equal to 6730 (4 terms = 5 terms).

    python poly_date.py [d m y maxDeg]   (default 8 11 15 5)

Port of examples/c/poly_date.c.
"""

import math
import sys

from primalsolver import Model, BK, SENSE, VAR_TYPE


def gen_monomials(d, m, y, k):
    """The degree-k monomials of (d,m,y) as (a, b, c, value)."""
    return [(a, b, k - a - b, d ** a * m ** b * y ** (k - a - b))
            for a in range(k + 1) for b in range(k - a + 1)]


def solve_degree(d, m, y, k):
    """MILP for one degree; returns (ok, nmono, n1, n2, L, R)."""
    mo = gen_monomials(d, m, y, k)
    n = len(mo)
    s1, s2, nv = 0, n, 2 * n
    with Model() as mdl:
        mdl.add_vars(nv)
        mdl.sense(SENSE.MIN)
        for i in range(nv):
            mdl.var_type(i, VAR_TYPE.INT)
            mdl.var_bounds(i, BK.RA, 0.0, 1.0)
            mdl.cj(i, 1.0)
        mdl.add_cons(1 + n + 1)
        mdl.a_row(0, [s1 + i for i in range(n)] + [s2 + i for i in range(n)],
                  [float(v[3]) for v in mo] + [-float(v[3]) for v in mo])
        mdl.con_bounds(0, BK.FX, 0.0, 0.0)
        for i in range(n):
            mdl.a_row(1 + i, [s1 + i, s2 + i], [1.0, 1.0])
            mdl.con_bounds(1 + i, BK.UP, -math.inf, 1.0)
        mdl.a_row(1 + n, [s1 + i for i in range(n)], [1.0] * n)
        mdl.con_bounds(1 + n, BK.LO, 1.0, math.inf)
        r = mdl.solve()
    n1 = n2 = left = right = 0
    ok = False
    if r.rc == 0:
        for i in range(n):
            if r.x[s1 + i] > 0.5:
                left += mo[i][3]
                n1 += 1
            if r.x[s2 + i] > 0.5:
                right += mo[i][3]
                n2 += 1
        ok = n1 > 0 and left == right
    return ok, n, n1, n2, left, right


def main() -> int:
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    y = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    max_deg = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    print(f"poly_date  d={d} m={m} y={y} maxDeg={max_deg}")
    all_ok, found = True, False
    for k in range(1, max_deg + 1):
        ok, n, n1, n2, left, right = solve_degree(d, m, y, k)
        if ok:
            good = left == right and left > 0 and n1 > 0
            print(f"  degree {k}: {n} monomials, {n1} + {n2} terms, LHS=RHS={left}  {'OK' if good else 'FAIL'}")
            all_ok &= good
            found = True
            break
        print(f"  degree {k}: no equation")
    if not found:
        print(f"  no equation up to degree {max_deg}")
        all_ok = False
    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
