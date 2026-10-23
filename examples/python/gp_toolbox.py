#!/usr/bin/env python3
"""A small geometric program (MOSEK Tutorials, gp-toolbox).

    minimize    x + y^2 z
    subject to  0.1 sqrt(x) + 2 y^-1 <= 1
                z^-1 + y x^-2 <= 1

A GP is convex under y = log x: every posynomial becomes a log-sum-exp,
    sum_i c_i exp(a_i . y) <= 1  <=>  p_i >= exp(a_i . y + ln c_i), sum p_i <= 1
i.e. one exponential cone per monomial plus a linear row. The objective adds a -t
and minimizes t. Port of examples/c/gp_toolbox.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

NV = 3
# each posynomial is a list of monomials (coef, [(var, exponent), ...])
OBJ = [(1.0, [(0, 1)]), (1.0, [(1, 2), (2, 1)])]          # x + y^2 z
C0 = [(0.1, [(0, 0.5)]), (2.0, [(1, -1)])]                # 0.1 sqrt(x) + 2/y
C1 = [(1.0, [(2, -1)]), (1.0, [(0, -2), (1, 1)])]         # 1/z + y/x^2
POSY = [(OBJ, True), (C0, False), (C1, False)]


def main() -> int:
    nmulti = sum(len(p) for p, _ in POSY)
    # vars: y[NV] | t | e_i (per monomial) | p_i (per monomial) | ONE
    ybase, tv, ebase = 0, NV, NV + 1
    pbase = ebase + nmulti
    onev = pbase + nmulti
    with Model() as m:
        m.add_vars(onev + 1)
        for i in range(NV):
            m.var_bounds(ybase + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(tv, BK.FR, -math.inf, math.inf)
        for i in range(nmulti):
            m.var_bounds(ebase + i, BK.FR, -math.inf, math.inf)
        for i in range(nmulti):
            m.var_bounds(pbase + i, BK.LO, 0.0, math.inf)
        m.var_bounds(onev, BK.FX, 1.0, 1.0)
        m.add_cons(len(POSY) + 2 * nmulti)          # one sum row per posynomial + one link per monomial
        r = ecur = pcur = 0
        for monos, isobj in POSY:
            p0 = pcur
            for coef, terms in monos:
                sub = [ebase + ecur] + [ybase + v for v, _ in terms]
                val = [1.0] + [-a for _, a in terms]
                if isobj:
                    sub.append(tv)
                    val.append(1.0)
                m.a_row(r, sub, val)
                m.con_bounds(r, BK.FX, math.log(coef), math.log(coef))
                r += 1
                m.cone(CT.PEXP, [pbase + pcur, onev, ebase + ecur])
                ecur += 1
                pcur += 1
            m.a_row(r, [pbase + p0 + q for q in range(len(monos))], [1.0] * len(monos))
            m.con_bounds(r, BK.UP, -math.inf, 1.0)
            r += 1
        m.cj(tv, 1.0)
        m.sense(SENSE.MIN)
        res = m.solve()
    ok = res.rc == 0
    x, topt = [0.0] * NV, 0.0
    if ok:
        x = [math.exp(res.x[ybase + i]) for i in range(NV)]
        topt = res.x[tv]
    # objective and the two constraint posynomials, by direct substitution
    obj = x[0] + x[1] * x[1] * x[2]
    c0 = 0.1 * math.sqrt(x[0]) + 2.0 / x[1] if ok else 0.0
    c1 = 1.0 / x[2] + x[1] / (x[0] * x[0]) if ok else 0.0
    okall = ok and c0 < 1.0 + 1e-8 and c1 < 1.0 + 1e-8 and abs(obj - math.exp(topt)) < 1e-7 * (1.0 + obj)
    print(f"gp_toolbox  x={x[0]:.6f} y={x[1]:.6f} z={x[2]:.6f}  obj = {obj:.6f}")
    print(f"  c0 = {c0:.9f}  c1 = {c1:.9f}  (both <= 1)   exp(t) = {math.exp(topt):.6f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
