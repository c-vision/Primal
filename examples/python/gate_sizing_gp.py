#!/usr/bin/env python3
"""Gate sizing of a digital circuit as a geometric program (lrakai/gate-sizing, from
Boyd's "A Tutorial on Geometric Programming").

Variables: the gate scale factors s_i, the arrival times T_i, the "max" auxiliaries
V_i and the objective D (the critical path delay). Gate i's delay is the posynomial
    D_i(s) = gamma_i * s_i^-1 * sum_{j driven by i} (beta_j s_j + alpha_j);
the model minimises D subject to D_i <= T_i, T_anc <= V_i, T_i <= D and a total
area bound sum_i a_i s_i <= Amax. Under y = log x each posynomial becomes one
exponential cone per monomial plus a linear row; a single-monomial one is linear.

Hand case: two inverters INV -> INV -> output of capacitance 10, Amax = 10:
D(s) = (s2 + 1)/s1 + 10/s2, and the optimum is D* = 3.19761770 at
s2 = 10 sqrt(10)/(sqrt(11)+sqrt(10)). Port of examples/c/gate_sizing_gp.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

NV = 6                                              # s1 s2 T1 T2 V2 D
S1, S2, T1, T2, V2, D = range(NV)
AMAX = 10.0
# posynomial constraints <= 1, each a list of monomials (coef, [(var, exponent), ...])
POSY = [
    [(1.0, [(S2, 1), (S1, -1), (T1, -1)]), (1.0, [(S1, -1), (T1, -1)])],
    [(10.0, [(S2, -1), (T2, -1)]), (1.0, [(V2, 1), (T2, -1)])],
    [(1.0, [(T1, 1), (V2, -1)])],
    [(1.0, [(T2, 1), (D, -1)])],
    [(1.0 / AMAX, [(S1, 1)]), (1.0 / AMAX, [(S2, 1)])],
]


def main() -> int:
    nmulti = sum(len(p) for p in POSY if len(p) > 1)
    nrq = sum(1 for p in POSY if len(p) == 1)
    ybase, ebase = 0, NV
    pbase = ebase + nmulti
    onev = pbase + nmulti
    nv = onev + 1
    with Model() as m:
        m.add_vars(nv)
        m.sense(SENSE.MIN)
        for i in range(nv):
            m.var_bounds(i, BK.FR, -math.inf, math.inf)
        for i in range(nmulti):
            m.var_bounds(pbase + i, BK.LO, 0.0, math.inf)
        m.var_bounds(onev, BK.FX, 1.0, 1.0)
        m.add_cons(len(POSY) + 2 * nmulti + nrq)    # as many rows as the C example reserves
        r = ecur = pcur = 0
        for monos in POSY:
            if len(monos) == 1:                     # linear: a.y + ln c <= 0
                coef, terms = monos[0]
                m.a_row(r, [ybase + v for v, _ in terms], [a for _, a in terms])
                m.con_bounds(r, BK.UP, -math.inf, -math.log(coef))
                r += 1
                continue
            p0 = pcur
            for coef, terms in monos:
                m.a_row(r, [ebase + ecur] + [ybase + v for v, _ in terms], [1.0] + [-a for _, a in terms])
                m.con_bounds(r, BK.FX, math.log(coef), math.log(coef))
                r += 1
                m.cone(CT.PEXP, [pbase + pcur, onev, ebase + ecur])
                ecur += 1
                pcur += 1
            m.a_row(r, [pbase + p0 + q for q in range(len(monos))], [1.0] * len(monos))
            m.con_bounds(r, BK.UP, -math.inf, 1.0)
            r += 1
        m.cj(ybase + D, 1.0)                        # minimize log D
        res = m.solve()
    ok = res.rc == 0
    y = [math.exp(res.x[ybase + i]) for i in range(NV)] if ok else [0.0] * NV
    s1, s2 = y[S1], y[S2]                           # independent checks at the recovered sizes
    delay = (s2 + 1.0) / s1 + 10.0 / s2 if ok else 0.0
    area = s1 + s2
    want = 10.0 * math.sqrt(10.0) / (math.sqrt(11.0) + math.sqrt(10.0))   # s2*
    good = (ok and abs(delay - 3.1976177) < 1e-4 and abs(area - AMAX) < 1e-6
            and abs(s2 - want) < 1e-2 and abs(y[D] - delay) < 1e-4)
    print(f"gate_sizing_gp  delay={delay:.8f} (expected 3.19761770)  s1={s1:.6f} s2={s2:.6f} (expected {want:.6f})")
    print(f"  area={area:.6f} (<= {AMAX:.0f})  D(var)={y[D]:.8f}  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
