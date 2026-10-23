#!/usr/bin/env python3
"""High-frequency transformer, minimum-loss design, as a geometric program
(MOSEK Tutorials, transformer-design).

A GP  min f0(x)  s.t.  f_k(x) <= 1  with posynomials f is convex after y = log x:
each posynomial becomes a log-sum-exp,
    sum_i c_i exp(a_i . y) <= 1   <=>   p_i >= exp(a_i . y + ln c_i), sum p_i <= 1,
one exponential cone per monomial plus one linear row; a single-monomial constraint
is linear, a . y + ln c <= 0, and the objective adds -t, minimizing t.
This ports the notebook's 15-variable, 28-constraint HF ferrite model.
DECLARED DEVIATION: the reference optimum is not reproduced; the solution is checked
by GP primal feasibility (every posynomial <= 1) at x = exp(y).
Port of examples/c/transformer_design.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

(VC, VT, VBW, VHW, VNP, VNS, VNLS, VB, VJ, VPC, VPCU, VVRR, VVRX, VIM, VHB) = range(15)
NGP = 15
MCON = 28


def mk(coef, *pairs):
    """One monomial: coefficient and (variable, exponent) pairs."""
    return coef, [(int(pairs[2 * i]), float(pairs[2 * i + 1])) for i in range(len(pairs) // 2)]


def constraints():
    """The 28 posynomial constraint slots and the objective posynomial."""
    c0, t0, bw0, hw0 = 0.005, 0.015, 0.010, 0.025   # constants
    s_va, ep, es, f, pf = 1200.0, 300.0, 75.0, 100e3, 0.80
    rf = math.sqrt(1.0 - pf * pf)
    vrm, k_phi, dt = 0.03, 0.02, 65.0
    ip = s_va / ep
    m_layers, d_foil, kcu = 2, 0.238e-3, 0.6
    kc, alpha, beta, bsat, rho_c = 1.9e-3, 1.24, 2.0, 0.4, 4800.0
    rho_w = 2.26e-8
    mu0 = 4e-7 * math.pi
    mu = mu0 * 2000.0
    kf, ka, fr, fl = 0.95, 8.0 * math.pow(100.0, 1.75), 1.32, 0.9
    pi = math.pi
    c33 = math.pow(2.0, 1.5) * pi * f * kf / ep     # derived coefficients
    c34 = es / ep
    c35 = 2.0 * ip / kcu
    c36 = (1.0 / fl) * (1.0 / d_foil) * ip
    denom_t = 1.42 * math.sqrt(2.0) * ka * math.pow(dt, 1.25)
    ccore = kf * kc * rho_c * math.pow(f, alpha)
    ccu = kcu * fr * rho_w
    cim = math.sqrt(2.0) / mu
    cnl = (1.0 / (k_phi * k_phi)) * (1.0 / (ip * ip))
    cr = fr * rho_w * pf
    inv_p2 = (m_layers * m_layers) / 4.0
    cx = ip * rf * f * mu0 * inv_p2
    coeff_vr = (1.0 / ep) * (1.0 / vrm)
    p = [
        # fix_var c,t,bw,hw: one monomial per slot (x/v <= 1 and v/x <= 1)
        [mk(1.0 / c0, VC, 1)], [mk(c0, VC, -1)], [mk(1.0 / t0, VT, 1)], [mk(t0, VT, -1)],
        [mk(1.0 / bw0, VBW, 1)], [mk(bw0, VBW, -1)], [mk(1.0 / hw0, VHW, 1)], [mk(hw0, VHW, -1)],
        # (33)-(36) equalities as two single-monomial slots each
        [mk(c33, VNP, 1, VB, 1, VC, 1, VT, 1)], [mk(1.0 / c33, VNP, -1, VB, -1, VC, -1, VT, -1)],
        [mk(c34, VNP, 1, VNS, -1)], [mk(1.0 / c34, VNP, -1, VNS, 1)],
        [mk(c35, VNP, 1, VJ, -1, VBW, -1, VHW, -1)], [mk(1.0 / c35, VNP, -1, VJ, 1, VBW, 1, VHW, 1)],
        [mk(c36, VJ, -1, VNP, 1, VNS, -1, VNLS, 1, VHW, -1)], [mk(1.0 / c36, VJ, 1, VNP, -1, VNS, 1, VNLS, -1, VHW, 1)],
        # (37) temperature rise
        [mk(1.0 / denom_t, VPC, 1, VHB, 0.25, VC, -0.5, VT, -0.5, VBW, -0.5, VHW, -0.5),
         mk(1.0 / denom_t, VPCU, 1, VHB, 0.25, VC, -0.5, VT, -0.5, VBW, -0.5, VHW, -0.5)],
        # (38) hw + 2c <= hbar
        [mk(1.0, VHW, 1, VHB, -1), mk(2.0, VC, 1, VHB, -1)],
        # (39) core loss
        [mk(4.0 * ccore, VB, beta, VC, 1, VT, 1, VHW, 1, VPC, -1), mk(8.0 * ccore, VB, beta, VC, 2, VT, 1, VPC, -1),
         mk(4.0 * ccore, VB, beta, VC, 1, VT, 1, VBW, 1, VPC, -1)],
        # (40) copper loss
        [mk(4.0 * ccu, VC, 1, VBW, 1, VHW, 1, VJ, 2, VPCU, -1), mk(2.0 * ccu, VT, 1, VBW, 1, VHW, 1, VJ, 2, VPCU, -1),
         mk(pi * ccu, VBW, 2, VHW, 1, VJ, 2, VPCU, -1)],
        # (42) magnetizing current
        [mk(cim, VB, 1, VNP, -1, VHW, 1, VIM, -1), mk(cim, VB, 1, VNP, -1, VBW, 1, VIM, -1),
         mk(2.0 * cim, VB, 1, VNP, -1, VC, 1, VIM, -1)],
        [mk(1.0 / bsat, VB, 1)],                    # (43) saturation
        [mk(cnl, VIM, 2), mk(cnl / (ep * ep), VPC, 2)],   # (44) no-load current ratio
        # (45) resistive regulation
        [mk(8.0 * cr, VNP, 1, VC, 1, VJ, 1, VVRR, -1), mk(4.0 * cr, VNP, 1, VT, 1, VJ, 1, VVRR, -1),
         mk(2.0 * pi * cr, VNP, 1, VBW, 1, VJ, 1, VVRR, -1)],
        # (46) reactive regulation
        [mk((8.0 * pi / 3.0) * cx, VNP, 2, VC, 1, VBW, 1, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2),
         mk((4.0 * pi / 3.0) * cx, VNP, 2, VT, 1, VBW, 1, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2),
         mk((2.0 * pi * pi / 3.0) * cx, VNP, 2, VBW, 2, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2)],
        [mk(1.0, VNLS, -1)], [mk(m_layers, VNLS, 1, VNS, -1)],   # (47) Nls >= 1, Ns >= m*Nls
        [mk(coeff_vr, VVRR, 1), mk(coeff_vr, VVRX, 1)],          # (48) total regulation
    ]
    return p, [mk(1.0, VPC, 1), mk(1.0, VPCU, 1)]


def posy_value(monos, x):
    s = 0.0
    for coef, terms in monos:
        v = coef
        for var, a in terms:
            v *= math.pow(x[var], a)
        s += v
    return s


def main() -> int:
    p, obj = constraints()
    if len(p) != MCON:
        print(f"transformer_design  slot count {len(p)} != {MCON} FAIL")
        return 1
    nmulti = sum(len(q) for q in p if len(q) > 1) + 2   # + objective
    ybase, tv, ebase = 0, NGP, NGP + 1               # vars: y | t | e_i | p_i | ONE
    pbase = ebase + nmulti
    onev = pbase + nmulti
    with Model() as m:
        m.add_vars(onev + 1)
        for i in range(NGP):
            m.var_bounds(ybase + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(tv, BK.FR, -math.inf, math.inf)
        for i in range(nmulti):
            m.var_bounds(ebase + i, BK.FR, -math.inf, math.inf)
        for i in range(nmulti):
            m.var_bounds(pbase + i, BK.LO, 0.0, math.inf)
        m.var_bounds(onev, BK.FX, 1.0, 1.0)
        m.add_cons(MCON + 1 + 2 * nmulti)
        nrow = ecur = pcur = 0

        def monomial_rows(monos, with_t):
            """e - a.y (+ t) = ln c per monomial, PEXP(p, ONE, e), and sum p <= 1."""
            nonlocal nrow, ecur, pcur
            p0 = pcur
            for coef, terms in monos:
                sub = [ebase + ecur] + [ybase + v for v, _ in terms]
                val = [1.0] + [-a for _, a in terms]
                if with_t:
                    sub.append(tv)
                    val.append(1.0)
                m.a_row(nrow, sub, val)
                m.con_bounds(nrow, BK.FX, math.log(coef), math.log(coef))
                nrow += 1
                m.cone(CT.PEXP, [pbase + pcur, onev, ebase + ecur])
                ecur += 1
                pcur += 1
            m.a_row(nrow, [pbase + p0 + q for q in range(len(monos))], [1.0] * len(monos))
            m.con_bounds(nrow, BK.UP, -math.inf, 1.0)
            nrow += 1

        for monos in p:
            if len(monos) == 1:                     # linear: a.y + ln c <= 0
                coef, terms = monos[0]
                m.a_row(nrow, [ybase + v for v, _ in terms], [a for _, a in terms])
                m.con_bounds(nrow, BK.UP, -math.inf, -math.log(coef))
                nrow += 1
            else:
                monomial_rows(monos, False)
        monomial_rows(obj, True)                    # objective: min t, p_k >= exp(a.y + ln c - t)
        m.cj(tv, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    x, topt = [0.0] * NGP, 0.0
    if ok:
        x = [math.exp(r.x[ybase + i]) for i in range(NGP)]
        topt = r.x[tv]
    worst_i, worst = -1, -1e300                     # independent check: every posynomial <= 1
    for i, monos in enumerate(p):
        s = posy_value(monos, x) if ok else 0.0
        if s - 1.0 > worst:
            worst, worst_i = s - 1.0, i
    objf = posy_value(obj, x) if ok else 0.0
    okall = ok and worst < 1e-6 and abs(objf - math.exp(topt)) < 1e-6 * (1.0 + objf)
    print(f"transformer_design  n={NGP} m={MCON}  loss = {objf:.6f}")
    print(f"  Np={x[VNP]:.4f} Ns={x[VNS]:.4f} Nls={x[VNLS]:.4f} B={x[VB]:.6f} J={x[VJ]:.4e} hbar={x[VHB]:.4f}")
    print(f"  worst posynomial <= 1: {worst:.3e} (slot {worst_i})   exp(t)={math.exp(topt):.6f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
