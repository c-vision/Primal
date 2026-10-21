#!/usr/bin/env python3
"""PrimalSolver - entropic value-at-risk portfolio (exponential cone).

Port of c_examples/finance/evar_portfolio.c (MOSEK Portfolio Optimization
Cookbook, ch. 8). Maximises m'x - delta*EVaR_alpha(loss) over x >= 0, sum x = 1,
with EVaR modelled by n exponential cones (PRIMAL_CT_PEXP) plus a linear row:

    (u_i, s, L_i - z) in K_exp;   sum_i p_i u_i <= s;   EVaR = z - s log(1-alpha).

Small deterministic instance (2 assets, 4 equiprobable scenarios), checked
against a brute force over the weight simplex with EVaR minimised by golden
section (independent of the conic solver).

    python python_examples/evar_portfolio.py
"""

import math

from primalsolver import Model, BK, CT, SENSE, SOLSTA

NA, NS = 2, 4
RET = [[0.10, 0.02], [-0.05, 0.04], [0.20, 0.01], [-0.10, 0.03]]
PROB = [0.25, 0.25, 0.25, 0.25]
ALPHA, DELTA = 0.95, 1.0


def loss(i, x):
    return -sum(RET[i][a] * x[a] for a in range(NA))


def evar_g(s, x):
    mx = max(loss(i, x) for i in range(NS))
    acc = sum(PROB[i] * math.exp((loss(i, x) - mx) / s) for i in range(NS))
    return mx + s * (math.log(acc) - math.log(1.0 - ALPHA))


def evar(x):
    a, b = -40.0, 12.0
    gr = 0.6180339887498949
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = evar_g(math.exp(c), x), evar_g(math.exp(d), x)
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = evar_g(math.exp(c), x)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = evar_g(math.exp(d), x)
    return evar_g(math.exp(0.5 * (a + b)), x)


def main() -> int:
    x0, z, s, u, v = 0, NA, NA + 1, NA + 2, NA + 2 + NS
    nv = NA + 2 + 2 * NS
    with Model(maxcon=2 + NS, maxvar=nv) as m:
        for a in range(NA):
            m.var_bounds(x0 + a, BK.LO, 0.0)
        m.var_bounds(z, BK.FR)
        m.var_bounds(s, BK.LO, 0.0)
        for i in range(NS):
            m.var_bounds(u + i, BK.LO, 0.0)
            m.var_bounds(v + i, BK.FR)
        # budget
        m.a_row(0, [x0 + a for a in range(NA)], [1.0] * NA)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        # sum p_i u_i - s <= 0
        m.a_row(1, [u + i for i in range(NS)] + [s], PROB + [-1.0])
        m.con_bounds(1, BK.UP, up=0.0)
        # v_i + sum R[i][a] x_a + z = 0
        for i in range(NS):
            m.a_row(2 + i, [v + i] + [x0 + a for a in range(NA)] + [z],
                       [1.0] + RET[i] + [1.0])
            m.con_bounds(2 + i, BK.FX, 0.0, 0.0)
        for i in range(NS):
            m.cone(CT.PEXP, [u + i, s, v + i])
        mu = [sum(PROB[i] * RET[i][a] for i in range(NS)) for a in range(NA)]
        for a in range(NA):
            m.cj(x0 + a, mu[a])
        m.cj(z, -DELTA)
        m.cj(s, DELTA * math.log(1.0 - ALPHA))
        m.sense(SENSE.MAX)
        r = m.solve()

    x = [r.x[x0], r.x[x0 + 1]]
    ev = r.x[z] - r.x[s] * math.log(1.0 - ALPHA)
    obj = sum(mu[a] * x[a] for a in range(NA)) - DELTA * ev

    best, bw = -1e300, 0.0
    for k in range(20001):
        w = k / 20000.0
        xr = [w, 1.0 - w]
        ret = sum(sum(PROB[i] * RET[i][a] for i in range(NS)) * xr[a] for a in range(NA))
        f = ret - DELTA * evar(xr)
        if f > best:
            best, bw = f, w
    ev_direct = evar(x)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(obj - best) < 1e-3 * (1.0 + abs(best))
          and abs(ev - ev_direct) < 1e-3 * (1.0 + abs(ev_direct))
          and abs(x[0] + x[1] - 1.0) < 1e-7 and min(x) >= -1e-9)
    print(f"evar_portfolio N={NA} T={NS} alpha={ALPHA} x=({x[0]:.6f},{x[1]:.6f}) "
          f"EVaR={ev:.6f} obj={obj:.6f} [brute {best:.6f} at w={bw:.5f}] "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
