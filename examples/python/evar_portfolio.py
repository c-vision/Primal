#!/usr/bin/env python3
"""Entropic value-at-risk portfolio (MOSEK Portfolio Optimization Cookbook, chapter 8).

Maximize m'x - delta EVaR_alpha(loss) over the simplex, with

    EVaR_alpha(L) = inf_{s>0} s (log(sum_i p_i exp(L_i/s)) - log(1-alpha))

modelled with one exponential cone per scenario plus a linear row:
(u_i, s, L_i - z) in K_exp, sum p_i u_i <= s, EVaR = z - s log(1-alpha). Checked
against a brute force over the simplex with EVaR minimized by golden section.
Port of examples/c/evar_portfolio.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

NA = 2                                              # assets
NS = 4                                              # scenarios
RET = [[0.10, 0.02], [-0.05, 0.04], [0.20, 0.01], [-0.10, 0.03]]
PROB = [0.25, 0.25, 0.25, 0.25]
ALPHA = 0.95
DELTA = 1.0


def loss(i, x):
    """Loss in scenario i for weights x: L_i = -sum_a R[i][a] x[a]."""
    r = 0.0
    for a in range(NA):
        r += RET[i][a] * x[a]
    return -r


def evar_g(s, x):
    """g(s) = s (logsumexp(L/s) - log(1-alpha)), written stably around max(L)."""
    mx = loss(0, x)
    for i in range(1, NS):
        l = loss(i, x)
        if l > mx:
            mx = l
    acc = 0.0
    for i in range(NS):
        acc += PROB[i] * math.exp((loss(i, x) - mx) / s)
    return mx + s * (math.log(acc) - math.log(1.0 - ALPHA))


def evar(x):
    """EVaR by golden section on log(s)."""
    a, b = -40.0, 12.0                              # log(s) bracket (the infimum may be at s -> 0)
    gr = 0.6180339887498949
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = evar_g(math.exp(c), x), evar_g(math.exp(d), x)
    for _ in range(200):
        if fc < fd:
            b = d; d = c; fd = fc; c = b - gr * (b - a); fc = evar_g(math.exp(c), x)
        else:
            a = c; c = d; fc = fd; d = a + gr * (b - a); fd = evar_g(math.exp(d), x)
    return evar_g(math.exp(0.5 * (a + b)), x)


def mean_return(xr):
    """Expected return of the weights xr."""
    ret = 0.0
    for a in range(NA):
        mr = 0.0
        for i in range(NS):
            mr += PROB[i] * RET[i][a]
        ret += mr * xr[a]
    return ret


def evar_conic():
    """The conic model: max m'x - delta EVaR; returns (ok, objective, x, evar)."""
    X0, Z, S, U, V = 0, NA, NA + 1, NA + 2, NA + 2 + NS
    NV = NA + 2 + 2 * NS
    R_BUD, R_ULIN, R_V = 0, 1, 2
    mvec = [0.0] * NA
    for a in range(NA):
        for i in range(NS):
            mvec[a] += PROB[i] * RET[i][a]
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(2 + NS)
        for a in range(NA):
            m.var_bounds(X0 + a, BK.LO, 0.0, math.inf)
        m.var_bounds(Z, BK.FR, -math.inf, math.inf)
        m.var_bounds(S, BK.LO, 0.0, math.inf)
        for i in range(NS):
            m.var_bounds(U + i, BK.LO, 0.0, math.inf)
        for i in range(NS):
            m.var_bounds(V + i, BK.FR, -math.inf, math.inf)
        m.a_row(R_BUD, list(range(X0, X0 + NA)), [1.0] * NA)   # budget
        m.con_bounds(R_BUD, BK.FX, 1.0, 1.0)
        m.a_row(R_ULIN, list(range(U, U + NS)) + [S], PROB + [-1.0])   # sum p_i u_i - s <= 0
        m.con_bounds(R_ULIN, BK.UP, -math.inf, 0.0)
        for i in range(NS):                         # v_i + R x + z = 0 (v_i = loss_i - z)
            m.a_row(R_V + i, [V + i] + list(range(X0, X0 + NA)) + [Z], [1.0] + RET[i] + [1.0])
            m.con_bounds(R_V + i, BK.FX, 0.0, 0.0)
        for i in range(NS):
            m.cone(CT.PEXP, [U + i, S, V + i])
        for a in range(NA):                         # max m'x - delta z + delta log(1-alpha) s
            m.cj(X0 + a, mvec[a])
        m.cj(Z, -DELTA)
        m.cj(S, DELTA * math.log(1.0 - ALPHA))
        m.sense(SENSE.MAX)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, [0.0] * NA, 0.0
    x = r.x
    obj = 0.0
    for a in range(NA):
        obj += mvec[a] * x[X0 + a]
    obj -= DELTA * (x[Z] - x[S] * math.log(1.0 - ALPHA))
    return True, obj, x[X0:X0 + NA], x[Z] - x[S] * math.log(1.0 - ALPHA)


def main() -> int:
    ok, obj, x, ev = evar_conic()
    best, bw = -1e300, 0.0                          # reference: brute force over the simplex
    for k in range(20001):
        w = k / 20000.0
        xr = [w, 1.0 - w]
        f = mean_return(xr) - DELTA * evar(xr)
        if f > best:
            best, bw = f, w
    ret_sol = mean_return(x)
    ev_direct = evar(x)
    okall = (ok and abs(obj - best) < 1e-3 * (1.0 + abs(best)) and abs(ev - ev_direct) < 1e-3 * (1.0 + abs(ev_direct))
             and abs(x[0] + x[1] - 1.0) < 1e-7 and x[0] >= -1e-9 and x[1] >= -1e-9)
    print("evar_portfolio  N=%d T=%d alpha=%.2f delta=%.2f" % (NA, NS, ALPHA, DELTA))
    print("  x = (%.8f, %.8f)   return = %.8f" % (x[0], x[1], ret_sol))
    print("  EVaR = %.8f   [brute force obj %.8f at w=%.5f]" % (ev, best, bw))
    print("  conic objective = %.8f   [brute force %.8f]" % (obj, best))
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
