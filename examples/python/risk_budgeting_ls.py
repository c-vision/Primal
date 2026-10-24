#!/usr/bin/env python3
"""LONG-SHORT risk budgeting, mixed-integer conic (MOSEK Portfolio Optimization Cookbook, ch9).

    minimize s - a sum_i b_i t_i
    s.t.  s >= 1/2 x'Sx  ((s, 1, G'x) in Qr),  t_i <= log|x_i|  ((|x_i|, 1, t_i) in K_exp)
          x = xp - xm,  xp <= M yp,  xm <= M ym,  yp + ym <= 1,  y binary

The binaries keep |x_i| = xp_i + xm_i exact (without them the log term is unbounded).
Hand case S = I, a = 1, b = (1/2, 1/2): |x_i| = 1/sqrt(2), obj = 0.5 + 0.34657359.
Port of examples/c/risk_budgeting_ls.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE


def rb_ls(N, G, b, a):
    """Solve for N assets with S = G G' (G row-major); returns (ok, objective, |x|)."""
    # var layout: x(N) xp(N) xm(N) yp(N) ym(N) t(N) s(1) z(N) u(N) ONE(1)
    X0, XP, XM, YP, YM, T0, S0 = 0, N, 2 * N, 3 * N, 4 * N, 5 * N, 6 * N
    Z0, U0, ONE = 6 * N + 1, 7 * N + 1, 8 * N + 1
    nv = ONE + 1
    with Model() as m:
        m.add_vars(nv)
        m.sense(SENSE.MIN)
        for i in range(N):
            m.var_bounds(X0 + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(XP + i, BK.LO, 0.0, math.inf)
            m.var_bounds(XM + i, BK.LO, 0.0, math.inf)
            m.var_type(YP + i, VAR_TYPE.INT)        # binary via [0,1] + INT
            m.var_bounds(YP + i, BK.RA, 0.0, 1.0)
            m.var_type(YM + i, VAR_TYPE.INT)
            m.var_bounds(YM + i, BK.RA, 0.0, 1.0)
            m.var_bounds(T0 + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(Z0 + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(U0 + i, BK.LO, 0.0, math.inf)
        m.var_bounds(S0, BK.FR, -math.inf, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        m.add_cons(6 * N)
        bigM = 20.0
        for i in range(N):
            m.a_row(i, [X0 + i, XP + i, XM + i], [1.0, -1.0, 1.0])
            m.con_bounds(i, BK.FX, 0.0, 0.0)                   # x - xp + xm = 0
            m.a_row(N + i, [XP + i, YP + i], [1.0, -bigM])
            m.con_bounds(N + i, BK.UP, -math.inf, 0.0)         # xp <= M yp
            m.a_row(2 * N + i, [XM + i, YM + i], [1.0, -bigM])
            m.con_bounds(2 * N + i, BK.UP, -math.inf, 0.0)     # xm <= M ym
            m.a_row(3 * N + i, [YP + i, YM + i], [1.0, 1.0])
            m.con_bounds(3 * N + i, BK.UP, -math.inf, 1.0)     # yp + ym <= 1
            sub, val = [Z0 + i], [1.0]                         # z_i - (G'x)_i = 0
            for k in range(N):
                if G[k * N + i] != 0.0:
                    sub.append(X0 + k); val.append(-G[k * N + i])
            m.a_row(4 * N + i, sub, val)
            m.con_bounds(4 * N + i, BK.FX, 0.0, 0.0)
            m.a_row(5 * N + i, [U0 + i, XP + i, XM + i], [1.0, -1.0, -1.0])
            m.con_bounds(5 * N + i, BK.FX, 0.0, 0.0)           # u - xp - xm = 0
        m.cone(CT.RQUAD, [S0, ONE] + list(range(Z0, Z0 + N)))   # s >= ||z||^2/2 = 1/2 x'Sx
        for i in range(N):                          # t_i <= log(u_i) = log|x_i|
            m.cone(CT.PEXP, [U0 + i, ONE, T0 + i])
        m.cj(S0, 1.0)
        for i in range(N):
            m.cj(T0 + i, -a * b[i])
        r = m.solve()
    if r.rc != 0:
        return False, -1.0, [0.0] * N
    return True, r.objective, [abs(r.x[X0 + i]) for i in range(N)]


def main() -> int:
    print("risk_budgeting_ls (long-short MIO)")
    allok = True
    N = 2                                           # hand case N=2, G=I, a=1, b=(1/2,1/2)
    ok, obj, xabs = rb_ls(N, [1.0, 0.0, 0.0, 1.0], [0.5, 0.5], 1.0)
    want = 0.5 + 0.34657359027997264                # 0.5 - log(1/sqrt2)
    xw = 1.0 / math.sqrt(2.0)
    good = ok and abs(obj - want) < 1e-4 and abs(xabs[0] - xw) < 1e-3 and abs(xabs[1] - xw) < 1e-3
    print("  N=2 S=I: obj=%.6f (expected %.6f)  |x|=(%.6f,%.6f) (expected %.6f)  %s"
          % (obj, want, xabs[0], xabs[1], xw, "OK" if good else "FAIL"))
    allok &= good
    print("OK" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
