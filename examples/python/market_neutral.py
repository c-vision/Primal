#!/usr/bin/env python3
"""Market-neutral long-short portfolio with leverage, turnover and a variance cap
(fy23-art/Portfolio-Optimization, MOSEK):

    maximize m'x - cost_rate sum_i (d+_i + d-_i)
    s.t.  factor'x = 0,  sum_i |x_i| <= leverage,  x'Sx <= gamma^2,  x - x0 = d+ - d-

Two assets with factor = (1, 1), so x = (t, -t) and every value is hand-derived:
A) no cost -> t = Tmax (the risk/leverage bound binds); B) large cost -> t = t0 (no
trade). A dense 1-D brute force confirms both. Port of examples/c/market_neutral.c.
"""

import math

from primalsolver import Model, BK, SENSE

NA = 2
S = [[0.04, 0.01], [0.01, 0.09]]                    # covariance
MU = [0.002, 0.0]                                   # alpha
GAMMA2 = 0.01                                       # 0.1^2
LEV = 2.0                                           # sum |x| <= L


def tmax():
    """Feasible bound on t from sum|x| <= L and x'Sx <= gamma^2 (x = (t, -t))."""
    q = S[0][0] - 2.0 * S[0][1] + S[1][1]           # x'Sx = t^2 q
    tr, tg = LEV / 2.0, math.sqrt(GAMMA2 / q)
    return tr if tr < tg else tg


def solve_mn(c, t0):
    """Conic model; returns (ok, objective, x)."""
    XP, Z, DP, DM, NV = 0, NA, 2 * NA, 3 * NA, 4 * NA
    R_MN, R_ZP, R_ZN, R_LEV, R_TL, R_RSK = 0, 1, 1 + NA, 1 + 2 * NA, 2 + 2 * NA, 2 + 3 * NA
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(3 * NA + 3)
        for i in range(NA):
            m.var_bounds(XP + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(Z + i, BK.LO, 0.0, math.inf)
            m.var_bounds(DP + i, BK.LO, 0.0, math.inf)
            m.var_bounds(DM + i, BK.LO, 0.0, math.inf)
        m.a_row(R_MN, [XP, XP + 1], [1.0, 1.0])     # factor'x = 0, factor = (1,1)
        m.con_bounds(R_MN, BK.FX, 0.0, 0.0)
        for i in range(NA):                         # z_i >= x_i and z_i >= -x_i
            m.a_row(R_ZP + i, [Z + i, XP + i], [1.0, -1.0])
            m.con_bounds(R_ZP + i, BK.LO, 0.0, math.inf)
            m.a_row(R_ZN + i, [Z + i, XP + i], [1.0, 1.0])
            m.con_bounds(R_ZN + i, BK.LO, 0.0, math.inf)
        m.a_row(R_LEV, [Z, Z + 1], [1.0, 1.0])
        m.con_bounds(R_LEV, BK.UP, -math.inf, LEV)
        for i in range(NA):                         # x_i - d+_i + d-_i = x0_i
            x0 = t0 if i == 0 else -t0
            m.a_row(R_TL + i, [XP + i, DP + i, DM + i], [1.0, -1.0, 1.0])
            m.con_bounds(R_TL + i, BK.FX, x0, x0)
        # risk: 1/2 x'Qx <= gamma^2 with Q = 2S
        m.qconk(R_RSK, [(0, 0, 2.0 * S[0][0]), (1, 1, 2.0 * S[1][1]), (0, 1, 2.0 * S[0][1])])
        m.con_bounds(R_RSK, BK.UP, -math.inf, GAMMA2)
        for i in range(NA):
            m.cj(XP + i, MU[i])
            m.cj(DP + i, -c)
            m.cj(DM + i, -c)
        m.sense(SENSE.MAX)
        r = m.solve()
    if r.rc != 0:
        return False, 0.0, [0.0] * NA
    x = r.x
    obj = MU[0] * x[XP] + MU[1] * x[XP + 1] - c * (x[DP] + x[DP + 1] + x[DM] + x[DM + 1])
    return True, obj, x[XP:XP + 2]


def obj_at(c, t0, t):
    """True objective at x = (t, -t): a t - c sum_i |x_i - x0_i|."""
    return (MU[0] - MU[1]) * t - c * (abs(t - t0) + abs(-t + t0))


def main() -> int:
    T, a = tmax(), MU[0] - MU[1]
    allok = True
    print("market_neutral  N=%d  Tmax=%.8f  a=m1-m2=%.6f" % (NA, T, a))
    cs, t0s = [0.0, 0.002], [0.1, 0.1]
    for k in range(2):
        c, t0 = cs[k], t0s[k]
        ok, obj, x = solve_mn(c, t0)
        # hand value: t* = Tmax if a > 2c, -Tmax if a < -2c, else clamp(t0)
        if a > 2.0 * c:
            tstar = T
        elif a < -2.0 * c:
            tstar = -T
        else:
            tstar = -T if t0 < -T else (T if t0 > T else t0)
        ref = obj_at(c, t0, tstar)
        best = -1e300                               # dense 1-D brute force
        for j in range(200001):
            f = obj_at(c, t0, -T + 2.0 * T * j / 200000.0)
            if f > best:
                best = f
        okk = (ok and abs(obj - ref) < 1e-6 and abs(obj - best) < 1e-6 and abs(x[0] + x[1]) < 1e-7
               and abs(x[0] - tstar) < 1e-5)
        allok = allok and okk
        print("  case %c: c=%.4f t0=%.2f  x=(%.8f, %.8f)  obj=%.10f  [closed-form %.10f]"
              % (chr(ord('A') + k), c, t0, x[0], x[1], obj, ref))
    print("OK" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
