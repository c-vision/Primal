#!/usr/bin/env python3
"""PrimalSolver - market-neutral long-short portfolio with a variance cap (QP).

Port of c_examples/finance/market_neutral.c:

    max  m'x - c*sum(d+ + d-)
    s.t. factor'x = 0;  sum|x| <= L;  x'Sx <= gamma^2;  x - x0 = d+ - d-, d+-,>=0.

The risk constraint is quadratic: it uses `putqconk` (1/2 x'Qx <= gamma^2 with
Q = 2S). Deterministic two-asset instance with factor=(1,1) reduces to a scalar
t (x=(t,-t)); two cases exercise the turnover penalty, checked against a dense
1-D brute force.

    python python_examples/market_neutral.py
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA

NA = 2
S = [[0.04, 0.01], [0.01, 0.09]]
M = [0.002, 0.0]
GAMMA2 = 0.01
LEV = 2.0


def tmax() -> float:
    q = S[0][0] - 2.0 * S[0][1] + S[1][1]
    return min(LEV / 2.0, math.sqrt(GAMMA2 / q))


def solve(c: float, t0: float):
    xp, z, dp, dm = 0, NA, 2 * NA, 3 * NA
    nv = 4 * NA
    r_mn, r_zp, r_zn = 0, 1, 1 + NA
    r_lev, r_tl, r_rsk = 1 + 2 * NA, 2 + 2 * NA, 2 + 3 * NA
    with Model(maxcon=3 * NA + 3, maxvar=nv) as m:
        for i in range(NA):
            m.var_bounds(xp + i, BK.FR)
            m.var_bounds(z + i, BK.LO, 0.0)
            m.var_bounds(dp + i, BK.LO, 0.0)
            m.var_bounds(dm + i, BK.LO, 0.0)
        m.a_row(r_mn, [xp, xp + 1], [1.0, 1.0]); m.con_bounds(r_mn, BK.FX, 0.0, 0.0)
        for i in range(NA):
            m.a_row(r_zp + i, [z + i, xp + i], [1.0, -1.0]); m.con_bounds(r_zp + i, BK.LO, 0.0)
            m.a_row(r_zn + i, [z + i, xp + i], [1.0, 1.0]); m.con_bounds(r_zn + i, BK.LO, 0.0)
        m.a_row(r_lev, [z, z + 1], [1.0, 1.0]); m.con_bounds(r_lev, BK.UP, up=LEV)
        for i in range(NA):
            x0i = t0 if i == 0 else -t0
            m.a_row(r_tl + i, [xp + i, dp + i, dm + i], [1.0, -1.0, 1.0])
            m.con_bounds(r_tl + i, BK.FX, x0i, x0i)
        m.qconk(r_rsk, [(0, 0, 2.0 * S[0][0]), (1, 1, 2.0 * S[1][1]), (0, 1, 2.0 * S[0][1])])
        m.con_bounds(r_rsk, BK.UP, up=GAMMA2)
        for i in range(NA):
            m.cj(xp + i, M[i]); m.cj(dp + i, -c); m.cj(dm + i, -c)
        m.sense(SENSE.MAX)
        r = m.solve()
    x = [r.x[xp], r.x[xp + 1]]
    obj = M[0] * x[0] + M[1] * x[1] - c * (r.x[dp] + r.x[dp + 1] + r.x[dm] + r.x[dm + 1])
    return r.solsta == SOLSTA.OPTIMAL, obj, x


def obj_at(c: float, t0: float, t: float) -> float:
    return (M[0] - M[1]) * t - c * (abs(t - t0) + abs(-t + t0))


def main() -> int:
    T = tmax()
    a = M[0] - M[1]
    print(f"market_neutral N={NA} Tmax={T:.8f} a={a:.6f}")
    all_ok = True
    for k, (c, t0) in enumerate([(0.0, 0.1), (0.002, 0.1)]):
        ok, obj, x = solve(c, t0)
        if a > 2.0 * c:
            tstar = T
        elif a < -2.0 * c:
            tstar = -T
        else:
            tstar = min(max(t0, -T), T)
        ref = obj_at(c, t0, tstar)
        best = max(obj_at(c, t0, -T + 2.0 * T * j / 200000.0) for j in range(200001))
        okk = (ok and abs(obj - ref) < 1e-6 and abs(obj - best) < 1e-6
               and abs(x[0] + x[1]) < 1e-7 and abs(x[0] - tstar) < 1e-5)
        all_ok = all_ok and okk
        print(f"  case {'AB'[k]}: c={c:.4f} t0={t0:.2f} x=({x[0]:.8f}, {x[1]:.8f}) "
              f"obj={obj:.10f} [closed-form {ref:.10f}]")
    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
