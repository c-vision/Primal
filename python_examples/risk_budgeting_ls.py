#!/usr/bin/env python3
"""PrimalSolver - LONG-SHORT risk budgeting, mixed-integer conic.

Port of c_examples/finance/risk_budgeting_ls.c (MOSEK Cookbook ch9):

    min s - a sum_i b_i t_i
    s.t. (s, 1, G'x) in Qr;  (|x_i|, 1, t_i) in K_exp;
         x = xp - xm;  xp <= M yp;  xm <= M ym;  yp + ym <= 1;  yp,ym binary.

Hand case S=I, a=1, b=(1/2,1/2): |x_i| = 1/sqrt(2), s = 1/2,
obj = 0.5 - log(1/sqrt2).   python python_examples/risk_budgeting_ls.py
"""

import math

from primalsolver import Model, BK, CT, VAR_TYPE, SOLSTA


def rb_ls(N, G, b, a, M=20.0):
    x0, xp, xm, yp, ym = 0, N, 2 * N, 3 * N, 4 * N
    t0, s0, z0, u0, one = 5 * N, 6 * N, 6 * N + 1, 7 * N + 1, 8 * N + 1
    nv = one + 1
    nrow = 6 * N
    with Model(maxcon=nrow, maxvar=nv) as m:
        for i in range(N):
            m.var_bounds(x0 + i, BK.FR)
            m.var_bounds(xp + i, BK.LO, 0.0)
            m.var_bounds(xm + i, BK.LO, 0.0)
            m.var_bounds(yp + i, BK.RA, 0.0, 1.0); m.var_type(yp + i, VAR_TYPE.INT)
            m.var_bounds(ym + i, BK.RA, 0.0, 1.0); m.var_type(ym + i, VAR_TYPE.INT)
            m.var_bounds(t0 + i, BK.FR)
            m.var_bounds(z0 + i, BK.FR)
            m.var_bounds(u0 + i, BK.LO, 0.0)
        m.var_bounds(s0, BK.FR)
        m.var_bounds(one, BK.FX, 1.0)
        for i in range(N):
            m.a_row(i, [x0 + i, xp + i, xm + i], [1.0, -1.0, 1.0]); m.con_bounds(i, BK.FX, 0, 0)
            m.a_row(N + i, [xp + i, yp + i], [1.0, -M]); m.con_bounds(N + i, BK.UP, up=0.0)
            m.a_row(2 * N + i, [xm + i, ym + i], [1.0, -M]); m.con_bounds(2 * N + i, BK.UP, up=0.0)
            m.a_row(3 * N + i, [yp + i, ym + i], [1.0, 1.0]); m.con_bounds(3 * N + i, BK.UP, up=1.0)
            cols, vals = [z0 + i], [1.0]
            for k in range(N):
                if G[k][i] != 0.0:
                    cols.append(x0 + k); vals.append(-G[k][i])
            m.a_row(4 * N + i, cols, vals); m.con_bounds(4 * N + i, BK.FX, 0, 0)
            m.a_row(5 * N + i, [u0 + i, xp + i, xm + i], [1.0, -1.0, -1.0]); m.con_bounds(5 * N + i, BK.FX, 0, 0)
        m.cone(CT.RQUAD, [s0, one] + [z0 + i for i in range(N)])
        for i in range(N):
            m.cone(CT.PEXP, [u0 + i, one, t0 + i])
        m.cj(s0, 1.0)
        for i in range(N):
            m.cj(t0 + i, -a * b[i])
        r = m.solve()
    xabs = [abs(r.x[x0 + i]) for i in range(N)]
    return r.objective, xabs, r.solsta


def main() -> int:
    N = 2
    G = [[1.0, 0.0], [0.0, 1.0]]
    b = [0.5, 0.5]
    obj, xabs, sta = rb_ls(N, G, b, 1.0)
    want = 0.5 + 0.34657359027997264      # 0.5 - log(1/sqrt2)
    xw = 1.0 / math.sqrt(2.0)
    ok = (sta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL) and abs(obj - want) < 1e-4
          and abs(xabs[0] - xw) < 1e-3 and abs(xabs[1] - xw) < 1e-3)
    print(f"risk_budgeting_ls obj={obj:.6f} (expected {want:.6f}) "
          f"|x|=({xabs[0]:.6f},{xabs[1]:.6f}) (expected {xw:.6f}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
