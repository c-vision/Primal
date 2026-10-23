#!/usr/bin/env python3
"""Distributionally robust portfolio (Wasserstein DRO), LP (MOSEK/Tutorials,
dist-robust-portfolio, the "program in 9" model).

Given N training scenarios and a Wasserstein radius eps, minimize the worst-case
(within the ball) expected loss:

    minimize    eps * lambda + (1/N) sum_i s_i
    subject to  b_k * t + a_k * (data_i . x) <= s_i     for all i, k in {0,1}
                | a_k * x_j | <= lambda                   for all j, k
                sum_j x_j = 1,  x >= 0

with the loss pieces a = (-1,-51), b = (10,-40) as in the notebook. It is checked
against a brute force over the weight simplex with t minimized on a grid. The grid
is the C example's; since the objective is convex piecewise-linear in t, the
minimum over the t grid is found by bisection instead of a full scan (the same
grid point, 100x fewer evaluations). Port of examples/c/dist_robust.c.
"""

import math

from primalsolver import Model, BK, SENSE

M = 2
NS = 3
AK = (-1.0, -51.0)
BKK = (10.0, -40.0)
DATA = [[0.02, 0.01], [-0.01, 0.03], [0.04, -0.02]]
EPS = 0.05
X, S, LAM, TT = 0, M, M + NS, M + NS + 1
NV = M + NS + 2


def worst_objective(x, t):
    """Worst-case DRO objective at weights x with auxiliary t."""
    lam = 0.0
    for j in range(M):
        lam = max(lam, abs(AK[0] * x[j]), abs(AK[1] * x[j]))
    ssum = 0.0
    for i in range(NS):
        d = 0.0
        for j in range(M):
            d += DATA[i][j] * x[j]
        s0, s1 = BKK[0] * t + AK[0] * d, BKK[1] * t + AK[1] * d
        ssum += s0 if s0 > s1 else s1
    return EPS * lam + ssum / NS


def grid_min(x, lo=-2000, hi=2000):
    """min over t = 0.01*b, b in [lo, hi], of the convex worst_objective(x, t)."""
    f = lambda b: worst_objective(x, b * 0.01)      # noqa: E731
    while hi - lo > 2:                              # bisection on the discrete slope
        mid = (lo + hi) // 2
        if f(mid) <= f(mid + 1):
            hi = mid + 1
        else:
            lo = mid
    return min(f(b) for b in range(lo, hi + 1))


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(2 * NS + 4 * M + 1)
        for j in range(M):
            m.var_bounds(X + j, BK.LO, 0.0, math.inf)
        for i in range(NS):
            m.var_bounds(S + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(LAM, BK.FR, -math.inf, math.inf)
        m.var_bounds(TT, BK.FR, -math.inf, math.inf)
        row = 0
        for i in range(NS):                         # C1 for both k
            for k in range(2):
                m.a_row(row, [TT] + [X + j for j in range(M)] + [S + i],
                        [BKK[k]] + [AK[k] * DATA[i][j] for j in range(M)] + [-1.0])
                m.con_bounds(row, BK.UP, -math.inf, 0.0)
                row += 1
        for j in range(M):                          # C2 positive/negative
            for k in range(2):
                m.a_row(row, [X + j, LAM], [AK[k], -1.0]); m.con_bounds(row, BK.UP, -math.inf, 0.0); row += 1
                m.a_row(row, [X + j, LAM], [-AK[k], -1.0]); m.con_bounds(row, BK.UP, -math.inf, 0.0); row += 1
        m.a_row(row, [X + j for j in range(M)], [1.0] * M)
        m.con_bounds(row, BK.FX, 1.0, 1.0)
        m.cj(LAM, EPS)
        for i in range(NS):
            m.cj(S + i, 1.0 / NS)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    obj, x = 0.0, [0.0] * M
    if ok:
        x = [r.x[X + j] for j in range(M)]
        obj = EPS * r.x[LAM]
        for i in range(NS):
            obj += r.x[S + i] / NS

    best = 1e300                                    # brute force: x = (w, 1-w), t on the grid
    for a in range(4001):
        w = a / 4000.0
        best = min(best, grid_min([w, 1.0 - w]))
    okall = ok and abs(obj - best) < 2e-3 * (1.0 + abs(best)) and abs(x[0] + x[1] - 1.0) < 1e-7
    print(f"dist_robust  m={M} N={NS} eps={EPS:.2f}")
    print(f"  x = ({x[0]:.8f}, {x[1]:.8f})   LP objective = {obj:.8f}")
    print(f"  brute force = {best:.8f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
