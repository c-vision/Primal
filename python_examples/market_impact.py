#!/usr/bin/env python3
"""PrimalSolver - Markowitz with market impact (power cones).

Port of c_examples/finance/market_impact.c:

    max mu'x - delta' t
    s.t. (t_i, 1, x_i) in POW^{2/3,1/3}   (t_i >= x_i^{3/2})
         (sqrt(gamma), G'x) in QUAD       (risk <= sqrt(gamma))
         e'x = 1, x >= 0.

Verified: t_i >= x_i^{3/2}, budget, nonnegativity, risk bound.

    python python_examples/market_impact.py [n] [k] [gamma] [delta]  (default 15 4 .05 .5)
"""

import math
import sys

from primalsolver import Model, BK, CT, SENSE, SOLSTA

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
GAMMA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
DELTA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(778899)
    mu, F, D = [], [], []
    for _ in range(N):
        mu.append(0.02 + rng.next() * 0.15)
        D.append(0.001 + rng.next() * 0.004)
        F.append([(rng.next() - 0.5) * 0.2 for _ in range(K)])
    m2 = K + N

    x0, t0, z0, tc, one = 0, N, 2 * N, 2 * N + m2, 2 * N + m2 + 1
    with Model(maxcon=m2 + 1, maxvar=2 * N + m2 + 2) as m:
        m.sense(SENSE.MAX)
        for i in range(N):
            m.cj(x0 + i, mu[i]); m.var_bounds(x0 + i, BK.LO, 0.0, 1.0)
            m.cj(t0 + i, -DELTA); m.var_bounds(t0 + i, BK.LO, 0.0)
        for j in range(m2):
            m.var_bounds(z0 + j, BK.FR)
        m.var_bounds(tc, BK.FX, math.sqrt(GAMMA))
        m.var_bounds(one, BK.FX, 1.0)
        for j in range(m2):
            cols, vals = [], []
            for i in range(N):
                g = F[i][j] if j < K else (math.sqrt(D[i]) if j - K == i else 0.0)
                if g != 0.0:
                    cols.append(x0 + i); vals.append(-g)
            cols.append(z0 + j); vals.append(1.0)
            m.a_row(j, cols, vals)
            m.con_bounds(j, BK.FX, 0.0, 0.0)
        m.a_row(m2, [x0 + i for i in range(N)], [1.0] * N)
        m.con_bounds(m2, BK.FX, 1.0, 1.0)
        m.cone(CT.QUAD, [tc] + [z0 + j for j in range(m2)])
        for i in range(N):
            m.cone(CT.PPOW, [t0 + i, one, x0 + i], param=2.0 / 3.0)
        r = m.solve()

    x = r.x
    budget = sum(x[x0 + i] for i in range(N))
    minx = min(x[x0 + i] for i in range(N))
    powslack = max(x[x0 + i] ** 1.5 - x[t0 + i] for i in range(N))
    risk2 = sum(D[i] * x[x0 + i] ** 2 for i in range(N))
    for a in range(K):
        s = sum(F[i][a] * x[x0 + i] for i in range(N))
        risk2 += s * s
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(budget - 1.0) < 1e-5 and minx > -1e-6
          and powslack < 1e-4 and risk2 <= GAMMA * (1.0 + 1e-3))
    print(f"mktimpact n={N} k={K} gamma={GAMMA:.4f} delta={DELTA:.2f} obj={r.objective:.6f} "
          f"powslack={powslack:.2e} risk2={risk2:.5f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
