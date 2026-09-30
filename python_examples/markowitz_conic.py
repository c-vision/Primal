#!/usr/bin/env python3
"""PrimalSolver - Markowitz portfolio as a second-order cone program.

Port of c_examples/finance/markowitz_conic.c:

    max  mu'x   s.t.  ||G'x||_2 <= sqrt(gamma)   [(sqrt(gamma), G'x) in QUAD]
                      e'x = 1,  x >= 0
    factor model:  Sigma = F F' + diag(D),  G = [F, sqrt(D)].

Verified: budget, nonnegativity, risk <= sqrt(gamma); when the risk constraint
is inactive the optimum is the single best-mean asset (objective = max mu_i).

    python python_examples/markowitz_conic.py
"""

import math

from primalsolver import Model, BK, CT, SENSE, SOLSTA

N, K, GAMMA = 60, 4, 0.05


class RNG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(42424242)
    mu, D, F = [], [], []
    for _ in range(N):
        mu.append(0.02 + rng.next() * 0.15)
        D.append(0.001 + rng.next() * 0.004)
        F.append([(rng.next() - 0.5) * 0.2 for _ in range(K)])

    m2 = K + N
    x0, z0, tc = 0, N, N + m2
    with Model(maxcon=m2 + 1, maxvar=N + m2 + 1) as m:
        m.sense(SENSE.MAX)
        for i in range(N):
            m.cj(x0 + i, mu[i])
            m.var_bounds(x0 + i, BK.LO, 0.0, 1.0)
        for j in range(m2):
            m.var_bounds(z0 + j, BK.FR)
        m.var_bounds(tc, BK.FX, math.sqrt(GAMMA))
        for j in range(m2):                     # z_j - (G'x)_j = 0
            cols, vals = [], []
            for i in range(N):
                g = F[i][j] if j < K else (math.sqrt(D[i]) if j - K == i else 0.0)
                if g != 0.0:
                    cols.append(x0 + i)
                    vals.append(-g)
            cols.append(z0 + j)
            vals.append(1.0)
            m.a_row(j, cols, vals)
            m.con_bounds(j, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [tc] + [z0 + j for j in range(m2)])
        m.a_row(m2, [x0 + i for i in range(N)], [1.0] * N)
        m.con_bounds(m2, BK.FX, 1.0, 1.0)
        r = m.solve()

    x = r.x[:N]
    budget = sum(x)
    minx = min(x)
    risk2 = sum(D[i] * x[i] * x[i] for i in range(N))
    for a in range(K):
        s = sum(F[i][a] * x[i] for i in range(N))
        risk2 += s * s
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(budget - 1.0) < 1e-6 and minx > -1e-6
          and risk2 <= GAMMA * (1.0 + 1e-4))
    if ok and risk2 < GAMMA * (1.0 - 1e-3):
        ok = abs(r.objective - max(mu)) < 1e-5 * (1.0 + max(mu))
    print(f"markowitz n={N} k={K} gamma={GAMMA:.4f} obj={r.objective:.6f} "
          f"risk2={risk2:.6g} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
