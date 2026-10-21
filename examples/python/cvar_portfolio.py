#!/usr/bin/env python3
"""PrimalSolver - CVaR portfolio (Rockafellar-Uryasev) as an LP.

Port of c_examples/finance/cvar_portfolio.c:

    min  t + 1/(alpha*S) sum_s u_s
    s.t. u_s >= -(r_s'x) - t,  u_s >= 0
         e'x = 1,  x >= 0

Verified: the LP objective equals the empirical CVaR of x* (mean of the worst
alpha-fraction scenario losses), computed directly.

    python python_examples/cvar_portfolio.py
"""

import math

from primalsolver import Model, BK, SOLSTA

N, S, ALPHA = 40, 200, 0.10


class RNG:
    """32-bit LCG, identical to the C samples' `rnd()`."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(606060)
    R = [[(rng.next() - 0.5) * 0.2 + 0.01 for _ in range(N)] for _ in range(S)]

    x0, tv, u0 = 0, N, N + 1
    with Model(maxcon=S + 1, maxvar=N + 1 + S) as m:
        for i in range(N):
            m.var_bounds(x0 + i, BK.LO, 0.0, 1.0)
        m.var_bounds(tv, BK.FR)
        m.cj(tv, 1.0)
        for s in range(S):
            m.var_bounds(u0 + s, BK.LO, 0.0)
            m.cj(u0 + s, 1.0 / (ALPHA * S))
            m.a_row(s, [x0 + i for i in range(N)] + [tv, u0 + s], R[s] + [1.0, 1.0])
            m.con_bounds(s, BK.LO, 0.0)
        m.a_row(S, [x0 + i for i in range(N)], [1.0] * N)
        m.con_bounds(S, BK.FX, 1.0, 1.0)
        r = m.solve()

    x = r.x[:N]
    budget = sum(x)
    minx = min(x)
    losses = sorted((-sum(R[s][i] * x[i] for i in range(N)) for s in range(S)), reverse=True)
    ktail = int(math.ceil(ALPHA * S))
    cvar = sum(losses[:ktail]) / ktail
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(budget - 1.0) < 1e-6 and minx > -1e-6
          and abs(r.objective - cvar) < 1e-4 * (1.0 + abs(cvar)))
    print(f"cvar n={N} S={S} alpha={ALPHA:.2f} obj={r.objective:.6f} "
          f"empirical={cvar:.6f} budget={budget:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
