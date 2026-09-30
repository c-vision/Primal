#!/usr/bin/env python3
"""PrimalSolver - large Markowitz QP with a factor risk model.

Port of c_examples/portfolio_large.c (a complex / scaling example):

    max  r'x - gamma * x'Sx    s.t.  sum x = 1,  x >= 0
    factor model:  S = F F' + diag(D),  so x'Sx is quadratic.

The objective is written as r'x + 0.5 x'Qx with Q = -2*gamma*S (the solver's
convention). Verified: budget, nonnegativity and strong duality pobj ~ dobj.

    python python_examples/portfolio_large.py
"""

from primalsolver import Model, BK, SENSE, SOLSTA

N, K, GAMMA = 100, 5, 0.5


class RNG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(31415926)
    ret, D, F = [], [], []
    for _ in range(N):
        ret.append(rng.next() * 0.2 - 0.05)
        D.append(0.01 + rng.next() * 0.02)
        F.append([rng.next() - 0.5 for _ in range(K)])

    with Model(maxcon=1, maxvar=N) as m:
        m.sense(SENSE.MAX)
        for i in range(N):
            m.cj(i, ret[i])
            m.var_bounds(i, BK.LO, 0.0, 1.0)
        terms = []
        for i in range(N):
            for j in range(i, N):
                s = sum(F[i][l] * F[j][l] for l in range(K))
                if i == j:
                    s += D[i]
                v = -2.0 * GAMMA * s
                if v != 0.0:
                    terms.append((i, j, v))
        m.qobj(terms)
        m.a_row(0, list(range(N)), [1.0] * N)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        r = m.solve()

    x = r.x
    budget = sum(x)
    minx = min(x)
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(budget - 1.0) < 1e-6 and minx > -1e-6
          and abs(r.objective - r.dual) < 1e-4 * (1.0 + abs(r.objective)))
    print(f"portfolio n={N} k={K} obj={r.objective:.6f} dobj={r.dual:.6f} "
          f"budget={budget:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
