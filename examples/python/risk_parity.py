#!/usr/bin/env python3
"""PrimalSolver - equal risk contribution portfolio (exponential cones).

Port of c_examples/finance/risk_parity.c:

    min sqrt(x'Sx) - c*sum_i log(x_i)   (x_i > 0)
    conic:  (t, G'x) in QUAD (t >= sqrt(x'Sx), S = F F' + diag(D), G = [F, sqrt(D)])
            (x_i, 1, s_i) in EXP   (x_i >= exp(s_i)),   min t - c*sum_i s_i.

The minimiser is scale-invariant; after rescaling to sum(x)=1 the risk
contributions x_i (Sx)_i are equal (checked within 0.1%).

    python python_examples/risk_parity.py [n] [k]      (default 20 4)
"""

import math
import sys

from primalsolver import Model, BK, CT, SOLSTA

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
C = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(909090)
    F, D = [], []
    for _ in range(N):
        D.append(0.001 + rng.next() * 0.004)
        F.append([(rng.next() - 0.5) * 0.2 for _ in range(K)])
    m2 = K + N

    x0, z0, tv, s0, one = 0, N, N + m2, N + m2 + 1, N + m2 + 1 + N
    with Model(maxcon=m2, maxvar=N + m2 + 1 + N + 1) as m:
        for i in range(N):
            m.var_bounds(x0 + i, BK.RA, 1e-4, 1e4)
        for j in range(m2):
            m.var_bounds(z0 + j, BK.FR)
        m.var_bounds(tv, BK.LO, 0.0); m.cj(tv, 1.0)
        for i in range(N):
            m.var_bounds(s0 + i, BK.RA, -30.0, 10.0); m.cj(s0 + i, -C)
        m.var_bounds(one, BK.FX, 1.0, 1.0)
        for j in range(m2):
            cols, vals = [], []
            for i in range(N):
                g = F[i][j] if j < K else (math.sqrt(D[i]) if j - K == i else 0.0)
                if g != 0.0:
                    cols.append(x0 + i); vals.append(-g)
            cols.append(z0 + j); vals.append(1.0)
            m.a_row(j, cols, vals)
            m.con_bounds(j, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [tv] + [z0 + j for j in range(m2)])
        for i in range(N):
            m.cone(CT.PEXP, [x0 + i, one, s0 + i])
        r = m.solve()

    x = r.x[:N]
    s = sum(x)
    x = [v / s for v in x]
    Sx = [0.0] * N
    for i in range(N):
        for a in range(K):
            Sx[i] += F[i][a] * sum(F[l][a] * x[l] for l in range(N))
        Sx[i] += D[i] * x[i]
    rc = [x[i] * Sx[i] for i in range(N)]
    spread = max(rc) / min(rc) - 1.0 if min(rc) > 0 else 1.0
    ok = r.solsta == SOLSTA.OPTIMAL and spread < 1e-3
    print(f"riskparity n={N} k={K} RC spread={spread:.3e} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
