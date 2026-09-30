#!/usr/bin/env python3
"""PrimalSolver - the exact `lsq_pos` form of MosekRegression (SOCP + QP).

Port of c_examples/finance/lsq_pos.c:

    min ||X w - rhs||_2  (not squared)   s.t. e'w = 1, 0 <= w <= 1,

cross-checked against the QP-equivalent min ||Xw - rhs||^2, whose optimum must
equal v*^2.

    python python_examples/lsq_pos.py [n] [m]     (default 40 5)
"""

import math
import sys

from primalsolver import Model, BK, CT, SOLSTA

N, M = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (40, 5)


class RNG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def solve_socp(X, rhs):
    w0, r0, v = 0, M, M + N
    with Model(maxcon=N + 1, maxvar=M + N + 1) as m:
        for j in range(M):
            m.var_bounds(w0 + j, BK.RA, 0.0, 1.0)
        for i in range(N):
            m.var_bounds(r0 + i, BK.FR)
        m.var_bounds(v, BK.FR)
        m.cj(v, 1.0)
        for i in range(N):
            cols, vals = [], []
            for j in range(M):
                if X[i][j] != 0.0:
                    cols.append(w0 + j); vals.append(-X[i][j])
            cols.append(r0 + i); vals.append(1.0)
            m.a_row(i, cols, vals)
            m.con_bounds(i, BK.FX, -rhs[i], -rhs[i])
        m.a_row(N, [w0 + j for j in range(M)], [1.0] * M)
        m.con_bounds(N, BK.FX, 1.0, 1.0)
        m.cone(CT.QUAD, [v] + [r0 + i for i in range(N)])
        r = m.solve()
    return r.objective, r.x[:M], r.solsta


def solve_qp(X, rhs):
    with Model(maxcon=1, maxvar=M) as m:
        for j in range(M):
            m.var_bounds(j, BK.RA, 0.0, 1.0)
        for j in range(M):
            m.cj(j, -2.0 * sum(X[i][j] * rhs[i] for i in range(N)))
        m.qobj([(j, k, 2.0 * sum(X[i][j] * X[i][k] for i in range(N)))
                for j in range(M) for k in range(j, M)])
        m.a_row(0, list(range(M)), [1.0] * M)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        r = m.solve()
    return r.objective, r.x[:M], r.solsta


def main() -> int:
    rng = RNG(13103397)
    X = [[(rng.next() - 0.5) * 2.0 for _ in range(M)] for _ in range(N)]
    rhs = [(rng.next() - 0.5) * 2.0 for _ in range(N)]

    v, ws, _ = solve_socp(X, rhs)
    q, _, _ = solve_qp(X, rhs)
    budget = sum(ws)
    lo, hi = min(ws), max(ws)
    rhs2 = sum(r * r for r in rhs)
    v2_from_qp = q + rhs2
    ok = (v >= 0.0 and abs(budget - 1.0) <= 1e-6 and lo >= -1e-6 and hi <= 1.0 + 1e-6
          and abs(v * v - v2_from_qp) <= 1e-5 * (1.0 + abs(v2_from_qp)))
    print(f"lsq_pos n={N} m={M} v*={v:.8f} sqrt(QP)={math.sqrt(abs(v2_from_qp)):.8f} "
          f"e'w={budget:.2e} w in [{lo:.3f},{hi:.3f}] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
