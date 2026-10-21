#!/usr/bin/env python3
"""PrimalSolver - least-squares regression as a second-order cone program.

Port of c_examples/socp_robust.c (a complex / scaling example):

    min ||A x - b||_2   <=>   min t  s.t.  (t, r) in QUAD,  r = A x - b.

Verified against the normal equations: at the optimum the residual r = A x - b
satisfies ||r|| = t (cone tight) and A'r = 0 (stationarity), plus strong duality.

    python python_examples/socp_robust.py
"""

import math

from primalsolver import Model, BK, CT, SOLSTA

M, D = 300, 40


class RNG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.s >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    rng = RNG(16180339)
    A = [[rng.next() * 2.0 - 1.0 for _ in range(D)] for _ in range(M)]
    b = [rng.next() * 2.0 - 1.0 for _ in range(M)]

    x0, r0, T = 0, D, D + M
    with Model(maxcon=M, maxvar=D + M + 1) as m:
        for j in range(D):
            m.var_bounds(x0 + j, BK.FR)
        for i in range(M):
            m.var_bounds(r0 + i, BK.FR)
        m.var_bounds(T, BK.LO, 0.0)
        m.cj(T, 1.0)
        for i in range(M):
            m.a_row(i, [x0 + j for j in range(D)] + [r0 + i],
                       [-A[i][j] for j in range(D)] + [1.0])
            m.con_bounds(i, BK.FX, -b[i], -b[i])
        m.cone(CT.QUAD, [T] + [r0 + i for i in range(M)])
        r = m.solve()

    x = r.x[:D]
    res = [sum(A[i][j] * x[j] for j in range(D)) - b[i] for i in range(M)]
    norm = math.sqrt(sum(e * e for e in res))
    grad = [sum(A[i][j] * res[i] for i in range(M)) for j in range(D)]
    gnorm = math.sqrt(sum(e * e for e in grad))
    ok = (r.solsta == SOLSTA.OPTIMAL
          and abs(r.x[T] - norm) < 1e-4 * (1.0 + norm)      # cone tight: t = ||r||
          and gnorm < 1e-4 * (1.0 + math.sqrt(sum(v * v for v in b)))  # A'r = 0
          and abs(r.objective - r.dual) < 1e-4 * (1.0 + abs(r.objective)))
    print(f"socp_ls m={M} d={D} cone={M + 1} obj={r.objective:.6f} "
          f"||r||={norm:.6f} ||A'r||={gnorm:.2e} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
