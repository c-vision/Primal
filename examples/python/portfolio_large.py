#!/usr/bin/env python3
"""Large Markowitz QP with a factor risk model (scaling test).

    max r'x - gamma x'Sx  s.t.  sum x = 1,  x >= 0,  S = F F' + diag(D)

Verified: budget, nonnegativity, strong duality. The data come from the same 32-bit
linear congruential generator as the C example.

    python portfolio_large.py [n] [k]   (default 100 5)

Port of examples/c/portfolio_large.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, SENSE

_state = 31415926


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if n < 1 or k < 1:
        return 2
    gamma = 0.5
    r, F, D = [0.0] * n, [0.0] * (n * k), [0.0] * n
    for i in range(n):
        r[i] = rnd() * 0.2 - 0.05
        D[i] = 0.01 + rnd() * 0.02
        for j in range(k):
            F[i * k + j] = rnd() - 0.5
    # objective (MAXIMIZE): r'x - gamma x'Sx = r'x + 0.5 x'Qx with Q = -2 gamma S
    Q = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += F[i * k + l] * F[j * k + l]
            if i == j:
                s += D[i]
            Q[i * n + j] = -2.0 * gamma * s
    with Model() as m:
        m.add_vars(n)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        for i in range(n):
            m.cj(i, r[i])
            m.var_bounds(i, BK.LO, 0.0, 1.0)
        m.qobj([(i, j, Q[i * n + j]) for i in range(n) for j in range(i, n) if Q[i * n + j] != 0.0])
        m.a_row(0, list(range(n)), [1.0] * n)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        t0 = time.perf_counter()
        res = m.solve()
        sec = time.perf_counter() - t0
    obj, dobj = 0.0, 0.0
    ok = res.rc == 0
    if ok:
        obj, dobj, x = res.objective, res.dual, res.x
        budget, minx = 0.0, 0.0
        for i in range(n):
            budget += x[i]
            if x[i] < minx:
                minx = x[i]
        ok = abs(budget - 1.0) < 1e-6 and minx > -1e-6 and abs(obj - dobj) < 1e-4 * (1.0 + abs(obj))
    print("portfolio  n=%d k=%d vars=%d nnzQ=%d obj=%.6f dobj=%.6f t=%.4fs %s"
          % (n, k, n, n * (n + 1) // 2, obj, dobj, sec, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
