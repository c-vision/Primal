#!/usr/bin/env python3
"""Markowitz portfolio as a second-order cone program.

    max mu'x  s.t.  ||G'x||_2 <= sqrt(gamma),  e'x = 1,  x >= 0

with the factor model Sigma = F F' + diag(D), G = [F, sqrt(D)]. Verified: budget,
nonnegativity, risk bound; when the risk is not binding the optimum is the best-mean
asset. The data come from the same 32-bit linear congruential generator as the C example.

    python markowitz_conic.py [n] [k] [gamma]   (default 60 4 0.05)

Port of examples/c/markowitz_conic.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, CT, SENSE

_state = 42424242


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    if n < 1 or k < 1 or gamma <= 0.0:
        return 2
    mu, F, D = [0.0] * n, [0.0] * (n * k), [0.0] * n
    for i in range(n):
        mu[i] = 0.02 + rnd() * 0.15
        D[i] = 0.001 + rnd() * 0.004
        for j in range(k):
            F[i * k + j] = (rnd() - 0.5) * 0.2
    m2 = k + n                                      # G = [F | diag(sqrt(D))]
    X0, Z0, TC = 0, n, n + m2                       # vars: x (n), z (m2), tcone (1)
    with Model() as t:
        t.add_vars(n + m2 + 1)
        t.add_cons(m2)
        t.sense(SENSE.MAX)
        for i in range(n):
            t.cj(X0 + i, mu[i])
            t.var_bounds(X0 + i, BK.LO, 0.0, 1.0)
        for j in range(m2):
            t.var_bounds(Z0 + j, BK.FR, -math.inf, math.inf)
        t.var_bounds(TC, BK.FX, math.sqrt(gamma), math.sqrt(gamma))
        for j in range(m2):                         # z_j - (G'x)_j = 0
            sub, val = [], []
            for i in range(n):
                g = F[i * k + j] if j < k else (math.sqrt(D[i]) if j - k == i else 0.0)
                if g != 0.0:
                    sub.append(X0 + i); val.append(-g)
            sub.append(Z0 + j); val.append(1.0)
            t.a_row(j, sub, val)
            t.con_bounds(j, BK.FX, 0.0, 0.0)
        t.cone(CT.QUAD, [TC] + list(range(Z0, Z0 + m2)))   # (tcone, z) in QUAD
        t.add_cons(1)                               # budget
        t.a_row(m2, list(range(X0, X0 + n)), [1.0] * n)
        t.con_bounds(m2, BK.FX, 1.0, 1.0)
        t0 = time.perf_counter()
        r = t.solve()
        sec = time.perf_counter() - t0
    obj = 0.0
    ok = r.rc == 0
    if ok:
        obj, x = r.objective, r.x
        budget, minx, risk2, maxmu = 0.0, 0.0, 0.0, mu[0]
        for i in range(n):
            budget += x[i]
            if x[i] < minx:
                minx = x[i]
            if mu[i] > maxmu:
                maxmu = mu[i]
            risk2 += D[i] * x[i] * x[i]
        for a2 in range(k):
            s = 0.0
            for i in range(n):
                s += F[i * k + a2] * x[i]
            risk2 += s * s
        ok = abs(budget - 1.0) < 1e-6 and minx > -1e-6 and risk2 <= gamma * (1.0 + 1e-4)
        # when the risk budget is not binding the optimum is the best asset
        if ok and risk2 < gamma * (1.0 - 1e-3):
            ok = abs(obj - maxmu) < 1e-5 * (1.0 + maxmu)
    print("markowitz  n=%d k=%d gamma=%.4f obj=%.6f t=%.4fs %s" % (n, k, gamma, obj, sec, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
