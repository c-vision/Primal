#!/usr/bin/env python3
"""Markowitz with market impact (power cones).

    max mu'x - delta't
    s.t.  (t_i, 1, x_i) in POW^{2/3,1/3} (t_i >= x_i^{3/2}),  (sqrt(gamma), G'x) in Q,
          e'x = 1,  x >= 0

with Sigma = F F' + diag(D), G = [F, sqrt(D)]. Verified: power-cone feasibility,
budget, nonnegativity, risk bound. The data come from the same 32-bit linear
congruential generator as the C example.

    python market_impact.py [n] [k] [gamma] [delta]   (default 15 4 0.05 0.5)

Port of examples/c/market_impact.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, CT, SENSE

_state = 778899


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    if n < 1 or k < 1 or gamma <= 0.0:
        return 2
    mu, F, D = [0.0] * n, [0.0] * (n * k), [0.0] * n
    for i in range(n):
        mu[i] = 0.02 + rnd() * 0.15
        D[i] = 0.001 + rnd() * 0.004
        for j in range(k):
            F[i * k + j] = (rnd() - 0.5) * 0.2
    m2 = k + n
    X0, T0, Z0, TC, ONE = 0, n, 2 * n, 2 * n + m2, 2 * n + m2 + 1   # x (n), t (n), z (m2), tc, one
    with Model() as t:
        t.add_vars(2 * n + m2 + 2)
        t.add_cons(n + m2 + 1)
        t.sense(SENSE.MAX)
        for i in range(n):
            t.cj(X0 + i, mu[i])
            t.var_bounds(X0 + i, BK.LO, 0.0, 1.0)
            t.cj(T0 + i, -delta)
            t.var_bounds(T0 + i, BK.LO, 0.0, math.inf)
        for j in range(m2):
            t.var_bounds(Z0 + j, BK.FR, -math.inf, math.inf)
        t.var_bounds(TC, BK.FX, math.sqrt(gamma), math.sqrt(gamma))
        t.var_bounds(ONE, BK.FX, 1.0, 1.0)
        for j in range(m2):                         # z_j = (G'x)_j
            sub, val = [], []
            for i in range(n):
                g = F[i * k + j] if j < k else (math.sqrt(D[i]) if j - k == i else 0.0)
                if g != 0.0:
                    sub.append(X0 + i); val.append(-g)
            sub.append(Z0 + j); val.append(1.0)
            t.a_row(j, sub, val)
            t.con_bounds(j, BK.FX, 0.0, 0.0)
        t.a_row(m2, list(range(X0, X0 + n)), [1.0] * n)   # budget
        t.con_bounds(m2, BK.FX, 1.0, 1.0)
        t.cone(CT.QUAD, [TC] + list(range(Z0, Z0 + m2)))  # risk cone
        for i in range(n):                          # (t_i, one, x_i) in POW^{2/3,1/3}
            t.cone(CT.PPOW, [T0 + i, ONE, X0 + i], 2.0 / 3.0)
        t0 = time.perf_counter()
        r = t.solve()
        sec = time.perf_counter() - t0
    ok = r.rc == 0
    if ok:
        obj, x = r.objective, r.x
        budget, minx, maxpow, risk2 = 0.0, 0.0, 0.0, 0.0
        for i in range(n):
            xi = x[X0 + i]
            budget += xi
            if xi < minx:
                minx = xi
            slack = math.pow(xi, 1.5) - x[T0 + i]   # t >= x^1.5
            if slack > maxpow:
                maxpow = slack
            risk2 += D[i] * xi * xi
        for a2 in range(k):
            s = 0.0
            for i in range(n):
                s += F[i * k + a2] * x[X0 + i]
            risk2 += s * s
        ok = abs(budget - 1.0) < 1e-5 and minx > -1e-6 and maxpow < 1e-4 and risk2 <= gamma * (1.0 + 1e-3)
        print("mktimpact  n=%d k=%d gamma=%.4f delta=%.2f obj=%.6f powslack=%.2e risk2=%.5f t=%.4fs %s"
              % (n, k, gamma, delta, obj, maxpow, risk2, sec, "OK" if ok else "FAIL"))
    else:
        print("mktimpact  n=%d rc=%d FAIL" % (n, r.rc))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
