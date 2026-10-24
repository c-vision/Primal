#!/usr/bin/env python3
"""CVaR portfolio optimization (Rockafellar-Uryasev) as an LP.

    min t + 1/(alpha S) sum_s u_s  s.t.  u_s >= -(r_s'x) - t,  u >= 0,  e'x = 1,  x >= 0

Verified: the LP objective equals the empirical CVaR of x* (mean of the worst
alpha-fraction of the scenario losses). The returns come from the same 32-bit linear
congruential generator as the C example.

    python cvar_portfolio.py [n] [S] [alpha]   (default 40 200 0.10)

Port of examples/c/cvar_portfolio.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 606060


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    if n < 1 or S < 1 or alpha <= 0.0 or alpha >= 1.0:
        return 2
    R = [(rnd() - 0.5) * 0.2 + 0.01 for _ in range(S * n)]   # scenario by scenario
    X0, TV, U0 = 0, n, n + 1                        # vars: x (n), t (1), u (S)
    with Model() as m:
        m.add_vars(n + 1 + S)
        m.add_cons(S + 1)
        for i in range(n):
            m.var_bounds(X0 + i, BK.LO, 0.0, 1.0)
        m.var_bounds(TV, BK.FR, -math.inf, math.inf)
        m.cj(TV, 1.0)
        for s in range(S):
            m.var_bounds(U0 + s, BK.LO, 0.0, math.inf)
            m.cj(U0 + s, 1.0 / (alpha * S))
            # u_s + r_s'x + t >= 0
            m.a_row(s, list(range(X0, X0 + n)) + [TV, U0 + s], R[s * n:s * n + n] + [1.0, 1.0])
            m.con_bounds(s, BK.LO, 0.0, math.inf)
        m.a_row(S, list(range(X0, X0 + n)), [1.0] * n)   # budget
        m.con_bounds(S, BK.FX, 1.0, 1.0)
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    ok = r.rc == 0
    if ok:
        obj, x = r.objective, r.x
        budget, minx = 0.0, 0.0
        for i in range(n):
            budget += x[i]
            if x[i] < minx:
                minx = x[i]
        loss = []                                   # empirical CVaR of x*: mean of the worst alpha*S losses
        for s in range(S):
            l = 0.0
            for i in range(n):
                l -= R[s * n + i] * x[i]
            loss.append(l)
        loss.sort(reverse=True)
        ktail = math.ceil(alpha * S)
        cvar = 0.0
        for s in range(ktail):
            cvar += loss[s]
        cvar /= float(ktail)
        ok = abs(budget - 1.0) < 1e-6 and minx > -1e-6 and abs(obj - cvar) < 1e-4 * (1.0 + abs(cvar))
        print("cvar       n=%d S=%d alpha=%.2f obj=%.6f empirical=%.6f t=%.4fs %s"
              % (n, S, alpha, obj, cvar, sec, "OK" if ok else "FAIL"))
    else:
        print("cvar       n=%d S=%d rc=%d FAIL" % (n, S, r.rc))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
