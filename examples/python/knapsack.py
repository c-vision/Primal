#!/usr/bin/env python3
"""0/1 knapsack MILP (scaling test).

    max sum_i p_i x_i   s.t.  sum_i w_i x_i <= C,  x_i in {0,1}

Verified against an exact O(nC) dynamic program. The data come from the same 32-bit
linear congruential generator as the C example.

    python knapsack.py [n] [C]   (default 120 3000)

Port of examples/c/knapsack.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, SENSE, VAR_TYPE

_state = 24681357


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    cap_max = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    if n < 1 or cap_max < 1:
        return 2
    w, p = [], []
    for _ in range(n):
        w.append(1 + int(rnd() * 100.0))
        p.append(1 + int(rnd() * 100.0))
    with Model() as m:
        m.add_vars(n)
        m.add_cons(1)
        m.sense(SENSE.MAX)
        for i in range(n):
            m.cj(i, float(p[i]))
            m.var_bounds(i, BK.RA, 0.0, 1.0)
            m.var_type(i, VAR_TYPE.INT_BIN)
        m.a_row(0, list(range(n)), [float(v) for v in w])
        m.con_bounds(0, BK.UP, -math.inf, float(cap_max))
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    if r.rc != 0:
        print(f"knapsack   n={n} C={cap_max} rc={r.rc} FAIL")
        return 1
    obj, x = r.objective, r.x
    ok = True
    cap, val = 0, 0.0
    for i in range(n):
        xi = int(math.floor(x[i] + 0.5))
        if abs(x[i] - xi) > 1e-6:
            ok = False
        cap += xi * w[i]
        val += float(xi) * p[i]
    if cap > cap_max or abs(val - obj) > 1e-6:
        ok = False
    dp = [0] * (cap_max + 1)                        # exact DP reference
    for i in range(n):
        for c in range(cap_max, w[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - w[i]] + p[i])
    dpbest = float(dp[cap_max])
    if abs(obj - dpbest) > 1e-6:
        ok = False
    print(f"knapsack   n={n} C={cap_max} vars={n} obj={obj:.0f} dp={dpbest:.0f} t={sec:.4f}s "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
