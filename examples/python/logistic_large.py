#!/usr/bin/env python3
"""Sum-of-exponentials regression via exponential cones (scaling test for the conic
path with many PEXP cones).

    min sum_i v_i + 0.5*gamma*||x||^2
    s.t. v_i >= exp(a_i'x + c_i)      (PEXP cone per i)

The quadratic term is absorbed exactly into a RQUAD cone, keeping the whole model in
the conic path. Verified against gradient descent on
f(x) = sum_i exp(a_i'x + c_i) + 0.5*gamma*||x||^2. The data come from the same
32-bit linear congruential generator as the C example.

    python logistic_large.py [n] [d]   (default 40 8)

Port of examples/c/logistic_large.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, CT

GAMMA = 0.1
_state = 11235813


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def fval(a_rows, c, x):
    """Objective value of the sum-of-exponentials model at x."""
    s = 0.0
    for ai, ci in zip(a_rows, c):
        u = ci
        for aij, xj in zip(ai, x):
            u += aij * xj
        s += math.exp(u)
    for xj in x:
        s += 0.5 * GAMMA * xj * xj
    return s


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if n < 1 or d < 1:
        return 2
    a = [(rnd() * 2.0 - 1.0) * 0.5 for _ in range(n * d)]
    c = [rnd() * 2.0 - 1.0 for _ in range(n)]
    a_rows = [a[i * d:(i + 1) * d] for i in range(n)]
    x0, v0, u0 = 0, d, d + n                        # vars: x (D), v (N), u (N), one, w
    one = d + 2 * n
    w = one + 1
    with Model() as m:
        m.add_vars(d + 2 * n + 2)
        for j in range(d):
            m.var_bounds(x0 + j, BK.FR, -math.inf, math.inf)
        for i in range(n):
            m.var_bounds(v0 + i, BK.LO, 0.0, math.inf)
            m.cj(v0 + i, 1.0)
            m.var_bounds(u0 + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(one, BK.FX, 1.0, 1.0)
        m.var_bounds(w, BK.LO, 0.0, math.inf)
        m.cj(w, GAMMA)
        m.add_cons(n)
        for i in range(n):                          # a_i'x - u_i = -c_i
            m.a_row(i, [x0 + j for j in range(d)] + [u0 + i], a_rows[i] + [-1.0])
            m.con_bounds(i, BK.FX, -c[i], -c[i])
        for i in range(n):
            m.cone(CT.PEXP, [v0 + i, one, u0 + i])
        m.cone(CT.RQUAD, [w, one] + [x0 + j for j in range(d)])   # w >= 0.5*||x||^2
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    if r.rc != 0:
        print(f"logistic   n={n} d={d} rc={r.rc} FAIL")
        return 1
    obj = r.objective
    xg = [0.0] * d                                  # gradient descent reference
    fg, step = fval(a_rows, c, xg), 0.5
    for _ in range(20000):
        g = []
        for j in range(d):
            h = 1e-6
            xp, xm = list(xg), list(xg)
            xp[j] += h
            xm[j] -= h
            g.append((fval(a_rows, c, xp) - fval(a_rows, c, xm)) / (2 * h))
        xn = [xg[j] - step * g[j] for j in range(d)]
        fn = fval(a_rows, c, xn)
        if fn < fg:
            xg, fg = xn, fn
            step *= 1.05
        elif fn < fg + 1e-12:
            step *= 0.7
        else:
            step *= 0.5
        if step < 1e-12:
            break
    ok = abs(obj - fg) < 1e-3 * (1.0 + abs(fg))
    print(f"logistic   n={n} d={d} cones={n} obj={obj:.6f} grad={fg:.6f} t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
