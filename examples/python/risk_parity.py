#!/usr/bin/env python3
"""Equal risk contribution portfolio via exponential cones.

    min sqrt(x'Sx) - c sum_i log(x_i)
    conic form: (t, G'x) in Q,  (x_i, 1, s_i) in EXP (s_i <= log x_i),  min t - c sum s_i

with S = F F' + diag(D), G = [F, sqrt(D)]. After rescaling to sum(x) = 1 the risk
contributions x_i (Sx)_i are equal (verified). The data come from the same 32-bit
linear congruential generator as the C example.

    python risk_parity.py [n] [k] [c]   (default 20 4 0.01)

Port of examples/c/risk_parity.c.
"""

import math
import sys
import time

from primalsolver import Model, BK, CT

_state = 909090


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if n < 2 or k < 1:
        return 2
    c = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    F, D = [0.0] * (n * k), [0.0] * n
    for i in range(n):
        D[i] = 0.001 + rnd() * 0.004
        for j in range(k):
            F[i * k + j] = (rnd() - 0.5) * 0.2
    m2 = k + n
    X0, Z0, TV, S0, ONE = 0, n, n + m2, n + m2 + 1, n + m2 + 1 + n   # x (n), z (m2), t, s (n), one
    with Model() as t:
        t.add_vars(n + m2 + 1 + n + 1)
        t.add_cons(m2)
        for i in range(n):
            t.var_bounds(X0 + i, BK.RA, 1e-4, 1e4)
        for j in range(m2):
            t.var_bounds(Z0 + j, BK.FR, -math.inf, math.inf)
        t.var_bounds(TV, BK.LO, 0.0, math.inf)
        t.cj(TV, 1.0)
        for i in range(n):
            # big-M on the log variable: keeps the round-0 tangent LP bounded
            t.var_bounds(S0 + i, BK.RA, -30.0, 10.0)
            t.cj(S0 + i, -c)
        t.var_bounds(ONE, BK.FX, 1.0, 1.0)
        for j in range(m2):                         # z_j - (G'x)_j = 0
            sub, val = [], []
            for i in range(n):
                g = F[i * k + j] if j < k else (math.sqrt(D[i]) if j - k == i else 0.0)
                if g != 0.0:
                    sub.append(X0 + i); val.append(-g)
            sub.append(Z0 + j); val.append(1.0)
            t.a_row(j, sub, val)
            t.con_bounds(j, BK.FX, 0.0, 0.0)
        t.cone(CT.QUAD, [TV] + list(range(Z0, Z0 + m2)))   # (t, z) in QUAD
        for i in range(n):                          # x_i >= exp(s_i)
            t.cone(CT.PEXP, [X0 + i, ONE, S0 + i])
        t0 = time.perf_counter()
        r = t.solve()
        sec = time.perf_counter() - t0
    ok = r.rc == 0
    spread = 0.0
    if ok:
        x = r.x[:n]
        s = 0.0                                     # rescale to sum 1
        for i in range(n):
            s += x[i]
        x = [xi / s for xi in x]
        Sx = [0.0] * n
        for i in range(n):
            for a2 in range(k):
                ss = 0.0
                for l in range(n):
                    ss += F[l * k + a2] * x[l]
                Sx[i] += F[i * k + a2] * ss
            Sx[i] += D[i] * x[i]
        rcmin, rcmax = 1e300, -1e300
        for i in range(n):
            rci = x[i] * Sx[i]
            if rci < rcmin:
                rcmin = rci
            if rci > rcmax:
                rcmax = rci
        spread = (rcmax / rcmin - 1.0) if rcmin > 0 else 1.0
        ok = spread < 1e-3                          # risk contributions equal within 0.1%
    print("riskparity n=%d k=%d RC spread=%.3e rc=%d t=%.4fs %s" % (n, k, spread, r.rc, sec, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
