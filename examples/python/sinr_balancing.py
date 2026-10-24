#!/usr/bin/env python3
"""Max-min SINR power control (uplink), by bisection on t.

Source: MOSEK Modeling Cookbook / MOSEK Tutorials, "SINR". With per-user power p_i,
channel gains G, noise sigma_i^2 and a total power budget P:

    SINR_i = G_ii p_i / ( sum_{j!=i} G_ij p_j + sigma_i^2 )
    max t  s.t.  SINR_i >= t  for all i,  sum_i p_i <= P,  p >= 0.

For a FIXED t the constraint is linear in p, so the optimal t is the largest t for
which an LP is feasible (bisection). Symmetric instance G = [[1, 0.1], [0.1, 1]],
sigma^2 = (1, 1), P = 1: hand-derived optimum t* = 10/21.
Port of examples/c/sinr_balancing.c.
"""

import math

from primalsolver import Model, BK


def csum(values, start=0.0):
    """Left-to-right sum, as the C loop (sum() is compensated since Python 3.12)."""
    s = start
    for v in values:
        s += v
    return s


N = 2
G = [[1.0, 0.1], [0.1, 1.0]]
SIGMA2 = [1.0, 1.0]
P = 1.0


def feasible(t):
    """Feasibility LP for a fixed t: returns (feasible, p)."""
    with Model() as m:
        m.add_vars(N)
        m.add_cons(N + 1)
        for i in range(N):
            m.var_bounds(i, BK.LO, 0.0, math.inf)
        for i in range(N):
            m.cj(i, 0.0)                            # feasibility only
        for i in range(N):                          # p_i - t*sum_{j!=i} G_ij p_j >= t*sigma_i^2
            m.a_row(i, list(range(N)), [1.0 if j == i else -t * G[i][j] for j in range(N)])
            m.con_bounds(i, BK.LO, t * SIGMA2[i], math.inf)
        m.a_row(N, list(range(N)), [1.0] * N)       # sum_i p_i <= P
        m.con_bounds(N, BK.UP, -math.inf, P)
        r = m.solve()
    return (r.rc == 0), (r.x[:N] if r.rc == 0 else None)


def main() -> int:
    lo, hi = 0.0, 100.0                             # hi: an obvious upper bound
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(mid)[0]:
            lo = mid
        else:
            hi = mid
    ok_lo, p = feasible(lo)
    if not ok_lo:
        p = [0.0, 0.0]
    want = 10.0 / 21.0                              # 0.476190...
    gmin, budget = 1e30, 0.0
    for i in range(N):
        inter = csum((G[i][j] * p[j] for j in range(N) if j != i), SIGMA2[i])
        gmin = min(gmin, G[i][i] * p[i] / inter)
        budget += p[i]
    ok = abs(lo - want) < 1e-6 and abs(gmin - lo) < 1e-6 and budget <= P + 1e-7
    print(f"sinr_balancing  t*={lo:.9f} (expected 10/21={want:.9f})  min_SINR={gmin:.9f}  "
          f"p=({p[0]:.6f},{p[1]:.6f}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
