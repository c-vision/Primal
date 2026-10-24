#!/usr/bin/env python3
"""No-arbitrage price bounds as an LP.

In a one-period model a probability vector q >= 0, sum q = 1, reproducing the traded
prices is a risk-neutral measure; an option's arbitrage-free price bounds are the min
and max of E_q[payoff] over those measures -- two LPs that differ only in the sense.
Hand case: stock S = (0.5, 1.0, 1.5) traded at 1.0, so q0 = q2 <= 1/2; a call struck at
1.0 pays (0, 0, 0.5): price 0.5 q2, bounds [0, 0.25]. Port of examples/c/option_pricing.c.
"""

import math

from primalsolver import Model, BK, SENSE

STOCK = [0.5, 1.0, 1.5]


def solve_bound(maximize):
    """Solve the risk-neutral LP for the call price bound (min or max)."""
    with Model() as m:
        m.add_vars(3)
        m.add_cons(2)
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])  # sum q = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], STOCK)            # E_q[S] = 1 (the traded stock)
        m.con_bounds(1, BK.FX, 1.0, 1.0)
        m.cj(2, 0.5)                            # the call payoff max(S-1, 0) = (0, 0, 0.5)
        if maximize:
            m.sense(SENSE.MAX)
        r = m.solve()
    return r.objective if r.rc == 0 else 0.0


def main() -> int:
    lo = solve_bound(False)
    hi = solve_bound(True)
    ok = abs(lo - 0.0) < 1e-9 and abs(hi - 0.25) < 1e-9
    print("call price K=1: [%.6f, %.6f] (expected [0, 0.25])" % (lo, hi))
    print("%s (no-arbitrage bounds as two LPs)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
