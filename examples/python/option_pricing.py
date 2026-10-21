#!/usr/bin/env python3
"""PrimalSolver - no-arbitrage option price bounds as two LPs.

Port of c_examples/finance/option_pricing.c. A risk-neutral measure is a
q >= 0 with sum q = 1 reproducing the traded prices (E_q[S] = price). The call
price lower/upper bounds are the min/max of E_q[payoff] over that polytope.

Hand case: 3 states, stock (0.5, 1.0, 1.5) traded at 1.0; call struck at 1.0
pays (0, 0, 0.5), so its price is 0.5 q2 with bounds [0, 0.25].

    python python_examples/option_pricing.py
"""

from primalsolver import Model, BK, SENSE

STOCK = [0.5, 1.0, 1.5]


def solve(maximize: bool) -> float:
    with Model(maxcon=2, maxvar=3) as m:
        for j in range(3):
            m.var_bounds(j, BK.LO, 0.0)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [0, 1, 2], STOCK)
        m.con_bounds(1, BK.FX, 1.0, 1.0)
        m.cj(2, 0.5)                     # call payoff max(S-1, 0) = (0, 0, 0.5)
        if maximize:
            m.sense(SENSE.MAX)
        return m.solve().objective


def main() -> int:
    lo, hi = solve(False), solve(True)
    ok = abs(lo - 0.0) < 1e-9 and abs(hi - 0.25) < 1e-9
    print(f"call price K=1: [{lo:.6f}, {hi:.6f}] (expected [0, 0.25])")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
