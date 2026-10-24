#!/usr/bin/env julia
# option_pricing.jl — no-arbitrage price bounds as an LP.
#
# In a one-period model with states s = 0..S-1, a probability vector q >= 0
# with sum q = 1 reproducing the traded prices (E_q[payoff_k] = price_k) is a
# risk-neutral measure; an option's arbitrage-free price bounds are the min
# and max of E_q[payoff] over those measures — two LPs differing only in sense.
#
# Hand case: stock S = (0.5, 1.0, 1.5) traded at 1.0, so q0 = q2 <= 1/2. A call
# struck at 1.0 pays (0, 0, 0.5): price 0.5 q2, bounds [0, 0.25].
# Port of examples/c/option_pricing.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const STOCK = [0.5, 1.0, 1.5]

# Solve the risk-neutral LP for the call price bound (min or max).
function solve_bound(maximize)
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 2)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # sum q = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [0, 1, 2], STOCK)              # E_q[S] = 1 (the traded stock)
        con_bounds(m, 1, BK_FX; lo=1.0, up=1.0)
        cj(m, 2, 0.5)                               # the call payoff max(S-1, 0) = (0, 0, 0.5)
        maximize && sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0 ? r.objective : 0.0
end

function main()
    lo = solve_bound(false)
    hi = solve_bound(true)
    ok = abs(lo - 0.0) < 1e-9 && abs(hi - 0.25) < 1e-9
    @printf("call price K=1: [%.6f, %.6f] (expected [0, 0.25])\n", lo, hi)
    @printf("%s (no-arbitrage bounds as two LPs)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
