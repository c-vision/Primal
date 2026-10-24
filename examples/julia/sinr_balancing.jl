#!/usr/bin/env julia
# sinr_balancing.jl — max-min SINR power control (uplink), by bisection on t
# (MOSEK Modeling Cookbook / Tutorials, "SINR"). With powers p_i, gains G,
# noise sigma_i^2 and a total budget P:
#
#   SINR_i = G_ii p_i / (sum_{j!=i} G_ij p_j + sigma_i^2)
#   max t  s.t.  SINR_i >= t,  sum_i p_i <= P,  p >= 0
#
# For a FIXED t the constraint is linear in p, so the optimal t is the largest
# t whose LP is feasible (bisection). Instance G = [1 0.1; 0.1 1],
# sigma^2 = (1, 1), P = 1: t* = 10/21. Port of examples/c/sinr_balancing.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 2
const G = [1.0 0.1; 0.1 1.0]
const sigma2 = [1.0, 1.0]
const P = 1.0

# Feasibility LP for a fixed t: returns (feasible, p).
function feasible(t)
    m = Model(N + 1, N)
    r = try
        for i in 0:N-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:N-1
            cj(m, i, 0.0)                           # feasibility only
        end
        row = 0
        for i in 0:N-1                              # p_i - t*sum_{j!=i} G_ij p_j >= t*sigma_i^2
            a_row(m, row, collect(0:N-1), [j == i ? 1.0 : -t * G[i+1, j+1] for j in 0:N-1])
            con_bounds(m, row, BK_LO; lo=t * sigma2[i+1], up=Inf)
            row += 1
        end
        a_row(m, row, collect(0:N-1), ones(N))      # sum_i p_i <= P
        con_bounds(m, row, BK_UP; lo=-Inf, up=P)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, r.x
end

function main()
    lo, hi = 0.0, 100.0                             # hi: an obvious upper bound
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        feasible(mid)[1] ? (lo = mid) : (hi = mid)
    end
    ok_lo, pl = feasible(lo)
    p = ok_lo ? pl : [0.0, 0.0]

    want = 10.0 / 21.0                              # 0.476190...
    gmin, budget = 1e30, 0.0
    for i in 1:N
        inter = sigma2[i]
        for j in 1:N
            j != i && (inter += G[i, j] * p[j])
        end
        s = G[i, i] * p[i] / inter
        s < gmin && (gmin = s)
        budget += p[i]
    end
    ok = abs(lo - want) < 1e-6 && abs(gmin - lo) < 1e-6 && budget <= P + 1e-7
    @printf("sinr_balancing  t*=%.9f (expected 10/21=%.9f)  min_SINR=%.9f  p=(%.6f,%.6f) %s\n",
            lo, want, gmin, p[1], p[2], ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
