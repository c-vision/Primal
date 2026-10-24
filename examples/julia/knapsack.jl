#!/usr/bin/env julia
# knapsack.jl — 0/1 knapsack MILP (scaling test).
#
#   max sum_i p_i x_i  s.t.  sum_i w_i x_i <= C,  x_i in {0,1}
#
# Verified against an exact O(nC) dynamic program. The data come from the same
# 32-bit linear congruential generator as the C example.
#
#   julia knapsack.jl [n] [C]   (default 120 3000)
#
# Port of examples/c/knapsack.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(24681357)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 120
    C = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 3000
    (n < 1 || C < 1) && return 2
    w, p = zeros(Int, n), zeros(Int, n)
    for i in 1:n
        w[i] = 1 + trunc(Int, rnd() * 100.0)
        p[i] = 1 + trunc(Int, rnd() * 100.0)
    end

    m = Model(0, 0)
    try
        add_vars(m, n)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        for i in 0:n-1
            cj(m, i, Float64(p[i+1]))
            var_bounds(m, i, BK_RA; lo=0.0, up=1.0)
            var_type(m, i, VT_INT_BIN)
        end
        a_row(m, 0, collect(0:n-1), Float64.(w))
        con_bounds(m, 0, BK_UP; lo=-Inf, up=Float64(C))

        t0 = time()
        r = solve(m)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj, x = r.objective, r.x
            cap, val = 0, 0.0
            for i in 1:n
                xi = floor(Int, x[i] + 0.5)
                abs(x[i] - xi) > 1e-6 && (ok = false)
                cap += xi * w[i]
                val += Float64(xi) * p[i]
            end
            cap > C && (ok = false)
            abs(val - obj) > 1e-6 && (ok = false)
            dp = zeros(Int, C + 1)                  # exact DP reference
            for i in 1:n, c in C:-1:w[i]
                cand = dp[c-w[i]+1] + p[i]
                cand > dp[c+1] && (dp[c+1] = cand)
            end
            dpbest = Float64(dp[C+1])
            abs(obj - dpbest) > 1e-6 && (ok = false)
            @printf("knapsack   n=%d C=%d vars=%d obj=%.0f dp=%.0f t=%.4fs %s\n",
                    n, C, n, obj, dpbest, sec, ok ? "OK" : "FAIL")
        else
            @printf("knapsack   n=%d C=%d rc=%d %s\n", n, C, r.rc, "FAIL")
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
