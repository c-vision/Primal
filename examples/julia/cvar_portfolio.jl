#!/usr/bin/env julia
# cvar_portfolio.jl — CVaR portfolio optimization (Rockafellar-Uryasev) as an LP.
#
#   min t + 1/(alpha S) sum_s u_s  s.t.  u_s >= -(r_s'x) - t,  u >= 0,  e'x = 1,  x >= 0
#
# Verified: the LP objective equals the empirical CVaR of x* (mean of the worst
# alpha-fraction of the scenario losses). The returns come from the same
# 32-bit linear congruential generator as the C example.
#
#   julia cvar_portfolio.jl [n] [S] [alpha]   (default 40 200 0.10)
#
# Port of examples/c/cvar_portfolio.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(606060)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 40
    S = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 200
    alpha = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.10
    (n < 1 || S < 1 || alpha <= 0.0 || alpha >= 1.0) && return 2
    R = [(rnd() - 0.5) * 0.2 + 0.01 for _ in 1:S*n]   # row-major, scenario by scenario

    X0, TV, U0 = 0, n, n + 1                        # vars: x (n), t (1), u (S)
    m = Model(0, 0)
    try
        add_vars(m, n + 1 + S)
        add_cons(m, S + 1)
        for i in 0:n-1
            var_bounds(m, X0 + i, BK_LO; lo=0.0, up=1.0)
        end
        var_bounds(m, TV, BK_FR; lo=-Inf, up=Inf)
        cj(m, TV, 1.0)
        for s in 0:S-1
            var_bounds(m, U0 + s, BK_LO; lo=0.0, up=Inf)
            cj(m, U0 + s, 1.0 / (alpha * S))
            # u_s + r_s'x + t >= 0
            a_row(m, s, [collect(X0:X0+n-1); TV; U0 + s], [R[s*n+1:s*n+n]; 1.0; 1.0])
            con_bounds(m, s, BK_LO; lo=0.0, up=Inf)
        end
        a_row(m, S, collect(X0:X0+n-1), ones(n))   # budget
        con_bounds(m, S, BK_FX; lo=1.0, up=1.0)

        t0 = time()
        r = solve(m)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj, x = r.objective, r.x
            budget, minx = 0.0, 0.0
            for i in 1:n
                budget += x[i]
                x[i] < minx && (minx = x[i])
            end
            loss = zeros(S)                         # empirical CVaR of x*: mean of the worst alpha*S losses
            for s in 0:S-1
                l = 0.0
                for i in 0:n-1
                    l -= R[s*n+i+1] * x[i+1]
                end
                loss[s+1] = l
            end
            sort!(loss; rev=true)
            ktail = ceil(Int, alpha * S)
            cvar = 0.0
            for s in 1:ktail
                cvar += loss[s]
            end
            cvar /= Float64(ktail)
            ok = abs(budget - 1.0) < 1e-6 && minx > -1e-6 && abs(obj - cvar) < 1e-4 * (1.0 + abs(cvar))
            @printf("cvar       n=%d S=%d alpha=%.2f obj=%.6f empirical=%.6f t=%.4fs %s\n",
                    n, S, alpha, obj, cvar, sec, ok ? "OK" : "FAIL")
        else
            @printf("cvar       n=%d S=%d rc=%d FAIL\n", n, S, r.rc)
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
