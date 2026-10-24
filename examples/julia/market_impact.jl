#!/usr/bin/env julia
# market_impact.jl — Markowitz with market impact (power cones):
#
#   max mu'x - delta't
#   s.t.  (t_i, 1, x_i) in POW^{2/3,1/3} (t_i >= x_i^{3/2}),  (sqrt(gamma), G'x) in Q,
#         e'x = 1,  x >= 0
#
# with Sigma = F F' + diag(D), G = [F, sqrt(D)]. Verified: power-cone
# feasibility, budget, nonnegativity, risk bound. The data come from the same
# 32-bit linear congruential generator as the C example.
#
#   julia market_impact.jl [n] [k] [gamma] [delta]   (default 15 4 0.05 0.5)
#
# Port of examples/c/market_impact.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(778899)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end
cpow(x, y) = ccall(:pow, Cdouble, (Cdouble, Cdouble), x, y)   # the C library's pow

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 15
    k = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
    gamma = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.05
    delta = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.5
    (n < 1 || k < 1 || gamma <= 0.0) && return 2
    mu, F, D = zeros(n), zeros(n * k), zeros(n)
    for i in 0:n-1
        mu[i+1] = 0.02 + rnd() * 0.15
        D[i+1] = 0.001 + rnd() * 0.004
        for j in 0:k-1
            F[i*k+j+1] = (rnd() - 0.5) * 0.2
        end
    end
    m2 = k + n
    X0, T0, Z0, TC, ONE = 0, n, 2n, 2n + m2, 2n + m2 + 1   # x (n), t (n), z (m2), tc, one
    t = Model(0, 0)
    try
        add_vars(t, 2n + m2 + 2)
        add_cons(t, n + m2 + 1)
        sense(t, SENSE_MAX)
        for i in 0:n-1
            cj(t, X0 + i, mu[i+1])
            var_bounds(t, X0 + i, BK_LO; lo=0.0, up=1.0)
            cj(t, T0 + i, -delta)
            var_bounds(t, T0 + i, BK_LO; lo=0.0, up=Inf)
        end
        for j in 0:m2-1
            var_bounds(t, Z0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, TC, BK_FX; lo=sqrt(gamma), up=sqrt(gamma))
        var_bounds(t, ONE, BK_FX; lo=1.0, up=1.0)
        for j in 0:m2-1                             # z_j = (G'x)_j
            sub, val = Int[], Float64[]
            for i in 0:n-1
                g = j < k ? F[i*k+j+1] : (j - k == i ? sqrt(D[i+1]) : 0.0)
                g != 0.0 && (push!(sub, X0 + i); push!(val, -g))
            end
            push!(sub, Z0 + j); push!(val, 1.0)
            a_row(t, j, sub, val)
            con_bounds(t, j, BK_FX; lo=0.0, up=0.0)
        end
        a_row(t, m2, collect(X0:X0+n-1), ones(n))  # budget
        con_bounds(t, m2, BK_FX; lo=1.0, up=1.0)
        cone(t, CT_QUAD, [TC; collect(Z0:Z0+m2-1)]) # risk cone
        for i in 0:n-1                              # (t_i, one, x_i) in POW^{2/3,1/3}
            cone(t, CT_PPOW, [T0 + i, ONE, X0 + i]; param=2.0 / 3.0)
        end

        t0 = time()
        r = solve(t)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj, x = r.objective, r.x
            budget, minx, maxpow, risk2 = 0.0, 0.0, 0.0, 0.0
            for i in 0:n-1
                xi = x[X0+i+1]
                budget += xi
                xi < minx && (minx = xi)
                slack = cpow(xi, 1.5) - x[T0+i+1]   # t >= x^1.5
                slack > maxpow && (maxpow = slack)
                risk2 += D[i+1] * xi * xi
            end
            for a2 in 0:k-1
                s = 0.0
                for i in 0:n-1
                    s += F[i*k+a2+1] * x[X0+i+1]
                end
                risk2 += s * s
            end
            ok = abs(budget - 1.0) < 1e-5 && minx > -1e-6 && maxpow < 1e-4 && risk2 <= gamma * (1.0 + 1e-3)
            @printf("mktimpact  n=%d k=%d gamma=%.4f delta=%.2f obj=%.6f powslack=%.2e risk2=%.5f t=%.4fs %s\n",
                    n, k, gamma, delta, obj, maxpow, risk2, sec, ok ? "OK" : "FAIL")
        else
            @printf("mktimpact  n=%d rc=%d FAIL\n", n, r.rc)
        end
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
