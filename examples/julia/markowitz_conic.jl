#!/usr/bin/env julia
# markowitz_conic.jl — Markowitz portfolio as a second-order cone program:
#
#   max mu'x  s.t.  ||G'x||_2 <= sqrt(gamma),  e'x = 1,  x >= 0
#
# with the factor model Sigma = F F' + diag(D), G = [F, sqrt(D)]. Verified:
# budget, nonnegativity, risk bound; when the risk is not binding the optimum
# is the best-mean asset. The data come from the same 32-bit linear
# congruential generator as the C example.
#
#   julia markowitz_conic.jl [n] [k] [gamma]   (default 60 4 0.05)
#
# Port of examples/c/markowitz_conic.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(42424242)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 60
    k = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
    gamma = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.05
    (n < 1 || k < 1 || gamma <= 0.0) && return 2
    mu, F, D = zeros(n), zeros(n * k), zeros(n)
    for i in 0:n-1
        mu[i+1] = 0.02 + rnd() * 0.15
        D[i+1] = 0.001 + rnd() * 0.004
        for j in 0:k-1
            F[i*k+j+1] = (rnd() - 0.5) * 0.2
        end
    end
    m2 = k + n                                      # G = [F | diag(sqrt(D))]
    X0, Z0, TC = 0, n, n + m2                       # vars: x (n), z (m2), tcone (1)
    t = Model(0, 0)
    try
        add_vars(t, n + m2 + 1)
        add_cons(t, m2)
        sense(t, SENSE_MAX)
        for i in 0:n-1
            cj(t, X0 + i, mu[i+1])
            var_bounds(t, X0 + i, BK_LO; lo=0.0, up=1.0)
        end
        for j in 0:m2-1
            var_bounds(t, Z0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, TC, BK_FX; lo=sqrt(gamma), up=sqrt(gamma))
        for j in 0:m2-1                             # z_j - (G'x)_j = 0
            sub, val = Int[], Float64[]
            for i in 0:n-1
                g = j < k ? F[i*k+j+1] : (j - k == i ? sqrt(D[i+1]) : 0.0)
                g != 0.0 && (push!(sub, X0 + i); push!(val, -g))
            end
            push!(sub, Z0 + j); push!(val, 1.0)
            a_row(t, j, sub, val)
            con_bounds(t, j, BK_FX; lo=0.0, up=0.0)
        end
        cone(t, CT_QUAD, [TC; collect(Z0:Z0+m2-1)])   # (tcone, z) in QUAD
        add_cons(t, 1)                              # budget
        a_row(t, m2, collect(X0:X0+n-1), ones(n))
        con_bounds(t, m2, BK_FX; lo=1.0, up=1.0)

        t0 = time()
        r = solve(t)
        sec = time() - t0
        obj = 0.0
        ok = r.rc == 0
        if ok
            obj, x = r.objective, r.x
            budget, minx, risk2, maxmu = 0.0, 0.0, 0.0, mu[1]
            for i in 1:n
                budget += x[i]
                x[i] < minx && (minx = x[i])
                mu[i] > maxmu && (maxmu = mu[i])
                risk2 += D[i] * x[i] * x[i]
            end
            for a2 in 0:k-1
                s = 0.0
                for i in 0:n-1
                    s += F[i*k+a2+1] * x[i+1]
                end
                risk2 += s * s
            end
            ok = abs(budget - 1.0) < 1e-6 && minx > -1e-6 && risk2 <= gamma * (1.0 + 1e-4)
            # when the risk budget is not binding the optimum is the best asset
            if ok && risk2 < gamma * (1.0 - 1e-3)
                ok = abs(obj - maxmu) < 1e-5 * (1.0 + maxmu)
            end
        end
        @printf("markowitz  n=%d k=%d gamma=%.4f obj=%.6f t=%.4fs %s\n", n, k, gamma, obj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
