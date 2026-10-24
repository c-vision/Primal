#!/usr/bin/env julia
# portfolio_large.jl — large Markowitz QP with a factor risk model (scaling).
#
#   max r'x - gamma x'Sx  s.t.  sum x = 1,  x >= 0,  S = F F' + diag(D)
#
# Verified: budget, nonnegativity, strong duality. The data come from the same
# 32-bit linear congruential generator as the C example.
#
#   julia portfolio_large.jl [n] [k]   (default 100 5)
#
# Port of examples/c/portfolio_large.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(31415926)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 100
    k = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 5
    (n < 1 || k < 1) && return 2
    gamma = 0.5
    r, F, D = zeros(n), zeros(n * k), zeros(n)
    for i in 0:n-1
        r[i+1] = rnd() * 0.2 - 0.05
        D[i+1] = 0.01 + rnd() * 0.02
        for j in 0:k-1
            F[i*k+j+1] = rnd() - 0.5
        end
    end
    # objective (MAXIMIZE): r'x - gamma x'Sx = r'x + 0.5 x'Qx with Q = -2 gamma S
    Q = zeros(n * n)
    for i in 0:n-1, j in 0:n-1
        s = 0.0
        for l in 0:k-1
            s += F[i*k+l+1] * F[j*k+l+1]
        end
        i == j && (s += D[i+1])
        Q[i*n+j+1] = -2.0 * gamma * s
    end

    m = Model(0, 0)
    try
        add_vars(m, n)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        for i in 0:n-1
            cj(m, i, r[i+1])
            var_bounds(m, i, BK_LO; lo=0.0, up=1.0)
        end
        qobj(m, [(i, j, Q[i*n+j+1]) for i in 0:n-1 for j in i:n-1 if Q[i*n+j+1] != 0.0])
        a_row(m, 0, collect(0:n-1), ones(n))
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)

        t0 = time()
        res = solve(m)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = res.rc == 0
        if ok
            obj, dobj, x = res.objective, res.dual, res.x
            budget, minx = 0.0, 0.0
            for i in 1:n
                budget += x[i]
                x[i] < minx && (minx = x[i])
            end
            ok = abs(budget - 1.0) < 1e-6 && minx > -1e-6 && abs(obj - dobj) < 1e-4 * (1.0 + abs(obj))
        end
        @printf("portfolio  n=%d k=%d vars=%d nnzQ=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                n, k, n, n * (n + 1) ÷ 2, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
