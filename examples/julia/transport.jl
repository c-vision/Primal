#!/usr/bin/env julia
# transport.jl — large transportation LP (scaling test).
#
#   min sum_ij c_ij x_ij  s.t.  sum_j x_ij = supply_i,  sum_i x_ij = demand_j,  x >= 0
#
# Feasible by construction (supplies/demands come from a reference flow).
# Verified: row/column sums, nonnegativity, strong duality. The data come from
# the same 32-bit linear congruential generator as the C example.
#
#   julia transport.jl [M] [N]   (default 60 60)
#
# Port of examples/c/transport.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(12345)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    M = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 60
    N = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 60
    (M < 1 || N < 1) && return 2
    nvar = M * N
    c, sup, dem = zeros(nvar), zeros(M), zeros(N)
    for i in 0:M-1, j in 0:N-1
        x0 = rnd() * 10.0
        c[i*N+j+1] = rnd() * 5.0 + 0.1
        sup[i+1] += x0
        dem[j+1] += x0
    end

    m = Model(0, 0)
    try
        add_vars(m, nvar)
        add_cons(m, M + N)
        for k in 0:nvar-1
            cj(m, k, c[k+1])
            var_bounds(m, k, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:M-1
            a_row(m, i, [i * N + j for j in 0:N-1], ones(N))
            con_bounds(m, i, BK_FX; lo=sup[i+1], up=sup[i+1])
        end
        for j in 0:N-1
            a_row(m, M + j, [i * N + j for i in 0:M-1], ones(M))
            con_bounds(m, M + j, BK_FX; lo=dem[j+1], up=dem[j+1])
        end

        t0 = time()
        r = solve(m)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, x = r.objective, r.dual, r.x
            maxsup, maxdem, minx = 0.0, 0.0, 0.0
            for i in 0:M-1
                s = 0.0
                for j in 0:N-1
                    s += x[i*N+j+1]
                end
                abs(s - sup[i+1]) > maxsup && (maxsup = abs(s - sup[i+1]))
            end
            for j in 0:N-1
                s = 0.0
                for i in 0:M-1
                    s += x[i*N+j+1]
                end
                abs(s - dem[j+1]) > maxdem && (maxdem = abs(s - dem[j+1]))
            end
            for k in 1:nvar
                x[k] < minx && (minx = x[k])
            end
            ok = maxsup < 1e-5 && maxdem < 1e-5 && minx > -1e-6 && abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
        end
        @printf("transport  M=%d N=%d vars=%d cons=%d nnz=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                M, N, nvar, M + N, 2nvar, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
