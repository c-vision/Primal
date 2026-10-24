#!/usr/bin/env julia
# assignment.jl — n x n assignment problem (LP, scaling test).
#
#   min sum_ij c_ij x_ij  s.t.  sum_j x_ij = 1,  sum_i x_ij = 1,  x >= 0
#
# The assignment polytope is integral, so the LP optimum is a permutation.
# Verified: row/column sums = 1, integrality, strong duality. The costs come
# from the same 32-bit linear congruential generator as the C example.
#
#   julia assignment.jl [n]   (default 80)
#
# Port of examples/c/assignment.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(987654321)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 80
    n < 1 && return 2
    nvar = n * n
    c = [rnd() * 100.0 for _ in 1:nvar]

    m = Model(0, 0)
    try
        add_vars(m, nvar)
        add_cons(m, 2n)
        for k in 0:nvar-1
            cj(m, k, c[k+1])
            var_bounds(m, k, BK_LO; lo=0.0, up=1.0)
        end
        for i in 0:n-1
            a_row(m, i, [i * n + j for j in 0:n-1], ones(n))
            con_bounds(m, i, BK_FX; lo=1.0, up=1.0)
            a_row(m, n + i, [j * n + i for j in 0:n-1], ones(n))
            con_bounds(m, n + i, BK_FX; lo=1.0, up=1.0)
        end

        t0 = time()
        r = solve(m)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, x = r.objective, r.dual, r.x
            maxrow, maxcol, maxfrac = 0.0, 0.0, 0.0
            for i in 0:n-1
                s = 0.0
                for j in 0:n-1
                    v = x[i*n+j+1]
                    s += v
                    fr = abs(v - floor(v + 0.5))
                    fr > maxfrac && (maxfrac = fr)
                end
                abs(s - 1.0) > maxrow && (maxrow = abs(s - 1.0))
            end
            for j in 0:n-1
                s = 0.0
                for i in 0:n-1
                    s += x[i*n+j+1]
                end
                abs(s - 1.0) > maxcol && (maxcol = abs(s - 1.0))
            end
            ok = maxrow < 1e-6 && maxcol < 1e-6 && maxfrac < 1e-6 && abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
        end
        @printf("assignment n=%d vars=%d cons=%d nnz=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                n, nvar, 2n, 2nvar, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
