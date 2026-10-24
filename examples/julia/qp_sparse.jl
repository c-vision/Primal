#!/usr/bin/env julia
# qp_sparse.jl — large sparse convex QP (scaling test for the sparse
# normal-equations QP interior point).
#
#   min 1/2 x'Qx + c'x  s.t.  A x = b,  x >= 0
#
# Q is the 1-D graph Laplacian + 0.5 I (tridiagonal, PD), A has m rows, and
# b = A x_ref for a planted x_ref >= 0: feasible and strictly convex.
# Verified: Ax = b, x >= 0, strong duality. The data come from the same 32-bit
# linear congruential generator as the C example.
#
#   julia qp_sparse.jl [n] [m]   (default 2000 8)
#
# Port of examples/c/qp_sparse.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(13579)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2000
    m = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 8
    (n < 2 || m < 1) && return 2
    # Q = 1-D Laplacian + 0.5 I (tridiagonal, PD); lower-triangle triplets
    nq = 2n - 1
    terms = [(j, j, ((j == 0 || j == n - 1) ? 1.0 : 2.0) + 0.5) for j in 0:n-1]
    append!(terms, [(j + 1, j, -1.0) for j in 0:n-2])

    xref, c = zeros(n), zeros(n)
    for j in 1:n
        xref[j] = rnd() < 0.7 ? (rnd() * 5.0 + 0.1) : 0.0   # ~30% at the bound
        c[j] = rnd() * 4.0 - 2.0
    end
    A = zeros(m * n)                                # row-major, as the C example
    for i in 0:m-1, j in 0:n-1
        A[i*n+j+1] = i == 0 ? 1.0 : (rnd() * 2.0 - 1.0)
    end
    b = zeros(m)
    for i in 0:m-1, j in 0:n-1
        b[i+1] += A[i*n+j+1] * xref[j+1]
    end

    t = Model(0, 0)
    try
        add_vars(t, n); add_cons(t, m)
        for j in 0:n-1
            cj(t, j, c[j+1])
            var_bounds(t, j, BK_LO; lo=0.0, up=Inf)
        end
        qobj(t, terms)
        for i in 0:m-1
            a_row(t, i, collect(0:n-1), A[i*n+1:i*n+n])
            con_bounds(t, i, BK_FX; lo=b[i+1], up=b[i+1])
        end

        t0 = time()
        r = solve(t)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, x = r.objective, r.dual, r.x
            maxviol, minx = 0.0, 0.0
            for i in 0:m-1
                s = 0.0
                for j in 0:n-1
                    s += A[i*n+j+1] * x[j+1]
                end
                e = abs(s - b[i+1]); e > maxviol && (maxviol = e)
            end
            for j in 1:n
                x[j] < minx && (minx = x[j])
            end
            bn = 1.0
            for i in 1:m
                e = abs(b[i]); e > bn && (bn = e)
            end
            ok = maxviol < 1e-4 * bn && minx > -1e-6 && abs(obj - dobj) < 1e-4 * (1.0 + abs(obj))
        end
        @printf("qp_sparse  n=%d m=%d nnzQ=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                n, m, nq, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
