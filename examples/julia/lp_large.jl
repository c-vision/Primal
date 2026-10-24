#!/usr/bin/env julia
# lp_large.jl — large generic sparse LP (scaling test for the sparse
# normal-equations interior point):
#
#   min c'x  s.t.  A x = b,  x >= 0
#
# A is random sparse (d nonzeros per column) with a PLANTED optimum from the KKT
# conditions: x* >= 0, free y*, z* complementary to x*; b = A x*, c = A'y* + z*,
# so obj* = c'x* is known independently. Verified: obj = obj*, Ax = b, x >= 0,
# strong duality. The data come from the same 32-bit linear congruential
# generator as the C example.
#
#   julia lp_large.jl [rows] [cols] [density]   (default 300 6000 8)
#
# Port of examples/c/lp_large.c.

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
    m = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 300    # rows
    n = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 6000   # cols
    d = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 8      # nnz/col
    (m < 1 || n < 1 || d < 1 || d > m) && return 2

    nnz = n * d                                     # sparse A in column form: d distinct rows per column
    arow, aval = zeros(Int, nnz), zeros(nnz)
    for j in 0:n-1
        used = falses(m)
        for k in 0:d-1
            i = 0
            while true
                i = trunc(Int, rnd() * m)
                i >= m && (i = m - 1)
                used[i+1] || break
            end
            used[i+1] = true
            arow[j*d+k+1] = i
            aval[j*d+k+1] = 2.0 * rnd() - 1.0      # in [-1,1)
        end
    end

    xs, ys, zs = zeros(n), zeros(m), zeros(n)       # planted optimum: x*, y*, z*
    for j in 1:n
        xs[j] = rnd() < 0.6 ? (rnd() * 10.0 + 0.1) : 0.0   # ~60% positive
    end
    for i in 1:m
        ys[i] = 2.0 * rnd() - 1.0
    end
    for j in 1:n
        zs[j] = xs[j] > 0.0 ? 0.0 : (rnd() * 5.0 + 0.1)
    end
    b = zeros(m)                                    # b = A x*
    for j in 0:n-1
        xs[j+1] != 0.0 || continue
        for k in 0:d-1
            b[arow[j*d+k+1]+1] += aval[j*d+k+1] * xs[j+1]
        end
    end
    c = zeros(n)                                    # c = A'y* + z* ;  obj* = c'x*
    for j in 0:n-1
        s = zs[j+1]
        for k in 0:d-1
            s += aval[j*d+k+1] * ys[arow[j*d+k+1]+1]
        end
        c[j+1] = s
    end
    objstar = 0.0
    for j in 1:n
        objstar += c[j] * xs[j]
    end
    rsub = [Int[] for _ in 1:m]                     # rows for putarow, columns in order
    rval = [Float64[] for _ in 1:m]
    for j in 0:n-1, k in 0:d-1
        i = arow[j*d+k+1]
        push!(rsub[i+1], j); push!(rval[i+1], aval[j*d+k+1])
    end

    t = Model(0, 0)
    try
        add_vars(t, n)
        add_cons(t, m)
        for j in 0:n-1
            cj(t, j, c[j+1])
            var_bounds(t, j, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:m-1
            a_row(t, i, rsub[i+1], rval[i+1])
            con_bounds(t, i, BK_FX; lo=b[i+1], up=b[i+1])
        end

        t0 = time()
        r = solve(t)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, x = r.objective, r.dual, r.x
            Ax = zeros(m)
            for j in 0:n-1
                x[j+1] != 0.0 || continue
                for k in 0:d-1
                    Ax[arow[j*d+k+1]+1] += aval[j*d+k+1] * x[j+1]
                end
            end
            maxviol, minx = 0.0, 0.0
            for i in 1:m
                e = abs(Ax[i] - b[i]); e > maxviol && (maxviol = e)
            end
            for j in 1:n
                x[j] < minx && (minx = x[j])
            end
            bn = 1.0
            for i in 1:m
                e = abs(b[i]); e > bn && (bn = e)
            end
            ok = maxviol < 1e-4 * bn && minx > -1e-6 && abs(obj - objstar) < 1e-4 * (1.0 + abs(objstar)) &&
                 abs(obj - dobj) < 1e-4 * (1.0 + abs(obj))
        end
        @printf("lp_large  rows=%d cols=%d nnz=%d obj=%.6f obj*=%.6f dobj=%.6f t=%.4fs %s\n",
                m, n, nnz, obj, objstar, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
