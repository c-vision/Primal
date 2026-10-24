#!/usr/bin/env julia
# regression_ls.jl — least-squares regression as a second-order cone program
# (Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for Portfolio
# Optimisation using MOSEK", arXiv:1310.3397, Section 2):
#
#   min ||X w - y||_2  <=>  min v  s.t.  (v, Xw - y) in Q^{n+1}
#
# Two variants: unconstrained (checked against the normal equations solved
# with the library's dense LU) and the constrained portfolio form sum(w) = 1,
# w >= 0 (Section 2.6). The data come from the same 32-bit linear
# congruential generator as the C example.
#
#   julia regression_ls.jl [n] [m]   (default 60 6)
#
# Port of examples/c/regression_ls.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(13103397)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# Solve M x = rhs (d x d, row-major) in place with the library's dense LU.
function dmat_solve_lu!(M::Vector{Float64}, rhs::Vector{Float64}, d::Int)
    lib = Primal._lib()
    p = ccall(Libdl.dlsym(lib, :dmat_new), Ptr{Cvoid}, (Cint, Cint), d, d)
    rows = unsafe_load(Ptr{Ptr{Ptr{Cdouble}}}(p + 2 * sizeof(Cint)))   # DMat {int m, n; double **v}
    for i in 1:d
        row = unsafe_load(rows, i)
        for j in 1:d
            unsafe_store!(row, M[(i-1)*d+j], j)
        end
    end
    rc = ccall(Libdl.dlsym(lib, :dmat_solve_lu), Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Cint), p, rhs, d)
    ccall(Libdl.dlsym(lib, :dmat_free), Cvoid, (Ptr{Cvoid},), p)
    return Int(rc)
end

# Build and solve: vars w (M), r (N), v; rows r_i - (Xw)_i = -y_i; cone (v, r).
# Returns (rc, v, w).
function solve_ls(constrained, X, y, N, M)
    W0, R0, V = 0, M, M + N
    t = Model(0, 0)
    r, t0, sec = nothing, 0.0, 0.0
    try
        add_vars(t, M + N + 1)
        add_cons(t, N)
        for j in 0:M-1
            constrained ? var_bounds(t, W0 + j, BK_LO; lo=0.0, up=Inf) : var_bounds(t, W0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:N-1
            var_bounds(t, R0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, V, BK_FR; lo=-Inf, up=Inf)
        cj(t, V, 1.0)
        for i in 0:N-1
            sub, val = Int[], Float64[]
            for j in 0:M-1
                xij = X[i*M+j+1]
                xij != 0.0 && (push!(sub, W0 + j); push!(val, -xij))
            end
            push!(sub, R0 + i); push!(val, 1.0)
            a_row(t, i, sub, val)
            con_bounds(t, i, BK_FX; lo=-y[i+1], up=-y[i+1])
        end
        cone(t, CT_QUAD, [V; collect(R0:R0+N-1)])
        if constrained
            add_cons(t, 1)
            a_row(t, N, collect(W0:W0+M-1), ones(M))
            con_bounds(t, N, BK_FX; lo=1.0, up=1.0)
        end
        t0 = time()
        r = solve(t)
        sec = time() - t0
    finally
        free(t)
    end
    return r.rc, r.objective, r.x[W0+1:W0+M], sec
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 60
    M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 6
    (N < 1 || M < 1) && return 2
    X = [(rnd() - 0.5) * 2.0 for _ in 1:N*M]
    y = [(rnd() - 0.5) * 2.0 for _ in 1:N]
    ok = true

    rc, v1, w1, sec1 = solve_ls(false, X, y, N, M)  # unconstrained LS
    rc == 0 || (ok = false; v1 = 0.0)
    G, rhs = zeros(M * M), zeros(M)                 # normal-equations reference
    for j in 0:M-1, k in 0:M-1
        s = 0.0
        for i in 0:N-1
            s += X[i*M+j+1] * X[i*M+k+1]
        end
        G[j*M+k+1] = s
    end
    for j in 0:M-1
        s = 0.0
        for i in 0:N-1
            s += X[i*M+j+1] * y[i+1]
        end
        rhs[j+1] = s
    end
    if dmat_solve_lu!(G, rhs, M) == 0
        maximum(abs.(w1 .- rhs)) > 1e-6 && (ok = false)
    end

    rc2, v2, w2, sec2 = solve_ls(true, X, y, N, M)  # constrained (portfolio) LS
    budget, minw = 0.0, 0.0
    if rc2 == 0
        for j in 1:M
            budget += w2[j]
            w2[j] < minw && (minw = w2[j])
        end
    else
        ok = false; v2 = 0.0
    end
    ok && (abs(budget - 1.0) > 1e-6 || minw < -1e-6) && (ok = false)
    ok && v2 < v1 - 1e-6 && (ok = false)            # constraints cannot lower the residual
    @printf("regr_ls    n=%d m=%d uncons=%.6f cons=%.6f t=%.4f/%.4fs %s\n", N, M, v1, v2, sec1, sec2, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
