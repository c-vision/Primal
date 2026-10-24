#!/usr/bin/env julia
# regression_regularized.jl — regularised regression (Section 3 of
# Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for Portfolio
# Optimisation using MOSEK", arXiv:1310.3397):
#
#   ridge :  min ||Xw-y||^2 + lambda ||w-w0||^2
#   sparse:  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|          (LASSO)
#   3/2   :  min ||Xw-y||^2 + lambda sum_i |w_i-w0_i|^{3/2}
#
# The squared residual is a quadratic objective; the penalties use the
# paper's cones (t_i, p_i) in Q^2 and (t_i, 1, p_i) in POW^{2/3,1/3}. Ridge is
# checked against (X'X + lambda I) w = X'y + lambda w0 (the library's dense LU),
# sparse and 3/2 against cone feasibility. The data come from the same 32-bit
# linear congruential generator as the C example.
#
#   julia regression_regularized.jl [n] [m] [lambda]   (default 30 4 0.5)
#
# Port of examples/c/regression_regularized.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(27182818)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end
cpow(x, y) = ccall(:pow, Cdouble, (Cdouble, Cdouble), x, y)   # the C library's pow

# Solve M x = rhs (d x d, row-major) in place with the library's dense LU.
function dmat_solve_lu!(Mx::Vector{Float64}, rhs::Vector{Float64}, d::Int)
    lib = Primal._lib()
    p = ccall(Libdl.dlsym(lib, :dmat_new), Ptr{Cvoid}, (Cint, Cint), d, d)
    rows = unsafe_load(Ptr{Ptr{Ptr{Cdouble}}}(p + 2 * sizeof(Cint)))   # DMat {int m, n; double **v}
    for i in 1:d
        row = unsafe_load(rows, i)
        for j in 1:d
            unsafe_store!(row, Mx[(i-1)*d+j], j)
        end
    end
    rc = ccall(Libdl.dlsym(lib, :dmat_solve_lu), Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Cint), p, rhs, d)
    ccall(Libdl.dlsym(lib, :dmat_free), Cvoid, (Ptr{Cvoid},), p)
    return Int(rc)
end

const REG_RIDGE, REG_SPARSE, REG_POW32 = 0, 1, 2

# Add the squared-residual QP terms Q = 2 X'X, c = -2 X'y, cfix = y'y.
function put_qobj_residual(t, X, y, N, M)
    terms = Tuple{Int,Int,Float64}[]
    for a in 0:M-1, b in a:M-1
        s = 0.0
        for i in 0:N-1
            s += X[i*M+a+1] * X[i*M+b+1]
        end
        s *= 2.0
        s != 0.0 && push!(terms, (a, b, s))
    end
    qobj(t, terms)
    for a in 0:M-1
        s = 0.0
        for i in 0:N-1
            s += X[i*M+a+1] * y[i+1]
        end
        cj(t, a, -2.0 * s)
    end
    yy = 0.0
    for i in 1:N
        yy += y[i] * y[i]
    end
    cfix(t, yy)
end

# Build the ridge, sparse or 3/2 regularised regression model.
function build(t, kind, X, y, w0, N, M, LAM)
    if kind == REG_RIDGE                            # pure QP: Q = 2(X'X + lam I), c = -2(X'y + lam w0)
        add_vars(t, M)
        for j in 0:M-1
            var_bounds(t, j, BK_FR; lo=-Inf, up=Inf)
        end
        put_qobj_residual(t, X, y, N, M)
        qobj(t, [(j, j, 2.0 * LAM) for j in 0:M-1])   # accumulate
        for j in 0:M-1
            cj(t, j, getcj(t, j) - 2.0 * LAM * w0[j+1])
        end
        cf = getcfix(t)
        w02 = 0.0
        for j in 1:M
            w02 += w0[j] * w0[j]
        end
        cfix(t, cf + LAM * w02)
        return
    end
    W0, P0, T0, ONE = 0, M, 2M, 3M                  # sparse / 3-2: conic penalty on p = w - w0
    add_vars(t, kind == REG_SPARSE ? 3M : 3M + 1)
    add_cons(t, M)
    for j in 0:M-1
        var_bounds(t, W0 + j, BK_FR; lo=-Inf, up=Inf)
    end
    for j in 0:M-1
        var_bounds(t, P0 + j, BK_FR; lo=-Inf, up=Inf)
    end
    for j in 0:M-1
        var_bounds(t, T0 + j, BK_LO; lo=0.0, up=Inf)
        cj(t, T0 + j, LAM)
    end
    put_qobj_residual(t, X, y, N, M)
    for j in 0:M-1                                  # p_j - w_j = -w0_j
        a_row(t, j, [W0 + j, P0 + j], [-1.0, 1.0])
        con_bounds(t, j, BK_FX; lo=-w0[j+1], up=-w0[j+1])
    end
    if kind == REG_SPARSE
        for j in 0:M-1
            cone(t, CT_QUAD, [T0 + j, P0 + j])
        end
    else
        var_bounds(t, ONE, BK_FX; lo=1.0, up=1.0)
        for j in 0:M-1
            cone(t, CT_PPOW, [T0 + j, ONE, P0 + j]; param=2.0 / 3.0)
        end
    end
end

# Build and solve one variant; returns (rc, x, seconds).
function run_variant(kind, X, y, w0, N, M, LAM)
    t = Model(0, 0)
    try
        build(t, kind, X, y, w0, N, M, LAM)
        t0 = time()
        r = solve(t)
        return r.rc, r.x, time() - t0
    finally
        free(t)
    end
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 30
    M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
    LAM = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.5
    (N < 1 || M < 1 || LAM < 0.0) && return 2
    X = [(rnd() - 0.5) * 2.0 for _ in 1:N*M]
    y = [(rnd() - 0.5) * 2.0 for _ in 1:N]
    w0 = zeros(M)
    ok = true

    rc, x, sec = run_variant(REG_RIDGE, X, y, w0, N, M, LAM)   # ridge vs (X'X + lam I) w = X'y + lam w0
    w = rc == 0 ? x[1:M] : zeros(M)
    rc == 0 || (ok = false)
    G, rhs = zeros(M * M), zeros(M)
    for j in 0:M-1, k in 0:M-1
        s = j == k ? LAM : 0.0
        for i in 0:N-1
            s += X[i*M+j+1] * X[i*M+k+1]
        end
        G[j*M+k+1] = s
    end
    for j in 0:M-1
        s = LAM * w0[j+1]
        for i in 0:N-1
            s += X[i*M+j+1] * y[i+1]
        end
        rhs[j+1] = s
    end
    if dmat_solve_lu!(G, rhs, M) == 0
        d = 0.0
        for j in 1:M
            abs(w[j] - rhs[j]) > d && (d = abs(w[j] - rhs[j]))
        end
        d > 1e-5 && (ok = false)
        @printf("ridge      n=%d m=%d lam=%.2f wdiff=%.2e t=%.4fs\n", N, M, LAM, d, sec)
    end

    rc, x, sec = run_variant(REG_SPARSE, X, y, w0, N, M, LAM)  # sparse: (t_j, p_j) in Q^2
    worst = 0.0
    if rc == 0
        for j in 1:M
            slack = abs(x[M+j]) - x[2M+j]
            slack > worst && (worst = slack)
        end
    else
        ok = false
    end
    worst > 1e-5 && (ok = false)
    @printf("sparse     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs\n", N, M, LAM, rc, worst, sec)

    rc, x, sec = run_variant(REG_POW32, X, y, w0, N, M, LAM)   # 3/2: t_j >= |p_j|^{3/2}
    worst = 0.0
    if rc == 0
        for j in 1:M
            slack = cpow(abs(x[M+j]), 1.5) - x[2M+j]
            slack > worst && (worst = slack)
        end
    else
        ok = false
    end
    worst > 1e-4 && (ok = false)
    @printf("pow3_2     n=%d m=%d lam=%.2f rc=%d worst=%.2e t=%.4fs\n", N, M, LAM, rc, worst, sec)

    @printf("regr_reg   n=%d m=%d lam=%.2f %s\n", N, M, LAM, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
