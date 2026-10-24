#!/usr/bin/env julia
# lsq_pos.jl — the exact `lsq_pos` form of the MosekRegression repository
# (tschm/MosekRegression, the code of arXiv:1310.3397):
#
#   min ||X w - rhs||_2  (not squared: (v, X w - rhs) in Q^{N+1})
#   s.t.  e'w = 1,  0 <= w <= 1
#
# The independent check is a second solve of the QP min ||Xw - rhs||^2, whose
# optimum plus ||rhs||^2 must be v*^2. The data come from the same 32-bit
# linear congruential generator as the C example.
#
#   julia lsq_pos.jl [n] [m]   (default 40 5)
#
# Port of examples/c/lsq_pos.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(13103397)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# min v s.t. (v, Xw - rhs) in Q, e'w = 1, 0 <= w <= 1; returns (v, w).
function solve_socp(X, rhs, N, M)
    W0, R0, V = 0, M, M + N
    t = Model(0, 0)
    r = try
        add_vars(t, M + N + 1)
        add_cons(t, N + 1)
        for j in 0:M-1
            var_bounds(t, W0 + j, BK_RA; lo=0.0, up=1.0)
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
            con_bounds(t, i, BK_FX; lo=-rhs[i+1], up=-rhs[i+1])
        end
        for j in 0:M-1
            cj(t, W0 + j, 0.0)
        end
        a_row(t, N, collect(W0:W0+M-1), ones(M))   # e'w = 1
        con_bounds(t, N, BK_FX; lo=1.0, up=1.0)
        cone(t, CT_QUAD, [V; collect(R0:R0+N-1)])  # (v, r) in Q
        solve(t)
    finally
        free(t)
    end
    r.rc == 0 || return -1.0, zeros(M)
    return r.objective, r.x[W0+1:W0+M]
end

# min ||Xw - rhs||^2 = w'(X'X)w - 2 rhs'X w + const, as 1/2 w'Qw + c'w with
# Q = 2 X'X and c = -2 X'rhs; returns (objective, w).
function solve_qp(X, rhs, N, M)
    t = Model(0, 0)
    r = try
        add_vars(t, M)
        add_cons(t, 1)
        for j in 0:M-1
            var_bounds(t, j, BK_RA; lo=0.0, up=1.0)
        end
        for j in 0:M-1
            s = 0.0
            for i in 0:N-1
                s += X[i*M+j+1] * rhs[i+1]
            end
            cj(t, j, -2.0 * s)
        end
        for j in 0:M-1, k in j:M-1
            s = 0.0
            for i in 0:N-1
                s += X[i*M+j+1] * X[i*M+k+1]
            end
            qobj(t, [(j, k, 2.0 * s)])
        end
        a_row(t, 0, collect(0:M-1), ones(M))       # e'w = 1
        con_bounds(t, 0, BK_FX; lo=1.0, up=1.0)
        solve(t)
    finally
        free(t)
    end
    r.rc == 0 || return 0.0, zeros(M)
    return r.objective, r.x[1:M]
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 40
    M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 5
    (N < 1 || M < 1 || M > 60) && return 2
    X = [(rnd() - 0.5) * 2.0 for _ in 1:N*M]
    rhs = [(rnd() - 0.5) * 2.0 for _ in 1:N]
    v, ws = solve_socp(X, rhs, N, M)
    q, _ = solve_qp(X, rhs, N, M)
    budget, lo, hi = 0.0, 0.0, 0.0
    for j in 1:M
        budget += ws[j]
        ws[j] < lo && (lo = ws[j])
        ws[j] > hi && (hi = ws[j])
    end
    rhs2 = 0.0
    for i in 1:N
        rhs2 += rhs[i] * rhs[i]
    end
    v2_from_qp = q + rhs2                           # v*^2 == pobj_qp + ||rhs||^2
    ok = v >= 0.0 && abs(budget - 1.0) <= 1e-6 && lo >= -1e-6 && hi <= 1.0 + 1e-6 &&
         abs(v * v - v2_from_qp) <= 1e-5 * (1.0 + abs(v2_from_qp))
    @printf("lsq_pos  n=%d m=%d  v*=%.8f  sqrt(QP)=%.8f  e'w=%.2e  w in [%.3f,%.3f] %s\n",
            N, M, v, sqrt(abs(v2_from_qp)), budget, lo, hi, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
