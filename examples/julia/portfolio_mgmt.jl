#!/usr/bin/env julia
# portfolio_mgmt.jl — portfolio management as conic programs (Section 4 of
# Schmelzer-Hauser-Andersen-Dahl, "Regression techniques for Portfolio
# Optimisation using MOSEK", arXiv:1310.3397):
#
#   min variance :  min ||Xw||^2                 s.t. e'w = 1, w >= 0
#   tracking     :  min ||Xw - rM||^2            s.t. e'w = 1, w >= 0
#   max return   :  max mu'w  s.t. (sigma, Xw) in Q, e'w = 1, w >= 0
#   130/30       :  max mu'w  s.t. sum t_i <= 1.6, (t_i, w_i) in Q^2, e'w = 1
#   robust       :  max mu0'w - u  s.t. (u, A w) in Q, e'w = 1, w >= 0
#
# Each is solved and verified (budget, bounds, cone feasibility). The data
# come from the same 32-bit linear congruential generator as the C example.
#
#   julia portfolio_mgmt.jl [n] [m]   (default 40 6)
#
# Port of examples/c/portfolio_mgmt.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(314159)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# Build, time and solve one model; returns (Result, seconds).
function timed_solve(build)
    t = Model(0, 0)
    try
        build(t)
        t0 = time()
        r = solve(t)
        return r, time() - t0
    finally
        free(t)
    end
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 40
    M = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 6
    (N < 1 || M < 1) && return 2
    X = [(rnd() - 0.5) * 0.1 for _ in 1:N*M]
    rM = [(rnd() - 0.5) * 0.08 for _ in 1:N]
    mu = [0.001 + rnd() * 0.02 for _ in 1:M]
    ok = true
    budget(x) = (s = 0.0; for j in 1:M; s += x[j]; end; s)

    # min ||Xw||^2 (or ||Xw - rM||^2) via QP
    function build_ls(t, tracking)
        add_vars(t, M)
        add_cons(t, 1)
        for j in 0:M-1
            var_bounds(t, j, BK_LO; lo=0.0, up=1.0)
        end
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
        cf = 0.0
        for a in 0:M-1
            c = 0.0
            for i in 0:N-1
                c += X[i*M+a+1] * (tracking ? rM[i+1] : 0.0)
            end
            cj(t, a, -2.0 * c)
        end
        if tracking
            for i in 1:N
                cf += rM[i] * rM[i]
            end
        end
        cfix(t, cf)
        a_row(t, 0, collect(0:M-1), ones(M))
        con_bounds(t, 0, BK_FX; lo=1.0, up=1.0)
    end

    r, sec = timed_solve(t -> build_ls(t, false))   # min variance
    bud = budget(r.x)
    (r.rc != 0 || abs(bud - 1.0) > 1e-6) && (ok = false)
    @printf("minvar     n=%d m=%d budget=%.6f t=%.4fs\n", N, M, bud, sec)

    r, sec = timed_solve(t -> build_ls(t, true))    # tracking
    bud = budget(r.x)
    (r.rc != 0 || abs(bud - 1.0) > 1e-6) && (ok = false)
    @printf("tracking   n=%d m=%d budget=%.6f t=%.4fs\n", N, M, bud, sec)

    sigma = 0.5                                     # max return with risk cone (sigma, Xw) in Q
    W0, R0, TC = 0, M, M + N
    r, sec = timed_solve() do t
        add_vars(t, M + N + 1)
        add_cons(t, N + 1)
        sense(t, SENSE_MAX)
        for j in 0:M-1
            cj(t, W0 + j, mu[j+1])
            var_bounds(t, W0 + j, BK_LO; lo=0.0, up=1.0)
        end
        for i in 0:N-1
            var_bounds(t, R0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, TC, BK_FX; lo=sigma, up=sigma)
        for i in 0:N-1
            sub, val = Int[], Float64[]
            for j in 0:M-1
                X[i*M+j+1] != 0.0 && (push!(sub, W0 + j); push!(val, -X[i*M+j+1]))
            end
            push!(sub, R0 + i); push!(val, 1.0)
            a_row(t, i, sub, val)
            con_bounds(t, i, BK_FX; lo=0.0, up=0.0)
        end
        a_row(t, N, collect(W0:W0+M-1), ones(M))   # budget
        con_bounds(t, N, BK_FX; lo=1.0, up=1.0)
        cone(t, CT_QUAD, [TC; collect(R0:R0+N-1)])
    end
    risk, bud = 0.0, 0.0
    if r.rc == 0
        for i in 0:N-1
            risk += r.x[R0+i+1] * r.x[R0+i+1]
        end
        bud = budget(r.x)
        risk = sqrt(risk)
    else
        ok = false
    end
    (risk > sigma * (1.0 + 1e-4) || abs(bud - 1.0) > 1e-6) && (ok = false)
    @printf("maxreturn  n=%d m=%d risk=%.6f budget=%.6f t=%.4fs\n", N, M, risk, bud, sec)

    T0 = M                                          # 130/30: (t_i, w_i) in Q^2, sum t <= 1.6, sum w = 1
    r, sec = timed_solve() do t
        add_vars(t, 2M)
        add_cons(t, 2)
        sense(t, SENSE_MAX)
        for j in 0:M-1
            cj(t, W0 + j, mu[j+1])
            var_bounds(t, W0 + j, BK_FR; lo=-Inf, up=Inf)
            var_bounds(t, T0 + j, BK_LO; lo=0.0, up=Inf)
            cone(t, CT_QUAD, [T0 + j, W0 + j])
        end
        a_row(t, 0, collect(T0:T0+M-1), ones(M))   # sum t <= 1.6
        con_bounds(t, 0, BK_UP; lo=-Inf, up=1.6)
        a_row(t, 1, collect(W0:W0+M-1), ones(M))   # sum w = 1
        con_bounds(t, 1, BK_FX; lo=1.0, up=1.0)
    end
    lev, bud = 0.0, 0.0
    if r.rc == 0
        for j in 0:M-1
            lev += r.x[T0+j+1]; bud += r.x[W0+j+1]
        end
    else
        ok = false
    end
    (lev > 1.6 + 1e-4 || abs(bud - 1.0) > 1e-6) && (ok = false)
    @printf("130_30     n=%d m=%d leverage=%.6f budget=%.6f t=%.4fs\n", N, M, lev, bud, sec)

    Z0, UV = M, 2M                                  # robust: max mu0'w - u, (u, A w) in Q
    r, sec = timed_solve() do t
        add_vars(t, 2M + 1)
        add_cons(t, M + 1)
        sense(t, SENSE_MAX)
        for j in 0:M-1
            cj(t, W0 + j, mu[j+1])
            var_bounds(t, W0 + j, BK_LO; lo=0.0, up=1.0)
        end
        for j in 0:M-1
            var_bounds(t, Z0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, UV, BK_LO; lo=0.0, up=Inf)
        cj(t, UV, -1.0)
        for i in 0:M-1                              # z_i - (A w)_i = 0
            sub, val = Int[], Float64[]
            for j in 0:M-1
                a = i == j ? 0.01 : (j == (i + 1) % M ? 0.002 : 0.0)
                a != 0.0 && (push!(sub, W0 + j); push!(val, -a))
            end
            push!(sub, Z0 + i); push!(val, 1.0)
            a_row(t, i, sub, val)
            con_bounds(t, i, BK_FX; lo=0.0, up=0.0)
        end
        a_row(t, M, collect(W0:W0+M-1), ones(M))   # budget
        con_bounds(t, M, BK_FX; lo=1.0, up=1.0)
        cone(t, CT_QUAD, [UV; collect(Z0:Z0+M-1)])
    end
    u, normz, bud = 0.0, 0.0, 0.0
    if r.rc == 0
        u = r.x[UV+1]
        for j in 0:M-1
            normz += r.x[Z0+j+1] * r.x[Z0+j+1]; bud += r.x[W0+j+1]
        end
        normz = sqrt(normz)
    else
        ok = false
    end
    (u < normz - 1e-5 || abs(bud - 1.0) > 1e-6) && (ok = false)
    @printf("robust     n=%d m=%d u=%.6f |Aw|=%.6f budget=%.6f t=%.4fs\n", N, M, u, normz, bud, sec)

    @printf("portfolio  n=%d m=%d %s\n", N, M, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
