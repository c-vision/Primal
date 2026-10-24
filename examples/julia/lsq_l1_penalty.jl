#!/usr/bin/env julia
# lsq_l1_penalty.jl — positivity least squares with a weighted L1 penalty
# (tschm/MosekRegression, lsq_pos_l1_penalty):
#
#   min ||X w - rhs||_2^2 + sum_i gamma_i |w_i - w0_i|  s.t.  e'w = 1,  w >= 0
#
# Conic form: (1/2, v, Xw - rhs) in Qr, (t_i, p_i) in Q^2, min v + sum t_i.
# Checks: A) w0 = 0 and uniform gamma reduce to the QP + gamma (same w);
# B) M = 2 with general gamma and w0, against a fine brute-force scan.
# Port of examples/c/lsq_l1_penalty.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Conic model; returns (objective or -1, w).
function l1_solve(N, M, X, b, gam, w0)
    W0, R0, V, P0, T0 = 0, M, M + N, M + N + 1, M + N + 1 + M
    HALF = M + N + 1 + 2M
    m = Model(0, 0)
    r = try
        add_vars(m, HALF + 1)
        add_cons(m, 1 + N + M)
        sense(m, SENSE_MIN)
        for i in 0:M-1
            var_bounds(m, W0 + i, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:N-1
            var_bounds(m, R0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, V, BK_FR; lo=-Inf, up=Inf)
        for i in 0:M-1
            var_bounds(m, P0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:M-1
            var_bounds(m, T0 + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, HALF, BK_FX; lo=0.5, up=0.5)
        a_row(m, 0, collect(W0:W0+M-1), ones(M))   # e'w = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for i in 0:N-1                              # r - X w = -b
            sub, val = [R0 + i], [1.0]
            for j in 0:M-1
                X[i+1, j+1] != 0.0 && (push!(sub, W0 + j); push!(val, -X[i+1, j+1]))
            end
            a_row(m, 1 + i, sub, val)
            con_bounds(m, 1 + i, BK_FX; lo=-b[i+1], up=-b[i+1])
        end
        for i in 0:M-1                              # p_i - gamma_i w_i = -gamma_i w0_i
            a_row(m, 1 + N + i, [P0 + i, W0 + i], [1.0, -gam[i+1]])
            con_bounds(m, 1 + N + i, BK_FX; lo=-gam[i+1] * w0[i+1], up=-gam[i+1] * w0[i+1])
        end
        cone(m, CT_RQUAD, [HALF; V; collect(R0:R0+N-1)])   # v >= ||r||^2
        for i in 0:M-1                              # t_i >= |p_i|
            cone(m, CT_QUAD, [T0 + i, P0 + i])
        end
        cj(m, V, 1.0)
        for i in 0:M-1
            cj(m, T0 + i, 1.0)
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return -1.0, zeros(M)
    return r.objective, r.x[W0+1:W0+M]
end

# QP reference: min ||Xw - b||^2 s.t. e'w = 1, w >= 0; returns (objective, w).
function qp_solve(N, M, X, b)
    m = Model(0, 0)
    r = try
        add_vars(m, M)
        add_cons(m, 1)
        sense(m, SENSE_MIN)
        for i in 0:M-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        a_row(m, 0, collect(0:M-1), ones(M))
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        # ||Xw-b||^2 = 0.5 w'(2X'X)w - 2(X'b)'w + b'b, lower triangle, full coefficient
        for i in 0:M-1, j in 0:i
            q = 0.0
            for k in 1:N
                q += 2.0 * X[k, i+1] * X[k, j+1]
            end
            qobj_ij(m, i, j, q)
        end
        cf = 0.0
        for k in 1:N
            cf += b[k] * b[k]
        end
        cfix(m, cf)
        for i in 0:M-1
            c = 0.0
            for k in 1:N
                c += -2.0 * b[k] * X[k, i+1]
            end
            cj(m, i, c)
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return -1.0, zeros(M)
    return r.objective, r.x[1:M]
end

function main()
    allok = true
    println("lsq_l1_penalty")
    # case A: w0 = 0, uniform gamma -> reduce to the QP + gamma
    N, M = 8, 3
    X, b = zeros(N, M), zeros(N)
    st = UInt32(987654321)
    for i in 1:N
        st = st * 0x41c64e6d + 0x00003039; b[i] = Float64((st >> 16) & 0x3ff) / 1024.0
        for j in 1:M
            st = st * 0x41c64e6d + 0x00003039
            X[i, j] = 0.2 + Float64((st >> 16) & 0x3ff) / 1024.0
        end
    end
    gamma = 0.7
    ol, wl = l1_solve(N, M, X, b, fill(gamma, M), zeros(M))
    oq, wq = qp_solve(N, M, X, b)
    dw = maximum(abs.(wl .- wq))
    good = ol > 0 && oq > 0 && dw < 1e-6 && abs(ol - (oq + gamma)) < 1e-6
    @printf("  A (w0=0, gamma=%.2f): L1 obj=%.8f  QP obj+gamma=%.8f  |dw|=%.2e  %s\n",
            gamma, ol, oq + gamma, dw, good ? "OK" : "FAIL")
    allok &= good

    # case B: M = 2, general gamma and w0 -> brute-force scan of w1
    N, M = 5, 2
    X = [0.5 1.0; 1.0 0.4; 0.2 0.9; 0.8 0.6; 0.4 0.4]
    b = [0.7, 0.5, 0.3, 0.9, 0.4]
    gam, w0 = [0.3, 1.1], [0.6, 0.4]
    ol, wl = l1_solve(N, M, X, b, gam, w0)
    best, bestw1 = Inf, 0.0
    for k in 0:2000000
        w1 = k / 2000000.0
        w2 = 1.0 - w1
        f = 0.0
        for i in 1:N
            rr = X[i, 1] * w1 + X[i, 2] * w2 - b[i]
            f += rr * rr
        end
        f += gam[1] * abs(w1 - w0[1]) + gam[2] * abs(w2 - w0[2])
        f < best && ((best, bestw1) = (f, w1))
    end
    good = ol > 0 && abs(ol - best) < 1e-5 && abs(wl[1] - bestw1) < 1e-4
    @printf("  B (M=2, gamma=(%.1f,%.1f)): conic obj=%.8f  brute obj=%.8f  w1=%.6f (bf %.6f)  %s\n",
            gam[1], gam[2], ol, best, wl[1], bestw1, good ? "OK" : "FAIL")
    allok &= good
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
