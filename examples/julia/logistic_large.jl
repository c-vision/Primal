#!/usr/bin/env julia
# logistic_large.jl — sum-of-exponentials regression via exponential cones
# (scaling test for the conic path with many PEXP cones):
#
#   min sum_i v_i + 0.5 gamma ||x||^2  s.t.  v_i >= exp(a_i'x + c_i)
#
# The quadratic term goes exactly into a rotated cone. Verified against
# gradient descent on f(x). The data come from the same 32-bit linear
# congruential generator as the C example.
#
#   julia logistic_large.jl [n] [d]   (default 40 8)
#
# Port of examples/c/logistic_large.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(11235813)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)      # the C library's exp
const GAMMA = 0.1

# Objective value of the sum-of-exponentials model at x.
function fval(x, A, c, N, D)
    s = 0.0
    for i in 0:N-1
        u = c[i+1]
        for j in 0:D-1
            u += A[i*D+j+1] * x[j+1]
        end
        s += cexp(u)
    end
    for j in 1:D
        s += 0.5 * GAMMA * x[j] * x[j]
    end
    return s
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 40
    D = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 8
    (N < 1 || D < 1) && return 2
    A = [(rnd() * 2.0 - 1.0) * 0.5 for _ in 1:N*D]
    c = [rnd() * 2.0 - 1.0 for _ in 1:N]

    X0, V0, U0, ONE = 0, D, D + N, D + 2N          # vars: x (D), v (N), u (N), one, w
    W = ONE + 1
    t = Model(0, 0)
    try
        add_vars(t, D + 2N + 2)
        for j in 0:D-1
            var_bounds(t, X0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:N-1
            var_bounds(t, V0 + i, BK_LO; lo=0.0, up=Inf)
            cj(t, V0 + i, 1.0)
            var_bounds(t, U0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, ONE, BK_FX; lo=1.0, up=1.0)
        var_bounds(t, W, BK_LO; lo=0.0, up=Inf)
        cj(t, W, GAMMA)
        add_cons(t, N)
        for i in 0:N-1                              # a_i'x - u_i = -c_i
            a_row(t, i, [collect(X0:X0+D-1); U0 + i], [A[i*D+1:i*D+D]; -1.0])
            con_bounds(t, i, BK_FX; lo=-c[i+1], up=-c[i+1])
        end
        for i in 0:N-1
            cone(t, CT_PEXP, [V0 + i, ONE, U0 + i])
        end
        cone(t, CT_RQUAD, [W; ONE; collect(X0:X0+D-1)])   # W >= 0.5 ||x||^2

        t0 = time()
        r = solve(t)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj = r.objective
            xg = zeros(D)                           # gradient descent reference
            fg, step = fval(xg, A, c, N, D), 0.5
            for _ in 1:20000
                g = zeros(D)
                for j in 1:D
                    h = 1e-6
                    xp, xm = copy(xg), copy(xg)
                    xp[j] += h; xm[j] -= h
                    g[j] = (fval(xp, A, c, N, D) - fval(xm, A, c, N, D)) / (2 * h)
                end
                xn = [xg[j] - step * g[j] for j in 1:D]
                fn = fval(xn, A, c, N, D)
                if fn < fg
                    xg = xn; fg = fn; step *= 1.05
                elseif fn < fg + 1e-12
                    step *= 0.7
                else
                    step *= 0.5
                end
                step < 1e-12 && break
            end
            ok = abs(obj - fg) < 1e-3 * (1.0 + abs(fg))
            @printf("logistic   n=%d d=%d cones=%d obj=%.6f grad=%.6f t=%.4fs %s\n", N, D, N, obj, fg, sec, ok ? "OK" : "FAIL")
        else
            @printf("logistic   n=%d d=%d rc=%d FAIL\n", N, D, r.rc)
        end
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
