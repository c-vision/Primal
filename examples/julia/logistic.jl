#!/usr/bin/env julia
# logistic.jl — MOSEK port "logistic": conic regression with the softmax
# (sum of exponentials):
#
#   min t + 0.5 gamma ||x||^2  s.t.  y_i = a_i'x,  t >= sum_i exp(y_i)
#
# Each exponential is a cone (v_i, 1, y_i) in PEXP with t = sum v_i; the
# quadratic term goes into the rotated cone (w, 1, x) (0.5 ||x||^2 <= w).
# Checked against a finite-difference gradient descent on the smooth objective.
# Port of examples/c/logistic.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const M = 4
const NX = 2
const A = [2.0 1.0; -1.0 1.0; 1.0 -2.0; -2.0 -1.0]
const GAMMA = 0.1
const IPAR_INTPNT_MAX_ITERATIONS = 11
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)      # the C library's exp

# smooth objective f(x) = sum_i exp(a_i'x) + 0.5 gamma ||x||^2
function fval(x)
    s = 0.0
    for i in 1:M
        s += cexp(A[i, 1] * x[1] + A[i, 2] * x[2])
    end
    return s + 0.5 * GAMMA * (x[1] * x[1] + x[2] * x[2])
end

function main()
    # variables: [x0, x1, v_0..v_{M-1}, u_0..u_{M-1}, one, w]
    NV = 2 + 2M + 2
    V0, U0, ONE = 2, 2 + M, 2 + 2M
    W = ONE + 1
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, M)
        sense(m, SENSE_MIN)
        for i in 0:M-1
            cj(m, V0 + i, 1.0)
        end
        cj(m, W, GAMMA)
        for i in 0:M-1                              # a_i'x - u_i = 0
            a_row(m, i, [0, 1, U0 + i], [A[i+1, 1], A[i+1, 2], -1.0])
            con_bounds(m, i, BK_FX; lo=0.0, up=0.0)
        end
        for j in 0:NV-1
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        for i in 0:M-1                              # big-M on the u_i: bounds the round-0 LP
            var_bounds(m, U0 + i, BK_RA; lo=-30.0, up=30.0)
        end
        for i in 0:M-1
            cone(m, CT_PEXP, [V0 + i, ONE, U0 + i])
        end
        cone(m, CT_RQUAD, [W, ONE, 0, 1])
        int_param(m, IPAR_INTPNT_MAX_ITERATIONS, 5000)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, po = r.x, r.objective
    @printf("x = (%.6f, %.6f), obj = %.6f\n", xx[1], xx[2], po)

    xg = [0.0, 0.0]                                 # independent check: gradient descent on f
    fbest, step = fval(xg), 0.5
    for _ in 1:20000
        g = zeros(2)
        for d in 1:2
            h = 1e-6
            xp, xm = copy(xg), copy(xg)
            xp[d] += h; xm[d] -= h
            g[d] = (fval(xp) - fval(xm)) / (2 * h)
        end
        xn = [xg[1] - step * g[1], xg[2] - step * g[2]]
        fn = fval(xn)
        if fn < fbest
            fbest = fn; xg = xn; step *= 1.05
        elseif fn < fbest + 1e-12
            step *= 0.7                             # close: shrink the step
        else
            step *= 0.5
        end
        step < 1e-12 && break
    end
    @printf("numerical gradient: f* = %.6f at (%.6f, %.6f)\n", fbest, xg[1], xg[2])
    ok = abs(po - fbest) < 1e-3 && abs(xx[1] - xg[1]) < 1e-2 && abs(xx[2] - xg[2]) < 1e-2
    @printf("%s (conic = numerical gradient)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
