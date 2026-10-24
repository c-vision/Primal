#!/usr/bin/env julia
# gp1.jl — MOSEK port "gp1": geometric programming via the log-log transform
# and exponential cones.
#
#   min x0/x1  s.t.  x0 + x1 <= 1,  x0 >= 0.5,  x1 >= 0.5
#
# With x_i = e^{y_i}: t >= e^{y0-y1} is the cone (t, 1, y0 - y1), and the
# log-sum-exp row is u0 + u1 <= 1 with (u_k, 1, y_k) in PEXP. The only
# feasible point is x = (0.5, 0.5), objective 1. Port of examples/c/gp1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const IPAR_INTPNT_MAX_ITERATIONS = 11
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)

function main()
    Y0, Y1, T, ONE, U0, U1 = 0, 1, 2, 3, 4, 5      # y (log of x), t, one, u (lse)
    m = Model(0, 0)
    r = try
        add_vars(m, 6)
        var_bounds(m, Y0, BK_LO; lo=clog(0.5), up=Inf)
        var_bounds(m, Y1, BK_LO; lo=clog(0.5), up=Inf)
        var_bounds(m, T, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        var_bounds(m, U0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, U1, BK_LO; lo=0.0, up=Inf)
        cj(m, T, 1.0)                               # min t
        add_cons(m, 2)
        a_row(m, 0, [U0, U1], [1.0, 1.0])          # r0: u0 + u1 <= 1 (log-sum-exp <= log 1)
        con_bounds(m, 0, BK_UP; lo=-Inf, up=1.0)
        add_vars(m, 1)                              # r1: v - y0 + y1 = 0, v an extra variable
        V = numvar(m) - 1
        var_bounds(m, V, BK_FR; lo=-Inf, up=Inf)
        a_row(m, 1, [V, Y0, Y1], [1.0, -1.0, 1.0])
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        cone(m, CT_PEXP, [T, ONE, V])               # (t, one, v) in PEXP
        cone(m, CT_PEXP, [U0, ONE, Y0])             # (u_k, one, y_k) in PEXP
        cone(m, CT_PEXP, [U1, ONE, Y1])
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
    x0, x1 = cexp(xx[Y0+1]), cexp(xx[Y1+1])
    @printf("y = (%.4f, %.4f) -> x = (%.4f, %.4f), t = %.6f\n", xx[Y0+1], xx[Y1+1], x0, x1, xx[T+1])
    @printf("GP objective = x0/x1 = %.6f (expected 1), conic pobj = %.6f\n", x0 / x1, po)
    ok = abs(x0 + x1 - 1.0) < 1e-3 && x0 >= 0.5 - 1e-6 && x1 >= 0.5 - 1e-6 &&
         abs(po - 1.0) < 1e-3 && abs(x0 / x1 - 1.0) < 1e-3
    @printf("%s (GP log-log: x=(0.5,0.5), obj=1)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
