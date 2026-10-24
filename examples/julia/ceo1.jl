#!/usr/bin/env julia
# ceo1.jl — exponential cone (variant of the MOSEK Optimizer API "ceo1"
# example):
#
#   min x0  s.t.  (x0, x1, x2) in EXP,  x1 = 1,  -1 <= x2 <= -0.5
#   i.e. x0 >= x1 exp(x2/x1) = exp(x2)
#
# Expected solution: x0 = exp(-1) = 0.367879 at x2 = -1.
# Port of examples/c/ceo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_FX; lo=1.0, up=1.0)
        var_bounds(m, 2, BK_RA; lo=-1.0, up=-0.5)
        cj(m, 0, 1.0)                               # min x0
        sense(m, SENSE_MIN)
        cone(m, CT_PEXP, [0, 1, 2])
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.6f, %.6f, %.6f), obj = %.6f\n", xx[1], xx[2], xx[3], obj)
    ok = abs(obj - exp(-1.0)) < 1e-6 && abs(xx[3] - (-1.0)) < 1e-6
    @printf("%s (expected obj = e^-1 = 0.367879 in x2 = -1)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
