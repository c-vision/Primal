#!/usr/bin/env julia
# mico1.jl — MOSEK port "mico1": mixed-integer conic optimization.
#
#   max x0  s.t.  (t, x0, x1) in QUAD,  t = 1,  x0 + x1 >= 0.5,  x0 integer >= 0,  x1 >= 0
#
# The cone gives x0^2 + x1^2 <= 1, so x0 in {0, 1}; x0 = 1, x1 = 0 is optimal,
# obj = 1. Port of examples/c/mico1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)                              # t, x0, x1
        add_cons(m, 1)
        var_bounds(m, 0, BK_FX; lo=1.0, up=1.0)     # t = 1
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_LO; lo=0.0, up=Inf)
        var_type(m, 1, VT_INT)
        cj(m, 1, 1.0)
        sense(m, SENSE_MAX)
        a_row(m, 0, [1, 2], [1.0, 1.0])            # x0 + x1 >= 0.5
        con_bounds(m, 0, BK_LO; lo=0.5, up=Inf)
        cone(m, CT_QUAD, [0, 1, 2])                 # (t, x0, x1) in QUAD
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.4f, %.4f), obj = %.4f\n", xx[2], xx[3], obj)

    best = -1e30                                    # check by enumeration: x0 in {0,1}, cone + row
    for a in 0:1
        xmax = sqrt(1.0 - Float64(a) * a)          # cone: x1 <= sqrt(1-a^2)
        need = 0.5 - a                              # row: x1 >= 0.5-a
        need > xmax + 1e-12 && continue             # x1 cannot cover the row
        o = Float64(a)
        o > best && (best = o)
    end
    @printf("enumeration: best = %.4f\n", best)
    ok = abs(obj - best) < 1e-6 && abs(xx[2] - floor(xx[2] + 0.5)) < 1e-6
    @printf("%s (MIP+cone: optimum by enumeration)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
