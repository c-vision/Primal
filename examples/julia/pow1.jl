#!/usr/bin/env julia
# pow1.jl — power cone (equivalent of the MOSEK docs pow example:
# min x + y  s.t.  x^0.5 y^0.5 >= 1, x, y >= 0).
#
# PPOW cone form (t, u, v): t^a u^(1-a) >= |v|, a = 0.5. With (t, u, v) =
# (x, y, 1): x y >= 1, so the optimum is obj = 2 at (x, y) = (1, 1).
# Port of examples/c/pow1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)                              # t = x, u = y, v = 1 (fixed)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_FX; lo=1.0, up=1.0)
        cj(m, 0, 1.0)                               # min x + y
        cj(m, 1, 1.0)
        sense(m, SENSE_MIN)
        cone(m, CT_PPOW, [0, 1, 2]; param=0.5)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[1], xx[2], obj)
    ok = abs(obj - 2.0) < 1e-4 && abs(xx[1] - 1.0) < 1e-3 && abs(xx[2] - 1.0) < 1e-3
    @printf("%s (expected obj = 2.0 at x = y = 1)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
