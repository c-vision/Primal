#!/usr/bin/env julia
# mi1.jl — MIP (equivalent to the Julia Optimizer API example, A by columns):
#
#   max x1 + 0.64 x2  s.t.  50 x1 + 31 x2 <= 250,  3 x1 - 2 x2 >= -4,  x integer >= 0
#
# Expected solution: x1 = 5, x2 = 0, obj = 5.0000. Port of examples/c/mi1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_cons(m, 2)
        add_vars(m, 2)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, 1.0)
        cj(m, 1, 0.64)
        # A by columns: col 1 = (50, 3), col 2 = (31, -2)
        a_col(m, 0, [0, 1], [50.0, 3.0])
        a_col(m, 1, [0, 1], [31.0, -2.0])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=250.0)
        con_bounds(m, 1, BK_LO; lo=-4.0, up=Inf)
        sense(m, SENSE_MAX)
        var_type(m, 0, VT_INT)
        var_type(m, 1, VT_INT)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj, sta = r.x, r.objective, r.solsta
    @printf("solsta = %d\n", sta)
    @printf("x1 = %.0f, x2 = %.0f, obj = %.4f\n", xx[1], xx[2], obj)
    ok = abs(xx[1] - 5.0) < 1e-6 && abs(xx[2] - 0.0) < 1e-6 && abs(obj - 5.0) < 1e-6 &&
         sta == SOLSTA_INTEGER_OPTIMAL
    @printf("%s\n", ok ? "OK (expected x1=5, x2=0, obj=5.0000)" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
