#!/usr/bin/env julia
# mil1.jl — MOSEK port "mil1": a small pure-integer MIP.
#
#   max 5x0 + 4x1 + 3x2
#   s.t.  2x0 + 3x1 + x2 <= 5,  4x0 + x1 + 2x2 <= 11,  3x0 + 4x1 + 2x2 <= 8
#         x integer >= 0
#
# Expected solution: x = (2, 0, 1), obj = 13. Port of examples/c/mil1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 3)
        c = [5.0, 4.0, 3.0]
        for j in 0:2
            cj(m, j, c[j+1])
        end
        sense(m, SENSE_MAX)
        a_row(m, 0, [0, 1, 2], [2.0, 3.0, 1.0])
        a_row(m, 1, [0, 1, 2], [4.0, 1.0, 2.0])
        a_row(m, 2, [0, 1, 2], [3.0, 4.0, 2.0])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=5.0)
        con_bounds(m, 1, BK_UP; lo=-Inf, up=11.0)
        con_bounds(m, 2, BK_UP; lo=-Inf, up=8.0)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            var_type(m, j, VT_INT)
        end
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj, sta = r.x, r.objective, r.solsta
    @printf("x = (%.0f, %.0f, %.0f), obj = %.4f\n", xx[1], xx[2], xx[3], obj)
    ok = sta == SOLSTA_INTEGER_OPTIMAL && abs(xx[1] - 2) < 1e-6 && abs(xx[2] - 0) < 1e-6 &&
         abs(xx[3] - 1) < 1e-6 && abs(obj - 13.0) < 1e-6
    @printf("%s (expected x=(2,0,1), obj=13)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
