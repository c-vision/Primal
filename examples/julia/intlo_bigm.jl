#!/usr/bin/env julia
# intlo_bigm.jl — Big-M worked example, "Introduction to Linear Optimization"
# (A. Nemirovski, 2024), Exercise 4.4(2):
#
#   maximize -2 x1 + x2 + x3 - 6 x4
#   s.t.  x1 + x2 = 2,  x3 + x4 = 2,  x1 + x3 = 2,  x >= 0
#
# The book's Big-M simplex finds x = [0, 2, 2, 0], optimal value 4.
# Port of examples/c/intlo_bigm.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(3, 4)
    r = try
        for j in 0:3
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        sense(m, SENSE_MAX)
        cj(m, 0, -2.0); cj(m, 1, 1.0)
        cj(m, 2, 1.0);  cj(m, 3, -6.0)
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_FX; lo=2.0, up=2.0)
        a_row(m, 1, [2, 3], [1.0, 1.0]); con_bounds(m, 1, BK_FX; lo=2.0, up=2.0)
        a_row(m, 2, [0, 2], [1.0, 1.0]); con_bounds(m, 2, BK_FX; lo=2.0, up=2.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 0.0) < 1e-6 && abs(x[2] - 2.0) < 1e-6 &&
             abs(x[3] - 2.0) < 1e-6 && abs(x[4] - 0.0) < 1e-6 && abs(z - 4.0) < 1e-6
        @printf("intlo_bigm  x=(%.6g,%.6g,%.6g,%.6g) obj=%.6g (book [0,2,2,0], 4) %s\n",
                x[1], x[2], x[3], x[4], z, ok ? "OK" : "FAIL")
    else
        @printf("intlo_bigm  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
