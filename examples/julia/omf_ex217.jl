#!/usr/bin/env julia
# omf_ex217.jl — Exercise 2.17 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), the simplex method:
#
#   maximize 4 x1 + x2 - x3  s.t.  x1 + 3 x3 <= 6,  3 x1 + x2 + 3 x3 <= 9,  x >= 0
#
# The book's optimum: x = (3, 0, 0), objective 12. Port of examples/c/omf_ex217.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(2, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, -4.0); cj(m, 1, -1.0); cj(m, 2, 1.0)
        a_row(m, 0, [0, 2], [1.0, 3.0]);       con_bounds(m, 0, BK_UP; lo=-Inf, up=6.0)
        a_row(m, 1, [0, 1, 2], [3.0, 1.0, 3.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=9.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(z + 12.0) < 1e-6 && abs(x[1] - 3.0) < 1e-6 && abs(x[2]) < 1e-6 && abs(x[3]) < 1e-6
        @printf("omf_ex217  x=(%.6g,%.6g,%.6g)  obj=%.6g (book x=(3,0,0), 12) %s\n",
                x[1], x[2], x[3], -z, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex217  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
