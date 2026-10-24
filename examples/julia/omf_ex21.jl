#!/usr/bin/env julia
# omf_ex21.jl — Example 2.1 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007): the running bond-portfolio LP. $100,000 split
# between a corporate bond (4% yield, maturity 3y, rating 2) and a government
# bond (3%, 4y, rating 1); average rating <= 1.5, average maturity <= 3.6y.
# In thousands of dollars:
#
#   maximize Z = 4 x1 + 3 x2
#   s.t.  x1 + x2 <= 100,  2 x1 + x2 <= 150,  3 x1 + 4 x2 <= 360,  x >= 0
#
# The book's simplex ends at x = (50, 50), Z = 350. Port of examples/c/omf_ex21.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(3, 2)
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, -4.0); cj(m, 1, -3.0)             # maximize
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_UP; lo=-Inf, up=100.0)
        a_row(m, 1, [0, 1], [2.0, 1.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=150.0)
        a_row(m, 2, [0, 1], [3.0, 4.0]); con_bounds(m, 2, BK_UP; lo=-Inf, up=360.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 50.0) < 1e-6 && abs(x[2] - 50.0) < 1e-6 && abs(z + 350.0) < 1e-6
        @printf("omf_ex21  x=(%.6g,%.6g) Z=%.6g (book x=(50,50), Z=350) %s\n", x[1], x[2], -z, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex21  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
