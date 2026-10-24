#!/usr/bin/env julia
# omf_ex25.jl — Exercise 2.5 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), duality:
#
#   minimize 2 x1 + 3 x2  s.t.  x1 + x2 >= 5,  x1 >= 1,  x2 >= 2
#
# The book asks to prove x* = (3, 2) is optimal (value 12).
# Port of examples/c/omf_ex25.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(1, 2)
    r = try
        var_bounds(m, 0, BK_LO; lo=1.0, up=Inf)    # x1 >= 1
        var_bounds(m, 1, BK_LO; lo=2.0, up=Inf)    # x2 >= 2
        cj(m, 0, 2.0); cj(m, 1, 3.0)
        a_row(m, 0, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_LO; lo=5.0, up=Inf)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 3.0) < 1e-6 && abs(x[2] - 2.0) < 1e-6 && abs(z - 12.0) < 1e-6
        @printf("omf_ex25  x=(%.6g,%.6g) obj=%.6g (book x=(3,2), 12) %s\n", x[1], x[2], z, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex25  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
