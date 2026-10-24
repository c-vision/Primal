#!/usr/bin/env julia
# lo1.jl — simple LP (matches the Python Optimizer API "lo1" example):
#
#   min 2x + 3y  s.t.  x + y >= 1,  x - y <= 0,  x, y >= 0
#
# Expected solution: x = 0.5, y = 0.5, obj = 2.5 (read from the basic
# solution). Port of examples/c/lo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    numvar, numcon = 2, 2
    m = Model(0, 0)
    r = try
        add_vars(m, numvar)
        add_cons(m, numcon)
        for j in 0:numvar-1                         # x, y >= 0
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, 2.0)                               # min 2x + 3y
        cj(m, 1, 3.0)
        sense(m, SENSE_MIN)
        a_row(m, 0, [0, 1], [1.0, 1.0])            # x + y >= 1
        con_bounds(m, 0, BK_LO; lo=1.0, up=Inf)
        a_row(m, 1, [0, 1], [1.0, -1.0])           # x - y <= 0
        con_bounds(m, 1, BK_UP; lo=-Inf, up=0.0)
        solve(m, SOL_BAS)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj, sta = r.x, r.objective, r.solsta
    @printf("solsta = %d\n", sta)
    @printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[1], xx[2], obj)
    ok = abs(xx[1] - 0.5) < 1e-6 && abs(xx[2] - 0.5) < 1e-6 && abs(obj - 2.5) < 1e-6 && sta == SOLSTA_OPTIMAL
    @printf("%s\n", ok ? "OK (expected x=0.5, y=0.5, obj=2.5)" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
