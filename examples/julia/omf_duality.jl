#!/usr/bin/env julia
# omf_duality.jl — first duality example, "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 2.2:
#
#   minimize -x1 - x2  s.t.  2 x1 + x2 <= 12,  x1 + 2 x2 <= 9,  x >= 0
#
# The book finds x = (5, 2), objective -7, certified by the dual multipliers
# (1/3, 1/3). Port of examples/c/omf_duality.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(2, 2)
    r = try
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, -1.0); cj(m, 1, -1.0)
        a_row(m, 0, [0, 1], [2.0, 1.0]); con_bounds(m, 0, BK_UP; lo=-Inf, up=12.0)
        a_row(m, 1, [0, 1], [1.0, 2.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=9.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 5.0) < 1e-6 && abs(x[2] - 2.0) < 1e-6 && abs(z + 7.0) < 1e-6
        @printf("omf_duality  x=(%.6g,%.6g) obj=%.6g (book (5,2), -7) %s\n", x[1], x[2], z, ok ? "OK" : "FAIL")
    else
        @printf("omf_duality  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
