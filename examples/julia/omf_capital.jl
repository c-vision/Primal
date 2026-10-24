#!/usr/bin/env julia
# omf_capital.jl — capital budgeting (0-1) problem, "Optimization Methods in
# Finance" (Cornuejols & Tutuncu, 2007), Section 11.2:
#
#   maximize 8 x1 + 11 x2 + 6 x3 + 4 x4
#   s.t.  6.7 x1 + 10 x2 + 5.5 x3 + 3.4 x4 <= 19,  x in {0,1}^4
#
# The book's integer optimum: x = (0, 1, 1, 1), value 21000.
# Port of examples/c/omf_capital.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(1, 4)
    r = try
        for j in 0:3
            var_bounds(m, j, BK_RA; lo=0.0, up=1.0)
            var_type(m, j, VT_INT_BIN)
        end
        sense(m, SENSE_MAX)
        cj(m, 0, 8.0); cj(m, 1, 11.0)
        cj(m, 2, 6.0); cj(m, 3, 4.0)
        a_row(m, 0, [0, 1, 2, 3], [6.7, 10.0, 5.5, 3.4])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=19.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        for j in 1:4
            ok = ok && abs(x[j] - floor(x[j] + 0.5)) < 1e-6
        end
        ok = ok && abs(x[1]) < 1e-6 && abs(x[2] - 1.0) < 1e-6 &&
             abs(x[3] - 1.0) < 1e-6 && abs(x[4] - 1.0) < 1e-6 && abs(z - 21.0) < 1e-6
        @printf("omf_capital  x=(%.6g,%.6g,%.6g,%.6g) Z=%.6g (=%.0f) (book (0,1,1,1), 21000) %s\n",
                x[1], x[2], x[3], x[4], z, 1000.0 * z, ok ? "OK" : "FAIL")
    else
        @printf("omf_capital  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
