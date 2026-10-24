#!/usr/bin/env julia
# intlo_simplex.jl — worked primal-simplex example, "Introduction to Linear
# Optimization" (A. Nemirovski, 2024), Section 4.3.2:
#
#   maximize 5 x1 + 3 x2 + 6 x3
#   s.t.  x1 + x2 + 2 x3 <= 30,  4 x1 + x2 + 4 x3 <= 60,  2 x1 + x2 + x3 <= 30,  x >= 0
#
# The book's tableau simplex ends at x* = (6, 12, 6), optimal value 102.
# Port of examples/c/intlo_simplex.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(3, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        sense(m, SENSE_MAX)
        cj(m, 0, 5.0); cj(m, 1, 3.0); cj(m, 2, 6.0)
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 2.0]); con_bounds(m, 0, BK_UP; lo=-Inf, up=30.0)
        a_row(m, 1, [0, 1, 2], [4.0, 1.0, 4.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=60.0)
        a_row(m, 2, [0, 1, 2], [2.0, 1.0, 1.0]); con_bounds(m, 2, BK_UP; lo=-Inf, up=30.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 6.0) < 1e-6 && abs(x[2] - 12.0) < 1e-6 &&
             abs(x[3] - 6.0) < 1e-6 && abs(z - 102.0) < 1e-6
        @printf("intlo_simplex  x=(%.6g,%.6g,%.6g) obj=%.6g (book (6,12,6), 102) %s\n",
                x[1], x[2], x[3], z, ok ? "OK" : "FAIL")
    else
        @printf("intlo_simplex  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
