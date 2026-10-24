#!/usr/bin/env julia
# boyd_lp_unique.jl — LP with unique optimum, "Convex Optimization"
# (Boyd & Vandenberghe, 2004), Exercise 5.28:
#
#   minimize 47 x1 + 93 x2 + 17 x3 - 93 x4  s.t.  A x <= b
#
# The book states the optimum is unique, x* = (1, 1, 1, 1) (objective 64).
# Port of examples/c/boyd_lp_unique.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    A = [-1.0 -6 1 3; -1 -2 7 1; 0 3 -10 -1; -6 -11 -2 12; 1 6 -1 -3]
    b = [-3.0, 5, -8, -7, 4]
    c = [47.0, 93, 17, -93]
    m = Model(5, 4)
    r = try
        for j in 0:3
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
            cj(m, j, c[j+1])
        end
        for i in 0:4
            a_row(m, i, [0, 1, 2, 3], A[i+1, :])
            con_bounds(m, i, BK_UP; lo=-Inf, up=b[i+1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = all(j -> abs(x[j] - 1.0) < 1e-5, 1:4)
        @printf("boyd_lp_unique  x=(%.6g,%.6g,%.6g,%.6g) obj=%.6g (book (1,1,1,1)) %s\n",
                x[1], x[2], x[3], x[4], z, ok ? "OK" : "FAIL")
    else
        @printf("boyd_lp_unique  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
