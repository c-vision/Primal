#!/usr/bin/env julia
# boyd_qp_box2.jl — barrier-method example, "Convex Optimization" (Boyd &
# Vandenberghe, 2004), Exercise 11.1:
#
#   minimize x^2 + 1  subject to  2 <= x <= 4
#
# The book states the optimal point x* = 2 (objective x^2 = 4 plus the constant
# 1 gives 5). Port of examples/c/boyd_qp_box2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 1)
    r = try
        var_bounds(m, 0, BK_RA; lo=2.0, up=4.0)
        cj(m, 0, 0.0)
        qobj(m, [(0, 0, 2.0)])                     # 0.5*2*x^2 = x^2
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 2.0) < 1e-5
        @printf("boyd_qp_box2  x=%.6g f=%.6g (book x*=2, f=5) %s\n", x[1], z + 1.0, ok ? "OK" : "FAIL")
    else
        @printf("boyd_qp_box2  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
