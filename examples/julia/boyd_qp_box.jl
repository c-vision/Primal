#!/usr/bin/env julia
# boyd_qp_box.jl — box-constrained QP, "Convex Optimization" (Boyd &
# Vandenberghe, 2004), Exercise 4.3:
#
#   minimize (1/2) x'Px + q'x + r   subject to  -1 <= x_i <= 1
#
# with P = [13 12 -2; 12 17 6; -2 6 12], q = (-22, -14.5, 13), r = 1. The book
# asks to prove x* = (1, 1/2, -1) is optimal (objective -21.625).
# Port of examples/c/boyd_qp_box.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_RA; lo=-1.0, up=1.0)
        end
        cj(m, 0, -22.0); cj(m, 1, -14.5); cj(m, 2, 13.0)
        # upper triangle of P with the full coefficient (mirrored internally)
        qobj(m, [(0, 0, 13.0), (0, 1, 12.0), (0, 2, -2.0), (1, 1, 17.0), (1, 2, 6.0), (2, 2, 12.0)])
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 1.0) < 1e-5 && abs(x[2] - 0.5) < 1e-5 && abs(x[3] + 1.0) < 1e-5
        @printf("boyd_qp_box  x=(%.6g,%.6g,%.6g) f=%.6g (book (1,1/2,-1), f=-21.625) %s\n",
                x[1], x[2], x[3], z + 1.0, ok ? "OK" : "FAIL")
    else
        @printf("boyd_qp_box  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
