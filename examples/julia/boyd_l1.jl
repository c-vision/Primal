#!/usr/bin/env julia
# boyd_l1.jl — parametrized l1-norm approximation, "Convex Optimization"
# (Boyd & Vandenberghe, 2004), Exercise 5.33:
#
#   minimize || A x + b + eps d ||_1,  x in R^3
#
# For eps = 0 the book asks to prove x* = (1, 1, 1) is optimal. Solved as an
# LP (x free, z >= 0 with z_i >= +/-(Ax+b)_i, minimize sum z); value 4.
# Port of examples/c/boyd_l1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    A = [-2 7 1; -5 -1 3; -7 3 -5; -1 4 -4; 1 5 5; 2 -5 -1]
    b = [-4, 3, 9, 0, -11, 5]
    m = Model(12, 9)                                # 3 x + 6 z
    r = try
        for j in 0:2
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        for j in 3:8
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            cj(m, j, 1.0)
        end
        for i in 0:5
            a = A[i+1, :]
            # z_i >= (A x + b)_i  <=>  z_i - a_i.x >= b_i
            a_row(m, 2i, [0, 1, 2, 3 + i], [-a[1], -a[2], -a[3], 1.0])
            con_bounds(m, 2i, BK_LO; lo=b[i+1], up=Inf)
            # z_i >= -(A x + b)_i  <=>  z_i + a_i.x >= -b_i
            a_row(m, 2i + 1, [0, 1, 2, 3 + i], [a[1], a[2], a[3], 1.0])
            con_bounds(m, 2i + 1, BK_LO; lo=-b[i+1], up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 1.0) < 1e-5 && abs(x[2] - 1.0) < 1e-5 && abs(x[3] - 1.0) < 1e-5 && abs(z - 4.0) < 1e-5
        @printf("boyd_l1  x=(%.6g,%.6g,%.6g) ||Ax+b||_1=%.6g (book (1,1,1), 4) %s\n",
                x[1], x[2], x[3], z, ok ? "OK" : "FAIL")
    else
        @printf("boyd_l1  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
