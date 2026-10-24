#!/usr/bin/env julia
# solvelinear.jl — MOSEK port "solvelinear": solve a linear system A x = b as
# an optimization problem (min t  s.t.  -t <= (Ax - b)_i <= t).
#
#   A = [2 1 -1; -3 -1 2; -2 1 2],  b = (8, -11, -3)  ->  x = (2, 3, -1), t = 0
#
# Port of examples/c/solvelinear.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 4)                              # x0, x1, x2, t
        for j in 0:2
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, 3, BK_LO; lo=0.0, up=Inf)
        cj(m, 3, 1.0)                               # min t
        # rows: (Ax)_i - t <= b_i  and  -(Ax)_i - t <= -b_i
        add_cons(m, 6)
        A = [2.0 1 -1; -3 -1 2; -2 1 2]
        b = [8.0, -11, -3]
        for i in 0:2
            sub = [0, 1, 2, 3]
            a_row(m, i, sub, [A[i+1, 1], A[i+1, 2], A[i+1, 3], -1.0])
            con_bounds(m, i, BK_UP; lo=-Inf, up=b[i+1])
            a_row(m, 3 + i, sub, [-A[i+1, 1], -A[i+1, 2], -A[i+1, 3], -1.0])
            con_bounds(m, 3 + i, BK_UP; lo=-Inf, up=-b[i+1])
        end
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, po = r.x, r.objective
    @printf("x = (%.4f, %.4f, %.4f), t = %.6f (expected (2, 3, -1), t=0)\n", xx[1], xx[2], xx[3], xx[4])
    ok = abs(xx[1] - 2.0) < 1e-6 && abs(xx[2] - 3.0) < 1e-6 && abs(xx[3] + 1.0) < 1e-6 && abs(po) < 1e-6
    @printf("%s (linear system solved: residual ~ 0)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
