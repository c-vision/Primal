#!/usr/bin/env julia
# qo1.jl — QP (matches the Python Optimizer API "qo1" example):
#
#   min x^2 + y^2  s.t.  x + y >= 1,  x, y >= 0
#
# Expected solution: x = 0.5, y = 0.5, obj = 0.5. The quadratic term is
# 0.5 x'Qx with Q = diag(2) (lower triangle, full coefficient, as MOSEK).
# Port of examples/c/qo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    numvar, numcon = 2, 1
    m = Model(0, 0)
    r = try
        add_vars(m, numvar)
        add_cons(m, numcon)
        for j in 0:numvar-1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        qobj(m, [(0, 0, 2.0), (1, 1, 2.0)])        # Q = diag(2): 0.5 x'Qx = x^2 + y^2
        sense(m, SENSE_MIN)
        a_row(m, 0, [0, 1], [1.0, 1.0])            # x + y >= 1
        con_bounds(m, 0, BK_LO; lo=1.0, up=Inf)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = %.6f, y = %.6f, obj = %.6f\n", xx[1], xx[2], obj)
    ok = abs(xx[1] - 0.5) < 1e-6 && abs(xx[2] - 0.5) < 1e-6 && abs(obj - 0.5) < 1e-6
    @printf("%s\n", ok ? "OK (expected x=0.5, y=0.5, obj=0.5)" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
