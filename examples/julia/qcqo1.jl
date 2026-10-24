#!/usr/bin/env julia
# qcqo1.jl — MOSEK port "qcqo1": quadratic constraints.
#
#   max x0 + x1  s.t.  x0^2 + x1^2 <= 1,  x >= 0
#
# Optimum: x* = (1, 1)/sqrt(2), obj = sqrt(2) ~ 1.414214. The quadratic row
# uses the MOSEK convention 1/2 x'Qx with Q = 2I. Port of examples/c/qcqo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 2)
        add_cons(m, 1)
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, 1.0)
        cj(m, 1, 1.0)
        sense(m, SENSE_MAX)
        qconk(m, 0, [(0, 0, 2.0), (1, 1, 2.0)])    # quadratic constraint: x0^2 + x1^2 <= 1
        con_bounds(m, 0, BK_UP; lo=-Inf, up=1.0)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.6f, %.6f), obj = %.6f\n", xx[1], xx[2], obj)
    ok = abs(xx[1] - 1.0 / sqrt(2.0)) < 1e-4 && abs(xx[2] - 1.0 / sqrt(2.0)) < 1e-4 &&
         abs(obj - sqrt(2.0)) < 1e-4
    @printf("%s (expected x=(1,1)/sqrt2, obj=sqrt2)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
