#!/usr/bin/env julia
# cqo1.jl — MOSEK port "cqo1": small conic problem with one quadratic and one
# rotated cone (the ACCs of the original are cones over direct variables).
#
#   min x3 + x4 + x5
#   s.t.  x0 + x1 + 2 x2 = 1,  (x3, x0, x1) in QUAD_3,  (x4, x5, x2) in RQUAD_3
#         x0, x1, x2 >= 0
#
# Known optimum: obj = 1/sqrt(2) ~ 0.7071. Port of examples/c/cqo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 6)
        add_cons(m, 1)
        sense(m, SENSE_MIN)
        cj(m, 3, 1.0)
        cj(m, 4, 1.0)
        cj(m, 5, 1.0)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        for j in 3:5
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 2.0])    # x0 + x1 + 2 x2 = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        cone(m, CT_QUAD, [3, 0, 1])                 # QUAD_3: (x3, x0, x1)
        cone(m, CT_RQUAD, [4, 5, 2])                # RQUAD_3: (x4, x5, x2)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    x, po = r.x, r.objective
    @printf("x = (%.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n", x[1], x[2], x[3], x[4], x[5], x[6])
    @printf("obj = %.6f (expected 1/sqrt(2) = %.6f)\n", po, 1.0 / sqrt(2.0))
    ok = abs(x[1] + x[2] + 2 * x[3] - 1.0) < 1e-6 && abs(po - 1.0 / sqrt(2.0)) < 1e-6 &&
         x[4] * x[4] >= x[1] * x[1] + x[2] * x[2] - 1e-8 && 2 * x[5] * x[6] >= x[3] * x[3] - 1e-8 &&
         abs(x[4] + x[5] + x[6] - po) < 1e-9
    @printf("%s (equality + QUAD/RQUAD cones + optimum)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
