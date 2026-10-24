#!/usr/bin/env julia
# mil2.jl — port of the "mil2" example: semi-continuous and semi-integer
# variables.
#
#   min x0 + 2 x1  s.t.  x0 + x1 >= 3
#   x0 semi-continuous [2, 10] (x0 = 0 or 2 <= x0 <= 10)
#   x1 semi-integer [2.5, 10] (x1 = 0 or 2.5 <= x1 <= 10, integer)
#
# Enumeration: x1 = 0, x0 = 3 is best: x = (3, 0), obj = 3.
# Port of examples/c/mil2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 2)
        add_cons(m, 1)
        var_bounds(m, 0, BK_RA; lo=2.0, up=10.0)
        var_bounds(m, 1, BK_RA; lo=2.5, up=10.0)
        var_type(m, 0, VT_SEMI_CONT)
        var_type(m, 1, VT_SEMI_INT)
        cj(m, 0, 1.0)
        cj(m, 1, 2.0)
        a_row(m, 0, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_LO; lo=3.0, up=Inf)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.4f, %.4f), obj = %.4f\n", xx[1], xx[2], obj)

    best = Inf                                      # exhaustive enumeration of the semi cases
    for a in 0:1, b in 0:1                          # x0 active? x1 active?
        v1 = b == 1 ? 3.0 : 0.0
        x0 = a == 1 ? max(2.0, 3.0 - v1) : 0.0
        x1 = b == 1 ? ceil(max(2.5, 3.0 - (a == 1 ? 2.0 : 0.0))) : 0.0
        a == 0 && b == 0 && continue                # 0+0 < 3
        a == 0 && x1 < 3.0 - 1e-9 && b == 1 && continue
        cov = (a == 1 ? x0 : 0.0) + (b == 1 ? x1 : 0.0)
        if cov < 3.0 - 1e-9
            # the active member must cover alone or with its companion
            a == 1 && b == 0 && (x0 = 3.0)
            a == 0 && b == 1 && (x1 = 3.0)
            a == 1 && b == 1 && continue            # both active: minimum coverage 2+3=5, ok
            cov = (a == 1 ? x0 : 0.0) + (b == 1 ? x1 : 0.0)
        end
        c = (a == 1 ? x0 : 0.0) + 2.0 * (b == 1 ? x1 : 0.0)
        cov >= 3.0 - 1e-9 && c < best && (best = c)
    end
    @printf("enumeration: best = %.4f\n", best)
    ok = abs(obj - 3.0) < 1e-6 && abs(obj - best) < 1e-6 && abs(xx[1] - 3.0) < 1e-6 && abs(xx[2]) < 1e-6
    @printf("%s (expected x=(3,0), obj=3)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
