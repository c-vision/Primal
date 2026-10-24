#!/usr/bin/env julia
# milo1.jl — MOSEK port "milo1": mixed MIP (some variables integer, some
# continuous).
#
#   max 3x0 + 2x1 + 2x2
#   s.t.  x0 + x1 + x2 <= 10,  x0 + 2x1 <= 9,  -x0 + 3x1 + x2 <= 10
#         x0, x1 integer >= 0,  x2 continuous >= 0
#
# Checks: integrality, feasibility, a consistent objective, INTEGER_OPTIMAL,
# and optimality against a complete enumeration of (x0, x1).
# Port of examples/c/milo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 3)
        c = [3.0, 2.0, 2.0]
        for j in 0:2
            cj(m, j, c[j+1])
        end
        sense(m, SENSE_MAX)
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
        a_row(m, 1, [0, 1], [1.0, 2.0])
        a_row(m, 2, [0, 1, 2], [-1.0, 3.0, 1.0])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=10.0)
        con_bounds(m, 1, BK_UP; lo=-Inf, up=9.0)
        con_bounds(m, 2, BK_UP; lo=-Inf, up=10.0)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_type(m, 0, VT_INT)
        var_type(m, 1, VT_INT)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj, sta = r.x, r.objective, r.solsta
    @printf("x = (%.4f, %.4f, %.4f), obj = %.4f\n", xx[1], xx[2], xx[3], obj)

    ok = sta == SOLSTA_INTEGER_OPTIMAL
    best = -1e30                                    # enumeration of (x0, x1) with x2 optimized
    if ok
        for x0 in 0:20, x1 in 0:20
            x0 + 2x1 > 9.0 + 1e-9 && continue      # row1
            v = 10.0 - x0 - x1                      # from row0: x2 <= 10-x0-x1
            w = 10.0 + x0 - 3.0 * x1                # from row2
            w < v && (v = w)
            if v > 0 && 3.0 * x0 + 2.0 * x1 + 2.0 * v > best
                best = 3.0 * x0 + 2.0 * x1 + 2.0 * v
            end
        end
    end
    @printf("enumeration: best = %.4f, clone = %.4f\n", best, obj)
    abs(obj - best) > 1e-6 && (ok = false)
    (abs(xx[1] - floor(xx[1] + 0.5)) > 1e-6 || abs(xx[2] - floor(xx[2] + 0.5)) > 1e-6) && (ok = false)
    abs(obj - (3 * xx[1] + 2 * xx[2] + 2 * xx[3])) > 1e-6 && (ok = false)
    @printf("%s (optimum confirmed by enumeration)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
