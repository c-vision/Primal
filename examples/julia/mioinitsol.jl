#!/usr/bin/env julia
# mioinitsol.jl — MOSEK port "mioinitsol": provide an initial solution to the
# MIP to speed up branch & bound.
#
#   max 3x0 + 2x1 + 2x2
#   s.t.  x0 + x1 + x2 <= 10,  x0 + 2x1 <= 9,  -x0 + 3x1 + x2 <= 10
#         x0, x1 integer >= 0,  x2 >= 0
#
# Initial (feasible, integer) solution x = (3, 3, 1), obj 17; the optimum is
# confirmed by enumeration. Port of examples/c/mioinitsol.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    x0 = [3.0, 3.0, 1.0]
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 3)
        sense(m, SENSE_MAX)
        c = [3.0, 2.0, 2.0]
        for j in 0:2
            cj(m, j, c[j+1])
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0]);  con_bounds(m, 0, BK_UP; lo=-Inf, up=10.0)
        a_row(m, 1, [0, 1], [1.0, 2.0]);          con_bounds(m, 1, BK_UP; lo=-Inf, up=9.0)
        a_row(m, 2, [0, 1, 2], [-1.0, 3.0, 1.0]); con_bounds(m, 2, BK_UP; lo=-Inf, up=10.0)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_type(m, 0, VT_INT)
        var_type(m, 1, VT_INT)
        warm_start(m; x=x0)                         # initial solution (feasible and integer)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.4f, %.4f, %.4f), obj = %.4f (initial obj = %.4f)\n",
            xx[1], xx[2], xx[3], obj, 3.0 * x0[1] + 2.0 * x0[2] + 2.0 * x0[3])

    best = -1e30                                    # complete enumeration (x0,x1 integer, x2 continuous)
    for a in 0:10, b in 0:5
        a + 2b > 9.0 + 1e-9 && continue
        v = 10.0 - a - b
        w = 10.0 + a - 3.0 * b
        w < v && (v = w)
        if v > 1e-9
            o = 3.0 * a + 2.0 * b + 2.0 * v
            o > best && (best = o)
        end
    end
    @printf("enumeration: best = %.4f\n", best)
    ok = abs(obj - best) < 1e-6 && abs(xx[1] - floor(xx[1] + 0.5)) < 1e-6 && abs(xx[2] - floor(xx[2] + 0.5)) < 1e-6
    @printf("%s (optimum confirmed, initsol accepted)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
