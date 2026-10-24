#!/usr/bin/env julia
# simple.jl — MOSEK port "simple": the simplest possible LP, checking the
# whole API chain.
#
#   max x0 + x1  s.t.  x0 + x1 = 1,  0 <= x0, x1 <= 1
#
# Any point on the row is optimal with obj = 1 (degenerate: check the
# objective, not the point). Port of examples/c/simple.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 2)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        cj(m, 0, 1.0)
        cj(m, 1, 1.0)
        var_bounds(m, 0, BK_RA; lo=0.0, up=1.0)
        var_bounds(m, 1, BK_RA; lo=0.0, up=1.0)
        a_row(m, 0, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, y, po, dobj, sta = r.x, r.y, r.objective, r.dual, r.solsta
    @printf("x = (%.4f, %.4f), y = (%.4f)\n", xx[1], xx[2], y[1])
    @printf("primal obj = %.4f, dual obj = %.4f, sta = %d\n", po, dobj, sta)
    ok = sta == SOLSTA_OPTIMAL && abs(po - 1.0) < 1e-9 &&
         abs(dobj - 1.0) < 1e-6 &&                  # strong duality
         abs(xx[1] + xx[2] - 1.0) < 1e-6 &&         # feasibility
         abs(xx[1] - (1.0 - xx[2])) < 1e-6
    @printf("%s (expected obj=1, pobj=dobj, feasible)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
