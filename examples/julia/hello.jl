#!/usr/bin/env julia
# hello.jl — MOSEK port "hello": the simplest possible task, a free variable,
# min x, x in [2, 4]  ->  x = 2, obj = 2. Port of examples/c/hello.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 1)
        cj(m, 0, 1.0)
        var_bounds(m, 0, BK_RA; lo=2.0, up=4.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    @printf("x = %.6f, obj = %.6f\n", r.x[1], r.objective)
    ok = abs(r.x[1] - 2.0) < 1e-6 && abs(r.objective - 2.0) < 1e-6
    @printf("%s (expected x = 2)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
