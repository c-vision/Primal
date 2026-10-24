#!/usr/bin/env julia
# reoptimization.jl — MOSEK port "reoptimization": solve, change a bound,
# re-optimize.
#
#   1. max x0 + x1  s.t.  x0 + x1 <= 4,  x >= 0   -> obj 4
#   2. the row bound becomes 10                   -> obj 10
#   3. the row bound becomes 2                    -> obj 2
#
# Port of examples/c/reoptimization.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    try
        add_vars(m, 2)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        cj(m, 0, 1.0)
        cj(m, 1, 1.0)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        a_row(m, 0, [0, 1], [1.0, 1.0])

        steps = [4.0, 10.0, 2.0]
        expect = [4.0, 10.0, 2.0]
        ok = true
        for k in 1:3
            con_bounds(m, 0, BK_UP; lo=-Inf, up=steps[k])
            r = solve(m)
            if r.rc != 0
                @printf("optimize %d rc=%d\n", k - 1, r.rc)
                return 1
            end
            po = r.objective
            @printf("step %d (rhs=%g): obj = %.4f (expected %.4f)\n", k, steps[k], po, expect[k])
            abs(po - expect[k]) > 1e-6 && (ok = false)
        end
        @printf("%s (reoptimization: 3 successive solves)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
