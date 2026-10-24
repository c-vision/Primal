#!/usr/bin/env julia
# sos1.jl — port of the "sos1" example: SOS type-1 constraint.
#
#   min x0 + x1 + x2  s.t.  x0 + 2 x1 + 3 x2 >= 2.5,  SOS1 {x0, x1, x2},  0 <= x <= 1
#
# At most ONE member is nonzero; the cheapest cover is x2 = 2.5/3 = 5/6.
# Optimum: x = (0, 0, 5/6), obj = 5/6. Port of examples/c/sos1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 1)
        for j in 0:2
            var_bounds(m, j, BK_RA; lo=0.0, up=1.0)
            cj(m, j, 1.0)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 2.0, 3.0])
        con_bounds(m, 0, BK_LO; lo=2.5, up=Inf)
        sos(m, 1, [0, 1, 2]; weights=[1.0, 2.0, 3.0])
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj = r.x, r.objective
    @printf("x = (%.4f, %.4f, %.4f), obj = %.4f\n", xx[1], xx[2], xx[3], obj)

    # SOS1 check by enumerating the 3 members: only member k active with
    # x_k = 2.5/coverage_k if <= 1
    best = Inf
    cov = [1.0, 2.0, 3.0]
    for k in 1:3
        need = 2.5 / cov[k]
        if need <= 1.0 + 1e-9 && need < best
            best = need
        end
    end
    nact = count(j -> xx[j] > 1e-6, 1:3)
    ok = abs(obj - best) < 1e-6 && nact <= 1 && abs(obj - (xx[1] + xx[2] + xx[3])) < 1e-6
    @printf("enumeration: best = %.4f, nonzeros = %d\n", best, nact)
    @printf("%s (expected obj=5/6, a single nonzero member)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
