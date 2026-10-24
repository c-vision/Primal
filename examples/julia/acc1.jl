#!/usr/bin/env julia
# acc1.jl — MOSEK port "acc1": affine conic constraints (ACC).
#
#   min x1  s.t.  x0 = 3,  (v0, v1) in QUAD with v0 = 2 x0 + x1, v1 = x0 - 1
#
# v0 = 6 + x1 >= |v1| = 2, so x1 >= -4: optimum x = (3, -4), obj = -4.
# Port of examples/c/acc1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 2)
        add_cons(m, 1)
        var_bounds(m, 0, BK_FX; lo=3.0, up=3.0)
        var_bounds(m, 1, BK_FR; lo=-Inf, up=Inf)
        cj(m, 1, 1.0)
        a_row(m, 0, [0], [1.0])
        con_bounds(m, 0, BK_FX; lo=3.0, up=3.0)
        # ACC: (v0, v1) in QUAD, v0 = 2 x0 + x1, v1 = x0 - 1 (via AFE + domain)
        afes(m, 2)
        afe_entry(m, 0, 0, 2.0)
        afe_entry(m, 0, 1, 1.0)
        afe_entry(m, 1, 0, 1.0)
        afe_g(m, 1, -1.0)
        dom = Ref{Clonglong}(0)
        if B.appendquadraticconedomain(m.task[], 2, dom) != 0
            @printf("domain rc\n")
            return 1
        end
        acc(m, Int(dom[]), [0, 1]; b=[0.0, 0.0])
        solve(m)                                    # x includes the ACC auxiliaries
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx, obj, nv = r.x, r.objective, min(length(r.x), 16)
    @printf("x = (%.4f, %.4f), obj = %.4f (numvar=%d incl. ACC auxiliaries)\n", xx[1], xx[2], obj, nv)
    v0 = 2.0 * xx[1] + xx[2]                        # external check: v0 >= |v1|
    v1 = xx[1] - 1.0
    ok = abs(xx[1] - 3.0) < 1e-6 && abs(xx[2] + 4.0) < 1e-4 && abs(obj + 4.0) < 1e-4 && v0 >= abs(v1) - 1e-6
    @printf("v0 = %.4f, |v1| = %.4f (ACC satisfied)\n", v0, abs(v1))
    @printf("%s (expected x=(3,-4), obj=-4)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
