#!/usr/bin/env julia
# callback.jl — MOSEK port "callback": progress callback during optimization.
#
# Registers PRIMAL_setprogresscb + PRIMAL_setinfoconnname and solves a small
# minimum-variance QP:  min 0.5 (x0^2 + x1^2)  s.t.  x0 + x1 = 1,  x >= 0
# -> x* = (0.5, 0.5), var = 0.25. Checks that the callback receives >= 1
# progress notification. Port of examples/c/callback.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

const n_calls = Ref(0)
const last_info = Ref("")

# Progress callback counting calls and storing the latest info.
function progcb(handle::Ptr{Cvoid}, info::Cstring)::Cvoid
    n_calls[] += 1
    s = unsafe_string(info)
    last_info[] = String(codeunits(s)[1:min(end, 255)])
    @printf("  [progress] %s\n", s)
    return nothing
end

function main()
    m = Model(0, 0)
    try
        add_vars(m, 2)
        add_cons(m, 1)
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        qobj(m, [(0, 0, 1.0), (1, 1, 1.0)])
        a_row(m, 0, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)

        B.setprogresscb(m.task[], @cfunction(progcb, Cvoid, (Ptr{Cvoid}, Cstring)), C_NULL)
        B.setinfoconnname(m.task[], "callback-demo")
        n_calls[] = 0

        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        xx, obj = r.x, r.objective
        @printf("x = (%.6f, %.6f), obj = %.6f, callback calls = %d\n", xx[1], xx[2], obj, n_calls[])
        @printf("last info: %s\n", last_info[])
        ok = abs(xx[1] - 0.5) < 1e-6 && abs(xx[2] - 0.5) < 1e-6 && abs(obj - 0.25) < 1e-6 && n_calls[] >= 1
        @printf("%s (expected x=(0.5,0.5), obj=0.25, >=1 calls)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
