#!/usr/bin/env julia
# solutionquality.jl — MOSEK port "solutionquality": measuring solution
# quality (primal/dual violations) via PRIMAL_getprimalinfeas /
# PRIMAL_getdualinfeas.
#
#   1. optimal LP: min x0 + x1  s.t.  x0 + x1 >= 1,  x >= 0 -> both violations ~ 0
#   2. a task with no solution: the getters refuse it (ERR_ARG)
#
# Port of examples/c/solutionquality.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

function main()
    m = Model(0, 0)                                 # 1. optimal LP
    m2 = Model(0, 0)
    try
        add_vars(m, 2)
        add_cons(m, 1)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, 1.0)
        cj(m, 1, 1.0)
        a_row(m, 0, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_LO; lo=1.0, up=Inf)
        rc = Int(B.optimize(m.task[]))
        if rc != 0
            @printf("optimize rc=%d\n", rc)
            return 1
        end
        pinf, dinf = Ref{Cdouble}(0.0), Ref{Cdouble}(0.0)
        B.getprimalinfeas(m.task[], SOL_ITR, pinf)
        B.getdualinfeas(m.task[], SOL_ITR, dinf)
        @printf("primal violation = %.3e, dual = %.3e (expected ~ 0)\n", pinf[], dinf[])

        add_vars(m2, 1)                             # 2. without a solution: refused
        dummy = Ref{Cdouble}(0.0)
        rejected = Int(B.getprimalinfeas(m2.task[], SOL_ITR, dummy)) == RC[:ERR_ARG]

        ok = pinf[] < 1e-6 && dinf[] < 1e-6 && rejected
        @printf("%s (quality measured + protected getters)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m2)
        free(m)
    end
end

exit(main())
