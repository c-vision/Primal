#!/usr/bin/env julia
# mioinfeas1.jl — MOSEK port "mioinfeas1": an infeasible MIP.
#
#   min 0  s.t.  x >= 5,  x <= 2,  x integer >= 0
#
# Expected (reference table 7.3, integer problems): prosta = PRIM_INFEAS,
# solsta = UNKNOWN. A branch-and-bound produces no Farkas vector, so the
# certificate status is never published on a MIP; rc stays ERR_INFEASIBLE.
# Port of examples/c/mioinfeas1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const PRO_STA_PRIM_INFEAS = 4

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 1)
        add_cons(m, 2)
        a_row(m, 0, [0], [1.0])
        a_row(m, 1, [0], [1.0])
        con_bounds(m, 0, BK_LO; lo=5.0, up=Inf)
        con_bounds(m, 1, BK_UP; lo=-Inf, up=2.0)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_type(m, 0, VT_INT)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    @printf("rc = %d, prosta = %d, solsta = %d\n", r.rc, r.prosta, r.solsta)
    ok = r.rc == RC[:ERR_INFEASIBLE] && r.prosta == PRO_STA_PRIM_INFEAS && r.solsta == SOLSTA_UNKNOWN
    @printf("%s (expected MIP infeasible: prosta PRIM_INFEAS + solsta UNKNOWN)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
