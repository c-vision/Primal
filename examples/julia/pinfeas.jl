#!/usr/bin/env julia
# pinfeas.jl — MOSEK port "pinfeas": a primal-infeasible LP. The solver must
# detect the infeasibility and produce the certificate.
#
#   min 0  s.t.  x0 + x1 >= 4,  x0 + x1 <= 2,  x >= 0
#
# Expected: rc = ERR_INFEASIBLE, solsta = PRIM_INFEAS_CER.
# Port of examples/c/pinfeas.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 2)
        add_cons(m, 2)
        a_row(m, 0, [0, 1], [1.0, 1.0])
        a_row(m, 1, [0, 1], [1.0, 1.0])
        con_bounds(m, 0, BK_LO; lo=4.0, up=Inf)
        con_bounds(m, 1, BK_UP; lo=-Inf, up=2.0)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ERR_INFEASIBLE = RC[:ERR_INFEASIBLE]
    @printf("rc = %d (ERR_INFEASIBLE = %d), solsta = %d (PRIM_INFEAS_CER = %d)\n",
            r.rc, ERR_INFEASIBLE, r.solsta, SOLSTA_PRIM_INFEAS_CER)
    ok = r.rc == ERR_INFEASIBLE && r.solsta == SOLSTA_PRIM_INFEAS_CER
    @printf("%s (expected infeasible with certificate)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
