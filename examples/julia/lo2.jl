#!/usr/bin/env julia
# lo2.jl — MOSEK port "lo2": LP with 4 variables and 3 constraints (one
# equality, one upper-bounded, one lower-bounded).
#
#   max 3x0 + x1 + 5x2 + x3
#   s.t.  3x0 + x1 + 2x2 = 30,  2x0 + x1 + 3x2 + x3 <= 25,  x0 + 2x1 + x2 + 2x3 >= 15
#         x >= 0,  x1 <= 10
#
# Checks: primal = dual (no duality gap) and BAS vs ITR agreement.
# Port of examples/c/lo2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

function main()
    c = [3.0, 1.0, 5.0, 1.0]
    m = Model(0, 0)
    try
        add_vars(m, 4)
        add_cons(m, 3)
        for j in 0:3
            cj(m, j, c[j+1])
        end
        sense(m, SENSE_MAX)
        a_row(m, 0, [0, 1, 2], [3.0, 1.0, 2.0])    # A by rows
        a_row(m, 1, [0, 1, 2, 3], [2.0, 1.0, 3.0, 1.0])
        a_row(m, 2, [0, 1, 2, 3], [1.0, 2.0, 1.0, 2.0])
        con_bounds(m, 0, BK_FX; lo=30.0, up=30.0)
        con_bounds(m, 1, BK_UP; lo=-Inf, up=25.0)
        con_bounds(m, 2, BK_LO; lo=15.0, up=Inf)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_RA; lo=0.0, up=10.0)
        var_bounds(m, 2, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 3, BK_LO; lo=0.0, up=Inf)
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        xx, po, dobj, sta = r.x, r.objective, r.dual, r.solsta
        xb = zeros(4)
        B.getxx(m.task[], SOL_BAS, xb)
        @printf("x = (%.6f, %.6f, %.6f, %.6f)\n", xx[1], xx[2], xx[3], xx[4])
        @printf("pobj = %.6f, dobj = %.6f, solsta = %d\n", po, dobj, sta)

        ok = sta == SOLSTA_OPTIMAL                  # primal feasibility
        r0 = 3 * xx[1] + xx[2] + 2 * xx[3]
        r1 = 2 * xx[1] + xx[2] + 3 * xx[3] + xx[4]
        r2 = xx[1] + 2 * xx[2] + xx[3] + 2 * xx[4]
        (abs(r0 - 30.0) > 1e-6 || r1 > 25.0 + 1e-6 || r2 < 15.0 - 1e-6) && (ok = false)
        # zero duality gap + consistent BAS/ITR
        abs(po - dobj) > 1e-6 * (1 + abs(po)) && (ok = false)
        abs(po - (c[1] * xx[1] + c[2] * xx[2] + c[3] * xx[3] + c[4] * xx[4])) > 1e-6 && (ok = false)
        any(j -> abs(xx[j] - xb[j]) > 1e-6, 1:4) && (ok = false)
        @printf("%s (optimal certified by pobj = dobj)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
