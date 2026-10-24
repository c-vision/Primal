#!/usr/bin/env julia
# sdo2.jl — MOSEK port "sdo2": SDP with TWO bar variables (PSD matrices) and
# an objective constant.
#
#   min <I, X1> + <I, X2> + 1  s.t.  <J, X1> + <I, X2> = 1,  X1, X2 PSD
#
# X1 pays <J, X1> with tr(X1) >= 1/2 (rank one, X1 = J/4), X2 = 0.
# Expected: obj = 1.5. Port of examples/c/sdo2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    try
        add_cons(m, 1)
        mI = sparsesymmat(m, 2, [(0, 0, 1.0), (1, 1, 1.0)])
        mJ = sparsesymmat(m, 2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0)])
        barvars(m, [2, 2])
        cfix(m, 1.0)
        sense(m, SENSE_MIN)
        barcj(m, 0, [(mI, 1.0)])                    # <I, X1> + <I, X2> in the objective
        barcj(m, 1, [(mI, 1.0)])
        baraij(m, 0, 0, [(mJ, 1.0)])                # <J, X1> + <I, X2> = 1
        baraij(m, 0, 1, [(mI, 1.0)])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        X1, X2, po = getbarxj(m, 0), getbarxj(m, 1), r.objective
        @printf("X1 = [%.6f %.6f; %.6f %.6f]\n", X1[1], X1[2], X1[3], X1[4])
        @printf("X2 = [%.6f %.6f; %.6f %.6f]\n", X2[1], X2[2], X2[3], X2[4])
        @printf("obj = %.6f\n", po)
        ok = abs(po - 1.5) < 1e-4                   # obj = 1.5, X1 = J/4, X2 = 0
        any(v -> abs(v - 0.25) > 1e-4, X1) && (ok = false)
        any(v -> abs(v) > 1e-4, X2) && (ok = false)
        @printf("%s (expected obj = 1.5, X1 = J/4, X2 = 0)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
