#!/usr/bin/env julia
# sdo1.jl — SDP (variant of the MOSEK docs "sdo1" example):
#
#   min tr(X)  s.t.  <J, X> = 1 (J = all-ones 2x2),  X PSD
#
# With X = [a b; b c], a + c + 2b = 1 and PSD force tr(X) >= 1/2.
# Expected: tr(X) = 0.5 at X = J/4 (rank one). Port of examples/c/sdo1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        barvars(m, [2])                             # one 2x2 X
        # J = E00 + E11 + 2 E01: entry (0,1) counts twice in the inner product
        mJ = sparsesymmat(m, 2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0)])
        add_cons(m, 1)                              # <J, X> = 1
        baraij(m, 0, 0, [(mJ, 1.0)])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        mI = sparsesymmat(m, 2, [(0, 0, 1.0), (1, 1, 1.0)])
        barcj(m, 0, [(mI, 1.0)])                    # min tr(X) = <I, X>
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    obj = r.objective
    @printf("tr(X) = %.6f\n", obj)
    ok = abs(obj - 0.5) < 1e-6
    @printf("%s (expected tr(X) = 0.5, X = J/2 rank-1)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
