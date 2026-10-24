#!/usr/bin/env julia
# sdo_lmi.jl — MOSEK port "sdo_lmi": linear matrix inequality with scalar
# variables.
#
#   max x0 + x1  s.t.  F(x) = I3 - x0 E00 - x1 E11 PSD,  x0 + x1 <= 2.5,  x >= 0
#
# F(x) = diag(1 - x0, 1 - x1, 1): the LMI binds at x = (1, 1), obj = 2, with
# X = F(x) = diag(0, 0, 1). The 3x3 bar variable X is tied to F(x) entry by
# entry. Port of examples/c/sdo_lmi.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const D = 3

function main()
    m = Model(0, 0)
    try
        add_vars(m, 2)                              # scalar variables x0, x1
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, 1.0)
        cj(m, 1, 1.0)
        sense(m, SENSE_MAX)
        barvars(m, [D])                             # bar X 3x3 PSD
        mE = [sparsesymmat(m, D, [(i, i, 1.0)]) for i in 0:D-1]
        add_cons(m, 7)                              # 3 diagonal + 3 off-diagonal + 1 linear
        row = 0
        baraij(m, row, 0, [(mE[1], 1.0)])           # X_00 = 1 - x0
        a_row(m, row, [0], [1.0]); con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        baraij(m, row, 0, [(mE[2], 1.0)])           # X_11 = 1 - x1
        a_row(m, row, [1], [1.0]); con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        baraij(m, row, 0, [(mE[3], 1.0)])           # X_22 = 1
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        mO = [sparsesymmat(m, D, [(0, 1, 1.0)]), sparsesymmat(m, D, [(0, 2, 1.0)]), sparsesymmat(m, D, [(1, 2, 1.0)])]
        for k in 1:3                                # off-diagonals X_01 = X_02 = X_12 = 0
            baraij(m, row, 0, [(mO[k], 1.0)])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        a_row(m, row, [0, 1], [1.0, 1.0])          # x0 + x1 <= 2.5
        con_bounds(m, row, BK_UP; lo=-Inf, up=2.5); row += 1
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        xx, po, X = r.x, r.objective, getbarxj(m, 0)
        @printf("x = (%.4f, %.4f), obj = %.4f (expected (1,1), 2)\n", xx[1], xx[2], po)
        @printf("X = diag(%.4f, %.4f, %.4f) (expected (0,0,1))\n", X[1], X[5], X[9])
        ok = abs(xx[1] - 1.0) < 1e-4 && abs(xx[2] - 1.0) < 1e-4 && abs(po - 2.0) < 1e-4 &&
             abs(X[1]) < 1e-4 && abs(X[5]) < 1e-4 && abs(X[9] - 1.0) < 1e-4
        @printf("%s (LMI active: x=(1,1), X=diag(0,0,1) PSD at the boundary)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
