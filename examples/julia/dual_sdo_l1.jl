#!/usr/bin/env julia
# dual_sdo_l1.jl — port of the "dual_sdo_l1" example: SDP duality. The dual of
#
#   min <C, X>  s.t.  <A_i, X> = b_i,  X PSD    is    max b'y  s.t.  C - sum_i y_i A_i PSD.
#
# Primal: min X00 s.t. X11 = 1 (2x2) -> X* = diag(0, 1), pobj = 0. Dual:
# max y s.t. diag(0, -y) PSD -> y* = 0. Strong duality 0 = 0, each problem
# solved separately. Port of examples/c/dual_sdo_l1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve the primal SDP min X00 s.t. X11 = 1; returns (value, X).
function primal_sdp()
    m = Model(0, 0)
    try
        add_vars(m, 1)
        var_bounds(m, 0, BK_FX; lo=0.0, up=0.0)     # dummy scalar
        mC = sparsesymmat(m, 2, [(0, 0, 1.0)])
        mA = sparsesymmat(m, 2, [(1, 1, 1.0)])
        barvars(m, [2])
        barcj(m, 0, [(mC, 1.0)])                    # min X00
        add_cons(m, 1)
        baraij(m, 0, 0, [(mA, 1.0)])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)     # X11 = 1
        r = solve(m)
        r.rc == 0 || return -1e30, zeros(4)
        return r.objective, getbarxj(m, 0)
    finally
        free(m)
    end
end

# Solve the dual SDP max y s.t. S = C - y A_1 PSD; returns (value, y, S).
function dual_sdp()
    m = Model(0, 0)
    try
        add_vars(m, 1)                              # y
        var_bounds(m, 0, BK_FR; lo=-Inf, up=Inf)
        cj(m, 0, 1.0)                               # max y (b_1 = 1)
        sense(m, SENSE_MAX)
        barvars(m, [2])
        mC = sparsesymmat(m, 2, [(0, 0, 1.0)])
        mA = sparsesymmat(m, 2, [(1, 1, 1.0)])
        mO = sparsesymmat(m, 2, [(0, 1, 1.0)])
        add_cons(m, 3)
        baraij(m, 0, 0, [(mC, 1.0)])                # S_00 = 0 (from C - 0)
        con_bounds(m, 0, BK_FX; lo=0.0, up=0.0)
        baraij(m, 1, 0, [(mO, 1.0)])                # S_01 = 0
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        baraij(m, 2, 0, [(mA, 1.0)])                # S_11 = -y: <mA, S> + y = 0
        a_row(m, 2, [0], [1.0])
        con_bounds(m, 2, BK_FX; lo=0.0, up=0.0)
        r = solve(m)
        r.rc == 0 || return -1e30, 0.0, zeros(4)
        return r.objective, r.x[1], getbarxj(m, 0)
    finally
        free(m)
    end
end

function main()
    pobj, X = primal_sdp()
    dobj, y, S = dual_sdp()
    @printf("primal: pobj = %.6f (expected 0), X = diag(%.4f, %.4f)\n", pobj, X[1], X[4])
    @printf("dual:   dobj = %.6f (expected 0), y = %.4f, S = diag(%.4f, %.4f)\n", dobj, y, S[1], S[4])
    ok = abs(pobj) < 1e-6 && abs(dobj) < 1e-6 && abs(pobj - dobj) < 1e-6 &&   # strong duality
         abs(X[1]) < 1e-4 && abs(X[4] - 1.0) < 1e-4 && abs(S[1]) < 1e-4 && abs(S[4]) < 1e-4
    @printf("%s (strong duality: <C,X*> = b'y* = 0)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
