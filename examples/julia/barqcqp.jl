#!/usr/bin/env julia
# barqcqp.jl — symmetric bars and a quadratic part (constraint and objective)
# in the same model, solved by the QCQP-to-conic route; plus a nonconvex
# domain, which must be refused. Values derived by hand:
#
#   A) min -x0 - x1 - 4<E00,B>  s.t.  x0^2 + x1^2 + <E00,B> <= 3,  <I,B> = 1,  B PSD,  x >= 0
#      -> x* = (1, 1), B = diag(1, 0), obj = -6
#   B) min x0^2 + x1^2 - 2x0 - 2x1 - 3<E00,B>  s.t.  <I,B> = 1,  B PSD,  x >= 0
#      -> x* = (1, 1), B = diag(1, 0), obj = -5
#   C) as A with x0^2 - 2 x1^2 (nonconvex): rc = ERR_ARG, no solution published
#
# Port of examples/c/barqcqp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B_ = Primal.Bindings

# E00 and the 2x2 identity, shared by the three cases
put_syms(m) = (sparsesymmat(m, 2, [(0, 0, 1.0)]), sparsesymmat(m, 2, [(0, 0, 1.0), (1, 1, 1.0)]))

# Solve and read x, objective, B through the raw getters; returns (ok, x, B, obj).
function solve_read(m)
    t = m.task[]
    x, B, po = zeros(2), zeros(4), Ref{Cdouble}(0.0)
    ok = B_.optimize(t) == 0 && B_.getxx(t, SOL_ITR, x) == 0 &&
         B_.getprimalobj(t, SOL_ITR, po) == 0 && B_.getbarxj(t, SOL_ITR, 0, B) == 0
    return ok, x, B, po[]
end

function main()
    pass = true

    m = Model(0, 0)                                 # A) quadratic row and bar in the same row
    try
        add_cons(m, 2)
        m00, mI = put_syms(m)
        barvars(m, [2])
        add_vars(m, 2)
        sense(m, SENSE_MIN)
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, -1.0); cj(m, 1, -1.0)
        barcj(m, 0, [(m00, -4.0)])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=3.0)
        qconk(m, 0, [(0, 0, 2.0), (1, 1, 2.0)])
        baraij(m, 0, 0, [(m00, 1.0)])
        baraij(m, 1, 0, [(mI, 1.0)])
        con_bounds(m, 1, BK_FX; lo=1.0, up=1.0)
        ok, x, B, po = solve_read(m)
        ok = ok && abs(x[1] - 1.0) < 1e-4 && abs(x[2] - 1.0) < 1e-4 && abs(po + 6.0) < 1e-5 &&
             abs(B[1] - 1.0) < 1e-5 && abs(B[2]) < 1e-5 && abs(B[4]) < 1e-5
        @printf("barqcqp A  obj=%.6f (expected -6)  x=(%.6f, %.6f)  B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
                po, x[1], x[2], B[1], B[2], B[3], B[4], ok ? "OK" : "FAIL")
        pass &= ok
    finally
        free(m)
    end

    m = Model(0, 0)                                 # B) convex quadratic objective + bar
    try
        add_cons(m, 1)
        m00, mI = put_syms(m)
        barvars(m, [2])
        add_vars(m, 2)
        sense(m, SENSE_MIN)
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, -2.0); cj(m, 1, -2.0)
        qobj(m, [(0, 0, 2.0), (1, 1, 2.0)])
        barcj(m, 0, [(m00, -3.0)])
        baraij(m, 0, 0, [(mI, 1.0)])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        ok, x, B, po = solve_read(m)
        ok = ok && abs(x[1] - 1.0) < 1e-4 && abs(x[2] - 1.0) < 1e-4 && abs(po + 5.0) < 1e-5 &&
             abs(B[1] - 1.0) < 1e-5 && abs(B[2]) < 1e-5 && abs(B[4]) < 1e-5
        @printf("barqcqp B  obj=%.6f (expected -5)  x=(%.6f, %.6f)  B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
                po, x[1], x[2], B[1], B[2], B[3], B[4], ok ? "OK" : "FAIL")
        pass &= ok
    finally
        free(m)
    end

    m = Model(0, 0)                                 # C) nonconvex domain: refusal, not an answer
    try
        add_cons(m, 2)
        m00, mI = put_syms(m)
        barvars(m, [2])
        add_vars(m, 2)
        sense(m, SENSE_MIN)
        for j in 0:1
            var_bounds(m, j, BK_RA; lo=0.0, up=2.0)
        end
        cj(m, 0, -1.0); cj(m, 1, -1.0)
        con_bounds(m, 0, BK_UP; lo=-Inf, up=3.0)
        qconk(m, 0, [(0, 0, 2.0), (1, 1, -4.0)])
        baraij(m, 0, 0, [(m00, 1.0)])
        baraij(m, 1, 0, [(mI, 1.0)])
        con_bounds(m, 1, BK_FX; lo=1.0, up=1.0)
        t = m.task[]
        x, po = zeros(2), Ref{Cdouble}(0.0)
        ERR_ARG = RC[:ERR_ARG]
        rc = Int(B_.optimize(t))
        ok = rc == ERR_ARG && B_.getxx(t, SOL_ITR, x) == ERR_ARG && B_.getprimalobj(t, SOL_ITR, po) == ERR_ARG
        @printf("barqcqp C  rc=%d (expected %d = ERR_ARG)  obj not published  %s\n", rc, ERR_ARG, ok ? "OK" : "FAIL")
        pass &= ok
    finally
        free(m)
    end
    return pass ? 0 : 1
end

exit(main())
