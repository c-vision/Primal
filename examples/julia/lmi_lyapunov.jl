#!/usr/bin/env julia
# lmi_lyapunov.jl — Lyapunov LMI for the stability of a linear system
# (andel-vitorio/linear-matrix-inequality, Problem 1):
#
#   minimize trace(P)  s.t.  P >= 0,  A'P + PA + Q <= 0
#
# with Q = I. The minimum-trace solution solves A'P + PA + Q = 0. Case
# A = [0 -2; 1 -3]: P* = [1 -1/2; -1/2 1/2], trace 3/2. The LMI is a second
# PSD bar variable S = -(A'P + PA + I) with three entrywise equalities; <M,X>
# counts the off-diagonal of M twice. Port of examples/c/lmi_lyapunov.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)                                 # A = [0 -2; 1 -3]
    ok, obj, P, S = false, 0.0, zeros(4), zeros(4)
    try
        m00 = sparsesymmat(m, 2, [(0, 0, 1.0)])
        m01 = sparsesymmat(m, 2, [(0, 1, 1.0)])
        m11 = sparsesymmat(m, 2, [(1, 1, 1.0)])
        barvars(m, [2])                             # barvar 0 : P (2x2 PSD)
        barvars(m, [2])                             # barvar 1 : S (2x2 PSD) = -(A'P+PA+I)
        add_cons(m, 3)
        sense(m, SENSE_MIN)
        # Row 0: S00 + 2 P01 = -1  ->  <E00,S> + <E01,P> = -1
        baraij(m, 0, 1, [(m00, 1.0)])
        baraij(m, 0, 0, [(m01, 1.0)])
        con_bounds(m, 0, BK_FX; lo=-1.0, up=-1.0)
        # Row 1: S01 + P11 - 2 P00 - 3 P01 = 0
        baraij(m, 1, 1, [(m01, 0.5)])
        baraij(m, 1, 0, [(m11, 1.0)])
        baraij(m, 1, 0, [(m00, -2.0)])
        baraij(m, 1, 0, [(m01, -1.5)])
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        # Row 2: S11 - 4 P01 - 6 P11 = -1  ->  <E11,S> - 2<E01,P> - 6<E11,P> = -1
        baraij(m, 2, 1, [(m11, 1.0)])
        baraij(m, 2, 0, [(m01, -2.0)])
        baraij(m, 2, 0, [(m11, -6.0)])
        con_bounds(m, 2, BK_FX; lo=-1.0, up=-1.0)
        # objective: minimize trace(P) = <E00,P> + <E11,P>
        barcj(m, 0, [(m00, 1.0)])
        barcj(m, 0, [(m11, 1.0)])
        r = solve(m)
        ok = r.rc == 0
        if ok
            obj, P, S = r.objective, getbarxj(m, 0), getbarxj(m, 1)
        end
    finally
        free(m)
    end
    trP, detP = P[1] + P[4], P[1] * P[4] - P[2] * P[3]
    disc = sqrt(max(0.0, trP * trP / 4 - detP))
    lmin = trP / 2 - disc
    good = ok && abs(obj - 1.5) < 1e-6 && abs(P[1] - 1.0) < 1e-6 && abs(P[2] + 0.5) < 1e-6 &&
           abs(P[3] + 0.5) < 1e-6 && abs(P[4] - 0.5) < 1e-6 && lmin > -1e-9
    @printf("lmi_lyapunov  obj=%.8f (expected 1.5 = 3/2)  P=[[%.6f,%.6f],[%.6f,%.6f]]  lmin=%.2e\n",
            obj, P[1], P[2], P[3], P[4], lmin)
    @printf("  trace(P)=%.8f  S (=-(A'P+PA+I)) = [[%.2e,%.2e],[.,%.2e]]  %s\n",
            trP, S[1], S[2], S[3], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
