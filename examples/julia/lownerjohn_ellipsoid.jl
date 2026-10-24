#!/usr/bin/env julia
# lownerjohn_ellipsoid.jl — Lowner-John inner ellipsoidal approximation (MOSEK
# Cookbook 11.6). The max-volume ellipsoid {C u + d : ||u|| <= 1} inside
# {x : A x <= b} solves
#
#   maximize t  s.t.  t <= det(C)^(1/n),  ||C a_i|| <= b_i - a_i'd,  C PSD
#
# det_rootn for n = 2 uses the PSD block Y = [C Z; Z' diag(Z)] and the rotated
# cone 2 Z00 (Z11/2) >= t^2. Instance: the unit square -> C = I/2,
# d = (1/2, 1/2), t = 1/2. Port of examples/c/lownerjohn_ellipsoid.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    T, D0, D1, X00, X01, X11, Z00, Z11, UU, AA, BB, CC, EE, NV = 0:13
    R = 14
    m = Model(R, NV)
    r = try
        sense(m, SENSE_MAX)
        barvars(m, [4])
        var_bounds(m, T, BK_LO; lo=0.0, up=Inf)
        cj(m, T, 1.0)
        var_bounds(m, D0, BK_RA; lo=0.0, up=1.0)
        var_bounds(m, D1, BK_RA; lo=0.0, up=1.0)
        var_bounds(m, X00, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, X01, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, X11, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, Z00, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, Z11, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, UU, BK_LO; lo=0.0, up=Inf)
        for v in AA:EE
            var_bounds(m, v, BK_LO; lo=0.0, up=Inf)
        end
        E(i, j) = sparsesymmat(m, 4, [(i, j, 1.0)])
        E00, E01, E11, E02, E13, E03, E22, E33, E23 =
            E(0, 0), E(0, 1), E(1, 1), E(0, 2), E(1, 3), E(0, 3), E(2, 2), E(3, 3), E(2, 3)
        # Y entries = scalar variables (off-diagonal counted twice: coeff 1/2)
        for (row, v, Em, coef) in ((0, X00, E00, 1.0), (1, X01, E01, 0.5), (2, X11, E11, 1.0),
                                   (3, Z00, E02, 0.5), (4, Z11, E13, 0.5))
            a_row(m, row, [v], [-1.0])
            baraij(m, row, 0, [(Em, coef)])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0)
        end
        baraij(m, 5, 0, [(E03, 0.5)]);  con_bounds(m, 5, BK_FX; lo=0.0, up=0.0)   # Z01 = 0
        baraij(m, 6, 0, [(E22, 1.0)]); baraij(m, 6, 0, [(E02, -0.5)])
        con_bounds(m, 6, BK_FX; lo=0.0, up=0.0)                                   # Y22 = Z00
        baraij(m, 7, 0, [(E33, 1.0)]); baraij(m, 7, 0, [(E13, -0.5)])
        con_bounds(m, 7, BK_FX; lo=0.0, up=0.0)                                   # Y33 = Z11
        baraij(m, 8, 0, [(E23, 0.5)]);  con_bounds(m, 8, BK_FX; lo=0.0, up=0.0)   # Y23 = 0
        a_row(m, 9, [UU], [1.0]); baraij(m, 9, 0, [(E13, -0.25)])
        con_bounds(m, 9, BK_FX; lo=0.0, up=0.0)                                   # U = Z11/2
        a_row(m, 10, [AA, D0], [1.0, 1.0]);  con_bounds(m, 10, BK_FX; lo=1.0, up=1.0)   # A = 1-d0
        a_row(m, 11, [BB, D0], [1.0, -1.0]); con_bounds(m, 11, BK_FX; lo=0.0, up=0.0)   # B = d0
        a_row(m, 12, [CC, D1], [1.0, 1.0]);  con_bounds(m, 12, BK_FX; lo=1.0, up=1.0)   # C = 1-d1
        a_row(m, 13, [EE, D1], [1.0, -1.0]); con_bounds(m, 13, BK_FX; lo=0.0, up=0.0)   # E = d1
        cone(m, CT_RQUAD, [Z00, UU, T])             # Z00*Z11 >= T^2
        cone(m, CT_QUAD, [AA, X00, X01])
        cone(m, CT_QUAD, [BB, X00, X01])
        cone(m, CT_QUAD, [CC, X01, X11])
        cone(m, CT_QUAD, [EE, X01, X11])
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, x = ok ? (r.objective, r.x) : (0.0, zeros(16))
    good = ok && abs(obj - 0.5) < 1e-5 && abs(x[X00+1] - 0.5) < 1e-4 && abs(x[X11+1] - 0.5) < 1e-4 &&
           abs(x[X01+1]) < 1e-4 && abs(x[D0+1] - 0.5) < 1e-4 && abs(x[D1+1] - 0.5) < 1e-4
    @printf("lownerjohn: rc=%d t*=%.8f  C=(%.5f,%.5f,%.5f) d=(%.5f,%.5f)  (expected 0.5, C=0.5I, d=(0.5,0.5))  %s\n",
            r.rc, obj, x[X00+1], x[X01+1], x[X11+1], x[D0+1], x[D1+1], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
