#!/usr/bin/env julia
# truss_design.jl — minimum-volume truss by semidefinite programming (MOSEK
# Modeling Cookbook / Tutorials, truss topology design).
#
# Two symmetric bars A-C and B-C (A = (-1,1), B = (1,1) fixed, C = (0,0) free,
# loaded with F = (0,-1)), cross-sections t >= 0, E = 1, L = sqrt(2):
#
#   min L (t1 + t2)  s.t.  [K(t) f; f' 1] PSD,  K(t) = sum_i t_i (E/L) d_i d_i'
#
# Hand derivation: t1 = t2 = sqrt(2), volume 4. The compliance is re-derived
# from the published t. Port of examples/c/truss_design.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const DIM = 3                                       # [K f; f' 1]
const L = 1.4142135623730951                        # sqrt(2)
const K0 = 1.0 / (2.0 * 1.4142135623730951)         # 1/(2 sqrt2)

# add the LMI entry `coef * X[i][j]` to row `row` of the bar block
function barentry(m, row, i, j, coef)
    id = sparsesymmat(m, DIM, [(i, j, 1.0)])
    baraij(m, row, 0, [(id, coef)])
end

function main()
    m = Model(6, 2)                                 # scalar vars: 0 = t1, 1 = t2
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, L)                                 # min L (t1 + t2)
        cj(m, 1, L)
        barvars(m, [DIM])
        barentry(m, 0, 0, 0, 1.0)                   # row 0: X[0][0] - K0 (t1 + t2) = 0
        a_row(m, 0, [0, 1], [-K0, -K0]); con_bounds(m, 0, BK_FX; lo=0.0, up=0.0)
        barentry(m, 1, 1, 1, 1.0)                   # row 1: X[1][1] - K0 (t1 + t2) = 0
        a_row(m, 1, [0, 1], [-K0, -K0]); con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        barentry(m, 2, 0, 1, 0.5)                   # row 2: X[0][1] - K0 (t2 - t1) = 0
        a_row(m, 2, [0, 1], [K0, -K0]); con_bounds(m, 2, BK_FX; lo=0.0, up=0.0)
        barentry(m, 3, 0, 2, 0.5)                   # row 3: X[0][2] = 0   (f_x = 0)
        con_bounds(m, 3, BK_FX; lo=0.0, up=0.0)
        barentry(m, 4, 1, 2, 0.5)                   # row 4: X[1][2] = -1  (f_y = -1)
        con_bounds(m, 4, BK_FX; lo=-1.0, up=-1.0)
        barentry(m, 5, 2, 2, 1.0)                   # row 5: X[2][2] = 1
        con_bounds(m, 5, BK_FX; lo=1.0, up=1.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        obj = r.objective
        t1, t2 = r.x[1], r.x[2]
        # compliance recomputed from t: sqrt(2)/(2 t1 t2) * (t1+t2)
        comp = (t1 > 0 && t2 > 0) ? (L / (2.0 * t1 * t2)) * (t1 + t2) : 1e30
        ok = abs(obj - 4.0) < 1e-5 && comp <= 1.0 + 1e-6
        @printf("truss_design  t=(%.6f,%.6f) volume=%.6f (expected 4) compliance=%.6f %s\n",
                t1, t2, obj, comp, ok ? "OK" : "FAIL")
    else
        @printf("truss_design  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
