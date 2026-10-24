#!/usr/bin/env julia
# omf_mvo.jl — Markowitz mean-variance optimization of "Optimization Methods
# in Finance" (Cornuejols & Tutuncu, 2007), Section 8.1.1, with the geometric
# means and covariance of Tables 8.2-8.3 (US stocks, bonds, money market):
#
#   minimize x'Sigma x  s.t.  mu'x >= R,  xS + xB + xM = 1,  x >= 0
#
# Solved for R = 6.5% .. 8.5% and checked against the efficient portfolios of
# Table 8.3 (rounded to 2 decimals). Port of examples/c/omf_mvo.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const MU = [0.1073, 0.0737, 0.0627]
const SIG = [0.02778 0.00387 0.00021; 0.00387 0.01112 -0.00020; 0.00021 -0.00020 0.00115]
const R = [0.065, 0.070, 0.075, 0.080, 0.085]
const VAR = [0.0010, 0.0014, 0.0026, 0.0044, 0.0070]
const XS = [0.03, 0.13, 0.24, 0.35, 0.45]
const XB = [0.10, 0.12, 0.14, 0.16, 0.18]
const XM = [0.87, 0.75, 0.62, 0.49, 0.37]

# Solve the MVO QP for target R[k] and check the book portfolio.
function solve_mvo(k)
    m = Model(2, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        # objective x'Sigma x = 0.5 x'Qx with Q = 2 Sigma (upper triangle)
        qobj(m, [(0, 0, 2 * SIG[1, 1]), (0, 1, 2 * SIG[1, 2]), (0, 2, 2 * SIG[1, 3]),
                 (1, 1, 2 * SIG[2, 2]), (1, 2, 2 * SIG[2, 3]), (2, 2, 2 * SIG[3, 3])])
        cj(m, 0, 0.0); cj(m, 1, 0.0); cj(m, 2, 0.0)
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget: xS+xB+xM = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [0, 1, 2], MU)                  # return: mu'x >= R
        con_bounds(m, 1, BK_LO; lo=R[k], up=Inf)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        var, x = r.objective, r.x
        ok = abs(var - VAR[k]) < 1e-4 && abs(x[1] - XS[k]) < 0.02 && abs(x[2] - XB[k]) < 0.02 &&
             abs(x[3] - XM[k]) < 0.02
        @printf("omf_mvo R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) (book %.2f,%.2f,%.2f) %s\n",
                R[k], var, VAR[k], x[1], x[2], x[3], XS[k], XB[k], XM[k], ok ? "OK" : "FAIL")
    else
        @printf("omf_mvo R=%.3f rc=%d FAIL\n", R[k], r.rc)
    end
    return ok
end

function main()
    ok = true
    for k in 1:5
        ok &= solve_mvo(k)
    end
    return ok ? 0 : 1
end

exit(main())
