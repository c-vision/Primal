#!/usr/bin/env julia
# boyd_corr_sdp.jl — bounding a correlation coefficient, "Convex Optimization"
# (Boyd & Vandenberghe, 2004), Example 8.3. Find the range of rho_14 given
#   0.6 <= rho_12 <= 0.9,  0.8 <= rho_13 <= 0.9,  0.5 <= rho_24 <= 0.7,
#   -0.8 <= rho_34 <= -0.4,
# and the 4x4 correlation matrix PSD (an SDP). The book reports
# min rho_14 = -0.39, max rho_14 = 0.23.
#
# Model: bar variable X (4x4 PSD) with <E_ij, X> = 2 rho_ij and <E_ii, X> = 1.
# Port of examples/c/boyd_corr_sdp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# index of the correlation variable for pair (i,j), i<j
const RIDX = [-1 0 1 2; -1 -1 3 4; -1 -1 -1 5; -1 -1 -1 -1]

# Solve the correlation SDP for one objective sign; returns (ok, rho_14).
function solve_corr(sgn)
    m = Model(0, 0)
    try
        add_vars(m, 6)                              # rho_12, rho_13, rho_14, rho_23, rho_24, rho_34
        lo = [0.6, 0.8, -Inf, -Inf, 0.5, -0.8]
        up = [0.9, 0.9, Inf, Inf, 0.7, -0.4]
        for k in 0:5
            var_bounds(m, k, BK_RA; lo=lo[k+1], up=up[k+1])
        end
        cj(m, 2, sgn)                               # minimize (sgn=+1) or maximize (-1) rho_14
        msym = Int[]
        for i in 0:3, j in i:3
            push!(msym, sparsesymmat(m, 4, [(i, j, 1.0)]))
        end
        barvars(m, [4])
        add_cons(m, 10)
        row = 0
        for i in 0:3, j in i:3
            baraij(m, row, 0, [(msym[row+1], 1.0)])
            if i == j
                con_bounds(m, row, BK_FX; lo=1.0, up=1.0)
            else
                a_row(m, row, [RIDX[i+1, j+1]], [-2.0])
                con_bounds(m, row, BK_FX; lo=0.0, up=0.0)
            end
            row += 1
        end
        r = solve(m)
        return r.rc == 0, (r.rc == 0 ? r.x[3] : 0.0)
    finally
        free(m)
    end
end

function main()
    lo14, hi14 = 0.0, 0.0
    ok, v = solve_corr(1.0)
    ok && (lo14 = v)
    if ok
        ok, v = solve_corr(-1.0)
        ok && (hi14 = v)
    end
    ok = ok && abs(lo14 + 0.39) < 0.01 && abs(hi14 - 0.23) < 0.01
    @printf("boyd_corr_sdp  min rho14=%.4g max rho14=%.4g (book -0.39, 0.23) %s\n", lo14, hi14, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
