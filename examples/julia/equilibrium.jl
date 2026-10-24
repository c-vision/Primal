#!/usr/bin/env julia
# equilibrium.jl — equilibrium of masses hanging on strings (SOCP).
#
# Source: MOSEK/Tutorials, equilibrium. Masses at x_i in R^2 joined by strings
# of maximum length l_ij settle in the minimum-energy configuration:
#
#   minimize g sum_i w_i x_i,y  s.t.  ||x_i - x_j||_2 <= l_ij,  x_i = f_i (pinned)
#
# Each string is a quadratic cone (l_ij, x_i - x_j) in Q^3. Two masses pinned at
# (-1,0) and (1,0) hold one free mass on strings of length l = 2: it settles at
# (0, -sqrt(l^2 - 1)) = (0, -sqrt(3)). Port of examples/c/equilibrium.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const LSTR = 2.0

# One free mass at (x0, x1) joined to two fixed anchors; minimize x1.
function solve_mass()
    X0, X1, D0, D1, L, NV = 0, 1, 2, 4, 6, 7
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, 4)
        for k in 0:5
            var_bounds(m, k, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, L, BK_FX; lo=LSTR, up=LSTR)
        # D0 = (x0,x1) - (-1,0) ; D1 = (x0,x1) - (1,0)
        a_row(m, 0, [D0, X0], [1.0, -1.0]);     con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [D0 + 1, X1], [1.0, -1.0]); con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        a_row(m, 2, [D1, X0], [1.0, -1.0]);     con_bounds(m, 2, BK_FX; lo=-1.0, up=-1.0)
        a_row(m, 3, [D1 + 1, X1], [1.0, -1.0]); con_bounds(m, 3, BK_FX; lo=0.0, up=0.0)
        cone(m, CT_QUAD, [L, D0, D0 + 1])
        cone(m, CT_QUAD, [L, D1, D1 + 1])
        cj(m, X1, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, 0.0
    return true, r.x[X0+1], r.x[X1+1]
end

function main()
    ok, x0, y = solve_mass()
    want = -sqrt(LSTR * LSTR - 1.0)                 # -sqrt(3)
    okall = ok && abs(y - want) < 1e-7 && abs(x0) < 1e-6
    @printf("equilibrium  l=%.1f\n", LSTR)
    @printf("  free mass = (%.8f, %.8f)   (expected (0, -sqrt(3)=%.8f))\n", x0, y, want)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
