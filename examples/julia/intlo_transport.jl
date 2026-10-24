#!/usr/bin/env julia
# intlo_transport.jl — transportation counterexample, "Introduction to Linear
# Optimization" (A. Nemirovski, 2024), Exercise 5.3:
#
#   Opt = min { sum c_ij x_ij : sum_j x_ij = a_i, sum_i x_ij = b_j, x >= 0 }
#
# with c11 = c22 = 1, c12 = c21 = 100.
#   (a) a = b = (1, 1): x11 = x22 = 1, optimum 2.
#   (b) a' = (1/2, 1), b' = (1, 1/2): the optimum is "at least 50" (it grows!).
# Port of examples/c/intlo_transport.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# One transportation LP: min c11 x11 + c12 x12 + c21 x21 + c22 x22.
function solve_lp(a0, a1, b0, b1)
    c = [1.0, 100.0, 100.0, 1.0]                    # 11, 12, 21, 22
    m = Model(4, 4)
    r = try
        for j in 0:3
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            cj(m, j, c[j+1])
        end
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_FX; lo=a0, up=a0)   # row supplies
        a_row(m, 1, [2, 3], [1.0, 1.0]); con_bounds(m, 1, BK_FX; lo=a1, up=a1)
        a_row(m, 2, [0, 2], [1.0, 1.0]); con_bounds(m, 2, BK_FX; lo=b0, up=b0)   # column demands
        a_row(m, 3, [1, 3], [1.0, 1.0]); con_bounds(m, 3, BK_FX; lo=b1, up=b1)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, r.objective
end

function main()
    z1, z2 = -1.0, -1.0
    ok, z = solve_lp(1.0, 1.0, 1.0, 1.0)
    ok && (z1 = z)
    if ok
        ok, z = solve_lp(0.5, 1.0, 1.0, 0.5)
        ok && (z2 = z)
    end
    ok = ok && abs(z1 - 2.0) < 1e-6 && z2 >= 50.0 - 1e-6
    @printf("intlo_transport  z_initial=%.6g z_new=%.6g (book 2, >=50) %s\n", z1, z2, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
