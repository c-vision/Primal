#!/usr/bin/env julia
# portfolio_3.jl — MOSEK port "portfolio_3_impact", MIP version: portfolio
# optimization with linear transaction costs paid only on traded assets.
#
#   max r'x - gamma x'Sx - f'u
#   s.t.  sum(x) = 1,  0 <= x <= 1,  u_j >= |x_j - x0_j|,
#         u_j semi-continuous in {0} U [0.0001, 1]
#
# Data from the MOSEK tutorial, f = 0.01, x0 = (1/3, 1/3, 1/3). Checks:
# budget, semi-continuity, objective no better than without costs (0.1191).
# Port of examples/c/portfolio_3.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    n = 3
    r = [0.10717, 0.07502, 0.11902]
    GT = [0.1667, 0.0, 0.0, 0.0247, 0.1592, 0.0, 0.0197, 0.0158, 0.1327]
    gamma = 0.05
    f = 0.01                                        # linear transaction cost
    x0 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    S = zeros(9)
    for i in 0:2, j in 0:2
        S[i*3+j+1] = 0.0
        for k in 0:2
            S[i*3+j+1] += GT[i*3+k+1] * GT[j*3+k+1]
        end
    end
    m = Model(0, 0)
    res = try
        add_vars(m, 2n)                             # x (3), u (3)
        add_cons(m, 1 + 2n)                         # budget, u_j >= +-(x_j - x0_j)
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
            cj(m, n + j, -f)
            var_bounds(m, j, BK_RA; lo=0.0, up=1.0)
            var_bounds(m, n + j, BK_RA; lo=0.0001, up=1.0)   # u_j semi-continuous: 0 or [0.0001, 1]
            var_type(m, n + j, VT_SEMI_CONT)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget: sum x = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for j in 0:n-1                              # |x_j - x0_j| <= u_j
            a_row(m, 1 + j, [j, n + j], [1.0, -1.0])
            con_bounds(m, 1 + j, BK_UP; lo=-Inf, up=x0[j+1])
            a_row(m, 1 + n + j, [j, n + j], [-1.0, -1.0])
            con_bounds(m, 1 + n + j, BK_UP; lo=-Inf, up=-x0[j+1])
        end
        # obj = r'x - f'u - gamma x'Sx (Q = -2 gamma S)
        qobj(m, [(0, 0, -2.0 * gamma * S[1]), (0, 1, -2.0 * gamma * S[2]), (0, 2, -2.0 * gamma * S[3]),
                 (1, 1, -2.0 * gamma * S[5]), (1, 2, -2.0 * gamma * S[6]), (2, 2, -2.0 * gamma * S[9])])
        solve(m)
    finally
        free(m)
    end
    if res.rc != 0
        @printf("optimize rc=%d\n", res.rc)
        return 1
    end
    xx, obj = res.x, res.objective
    @printf("x = (%.4f, %.4f, %.4f)\n", xx[1], xx[2], xx[3])
    @printf("u = (%.4f, %.4f, %.4f)\n", xx[4], xx[5], xx[6])
    @printf("obj = %.6f\n", obj)
    ok = abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-5   # budget, |x-x0| <= u, semi-continuity of u
    for j in 1:n
        abs(xx[j] - x0[j]) > xx[n+j] + 1e-6 && (ok = false)
        (xx[n+j] > 1e-6 && xx[n+j] < 0.0001 - 1e-9) && (ok = false)
    end
    # without costs the portfolio dominates: pobj of portfolio_1 ~ 0.119066
    @printf("%s (budget+semi-continuity verified, obj <= 0.1191: %s)\n", ok ? "OK" : "FAIL", obj <= 0.1191 ? "yes" : "no")
    obj > 0.1191 && (ok = false)
    return ok ? 0 : 1
end

exit(main())
