#!/usr/bin/env julia
# portfolio_5.jl — MOSEK port "portfolio_5_card": portfolio optimization with
# a CARDINALITY constraint (at most k assets held), a MIP with binaries:
#
#   max r'x - gamma x'Sigma x  s.t.  sum(x) = 1,  0 <= x_j <= y_j,  y binary,  sum(y) <= k
#
# Data: the MOSEK tutorial (n = 3), gamma = 0.05, k = 2. Checks: budget,
# cardinality, and an objective no better than without the limit (0.119066).
# Port of examples/c/portfolio_5.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    n, k = 3, 2
    r = [0.10717, 0.07502, 0.11902]
    GT = [0.1667, 0.0, 0.0, 0.0247, 0.1592, 0.0, 0.0197, 0.0158, 0.1327]
    gamma = 0.05
    S = zeros(9)
    for i in 0:2, j in 0:2
        S[i*3+j+1] = 0.0
        for k2 in 0:2
            S[i*3+j+1] += GT[i*3+k2+1] * GT[j*3+k2+1]
        end
    end
    m = Model(0, 0)
    res = try
        add_vars(m, 2n)                             # x (3), y (3 binaries)
        add_cons(m, 2 + n)                          # budget, cardinality, x_j <= y_j
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, n + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, n + j, VT_INT_BIN)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget: sum x = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [3, 4, 5], [1.0, 1.0, 1.0])    # cardinality: sum y <= k
        con_bounds(m, 1, BK_UP; lo=-Inf, up=Float64(k))
        for j in 0:n-1                              # x_j - y_j <= 0
            a_row(m, 2 + j, [j, n + j], [1.0, -1.0])
            con_bounds(m, 2 + j, BK_UP; lo=-Inf, up=0.0)
        end
        # obj = r'x - gamma x'Sx (Q = -2 gamma S)
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
    @printf("y = (%.0f, %.0f, %.0f)\n", xx[4], xx[5], xx[6])
    @printf("obj = %.6f\n", obj)
    nact = 0                                        # budget, cardinality, x_j <= y_j, y binaries
    for j in 1:n
        xx[j] > 1e-6 && (nact += 1)
        (xx[n+j] > 1e-6 && xx[j] > xx[n+j] + 1e-6) && (nact = 99)
        abs(xx[n+j] - floor(xx[n+j] + 0.5)) > 1e-6 && (nact = 99)
    end
    ok = abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-5 && nact <= k && obj <= 0.119066 + 1e-6
    @printf("%s (budget, cardinality <= %d, obj <= 0.119066)\n", ok ? "OK" : "FAIL", k)
    return ok ? 0 : 1
end

exit(main())
