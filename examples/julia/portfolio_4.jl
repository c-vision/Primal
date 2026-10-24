#!/usr/bin/env julia
# portfolio_4.jl — MOSEK port "portfolio_4_transcost", MIQP version: a
# cardinality-limited portfolio with binaries plus linear transaction costs.
#
#   max r'x - gamma x'Sigma x - f'|x - x0|
#   s.t.  sum(x) = 1,  0 <= x_j <= y_j,  y binary,  sum(y) <= k
#
# Data from the MOSEK tutorial, gamma = 0.05, f = 0.01, x0 uniform, k = 3;
# |x - x0| uses auxiliary u_j. Checks: budget, binaries, obj <= portfolio_1.
# Port of examples/c/portfolio_4.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    n, k = 3, 3
    r = [0.10717, 0.07502, 0.11902]
    GT = [0.1667, 0.0, 0.0, 0.0247, 0.1592, 0.0, 0.0197, 0.0158, 0.1327]
    gamma, f = 0.05, 0.01
    x0 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    S = zeros(9)
    for i in 0:2, j in 0:2
        S[i*3+j+1] = 0.0
        for q in 0:2
            S[i*3+j+1] += GT[i*3+q+1] * GT[j*3+q+1]
        end
    end
    m = Model(0, 0)
    res = try
        add_vars(m, 3n)                             # x (0..2), u (3..5), y (6..8)
        add_cons(m, 2 + n + 2n)                     # budget, cardinality, x_j <= y_j, |x_j-x0_j| <= u_j
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
            cj(m, n + j, -f)
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, n + j, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, 2n + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, 2n + j, VT_INT_BIN)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget: sum x = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [6, 7, 8], [1.0, 1.0, 1.0])    # cardinality: sum y <= k
        con_bounds(m, 1, BK_UP; lo=-Inf, up=Float64(k))
        for j in 0:n-1                              # x_j - y_j <= 0
            a_row(m, 2 + j, [j, 2n + j], [1.0, -1.0])
            con_bounds(m, 2 + j, BK_UP; lo=-Inf, up=0.0)
        end
        for j in 0:n-1                              # |x_j - x0_j| <= u_j
            a_row(m, 2 + n + j, [j, n + j], [1.0, -1.0])
            con_bounds(m, 2 + n + j, BK_UP; lo=-Inf, up=x0[j+1])
            a_row(m, 2 + 2n + j, [j, n + j], [-1.0, -1.0])
            con_bounds(m, 2 + 2n + j, BK_UP; lo=-Inf, up=-x0[j+1])
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
    @printf("y = (%.0f, %.0f, %.0f)\n", xx[7], xx[8], xx[9])
    @printf("obj = %.6f\n", obj)
    ok = abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-5
    nact = 0
    for j in 1:n
        xx[j] > 1e-6 && (nact += 1)
        abs(xx[2n+j] - floor(xx[2n+j] + 0.5)) > 1e-6 && (ok = false)
        xx[j] > xx[2n+j] + 1e-6 && (ok = false)
    end
    nact > k && (ok = false)
    obj > 0.119066 + 1e-6 && (ok = false)
    @printf("%s (MIQP: budget, binaries, |x-x0|<=u, obj <= 0.119066)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
