#!/usr/bin/env julia
# portfolio_2.jl — MOSEK port "portfolio_2_frontier": mean-variance frontier.
# For a grid of gamma values it solves the Markowitz QP
#
#   max r'x - gamma x'Sigma x  s.t.  sum(x) = 1,  x >= 0
#
# and prints (risk, expected return): risk must not grow as gamma grows.
# Port of examples/c/portfolio_2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve the Markowitz QP for one gamma; returns (ok, x, risk, return).
function solve_gamma(gamma)
    n = 3
    r = [0.10717, 0.07502, 0.11902]
    GT = [0.1667, 0.0, 0.0, 0.0247, 0.1592, 0.0, 0.0197, 0.0158, 0.1327]
    S = zeros(9)
    for i in 0:2, j in 0:2
        S[i*3+j+1] = 0.0
        for k in 0:2
            S[i*3+j+1] += GT[i*3+k+1] * GT[j*3+k+1]
        end
    end
    m = Model(0, 0)
    res = try
        add_vars(m, n)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
        end
        qobj(m, [(0, 0, -2.0 * gamma * S[1]), (0, 1, -2.0 * gamma * S[2]), (0, 2, -2.0 * gamma * S[3]),
                 (1, 1, -2.0 * gamma * S[5]), (1, 2, -2.0 * gamma * S[6]), (2, 2, -2.0 * gamma * S[9])])
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for j in 0:n-1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    res.rc == 0 || return false, zeros(3), 0.0, 0.0
    x = res.x
    risk = 0.0
    for i in 0:n-1, j in 0:n-1
        risk += x[i+1] * S[i*3+j+1] * x[j+1]
    end
    ret = r[1] * x[1] + r[2] * x[2] + r[3] * x[3]
    return true, x, risk, ret
end

function main()
    gammas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    @printf("%8s %12s %12s   x\n", "gamma", "risk", "return")
    prev_risk = Inf
    ok = true
    for g in gammas
        good, x, risk, ret = solve_gamma(g)
        good || (ok = false; break)
        @printf("%8.2f %12.6f %12.6f   (%.4f, %.4f, %.4f)\n", g, risk, ret, x[1], x[2], x[3])
        abs(x[1] + x[2] + x[3] - 1.0) > 1e-6 && (ok = false)   # feasibility
        any(v -> v < -1e-9, x) && (ok = false)
        # monotonicity: risk does not grow as gamma grows (flat up to solver
        # precision while the optimum is the same vertex)
        risk > prev_risk + 1e-6 && (ok = false)
        prev_risk = risk
    end
    @printf("%s (monotone risk frontier)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
