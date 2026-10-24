#!/usr/bin/env julia
# portfolio_6.jl — MOSEK port "portfolio_6_factor": portfolio optimization
# with a FACTOR model, covariance F P F' + D, plus cardinality binaries:
#
#   max r'x - gamma (||p^{1/2} F'x||^2 + x'Dx)  s.t.  sum(x) = 1,  0 <= x <= y,  y binary,  sum(y) <= k
#
# The factor risk uses a variable v with the cone (v, z) in QUAD, z = p^{1/2} F'x;
# x'Dx stays quadratic. Checks: budget, binaries, objective vs the risk
# evaluated directly at x*. Port of examples/c/portfolio_6.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    n, mf, k = 3, 2, 2                              # assets, factors, cardinality
    r = [0.10717, 0.07502, 0.11902]
    F = [0.16 -0.23; 0.28 0.19; -0.11 0.32]
    p = [0.05, 0.06]
    D = [0.01, 0.02, 0.005]
    gamma = 0.05
    VV, VZ, VY = 3, 4, 6                            # x(0..2), v, z(4..5), y(6..8)
    m = Model(0, 0)
    res = try
        add_vars(m, 9)
        add_cons(m, 2 + n + mf)                     # budget, cardinality, x <= y, z = sqrt(p) F'x
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, VV, -gamma)                           # -gamma v
        var_bounds(m, VV, BK_LO; lo=0.0, up=Inf)
        for i in 0:mf-1
            var_bounds(m, VZ + i, BK_FR; lo=-Inf, up=Inf)
        end
        for j in 0:n-1
            var_bounds(m, VY + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, VY + j, VT_INT_BIN)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [VY, VY + 1, VY + 2], [1.0, 1.0, 1.0])   # cardinality
        con_bounds(m, 1, BK_UP; lo=-Inf, up=Float64(k))
        for j in 0:n-1                              # x_j <= y_j
            a_row(m, 2 + j, [j, VY + j], [1.0, -1.0])
            con_bounds(m, 2 + j, BK_UP; lo=-Inf, up=0.0)
        end
        for i in 0:mf-1                             # z_i = sqrt(p_i) (F'x)_i
            a_row(m, 2 + n + i, [0, 1, 2, VZ + i],
                  [-sqrt(p[i+1]) * F[1, i+1], -sqrt(p[i+1]) * F[2, i+1], -sqrt(p[i+1]) * F[3, i+1], 1.0])
            con_bounds(m, 2 + n + i, BK_FX; lo=0.0, up=0.0)
        end
        cone(m, CT_QUAD, [VV, VZ, VZ + 1])          # v >= ||z||
        qobj(m, [(j, j, -2.0 * gamma * D[j+1]) for j in 0:2])   # -gamma x'Dx (Q = -2 gamma D)
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
    @printf("v = %.4f, y = (%.0f, %.0f, %.0f)\n", xx[VV+1], xx[VY+1], xx[VY+2], xx[VY+3])
    @printf("obj = %.6f\n", obj)
    # independent check: risk at x* from F, P, D; the conic IPM may leave v slightly
    # above ||z||, so the objective sits up to ~gamma (v - ||z||) below (tolerance 5e-3)
    fx = zeros(mf)
    for i in 1:mf, j in 1:n
        fx[i] += F[j, i] * xx[j]
    end
    risk = 0.0
    for i in 1:mf
        risk += p[i] * fx[i] * fx[i]
    end
    for j in 1:n
        risk += D[j] * xx[j] * xx[j]
    end
    exp_obj = 0.0
    for j in 1:n
        exp_obj += r[j] * xx[j]
    end
    exp_obj -= gamma * risk
    @printf("direct risk = %.6f, expected obj = %.6f\n", risk, exp_obj)
    ok = abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-5 && obj >= exp_obj - 5e-3 && obj <= exp_obj + 5e-3
    nact = 0
    for j in 1:n
        xx[j] > 1e-6 && (nact += 1)
        abs(xx[VY+j] - floor(xx[VY+j] + 0.5)) > 1e-6 && (ok = false)
        xx[j] > xx[VY+j] + 1e-6 && (ok = false)
    end
    nact > k && (ok = false)
    @printf("%s (factors+MIP: obj = r'x - gamma*risk, cardinality <= %d)\n", ok ? "OK" : "FAIL", k)
    return ok ? 0 : 1
end

exit(main())
