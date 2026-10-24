#!/usr/bin/env julia
# risk_budgeting.jl — risk budgeting: target risk contributions.
#
# Find w > 0 with w_i (Sigma w)_i = b_i (w'Sigma w). The convex formulation
# (Spinu 2013) is  min (1/2) w'Sigma w - sum_i b_i ln(w_i),  normalised after.
# The logarithms use the exponential cone: s_i >= -ln(w_i) <=> (w_i, 1, -s_i) in K_exp.
#
# Hand case: Sigma = I, b = (0.8, 0.2): w ~ (sqrt(0.8), sqrt(0.2)), normalised
# (2/3, 1/3), risk contributions exactly (0.8, 0.2).
# Port of examples/c/risk_budgeting.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    b = [0.8, 0.2]
    m = Model(0, 0)
    r = try
        add_vars(m, 7)                              # w0, w1, s0, s1, v0, v1, one
        add_cons(m, 2)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        for j in 2:5
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, 6, BK_FX; lo=1.0, up=1.0)
        # (w_i, one, v_i) in K_exp  =>  w_i >= exp(v_i) = exp(-s_i)
        cone(m, CT_PEXP, [0, 6, 4])
        cone(m, CT_PEXP, [1, 6, 5])
        a_row(m, 0, [4, 2], [1.0, 1.0]); con_bounds(m, 0, BK_FX; lo=0.0, up=0.0)   # v_i + s_i = 0
        a_row(m, 1, [5, 3], [1.0, 1.0]); con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        # objective: (1/2)(w0^2 + w1^2) + b0 s0 + b1 s1
        qobj(m, [(0, 0, 2.0)])
        qobj(m, [(1, 1, 2.0)])
        cj(m, 2, b[1])
        cj(m, 3, b[2])
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    x = r.x
    w0 = x[1] / (x[1] + x[2])
    w1 = x[2] / (x[1] + x[2])
    rc0 = (w0 * w0) / (w0 * w0 + w1 * w1)
    rc1 = (w1 * w1) / (w0 * w0 + w1 * w1)
    # the conic IPM delivers ~1e-5 here; the budgets are the meaningful check
    ok = abs(w0 - 2.0 / 3.0) < 1e-3 && abs(w1 - 1.0 / 3.0) < 1e-3 &&
         abs(rc0 - b[1]) < 1e-4 && abs(rc1 - b[2]) < 1e-4
    @printf("w = (%.6f, %.6f) (expected 0.6667, 0.3333)\n", w0, w1)
    @printf("risk contributions = (%.6f, %.6f) (expected 0.8, 0.2)\n", rc0, rc1)
    @printf("%s (risk budgeting, log via exponential cone)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
