#!/usr/bin/env julia
# mean_cvar.jl — mean-CVaR portfolio.
#
#   max mu'w - lambda CVaR_beta(w)  s.t.  e'w = 1,  w >= 0
#
# with the Rockafellar-Uryasev CVaR linearised: one LP in (w, alpha, z) with
# z_s >= -r_s'w - alpha, z_s >= 0. Hand case: 2 assets, 2 equiprobable
# scenarios, beta = 1/2, lambda = 1, mu = (0.2, 0.1), r0 = (0.1, 0.3),
# r1 = (0.4, -0.1): the optimum sits at the kink r0'w = r1'w, w = (4/7, 3/7),
# value 12/35. Port of examples/c/mean_cvar.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    mu = [0.2, 0.1]
    R0 = [0.1, 0.3]
    R1 = [0.4, -0.1]
    m = Model(0, 0)
    r = try
        add_vars(m, 5)                              # w0, w1, alpha, z0, z1
        add_cons(m, 3)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, 3, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 4, BK_LO; lo=0.0, up=Inf)
        # objective: max mu'w - alpha - z0 - z1
        cj(m, 0, mu[1])
        cj(m, 1, mu[2])
        cj(m, 2, -1.0)
        cj(m, 3, -1.0)
        cj(m, 4, -1.0)
        sense(m, SENSE_MAX)
        # z_s + r_s'w + alpha >= 0
        a_row(m, 0, [3, 0, 1, 2], [1.0, R0[1], R0[2], 1.0]); con_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        a_row(m, 1, [4, 0, 1, 2], [1.0, R1[1], R1[2], 1.0]); con_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        a_row(m, 2, [0, 1], [1.0, 1.0]); con_bounds(m, 2, BK_FX; lo=1.0, up=1.0)   # e'w = 1
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    obj, x = r.objective, r.x
    w0, w1 = x[1], x[2]
    cvar = max(-(R0[1] * w0 + R0[2] * w1), -(R1[1] * w0 + R1[2] * w1))
    ok = abs(w0 - 4.0 / 7.0) < 1e-6 && abs(w1 - 3.0 / 7.0) < 1e-6 && abs(obj - 12.0 / 35.0) < 1e-6 &&
         abs((mu[1] * w0 + mu[2] * w1) - cvar - obj) < 1e-6
    @printf("w = (%.6f, %.6f) (expected 0.5714, 0.4286)\n", w0, w1)
    @printf("obj = %.6f (expected 12/35 = %.6f), CVaR = %.6f\n", obj, 12.0 / 35.0, cvar)
    @printf("%s (mean-CVaR, optimum at the kink r0'w = r1'w)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
