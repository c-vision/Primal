#!/usr/bin/env julia
# robust_cvar.jl — worst-case CVaR over a box of return uncertainty.
#
# CVaR (Rockafellar-Uryasev) is an LP in (w, alpha, z). If each scenario return
# may deviate in a box r_s + delta, |delta| <= d_s, and w >= 0, the worst case
# is the low corner r_s - d_s: the robust CVaR is the same LP on shifted returns.
#
# Hand case: 2 equiprobable scenarios, beta = 1/2, r0 = (0.10, 0.20),
# r1 = (0.30, 0.00), d = (0.05, 0.05). At w = (1/2, 1/2) the robust CVaR is
# -0.10, the nominal -0.15: robustness costs 0.05. Port of examples/c/robust_cvar.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# robust: subtract the box radius from each scenario return
function solve_cvar(robust)
    R0, R1, D = [0.10, 0.20], [0.30, 0.00], [0.05, 0.05]
    r0 = [R0[j] - (robust ? D[j] : 0.0) for j in 1:2]
    r1 = [R1[j] - (robust ? D[j] : 0.0) for j in 1:2]
    m = Model(0, 0)
    r = try
        add_vars(m, 5)                              # w0, w1, alpha, z0, z1
        add_cons(m, 3)
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, 3, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 4, BK_LO; lo=0.0, up=Inf)
        cj(m, 2, 1.0); cj(m, 3, 1.0); cj(m, 4, 1.0)
        # z_s + r_s'w + alpha >= 0
        a_row(m, 0, [3, 0, 1, 2], [1.0, r0[1], r0[2], 1.0]); con_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        a_row(m, 1, [4, 0, 1, 2], [1.0, r1[1], r1[2], 1.0]); con_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        a_row(m, 2, [0, 1], [1.0, 1.0]); con_bounds(m, 2, BK_FX; lo=1.0, up=1.0)   # e'w = 1
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return 1e30, [NaN, NaN]
    return r.objective, r.x[1:2]
end

function main()
    nom, wn = solve_cvar(false)
    rob, wr = solve_cvar(true)
    ok = abs(rob - (-0.10)) < 1e-7 && abs(wr[1] - 0.5) < 1e-6 && abs(wr[2] - 0.5) < 1e-6 &&
         abs(nom - (-0.15)) < 1e-7 && rob >= nom - 1e-9
    @printf("nominal CVaR = %.6f (expected -0.15), w = (%.4f, %.4f)\n", nom, wn[1], wn[2])
    @printf("robust  CVaR = %.6f (expected -0.10), w = (%.4f, %.4f)\n", rob, wr[1], wr[2])
    @printf("%s (robustness costs: robust CVaR >= nominal)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
