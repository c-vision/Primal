#!/usr/bin/env julia
# sharpe_ratio.jl — the max-Sharpe case (A3) of arXiv:2508.03704, Section 4.4.
#
#   max (mu'w - rf) / sqrt(w'Sigma w)  s.t.  e'w = 1
#
# is a ratio, not a QP. Charnes-Cooper: kappa = 1/sqrt(w'Sigma w), y = kappa w,
#
#   max mu'y - rf kappa  s.t.  y'Sigma y <= 1,  e'y = kappa,  kappa >= 0,
#
# and w = y/kappa. Hand case: mu = (3, 1), rf = 0, Sigma = I: max Sharpe
# ||mu|| = sqrt(10) at w = (3/4, 1/4). Port of examples/c/sharpe_ratio.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    mu = [3.0, 1.0]
    m = Model(0, 0)
    r = try
        add_vars(m, 3)                              # y0, y1, kappa
        add_cons(m, 2)                              # 0: y'Sigma y <= 1, 1: e'y - kappa = 0
        var_bounds(m, 0, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, 1, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, 2, BK_LO; lo=0.0, up=Inf)
        qconk(m, 0, [(0, 0, 2.0), (1, 1, 2.0)])    # Sigma = I: 1/2 x'Qx with Q = 2I
        con_bounds(m, 0, BK_UP; lo=-Inf, up=1.0)
        a_row(m, 1, [0, 1, 2], [1.0, 1.0, -1.0])
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        cj(m, 0, mu[1])
        cj(m, 1, mu[2])
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    sharpe, x = r.objective, r.x
    w0, w1 = x[1] / x[3], x[2] / x[3]
    nrm = sqrt(w0 * w0 + w1 * w1)
    ratio = (mu[1] * w0 + mu[2] * w1) / nrm
    ok = abs(sharpe - sqrt(10.0)) < 1e-6 && abs(w0 - 0.75) < 1e-5 && abs(w1 - 0.25) < 1e-5 &&
         abs(ratio - sharpe) < 1e-5 && abs(w0 + w1 - 1.0) < 1e-6
    @printf("Sharpe = %.6f (expected sqrt(10) = %.6f), w = (%.4f, %.4f) (expected .75,.25)\n",
            sharpe, sqrt(10.0), w0, w1)
    @printf("reconstructed (mu'w)/||w|| = %.6f, e'w = %.2e\n", ratio, w0 + w1)
    @printf("%s (A3 is not quadratic: solved with Charnes-Cooper)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
