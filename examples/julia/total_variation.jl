#!/usr/bin/env julia
# total_variation.jl — total-variation denoising (MOSEK Cookbook 11.3).
#
#   minimize sum of the TV terms t
#   s.t.  (t, du) in QUAD  (total variation),  (sigma, f - u) in QUAD  (fidelity),  u in [0,1]
#
# Exercised on the smallest non-trivial signal, f = (0, 1), sigma = 1/2:
#   minimize |u1 - u0|  s.t.  u0^2 + (1 - u1)^2 <= 1/4,
# whose optimum is 1 - 1/sqrt(2) = 0.29289322 at u0 = 1/(2 sqrt 2) = 0.35355339.
# Port of examples/c/total_variation.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    f, sigma = [0.0, 1.0], 0.5
    U, TT, DX, E0, E1, SG, NV = 0, 2, 3, 4, 5, 6, 7
    m = Model(3, NV)
    r = try
        sense(m, SENSE_MIN)
        for j in 0:1
            var_bounds(m, U + j, BK_RA; lo=0.0, up=1.0)
        end
        var_bounds(m, TT, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, DX, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, E0, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, E1, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, SG, BK_FX; lo=sigma, up=sigma)
        cj(m, TT, 1.0)
        # dx = u1 - u0 ; e_j = f_j - u_j
        a_row(m, 0, [DX, U, U + 1], [1.0, 1.0, -1.0]); con_bounds(m, 0, BK_FX; lo=0.0, up=0.0)
        a_row(m, 1, [E0, U], [1.0, 1.0]);              con_bounds(m, 1, BK_FX; lo=f[1], up=f[1])
        a_row(m, 2, [E1, U + 1], [1.0, 1.0]);          con_bounds(m, 2, BK_FX; lo=f[2], up=f[2])
        cone(m, CT_QUAD, [TT, DX])                  # TV cone (t, dx)
        cone(m, CT_QUAD, [SG, E0, E1])              # fidelity cone (sigma, e0, e1)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, u = 0.0, zeros(NV)
    if ok
        obj, u = r.objective, r.x
    end
    tv, a0 = 1.0 - 1.0 / sqrt(2.0), 1.0 / (2.0 * sqrt(2.0))
    good = ok && abs(obj - tv) < 1e-7 && abs(u[1] - a0) < 1e-6 && abs(u[2] - (1.0 - a0)) < 1e-6 &&
           u[1] * u[1] + (1.0 - u[2]) * (1.0 - u[2]) <= sigma * sigma + 1e-9
    @printf("total_variation: TV*=%.8f  u=(%.6f, %.6f)  (expected 1-1/sqrt2, u=(1/(2sqrt2),1-1/(2sqrt2)))  %s\n",
            obj, u[1], u[2], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
