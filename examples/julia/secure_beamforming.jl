#!/usr/bin/env julia
# secure_beamforming.jl — the convex core of the SCA subproblem of
# vkumar-ucd/secure_ISAC_GC23 (IRS-secure-ISAC beamforming, GLOBECOM 2023).
# One step of the successive convex approximation is the SOCP
#
#   maximize a'w  s.t.  h'w >= sqrt(gamma) ||w||  (SINR),  ||w|| <= sqrt(P)  (power)
#
# Hand instance: h = (1, 0), a = (1, 1)/sqrt(2), gamma = 1/2, P = 2
# -> w = (1, 1), a'w = sqrt(2), gain 2, both constraints active.
# Port of examples/c/secure_beamforming.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    h = [1.0, 0.0]
    a = [1.0 / sqrt(2.0), 1.0 / sqrt(2.0)]
    gamma, P = 0.5, 2.0
    sg, sP = sqrt(gamma), sqrt(P)
    W, PB, U, Q, NV, NROW = 0, 2, 3, 5, 6, 3
    m = Model(NROW, NV)
    r = try
        for v in 0:NV-1
            var_bounds(m, v, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, Q, BK_FX; lo=sP, up=sP)       # sqrt(P)
        sense(m, SENSE_MAX)
        for j in 0:1
            cj(m, W + j, a[j+1])                    # maximize a'w
        end
        a_row(m, 0, [PB, W, W + 1], [1.0, -h[1], -h[2]])   # p = h'w
        con_bounds(m, 0, BK_FX; lo=0.0, up=0.0)
        for j in 0:1                                # u_j = sqrt(gamma) w_j
            a_row(m, 1 + j, [U + j, W + j], [1.0, -sg])
            con_bounds(m, 1 + j, BK_FX; lo=0.0, up=0.0)
        end
        cone(m, CT_QUAD, [PB, U, U + 1])            # SINR: (h'w, sqrt(gamma) w) in QUAD
        cone(m, CT_QUAD, [Q, W, W + 1])             # power: (sqrt(P), w) in QUAD
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, w = 0.0, zeros(NV)
    if ok
        obj, w = r.objective, r.x
    end
    # the optimum is degenerate (the objective gradient is radial on the power
    # circle), so w is accurate to ~1e-4 while the objective is to 1e-7.
    good = ok && abs(obj - sqrt(2.0)) < 1e-6 && abs(w[1] - 1.0) < 1e-3 && abs(w[2] - 1.0) < 1e-3 &&
           abs(w[1] * w[1] + w[2] * w[2] - P) < 1e-6   # power active
    @printf("secure_beamforming: a'w=%.8f  w=(%.6f,%.6f)  gain=%.8f  (expected sqrt2, w=(1,1), gain=2)  %s\n",
            obj, w[1], w[2], obj * obj, good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
