#!/usr/bin/env julia
# oa_feed.jl — animal feed mix LP, "Optimization Algorithms: AI techniques for
# design, planning, and control problems" (2024), Listing 2.6 / Appendix C.1:
#
#   minimize 30.5 x1 + 10.0 x2 + 90 x3                            (cents per kg)
#   s.t.  0.008 <= 0.001 x1 + 0.38 x2 + 0.002 x3 <= 0.012         (calcium)
#         0.09 x1 + 0.50 x3 >= 0.22                               (protein)
#         0.02 x1 + 0.08 x3 <= 0.05                               (fiber)
#         x1 + x2 + x3 = 1,  0 <= x <= 1
#
# Corn = x1, Limestone = x2, Soybean meal = x3. The book's output: 65.0% /
# 3.0% / 32.0%, total cost 49.16 cents/kg. Port of examples/c/oa_feed.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(4, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_RA; lo=0.0, up=1.0)
        end
        cj(m, 0, 30.5); cj(m, 1, 10.0); cj(m, 2, 90.0)
        a_row(m, 0, [0, 1, 2], [0.001, 0.38, 0.002]); con_bounds(m, 0, BK_RA; lo=0.008, up=0.012)
        a_row(m, 1, [0, 2], [0.09, 0.50]);            con_bounds(m, 1, BK_LO; lo=0.22, up=Inf)
        a_row(m, 2, [0, 2], [0.02, 0.08]);            con_bounds(m, 2, BK_UP; lo=-Inf, up=0.05)
        a_row(m, 3, [0, 1, 2], [1, 1, 1]);            con_bounds(m, 3, BK_FX; lo=1.0, up=1.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 0.65) < 0.01 && abs(x[2] - 0.03) < 0.01 &&
             abs(x[3] - 0.32) < 0.01 && abs(z - 49.16) < 0.05
        @printf("oa_feed  corn=%.4g%% lime=%.4g%% soy=%.4g%% cost=%.4g c/kg (book 65/3/32, 49.16) %s\n",
                100.0 * x[1], 100.0 * x[2], 100.0 * x[3], z, ok ? "OK" : "FAIL")
    else
        @printf("oa_feed  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
