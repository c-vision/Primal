#!/usr/bin/env julia
# omf_financing.jl — the short-term financing LP of "Optimization Methods in
# Finance" (Cornuejols & Tutuncu, 2007), Section 3.1.1. Monthly cash needs
# (k$) Jan..Jun: 150, 100, -200, 200, -50, -300; a credit line x_i (1%/month,
# limit 100), commercial paper y_i in Jan-Mar (2% for three months), excess
# funds z_i (0.5%/month). Maximize the wealth v at the end of June.
#
# The book's sensitivity report gives shadow prices u_Jan = -1.0373,
# u_Feb = -1.030, u_Mar = -1.020 (the primal optimum is not unique).
# Port of examples/c/omf_financing.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(6, 14)                                # 0..4 = x1..x5, 5..7 = y1..y3, 8..12 = z1..z5, 13 = v
    r = try
        for i in 0:4
            var_bounds(m, i, BK_RA; lo=0.0, up=100.0)
        end
        for i in 5:12
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, 13, BK_FR; lo=-Inf, up=Inf)   # v
        cj(m, 13, -1.0)                             # max v
        rhs = [150.0, 100, -200, 200, -50, -300]
        ii = [[0, 5, 8], [1, 6, 0, 8, 9], [2, 7, 1, 9, 10], [3, 5, 2, 10, 11], [4, 6, 3, 11, 12], [7, 4, 12, 13]]
        vv = [[1.0, 1.0, -1.0], [1.0, 1.0, -1.01, 1.003, -1.0], [1.0, 1.0, -1.01, 1.003, -1.0],
              [1.0, -1.02, -1.01, 1.003, -1.0], [1.0, -1.02, -1.01, 1.003, -1.0], [-1.02, -1.01, 1.003, -1.0]]
        for k in 0:5
            a_row(m, k, ii[k+1], vv[k+1])
            con_bounds(m, k, BK_FX; lo=rhs[k+1], up=rhs[k+1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        obj, x, y = r.objective, r.x, r.y
        v = -obj                                    # the objective is max v, reported as min -v
        # book shadow prices Jan/Feb/Mar = -1.0373/-1.030/-1.020
        ok = abs(abs(y[1]) - 1.0373) < 1e-3 && abs(abs(y[2]) - 1.030) < 1e-3 && abs(abs(y[3]) - 1.020) < 1e-3
        @printf("omf_financing  v=%.4f  x=(%.2f,%.4f,%.2f,%.2f,%.2f)  uJan=%.4f uFeb=%.4f uMar=%.4f  %s\n",
                v, x[1], x[2], x[3], x[4], x[5], abs(y[1]), abs(y[2]), abs(y[3]), ok ? "OK" : "FAIL")
    else
        @printf("omf_financing  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
