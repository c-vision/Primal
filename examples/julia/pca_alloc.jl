#!/usr/bin/env julia
# pca_alloc.jl — portfolio allocation LP, "Portfolio Construction and
# Analytics" (2016), Section 6.3. A $10,000,000 endowment over four funds:
# growth (ret 20.69%, risk 4), index (5.87%, 2), corporate bond (10.52%, 2),
# money market (2.43%, 1). Maximize the return s.t. total = 10M, funds 1+3
# <= 6M, average risk <= 2 (4x1 + 2x2 + 2x3 + x4 <= 20M), each x_i <= 4M.
#
# The book's optimum: return $931,800 at x = (2M, 0, 4M, 4M).
# Port of examples/c/pca_alloc.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    mu = [0.2069, 0.0587, 0.1052, 0.0243]
    risk = [4.0, 2.0, 2.0, 1.0]
    m = Model(3, 4)
    r = try
        for j in 0:3
            var_bounds(m, j, BK_RA; lo=0.0, up=4.0e6)
            cj(m, j, -mu[j+1])                      # maximize return (c negated, minimize)
        end
        a_row(m, 0, [0, 1, 2, 3], [1, 1, 1, 1]); con_bounds(m, 0, BK_FX; lo=1.0e7, up=1.0e7)
        a_row(m, 1, [0, 2], [1, 1]);             con_bounds(m, 1, BK_UP; lo=-Inf, up=6.0e6)
        a_row(m, 2, [0, 1, 2, 3], risk);         con_bounds(m, 2, BK_UP; lo=-Inf, up=2.0e7)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(-z - 931800.0) < 1.0 && abs(x[1] - 2.0e6) < 1.0 && abs(x[2]) < 1.0 &&
             abs(x[3] - 4.0e6) < 1.0 && abs(x[4] - 4.0e6) < 1.0
        @printf("pca_alloc  x=(%.6g,%.6g,%.6g,%.6g)e6 return=%.6g (book 2,0,4,4 -> 931800) %s\n",
                x[1] / 1e6, x[2] / 1e6, x[3] / 1e6, x[4] / 1e6, -z, ok ? "OK" : "FAIL")
    else
        @printf("pca_alloc  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
