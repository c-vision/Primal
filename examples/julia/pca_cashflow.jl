#!/usr/bin/env julia
# pca_cashflow.jl — cash-flow matching, an independent LP exercise (model from
# "Portfolio Construction and Analytics" (2016), Section 15.2.1.4):
#
#   minimize sum_i p_i x_i  s.t.  sum_i c_it x_i >= m_t (t = 1..8),  x >= 0
#
# Five semi-annual-coupon bonds, eight obligations. The book's allocation
# matches ours to the cent, x = (6000, 28103.61, 3555.18, 0, 9661.84); its
# printed objective is ~$148 off those same numbers (ignored here).
# Port of examples/c/pca_cashflow.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    p = [102.36, 110.83, 96.94, 114.65, 96.63]
    c = [  2.50    5.00    3.00    4.00    3.50
           2.50    5.00    3.00    4.00    3.50
           2.50    5.00    3.00    4.00    3.50
           2.50    5.00    3.00    4.00    3.50
         102.50    5.00    3.00    4.00    3.50
           0.00  105.00    3.00    4.00    3.50
           0.00    0.00  103.00    4.00    3.50
           0.00    0.00    0.00  104.00  103.50]
    mt = [100000.0, 200000, 100000, 200000, 800000, 1200000, 400000, 1000000]
    m = Model(8, 5)
    r = try
        for j in 0:4
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            cj(m, j, p[j+1])
        end
        for i in 0:7
            a_row(m, i, [0, 1, 2, 3, 4], c[i+1, :])
            con_bounds(m, i, BK_LO; lo=mt[i+1], up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 6000.0) < 1.0 && abs(x[2] - 28103.61) < 1.0 && abs(x[3] - 3555.18) < 1.0 &&
             abs(x[4]) < 1.0 && abs(x[5] - 9661.84) < 1.0 && abs(z - 5007145.84) < 1.0
        @printf("pca_cashflow  x=(%.0f,%.2f,%.2f,%.0f,%.2f) cost=%.2f (independent LP; book allocation matches) %s\n",
                x[1], x[2], x[3], x[4], x[5], z, ok ? "OK" : "FAIL")
    else
        @printf("pca_cashflow  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
