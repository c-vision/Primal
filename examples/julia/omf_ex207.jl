#!/usr/bin/env julia
# omf_ex207.jl — relative robust portfolio, "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Exercise 20.7 / formulation (20.17):
#
#   minimize t
#   s.t.  5.662 - (6 x1 + 4 x2) <= t,  5.662 - (4 x1 + 6 x2) <= t,  5 - (5 x1 + 5 x2) <= t
#         TE(x) <= 0.10,  x1 + x2 + x3 = 1,  x >= 0
#
# TE(x) = sqrt(d'C d), d = (x1 - 0.5, x2 - 0.5, x3), written with the Cholesky
# factor of C's 2x2 block as the quadratic cone (0.10, y1, y2). The book states
# x* = (0.5, 0.5, 0), t* = 0.662. Port of examples/c/omf_ex207.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    L11, L21, L22 = 0.42, 0.231, 0.235678147
    m = Model(7, 7)                                 # x1,x2,x3 ; t ; s(=0.10) ; y1,y2
    r = try
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, 3, BK_FR; lo=-Inf, up=Inf)    # t
        var_bounds(m, 4, BK_FX; lo=0.10, up=0.10)   # s
        var_bounds(m, 5, BK_FR; lo=-Inf, up=Inf)    # y1
        var_bounds(m, 6, BK_FR; lo=-Inf, up=Inf)    # y2
        cj(m, 3, 1.0)                               # min t
        a_row(m, 0, [0, 1, 2], [1, 1, 1]); con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [0, 1, 3], [6, 4, 1]); con_bounds(m, 1, BK_LO; lo=5.662, up=Inf)
        a_row(m, 2, [0, 1, 3], [4, 6, 1]); con_bounds(m, 2, BK_LO; lo=5.662, up=Inf)
        a_row(m, 3, [0, 1, 3], [5, 5, 1]); con_bounds(m, 3, BK_LO; lo=5.0, up=Inf)
        a_row(m, 4, [0, 5], [-L11, 1])             # y1 = L11 (x1 - 0.5)
        con_bounds(m, 4, BK_FX; lo=-0.5 * L11, up=-0.5 * L11)
        a_row(m, 5, [0, 1, 6], [-L21, -L22, 1])    # y2 = L21 (x1 - 0.5) + L22 (x2 - 0.5)
        con_bounds(m, 5, BK_FX; lo=-0.5 * (L21 + L22), up=-0.5 * (L21 + L22))
        cone(m, CT_QUAD, [4, 5, 6])
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        te = ccall(:hypot, Cdouble, (Cdouble, Cdouble), x[6], x[7])
        ok = abs(x[1] - 0.5) < 1e-4 && abs(x[2] - 0.5) < 1e-4 && abs(x[3]) < 1e-4 &&
             abs(z - 0.662) < 1e-3 && te <= 0.10 + 1e-6
        @printf("omf_ex207  x=(%.6g,%.6g,%.6g) t=%.6g TE=%.6g (book (0.5,0.5,0), 0.662) %s\n",
                x[1], x[2], x[3], z, te, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex207  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
