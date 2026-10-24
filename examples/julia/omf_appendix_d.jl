#!/usr/bin/env julia
# omf_appendix_d.jl — revised-simplex worked LP, "Optimization Methods in
# Finance" (Cornuejols & Tutuncu, 2007), Appendix D:
#
#   maximize Z = x1 + 2 x2 + x3 - 2 x4
#   s.t.  -2 x1 +   x2 + x3 + 2 x4 + x6           = 2
#          -x1 + 2 x2 + x3      + x5 + x7         = 7
#           x1       + x3 + x4 + x5          + x8 = 3,   x >= 0
#
# The book reports x1 = 3, x2 = 5, x6 = 3, all other x = 0, Z = 13.
# Port of examples/c/omf_appendix_d.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(3, 8)
    r = try
        for j in 0:7
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        sense(m, SENSE_MAX)
        cj(m, 0, 1.0); cj(m, 1, 2.0)
        cj(m, 2, 1.0); cj(m, 3, -2.0)
        a_row(m, 0, [0, 1, 2, 3, 5], [-2, 1, 1, 2, 1]); con_bounds(m, 0, BK_FX; lo=2.0, up=2.0)
        a_row(m, 1, [0, 1, 2, 4, 6], [-1, 2, 1, 1, 1]); con_bounds(m, 1, BK_FX; lo=7.0, up=7.0)
        a_row(m, 2, [0, 2, 3, 4, 7], [1, 1, 1, 1, 1]);  con_bounds(m, 2, BK_FX; lo=3.0, up=3.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(x[1] - 3.0) < 1e-6 && abs(x[2] - 5.0) < 1e-6 && abs(x[6] - 3.0) < 1e-6
        for j in 2:7
            j != 5 && (ok = ok && abs(x[j+1]) < 1e-6)
        end
        ok = ok && abs(z - 13.0) < 1e-6
        @printf("omf_appendix_d  x1=%.6g x2=%.6g x6=%.6g Z=%.6g (book 3,5,3, 13) %s\n",
                x[1], x[2], x[6], z, ok ? "OK" : "FAIL")
    else
        @printf("omf_appendix_d  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
