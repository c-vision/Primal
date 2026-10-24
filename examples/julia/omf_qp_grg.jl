#!/usr/bin/env julia
# omf_qp_grg.jl — the QP solved by GRG in "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 5:
#
#   minimize (x1 - 1/2)^2 + (x2 - 5/2)^2  s.t.  x1 - x2 >= 0,  x1 >= 0,  0 <= x2 <= 2
#
# The book's optimum is x = (1.5, 1.5). Expanded, f = x1^2 + x2^2 - x1 - 5 x2 + 6.5:
# Q = 2 I, c = (-1, -5), and f = solver objective + 6.5 = 2.
# Port of examples/c/omf_qp_grg.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(1, 2)
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)    # x1
        var_bounds(m, 1, BK_RA; lo=0.0, up=2.0)    # 0 <= x2 <= 2
        cj(m, 0, -1.0); cj(m, 1, -5.0)
        qobj(m, [(0, 0, 2.0), (1, 1, 2.0)])        # Q = 2 I  (0.5 x'Qx = x1^2 + x2^2)
        a_row(m, 0, [0, 1], [1.0, -1.0])           # x1 - x2 >= 0
        con_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        f = z + 6.5   # add back the constant of (x1-1/2)^2+(x2-5/2)^2
        ok = abs(x[1] - 1.5) < 1e-6 && abs(x[2] - 1.5) < 1e-6 && abs(f - 2.0) < 1e-6 && x[1] - x[2] >= -1e-6
        @printf("omf_qp_grg  x=(%.6g,%.6g)  f=%.6g (book x=(1.5,1.5), f=2) %s\n", x[1], x[2], f, ok ? "OK" : "FAIL")
    else
        @printf("omf_qp_grg  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
