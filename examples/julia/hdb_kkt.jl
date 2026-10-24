#!/usr/bin/env julia
# hdb_kkt.jl — Lagrangian-duality LP example, "Handbook of Algorithms for
# Physical Design Automation" (2009), Section 6.3:
#
#   minimize x1 + x2  subject to  x1 + 2 x2 = 2,  (x1, x2) >= 0
#
# The book derives x* = (0, 1), p* = 1, and the equality multiplier nu* = 1/2.
# Port of examples/c/hdb_kkt.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(1, 2)
    r = try
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, 1.0); cj(m, 1, 1.0)
        a_row(m, 0, [0, 1], [1.0, 2.0])
        con_bounds(m, 0, BK_FX; lo=2.0, up=2.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, y, z = r.x, r.y, r.objective
        ok = abs(x[1] - 0.0) < 1e-6 && abs(x[2] - 1.0) < 1e-6 && abs(z - 1.0) < 1e-6 &&
             abs(abs(y[1]) - 0.5) < 1e-6
        @printf("hdb_kkt  x=(%.6g,%.6g) p*=%.6g |y|=%.6g (book (0,1), 1, 1/2) %s\n",
                x[1], x[2], z, abs(y[1]), ok ? "OK" : "FAIL")
    else
        @printf("hdb_kkt  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
