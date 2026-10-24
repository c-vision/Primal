#!/usr/bin/env julia
# npo_minimax.jl — minimax (L-infinity) regression as an LP, "Nonlinear
# Parameter Optimization Using R Tools" (2014), Section 14.4:
#
#   x* = argmin_x max_i |A x - y|_i
#
# with A 10x3 (columns 1, i, i^2 for i = 1..10) and the perturbed response y
# of the L1 example. The book reports coeffs = (2.73966, -1.92954, -1.00881),
# mad = 0.726338. LP form: x free, t free, -t <= (Ax - y)_i <= t.
# Port of examples/c/npo_minimax.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    y = [0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
         -45.193280, -60.690202, -77.243728, -96.406035, -117.241829]
    m = Model(20, 4)                                # 3 x + t
    r = try
        for j in 0:3
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        cj(m, 3, 1.0)                               # min t
        for i in 0:9
            a1, a2 = 1.0, Float64(i + 1)
            a3 = a2 * a2
            # (A x)_i - y_i <= t  =>  a_i.x - t <= y_i
            a_row(m, 2i, [0, 1, 2, 3], [a1, a2, a3, -1.0])
            con_bounds(m, 2i, BK_UP; lo=-Inf, up=y[i+1])
            # -(A x)_i + y_i <= t  =>  -a_i.x - t <= -y_i
            a_row(m, 2i + 1, [0, 1, 2, 3], [-a1, -a2, -a3, -1.0])
            con_bounds(m, 2i + 1, BK_UP; lo=-Inf, up=-y[i+1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        v, z = r.x, r.objective
        ok = abs(v[1] - 2.73966) < 1e-4 && abs(v[2] + 1.92954) < 1e-4 &&
             abs(v[3] + 1.00881) < 1e-4 && abs(z - 0.726338) < 1e-5
        @printf("npo_minimax  coeffs=(%.6g,%.6g,%.6g) mad=%.6g (book (2.73966,-1.92954,-1.00881), 0.726338) %s\n",
                v[1], v[2], v[3], z, ok ? "OK" : "FAIL")
    else
        @printf("npo_minimax  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
