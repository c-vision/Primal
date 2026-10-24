#!/usr/bin/env julia
# npo_lad.jl — L1 (least absolute deviations) regression as an LP, "Nonlinear
# Parameter Optimization Using R Tools" (2014), Section 14.3:
#
#   x* = argmin_x sum_i |A x - y|_i
#
# with A 10x3, columns (1, i, i^2) for i = 1..10, and a perturbed response y.
# The book reports coeffs = (3.489517, -2.174797, -0.989834), sad = 3.2951.
# LP form: x free, z >= 0, z >= +/-(Ax - y). Port of examples/c/npo_lad.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    y = [0.324886, -4.428305, -12.854561, -20.393083, -32.854574,
         -45.193280, -60.690202, -77.243728, -96.406035, -117.241829]
    m = Model(20, 13)                               # 3 x (free) + 10 z (>= 0)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        for j in 3:12
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            cj(m, j, 1.0)
        end
        for i in 0:9
            a1, a2 = 1.0, Float64(i + 1)           # row [1, i, i^2]
            a3 = a2 * a2
            # z_i - (A x)_i >= -y_i
            a_row(m, 2i, [0, 1, 2, 3 + i], [-a1, -a2, -a3, 1.0])
            con_bounds(m, 2i, BK_LO; lo=-y[i+1], up=Inf)
            # z_i + (A x)_i >= y_i
            a_row(m, 2i + 1, [0, 1, 2, 3 + i], [a1, a2, a3, 1.0])
            con_bounds(m, 2i + 1, BK_LO; lo=y[i+1], up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        v, z = r.x, r.objective
        ok = abs(v[1] - 3.489517) < 1e-4 && abs(v[2] + 2.174797) < 1e-4 &&
             abs(v[3] + 0.989834) < 1e-4 && abs(z - 3.2951) < 1e-4
        @printf("npo_lad  coeffs=(%.6g,%.6g,%.6g) sad=%.6g (book (3.489517,-2.174797,-0.989834), 3.2951) %s\n",
                v[1], v[2], v[3], z, ok ? "OK" : "FAIL")
    else
        @printf("npo_lad  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
