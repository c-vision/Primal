#!/usr/bin/env julia
# omf_workforce.jl — the workforce planning LP of "Optimization Methods in
# Finance" (Cornuejols & Tutuncu, 2007), Exercise 3.15. A restaurant needs
# 14, 13, 15, 16, 19, 18, 11 workers Mon..Sun; each worker works five
# consecutive days. Shift i starts on day i:
#
#   min sum_i x_i  s.t.  for each day, the five shifts covering it >= demand,  x >= 0
#
# The book's sensitivity report: x = (4, 7, 1, 4, 3, 3, 0), 22 workers, shadow
# prices (1/3, 0, 1/3, 0, 1/3, 1/3, 0), reduced cost 1/3 for Shift 7.
# Port of examples/c/omf_workforce.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const D = 7
const DEMAND = [14.0, 13, 15, 16, 19, 18, 11]
const XBOOK = [4.0, 7, 1, 4, 3, 3, 0]
const SPBOOK = [0.333333, 0, 0.333333, 0, 0.333333, 0.333333, 0]

function main()
    m = Model(D, D)                                 # vars x1..x7 (shift Mon..Sun); rows Mon..Sun
    try
        for i in 0:D-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
            cj(m, i, 1.0)                           # min sum x_i
        end
        for d in 0:D-1                              # day d is worked by the shifts starting d, d-1, ..., d-4
            a_row(m, d, [mod(d - k, D) for k in 0:4], ones(5))
            con_bounds(m, d, BK_LO; lo=DEMAND[d+1], up=Inf)
        end
        r = solve(m)
        ok = r.rc == 0
        if ok
            obj, x, y = r.objective, r.x, r.y
            x_ok = all(i -> abs(x[i] - XBOOK[i]) <= 1e-6, 1:D)
            sp_ok = all(d -> abs(abs(y[d]) - SPBOOK[d]) <= 1e-5, 1:D)   # dual sign on a >= row is negative
            rcs = getreducedcosts(m, 0, D)          # Shift 7 is unused; its reduced cost is 1/3
            rc7_ok = abs(rcs[7] - 0.333333) < 1e-5
            ok = abs(obj - 22.0) < 1e-6 && x_ok && sp_ok && rc7_ok
            @printf("omf_workforce  workers=%.4f (book 22)  %s\n", obj, ok ? "OK" : "FAIL")
            print("  x: ")
            for i in 1:D
                @printf("S%d=%.4g ", i, x[i])
            end
            @printf("  (book 4,7,1,4,3,3,0)  %s\n", x_ok ? "OK" : "FAIL")
            print("  shadow: ")
            for d in 1:D
                @printf("%.6f ", abs(y[d]))
            end
            @printf("  %s\n", sp_ok ? "OK" : "FAIL")
            @printf("  reduced cost Shift7 = %.6f (book 0.333333)  %s\n", rcs[7], rc7_ok ? "OK" : "FAIL")
        else
            @printf("omf_workforce  rc=%d FAIL\n", r.rc)
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
