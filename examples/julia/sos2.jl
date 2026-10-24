#!/usr/bin/env julia
# sos2.jl — port of the "sos2" example: SOS type-2 constraint (at most two
# nonzero members, ADJACENT in the weight order — the basis of piecewise-linear
# approximation).
#
# f(t) on breakpoints t = (0, 1, 2) with values (0, 1, 0.5): with t = sum w_i t_i,
# y = sum w_i f_i, sum w = 1, w >= 0 and SOS2{w0, w1, w2}, fixing t = 1.5 must
# give y = 1 + 0.5 (1.5 - 1) = 0.75. Port of examples/c/sos2.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 4)                              # w0, w1, w2, y
        add_cons(m, 3)
        for j in 0:3
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, 3, BK_FR; lo=-Inf, up=Inf)    # y free
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # w0 + w1 + w2 = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [0, 1, 2], [0.0, 1.0, 2.0])    # 0 w0 + 1 w1 + 2 w2 = 1.5 (t = 1.5)
        con_bounds(m, 1, BK_FX; lo=1.5, up=1.5)
        a_row(m, 2, [0, 1, 2, 3], [0.0, 1.0, 0.5, -1.0])   # y = 0 w0 + 1 w1 + 0.5 w2
        con_bounds(m, 2, BK_FX; lo=0.0, up=0.0)
        sos(m, 2, [0, 1, 2]; weights=[0.0, 1.0, 2.0])      # SOS2 on the weights, breakpoint order
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    xx = r.x
    @printf("w = (%.4f, %.4f, %.4f), y = %.4f (expected 0.75)\n", xx[1], xx[2], xx[3], xx[4])
    ok = abs(xx[4] - 0.75) < 1e-6 && abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-6 &&
         abs(xx[2] + 2.0 * xx[3] - 1.5) < 1e-6
    xx[1] > 1e-6 && xx[3] > 1e-6 && (ok = false)   # SOS2: w0 and w2 not both positive
    @printf("%s (piecewise-linear SOS2: y(1.5)=0.75, adjacency respected)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
