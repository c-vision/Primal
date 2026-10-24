#!/usr/bin/env julia
# index_tracking.jl — two-period index tracking with turnover.
#
#   min sum_t |w_t - b_t|_1 + c (|w_0 - w_init|_1 + |w_1 - w_0|_1)
#   s.t.  e'w_t = 1,  w_t >= 0
#
# with the absolute values linearised by z_t >= +-(w_t - b_t),
# v_0 >= +-(w_0 - w_init), v_1 >= +-(w_1 - w_0). Hand case: b_0 = (0.6, 0.4),
# b_1 = (0.3, 0.7), w_init = (0.5, 0.5), c = 0.1: w_t = b_t, value 0.08.
# Port of examples/c/index_tracking.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    b0, b1, winit = [0.6, 0.4], [0.3, 0.7], [0.5, 0.5]
    c = 0.1
    m = Model(0, 0)                                 # w0(0,1), w1(2,3), z0(4,5), z1(6,7), v0(8,9), v1(10,11)
    res = try
        add_vars(m, 12)
        add_cons(m, 18)
        for j in 0:11
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        sense(m, SENSE_MIN)
        for j in 0:1
            cj(m, 4 + j, 1.0)                       # z0
            cj(m, 6 + j, 1.0)                       # z1
            cj(m, 8 + j, c)                         # v0
            cj(m, 10 + j, c)                        # v1
        end
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)   # e'w_0 = 1
        a_row(m, 1, [2, 3], [1.0, 1.0]); con_bounds(m, 1, BK_FX; lo=1.0, up=1.0)   # e'w_1 = 1
        r = 2
        for j in 0:1                                # z0 >= +-(w0 - b0)
            a_row(m, r, [j, 4 + j], [1.0, -1.0]);  con_bounds(m, r, BK_UP; lo=-Inf, up=b0[j+1]); r += 1
            a_row(m, r, [j, 4 + j], [-1.0, -1.0]); con_bounds(m, r, BK_UP; lo=-Inf, up=-b0[j+1]); r += 1
        end
        for j in 0:1                                # z1 >= +-(w1 - b1)
            a_row(m, r, [2 + j, 6 + j], [1.0, -1.0]);  con_bounds(m, r, BK_UP; lo=-Inf, up=b1[j+1]); r += 1
            a_row(m, r, [2 + j, 6 + j], [-1.0, -1.0]); con_bounds(m, r, BK_UP; lo=-Inf, up=-b1[j+1]); r += 1
        end
        for j in 0:1                                # v0 >= +-(w0 - w_init)
            a_row(m, r, [j, 8 + j], [1.0, -1.0]);  con_bounds(m, r, BK_UP; lo=-Inf, up=winit[j+1]); r += 1
            a_row(m, r, [j, 8 + j], [-1.0, -1.0]); con_bounds(m, r, BK_UP; lo=-Inf, up=-winit[j+1]); r += 1
        end
        for j in 0:1                                # v1 >= +-(w1 - w0)
            a_row(m, r, [2 + j, j, 10 + j], [1.0, -1.0, -1.0]); con_bounds(m, r, BK_UP; lo=-Inf, up=0.0); r += 1
            a_row(m, r, [2 + j, j, 10 + j], [-1.0, 1.0, -1.0]); con_bounds(m, r, BK_UP; lo=-Inf, up=0.0); r += 1
        end
        solve(m)
    finally
        free(m)
    end
    if res.rc != 0
        @printf("optimize rc=%d\n", res.rc)
        return 1
    end
    obj, x = res.objective, res.x
    track0 = abs(x[1] - b0[1]) + abs(x[2] - b0[2])
    track1 = abs(x[3] - b1[1]) + abs(x[4] - b1[2])
    ok = abs(obj - 0.08) < 1e-7 && abs(x[1] - 0.6) < 1e-6 && abs(x[2] - 0.4) < 1e-6 &&
         abs(x[3] - 0.3) < 1e-6 && abs(x[4] - 0.7) < 1e-6 && abs(track0) < 1e-6 && abs(track1) < 1e-6
    @printf("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (expected 0.08)\n", x[1], x[2], x[3], x[4], obj)
    @printf("tracking = %.4f then %.4f (expected 0.0, 0.0)\n", track0, track1)
    @printf("%s (index tracking over two periods with turnover cost)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
