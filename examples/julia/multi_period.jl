#!/usr/bin/env julia
# multi_period.jl — two-period portfolio with transaction costs.
#
#   max sum_{t=0,1} mu_t'w_t - c (|w_0 - w_init|_1 + |w_1 - w_0|_1)
#   s.t.  e'w_t = 1,  w_t >= 0
#
# with the absolute values linearised by z_t >= +-(w_t - w_{t-1}), z_t >= 0.
# Hand case: mu = (0.5, 0.1), w_init = (0.5, 0.5), c = 0.1: move to (1, 0) in
# the first period (cost 0.1, gain 0.2), stay there: value 0.9.
# Port of examples/c/multi_period.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    mu = [0.5, 0.1]
    w0init = [0.5, 0.5]
    cost = 0.1
    m = Model(0, 0)
    r = try
        add_vars(m, 8)                              # w0(0,1), w1(2,3), z0(4,5), z1(6,7)
        add_cons(m, 10)
        for j in 0:7
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        sense(m, SENSE_MAX)
        for j in 0:1
            cj(m, j, mu[j+1])
            cj(m, 2 + j, mu[j+1])
            cj(m, 4 + j, -cost)
            cj(m, 6 + j, -cost)
        end
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)   # e'w_0 = 1
        a_row(m, 1, [2, 3], [1.0, 1.0]); con_bounds(m, 1, BK_FX; lo=1.0, up=1.0)   # e'w_1 = 1
        for j in 0:1                                # z0 >= +-(w0 - w_init)
            a_row(m, 2 + j, [j, 4 + j], [1.0, -1.0])
            con_bounds(m, 2 + j, BK_UP; lo=-Inf, up=w0init[j+1])
            a_row(m, 4 + j, [j, 4 + j], [-1.0, -1.0])
            con_bounds(m, 4 + j, BK_UP; lo=-Inf, up=-w0init[j+1])
        end
        for j in 0:1                                # z1 >= +-(w1 - w0)
            a_row(m, 6 + j, [2 + j, j, 6 + j], [1.0, -1.0, -1.0])
            con_bounds(m, 6 + j, BK_UP; lo=-Inf, up=0.0)
            a_row(m, 8 + j, [2 + j, j, 6 + j], [-1.0, 1.0, -1.0])
            con_bounds(m, 8 + j, BK_UP; lo=-Inf, up=0.0)
        end
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    obj, x = r.objective, r.x
    turnover0 = abs(x[1] - w0init[1]) + abs(x[2] - w0init[2])
    turnover1 = abs(x[3] - x[1]) + abs(x[4] - x[2])
    ok = abs(obj - 0.9) < 1e-7 && abs(x[1] - 1.0) < 1e-6 && abs(x[2]) < 1e-6 &&
         abs(x[3] - 1.0) < 1e-6 && abs(x[4]) < 1e-6 && abs(turnover0 - 1.0) < 1e-6 && abs(turnover1) < 1e-6
    @printf("w0 = (%.4f, %.4f), w1 = (%.4f, %.4f), obj = %.6f (expected 0.9)\n", x[1], x[2], x[3], x[4], obj)
    @printf("turnover = %.4f then %.4f (expected 1.0, 0.0)\n", turnover0, turnover1)
    @printf("%s (multi-period: the second period is already on target)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
