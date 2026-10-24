#!/usr/bin/env julia
# boyd_detector.jl — minimax detector, "Convex Optimization" (Boyd &
# Vandenberghe, 2004), Example 7.4. Binary hypothesis testing with n = 4 and
#   P = [0.70 0.10; 0.20 0.10; 0.05 0.70; 0.05 0.10]
# (column j = distribution of the observation under hypothesis j+1). The
# minimax randomized detector T^(4) = [1 2/3 0 0; 0 1/3 1 1] yields
# P_fp = P_fn = 1/6.
#
# LP form: T0_i, T1_i >= 0 with T0_i + T1_i = 1, and t with
#   t >= sum_i P[i][0] T1_i (false negative), t >= sum_i P[i][1] T0_i (false positive).
# Port of examples/c/boyd_detector.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    P0 = [0.70, 0.20, 0.05, 0.05]                   # hyp 1 column
    P1 = [0.10, 0.10, 0.70, 0.10]                   # hyp 2 column
    m = Model(6, 9)                                 # 4 col-balance + 2 error rows; 8 T + t
    r = try
        for j in 0:7
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, 8, BK_FR; lo=-Inf, up=Inf)
        cj(m, 8, 1.0)                               # minimize t
        for i in 0:3                                # T0_i + T1_i = 1
            a_row(m, i, [i, 4 + i], [1.0, 1.0])
            con_bounds(m, i, BK_FX; lo=1.0, up=1.0)
        end
        # false negative: t - sum_i P0[i] T1_i >= 0
        a_row(m, 4, [4, 5, 6, 7, 8], [-P0[1], -P0[2], -P0[3], -P0[4], 1.0])
        con_bounds(m, 4, BK_LO; lo=0.0, up=Inf)
        # false positive: t - sum_i P1[i] T0_i >= 0
        a_row(m, 5, [0, 1, 2, 3, 8], [-P1[1], -P1[2], -P1[3], -P1[4], 1.0])
        con_bounds(m, 5, BK_LO; lo=0.0, up=Inf)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, v = r.objective, r.x
        fn, fp = 0.0, 0.0                           # recomputed error probabilities
        for i in 1:4
            fn += P0[i] * v[4+i]
            fp += P1[i] * v[i]
        end
        ok = abs(z - 1.0 / 6.0) < 1e-6 && abs(fn - 1.0 / 6.0) < 1e-6 && abs(fp - 1.0 / 6.0) < 1e-6
        @printf("boyd_detector  t=%.6g Pfp=%.6g Pfn=%.6g (book 1/6 all) %s\n", z, fp, fn, ok ? "OK" : "FAIL")
    else
        @printf("boyd_detector  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
