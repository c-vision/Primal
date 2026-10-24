#!/usr/bin/env julia
# cardinality_portfolio.jl — minimum variance with a cardinality constraint:
#
#   min w'Sigma w  s.t.  e'w = 1,  w >= 0,  sum_j z_j <= K,  w_j <= z_j,  z binary
#
# A mixed-integer QP: z_j says whether asset j is used, at most K may be.
# Hand case: Sigma = I, N = 3, K = 2. Unconstrained the minimum is 1/3 at
# (1/3, 1/3, 1/3); with K = 2 it is 1/2 at (1/2, 1/2, 0) (any pair).
# Port of examples/c/cardinality_portfolio.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    m = Model(0, 0)
    r = try
        add_vars(m, 6)                              # 0..2 w, 3..5 z
        add_cons(m, 5)
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=1.0)
            var_bounds(m, 3 + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, 3 + j, VT_INT_BIN)
            a_row(m, j, [j, 3 + j], [1.0, -1.0])   # w_j - z_j <= 0
            con_bounds(m, j, BK_UP; lo=-Inf, up=0.0)
        end
        a_row(m, 3, [0, 1, 2], [1.0, 1.0, 1.0])    # e'w = 1
        con_bounds(m, 3, BK_FX; lo=1.0, up=1.0)
        a_row(m, 4, [3, 4, 5], [1.0, 1.0, 1.0])    # sum z <= 2
        con_bounds(m, 4, BK_UP; lo=-Inf, up=2.0)
        for j in 0:2                                # 1/2 w'Qw with Q = 2I -> w0^2+w1^2+w2^2
            qobj(m, [(j, j, 2.0)])
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
    used = count(j -> x[j] > 1e-6, 1:3)
    quad = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    ok = abs(obj - 0.5) < 1e-6 && used <= 2 && abs(x[1] + x[2] + x[3] - 1.0) < 1e-6 && abs(quad - obj) < 1e-6
    @printf("w = (%.4f, %.4f, %.4f), active = %d, obj = %.6f (expected 0.5)\n", x[1], x[2], x[3], used, obj)
    @printf("%s (MIQP: cardinality K=2 over 3 active)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
