#!/usr/bin/env julia
# portfolio_1.jl — MOSEK port "portfolio_1_basic": Markowitz portfolio
# optimization, basic version.
#
#   max r'x - gamma x'Sigma x  s.t.  sum(x) = 1,  x >= 0
#
# 3 assets from the MOSEK tutorial, Sigma = GT GT' with a lower-triangular
# factor, gamma = 0.05. Checks: feasibility, a consistent objective and a zero
# duality gap. Port of examples/c/portfolio_1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    n = 3
    r = [0.10717, 0.07502, 0.11902]
    GT = [0.1667, 0.0, 0.0, 0.0247, 0.1592, 0.0, 0.0197, 0.0158, 0.1327]
    gamma = 0.05
    S = zeros(9)                                    # Sigma = GT * GT'
    for i in 0:2, j in 0:2
        S[i*3+j+1] = 0.0
        for k in 0:2
            S[i*3+j+1] += GT[i*3+k+1] * GT[j*3+k+1]
        end
    end
    m = Model(0, 0)
    res = try
        add_vars(m, n)
        add_cons(m, 1)
        sense(m, SENSE_MAX)
        for j in 0:n-1
            cj(m, j, r[j+1])
        end
        # obj = r'x - gamma x'Sx = 0.5 x'Qx with Q = -2 gamma S (upper triangle, full coefficient)
        qobj(m, [(0, 0, -2.0 * gamma * S[1]), (0, 1, -2.0 * gamma * S[2]), (0, 2, -2.0 * gamma * S[3]),
                 (1, 1, -2.0 * gamma * S[5]), (1, 2, -2.0 * gamma * S[6]), (2, 2, -2.0 * gamma * S[9])])
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # sum(x) = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for j in 0:n-1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    if res.rc != 0
        @printf("optimize rc=%d\n", res.rc)
        return 1
    end
    xx, po, dobj = res.x, res.objective, res.dual
    @printf("x = (%.6f, %.6f, %.6f)\n", xx[1], xx[2], xx[3])
    @printf("pobj = %.6f, dobj = %.6f\n", po, dobj)
    ok = abs(xx[1] + xx[2] + xx[3] - 1.0) < 1e-6   # feasible + consistent pobj + no gap
    any(v -> v < -1e-9, xx) && (ok = false)
    q = 0.0
    for i in 0:2, j in 0:2
        q += xx[i+1] * S[i*3+j+1] * xx[j+1]
    end
    lin = r[1] * xx[1] + r[2] * xx[2] + r[3] * xx[3]
    expect = lin - gamma * q
    @printf("obj check: %.6f (expected %.6f)\n", po, expect)
    abs(po - expect) > 1e-6 && (ok = false)
    abs(po - dobj) > 1e-5 * (1 + abs(po)) && (ok = false)
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
