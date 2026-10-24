#!/usr/bin/env julia
# sharpe_ratio_sectors.jl — max-Sharpe portfolio, long-only, with a sector
# diversification cap (matimacazaga/portfolio_optimization_with_mosek). As a
# minimum-norm problem with the normalisation mu'y = 1 (rf = 0):
#
#   minimize s  s.t.  s >= ||G y||,  mu'y = 1,  e'y = z,  y, z >= 0,  sector caps
#
# and w = y/z, Sharpe = 1/s. Case: Sigma = I, mu = (3,1,3,1), sectors {0,1},
# {2,3}. Cap 0.5 does not bind (Sharpe = sqrt(20)); cap 0.4 binds and the
# Sharpe drops. Port of examples/c/sharpe_ratio_sectors.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NA = 4

# Solve with sector-{0,1} cap; returns (ok, Sharpe or -1, w).
function solve_case(cap)
    mu = [3.0, 1.0, 3.0, 1.0]
    Y0, Z, S = 0, NA, NA + 1                        # y(4), z, s
    m = Model(0, 0)
    r = try
        add_vars(m, S + 1)
        sense(m, SENSE_MIN)
        for i in 0:NA-1
            var_bounds(m, Y0 + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, Z, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, S, BK_FR; lo=-Inf, up=Inf)
        add_cons(m, 3)                              # mu'y = 1 ; e'y - z = 0 ; cap
        a_row(m, 0, [Y0, Y0 + 1, Y0 + 2, Y0 + 3], mu)
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [Y0, Y0 + 1, Y0 + 2, Y0 + 3, Z], [1, 1, 1, 1, -1])
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        a_row(m, 2, [Y0, Y0 + 1, Z], [1.0, 1.0, -cap])   # y0 + y1 - cap z <= 0
        con_bounds(m, 2, BK_UP; lo=-Inf, up=0.0)
        cone(m, CT_QUAD, [S, Y0, Y0 + 1, Y0 + 2, Y0 + 3])  # s >= ||y|| (G = I)
        cj(m, S, 1.0)
        solve(m)
    finally
        free(m)
    end
    w = zeros(NA)
    obj = -1.0
    if r.rc == 0
        z = r.x[Z+1]
        w = [z > 0 ? r.x[Y0+i+1] / z : 0.0 for i in 0:NA-1]
        obj = r.x[S+1]                              # ||y|| = 1/Sharpe
    end
    return r.rc == 0, (obj > 0 ? 1.0 / obj : -1.0), w
end

function main()
    println("sharpe_ratio_sectors (4 assets, 2 sectors, Sigma=I)")
    allok = true
    ok, sh, w = solve_case(0.5)                     # case A: the cap does not bind
    sw = w[1] + w[2] + w[3] + w[4]
    good = ok && abs(sh - sqrt(20.0)) < 1e-4 && abs(w[1] + w[2] - 0.5) < 1e-4 &&
           abs(w[1] - 0.375) < 1e-4 && abs(w[2] - 0.125) < 1e-4 &&
           abs(w[3] - 0.375) < 1e-4 && abs(w[4] - 0.125) < 1e-4 && abs(sw - 1.0) < 1e-6
    @printf("  A cap=0.50: Sharpe=%.6f (expected %.6f)  w=(%.4f,%.4f,%.4f,%.4f)  %s\n",
            sh, sqrt(20.0), w[1], w[2], w[3], w[4], good ? "OK" : "FAIL")
    allok &= good

    ok, sh2, w = solve_case(0.4)                    # case B: the cap binds
    s, s1 = w[1] + w[2] + w[3] + w[4], w[1] + w[2]
    good = ok && sh2 > 0 && sh2 < sqrt(20.0) - 1e-6 && all(v -> v >= -1e-9, w) &&
           abs(s - 1.0) < 1e-6 && s1 <= 0.4 + 1e-6
    @printf("  B cap=0.40: Sharpe=%.6f (< %.6f)  w=(%.4f,%.4f,%.4f,%.4f) sum=%.6f s1=%.6f  %s\n",
            sh2, sqrt(20.0), w[1], w[2], w[3], w[4], s, s1, good ? "OK" : "FAIL")
    allok &= good

    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
