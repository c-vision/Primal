#!/usr/bin/env julia
# omf_ex411.jl — Exercise 4.11 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 4.3: arbitrage in the currency market
# with the exchange rates of February 14, 2002 (Dollar, Euro, Pound, Yen).
#
#   max D  s.t.  one balance row per currency over the 12 conversions,
#                0 <= variables,  D <= 10000
#
# The dollar -> euro -> yen -> dollar cycle makes ~$0.0003 per dollar
# (1.1486 * 116.12 * 0.00750 = 1.000316), the reverse loses ~$0.0002; the LP
# drives D to its cap. Several cycles are optimal (this solver finds the
# direct dollar -> yen -> dollar one). Port of examples/c/omf_ex411.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NV = 13
const DE, DP, DY, ED, EP, EY, PD, PE, PY, YD, YE, YP, D_ = 0:12

function main()
    c_dey = 1.1486 * 116.12 * 0.00750               # > 1 : arbitrage
    c_dye = 133.38 * 0.00861 * 0.8706               # < 1 : loss
    cycles_ok = c_dey > 1.0 && c_dye < 1.0 && abs((c_dey - 1.0) - 0.0003) < 5e-5 &&
                abs((1.0 - c_dye) - 0.0002) < 5e-5

    m = Model(4, NV)
    r = try
        for j in 0:NV-1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, D_, BK_UP; lo=0.0, up=10000.0)
        cj(m, D_, -1.0)                             # max D
        a_row(m, 0, [D_, DE, DP, DY, ED, PD, YD], [1.0, 1.0, 1.0, 1.0, -0.8706, -1.4279, -0.00750])   # Dollar
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [ED, EP, EY, DE, PE, YE], [1.0, 1.0, 1.0, -1.1486, -1.6401, -0.00861])            # Euro
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        a_row(m, 2, [PD, PE, PY, DP, EP, YP], [1.0, 1.0, 1.0, -0.7003, -0.6097, -0.00525])            # Pound
        con_bounds(m, 2, BK_FX; lo=0.0, up=0.0)
        a_row(m, 3, [YD, YE, YP, DY, EY, PY], [1.0, 1.0, 1.0, -133.38, -116.12, -190.45])             # Yen
        con_bounds(m, 3, BK_FX; lo=0.0, up=0.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        d = -z
        ok = cycles_ok && abs(d - 10000.0) < 1.0
        @printf("omf_ex411  arbitrage D=%.4f (cap 10000)  %s\n", d, ok ? "OK" : "FAIL")
        @printf("  cycle \$->E->Yen->\$ = %.6f (>1, +%.4f/dollar)  \$->Yen->E->\$ = %.6f (<1, %.4f/dollar)  %s\n",
                c_dey, c_dey - 1.0, c_dye, c_dye - 1.0, cycles_ok ? "OK" : "FAIL")
        @printf("  flows (one of several optimal cycles): DE=%.6g DP=%.4g DY=%.4g ED=%.4g YD=%.4g\n",
                x[DE+1], x[DP+1], x[DY+1], x[ED+1], x[YD+1])
    else
        @printf("omf_ex411  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
