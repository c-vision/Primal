#!/usr/bin/env julia
# f_sparc.jl — subcarrier and power allocation as a mixed-integer exp-cone
# program (MOSEK Tutorials, f-sparc):
#
#   maximize sum z_ij
#   s.t.  sum_j x_ij <= 1,  p_ij <= x_ij,  x binary,  SIGMA t + sum p = 1,
#         t in [1/P, 1/SIGMA],  sum_i z_ij >= t d_j,
#         (t + p/n, t, z log(2)/BW) in K_exp  (t + p/n >= t 2^(z/(BW t)))
#
# Instance I = J = n = 1, d = 0.2 (BW = 1.25, P = 36, SIGMA = 10): checked
# against a dense scan of t (optimum z ~ 0.2206 at t ~ 0.058).
# Port of examples/c/f_sparc.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const SIGMA = 10.0
const PMAX = 36.0
const BW = 1.25
const NOISE = 1.0
const DEMAND = 0.2

clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/log2
clog2(x) = ccall(:log2, Cdouble, (Cdouble,), x)

function main()
    Z, X, PP, TT, TP, ZC, NV = 0, 1, 2, 3, 4, 5, 6
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, 5)
        var_bounds(m, Z, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, X, BK_RA; lo=0.0, up=1.0)
        var_type(m, X, VT_INT_BIN)
        var_bounds(m, PP, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, TT, BK_RA; lo=1.0 / PMAX, up=1.0 / SIGMA)
        var_bounds(m, TP, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, ZC, BK_FR; lo=-Inf, up=Inf)
        a_row(m, 0, [TT, PP], [SIGMA, 1.0]);                 con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)    # SIGMA t + p = 1
        a_row(m, 1, [PP, X], [1.0, -1.0]);                   con_bounds(m, 1, BK_UP; lo=-Inf, up=0.0)   # p - x <= 0
        a_row(m, 2, [TT, Z], [DEMAND, -1.0]);                con_bounds(m, 2, BK_UP; lo=-Inf, up=0.0)   # t d - z <= 0
        a_row(m, 3, [TP, PP, TT], [1.0, -1.0 / NOISE, -1.0]); con_bounds(m, 3, BK_FX; lo=0.0, up=0.0)  # tp - p/n - t = 0
        a_row(m, 4, [ZC, Z], [1.0, -clog(2.0) / BW]);        con_bounds(m, 4, BK_FX; lo=0.0, up=0.0)    # zc - z log2/BW = 0
        cone(m, CT_PEXP, [TP, TT, ZC])
        cj(m, Z, 1.0)
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    z, tv = 0.0, 0.0
    if ok
        z, tv = r.x[Z+1], r.x[TT+1]
    end

    best, bt = -1e300, 0.0                          # reference: dense scan of t (p = 1 - SIGMA t, x = 1)
    for q in 0:400000
        tt = 1.0 / PMAX + (1.0 / SIGMA - 1.0 / PMAX) * q / 400000.0
        p = 1.0 - SIGMA * tt
        p < 0.0 && break
        zz = BW * tt * clog2(1.0 + p / (NOISE * tt))
        zz < DEMAND * tt && continue                # demand constraint
        zz > best && ((best, bt) = (zz, tt))
    end
    okall = ok && abs(z - best) < 1e-4 * (1.0 + abs(best))
    @printf("f_sparc  I=1 J=1  n=%.1f d=%.1f\n", NOISE, DEMAND)
    @printf("  z* = %.8f  t = %.6f   [scan %.8f at t=%.6f]\n", z, tv, best, bt)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
