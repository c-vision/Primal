#!/usr/bin/env julia
# market_neutral.jl — market-neutral long-short portfolio with leverage,
# turnover and a variance cap (fy23-art/Portfolio-Optimization, MOSEK):
#
#   maximize m'x - cost_rate sum_i (d+_i + d-_i)
#   s.t.  factor'x = 0,  sum_i |x_i| <= leverage,  x'Sx <= gamma^2,  x - x0 = d+ - d-
#
# Two assets with factor = (1, 1), so x = (t, -t) and every value is
# hand-derived: A) no cost -> t = Tmax (risk/leverage bound binds);
# B) large cost -> t = t0 (no trade). A dense 1-D brute force confirms both.
# Port of examples/c/market_neutral.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NA = 2
const S = [0.04 0.01; 0.01 0.09]                    # covariance
const Mu = [0.002, 0.0]                             # alpha
const GAMMA2 = 0.01                                 # 0.1^2
const LEV = 2.0                                     # sum |x| <= L

# feasible bound on t from sum|x| <= L and x'Sx <= gamma^2 (x = (t, -t))
function tmax()
    q = S[1, 1] - 2.0 * S[1, 2] + S[2, 2]           # x'Sx = t^2 q
    tr, tg = LEV / 2.0, sqrt(GAMMA2 / q)
    return tr < tg ? tr : tg
end

# Conic model; returns (ok, objective, x).
function solve_mn(c, t0)
    XP, Z, DP, DM, NV = 0, NA, 2NA, 3NA, 4NA
    R_MN, R_ZP, R_ZN, R_LEV, R_TL, R_RSK = 0, 1, 1 + NA, 1 + 2NA, 2 + 2NA, 2 + 3NA
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, 3NA + 3)
        for i in 0:NA-1
            var_bounds(m, XP + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, Z + i, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, DP + i, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, DM + i, BK_LO; lo=0.0, up=Inf)
        end
        a_row(m, R_MN, [XP, XP + 1], [1.0, 1.0])   # factor'x = 0, factor = (1,1)
        con_bounds(m, R_MN, BK_FX; lo=0.0, up=0.0)
        for i in 0:NA-1                             # z_i >= x_i and z_i >= -x_i
            a_row(m, R_ZP + i, [Z + i, XP + i], [1.0, -1.0])
            con_bounds(m, R_ZP + i, BK_LO; lo=0.0, up=Inf)
            a_row(m, R_ZN + i, [Z + i, XP + i], [1.0, 1.0])
            con_bounds(m, R_ZN + i, BK_LO; lo=0.0, up=Inf)
        end
        a_row(m, R_LEV, [Z, Z + 1], [1.0, 1.0])
        con_bounds(m, R_LEV, BK_UP; lo=-Inf, up=LEV)
        for i in 0:NA-1                             # x_i - d+_i + d-_i = x0_i
            x0 = i == 0 ? t0 : -t0
            a_row(m, R_TL + i, [XP + i, DP + i, DM + i], [1.0, -1.0, 1.0])
            con_bounds(m, R_TL + i, BK_FX; lo=x0, up=x0)
        end
        # risk: 1/2 x'Qx <= gamma^2 with Q = 2S
        qconk(m, R_RSK, [(0, 0, 2.0 * S[1, 1]), (1, 1, 2.0 * S[2, 2]), (0, 1, 2.0 * S[1, 2])])
        con_bounds(m, R_RSK, BK_UP; lo=-Inf, up=GAMMA2)
        for i in 0:NA-1
            cj(m, XP + i, Mu[i+1])
            cj(m, DP + i, -c)
            cj(m, DM + i, -c)
        end
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, zeros(NA)
    x = r.x
    obj = Mu[1] * x[XP+1] + Mu[2] * x[XP+2] - c * (x[DP+1] + x[DP+2] + x[DM+1] + x[DM+2])
    return true, obj, x[XP+1:XP+2]
end

# true objective at x = (t, -t): a t - c sum_i |x_i - x0_i|
obj_at(c, t0, t) = (Mu[1] - Mu[2]) * t - c * (abs(t - t0) + abs(-t + t0))

function main()
    T, a = tmax(), Mu[1] - Mu[2]
    allok = true
    @printf("market_neutral  N=%d  Tmax=%.8f  a=m1-m2=%.6f\n", NA, T, a)
    cs, t0s = [0.0, 0.002], [0.1, 0.1]
    for k in 1:2
        c, t0 = cs[k], t0s[k]
        ok, obj, x = solve_mn(c, t0)
        # hand value: t* = Tmax if a > 2c, -Tmax if a < -2c, else clamp(t0)
        tstar = a > 2.0 * c ? T : a < -2.0 * c ? -T : (t0 < -T ? -T : (t0 > T ? T : t0))
        ref = obj_at(c, t0, tstar)
        best = -1e300                               # dense 1-D brute force
        for j in 0:200000
            f = obj_at(c, t0, -T + 2.0 * T * j / 200000.0)
            f > best && (best = f)
        end
        okk = ok && abs(obj - ref) < 1e-6 && abs(obj - best) < 1e-6 && abs(x[1] + x[2]) < 1e-7 && abs(x[1] - tstar) < 1e-5
        allok = allok && okk
        @printf("  case %c: c=%.4f t0=%.2f  x=(%.8f, %.8f)  obj=%.10f  [closed-form %.10f]\n",
                'A' + (k - 1), c, t0, x[1], x[2], obj, ref)
    end
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
