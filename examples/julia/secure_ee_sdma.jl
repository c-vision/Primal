#!/usr/bin/env julia
# secure_ee_sdma.jl — secure energy efficiency of a downlink SDMA beamformer
# (rezarhp/Secure-EE-SDMA, Hashempour et al., arXiv:2603.21889, "Part B"):
# an outer Dinkelbach loop on SEE = zeta/(P0 + ||p||^2) and an inner SCA loop
# on the two non-convex SINR ratios, with the surrogates of sca_funcs.py, the
# exact cones |h_e'p|^2 <= sigma^2 rho_e (RQUAD) and 1 + rho_u >= 2^f_p (PEXP),
# the power cone and the harvesting constraint |h_e'p|^2 >= E_req.
#
# Instance: one user, one eavesdropper/harvester, two real antennas; the cone
# solution is validated against a dense brute force over (p0, p1) of the true
# objective. Port of examples/c/secure_ee_sdma.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NT = 2
const HU = [1.0, 0.5]                               # legitimate-user channel
const HE = [0.3, 1.0]                               # eavesdropper/harvester channel
const SIGMA2 = 1.0
const PMAX = 2.0
const PCIRC = 1.0                                   # circuit power P0
const EH = 0.01                                     # harvested DC energy [W]
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp/pow
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)
cpow(x, y) = ccall(:pow, Cdouble, (Cdouble, Cdouble), x, y)

# sai_inverse(): the repository's inverse of the non-linear harvesting map
function sai_inverse(x)
    b0, b1, phi, k1 = 150.0, 0.014, 0.024, 1.0
    k2 = phi / (k1 * (1.0 + cexp(b0 * b1)))
    return b1 - clog(phi / (k1 * (x + k2)) - 1.0) / b0
end

dot2(a, b) = a[1] * b[1] + a[2] * b[2]

# true secrecy rate and EE at a precoder p
function evaluate(p)
    hu, he = dot2(HU, p), dot2(HE, p)
    gu, ge = hu * hu / SIGMA2, he * he / SIGMA2
    r = clog(1.0 + gu) / clog(2.0) - clog(1.0 + ge) / clog(2.0)
    return r, r / (PCIRC + dot2(p, p))
end

const PV0, PV1, Y, Z, FP, FE, RU, RE, OPRU, LFP, ZE, PW, HALF, ONE, TPS, HALFSIG, NVAR = 0:16
const R_ZE, R_OBJ, R_SEC, R_EXP, R_GAM, R_USR, R_HARV, R_LFP, NR = 0:8

# One SCA step: the SOCP built around the previous point; returns (ok, p, zeta).
function solve_sca(p_old, rho_u_old, f_e_old, eta)
    m = Model(NR, NVAR)
    r = try
        for j in 0:NVAR-1
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, Z, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, FP, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, FE, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, RU, BK_LO; lo=1e-9, up=Inf)
        var_bounds(m, RE, BK_LO; lo=1e-9, up=Inf)
        var_bounds(m, PW, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, HALF, BK_FX; lo=0.5, up=0.5)
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        var_bounds(m, TPS, BK_FX; lo=sqrt(PMAX), up=sqrt(PMAX))
        var_bounds(m, HALFSIG, BK_FX; lo=0.5 * SIGMA2, up=0.5 * SIGMA2)
        hu_old, he_old = dot2(HU, p_old), dot2(HE, p_old)
        A, LN2 = cpow(2.0, f_e_old), clog(2.0)
        a_row(m, R_ZE, [ZE, PV0, PV1], [1.0, -HE[1], -HE[2]])            # z_e - h_e'p = 0
        con_bounds(m, R_ZE, BK_FX; lo=0.0, up=0.0)
        a_row(m, R_OBJ, [Y, Z, PW], [1.0, -1.0, eta])                    # y - zeta + eta pw <= -eta P0
        con_bounds(m, R_OBJ, BK_UP; lo=-Inf, up=-eta * PCIRC)
        a_row(m, R_SEC, [Z, FP, FE], [1.0, -1.0, 1.0])                   # zeta - f_p + f_e <= 0
        con_bounds(m, R_SEC, BK_UP; lo=-Inf, up=0.0)
        a_row(m, R_EXP, [OPRU, RU], [1.0, -1.0])                         # onepru - rho_u = 1
        con_bounds(m, R_EXP, BK_FX; lo=1.0, up=1.0)
        a_row(m, R_GAM, [RE, FE], [1.0, -A * LN2])                       # 1 + rho_e <= A(1 + ln2 (f_e - f_e_old))
        con_bounds(m, R_GAM, BK_UP; lo=-Inf, up=A - 1.0 - A * LN2 * f_e_old)
        a_row(m, R_USR, [PV0, PV1, RU],                                  # user SCA surrogate >= sigma^2
              [2.0 * hu_old * HU[1] / rho_u_old, 2.0 * hu_old * HU[2] / rho_u_old,
               -hu_old * hu_old / (rho_u_old * rho_u_old)])
        con_bounds(m, R_USR, BK_LO; lo=SIGMA2, up=Inf)
        a_row(m, R_HARV, [PV0, PV1], [2.0 * he_old * HE[1], 2.0 * he_old * HE[2]])   # harvesting SCA
        con_bounds(m, R_HARV, BK_LO; lo=sai_inverse(EH) + he_old * he_old, up=Inf)
        a_row(m, R_LFP, [LFP, FP], [1.0, -LN2])                          # lfp - ln2 f_p = 0
        con_bounds(m, R_LFP, BK_FX; lo=0.0, up=0.0)
        cone(m, CT_PEXP, [OPRU, ONE, LFP])
        cone(m, CT_RQUAD, [RE, HALFSIG, ZE])
        cone(m, CT_RQUAD, [PW, HALF, PV0, PV1])
        cone(m, CT_QUAD, [TPS, PV0, PV1])
        cj(m, Y, 1.0)
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, zeros(2), 0.0
    return true, [r.x[PV0+1], r.x[PV1+1]], r.x[Z+1]
end

function main()
    ereq = sai_inverse(EH)
    nrm = sqrt(dot2(HU, HU))                        # MRT initialization toward the user
    p = [sqrt(PMAX) * HU[1] / nrm, sqrt(PMAX) * HU[2] / nrm]
    eta, ee_actual, rate_actual = 0.0, 0.0, 0.0     # outer Dinkelbach, inner SCA
    outer, inner = 0, 0
    while outer < 50
        y_prev = -Inf
        inner = 0
        while inner < 50
            hu, he = dot2(HU, p), dot2(HE, p)
            rho_u_old = hu * hu / SIGMA2
            f_e_old = clog(1.0 + he * he / SIGMA2) / clog(2.0)
            ok, pn, z = solve_sca(p, rho_u_old, f_e_old, eta)
            if !ok
                println("secure_ee_sdma  SCA subproblem rc!=OK FAIL")
                return 1
            end
            p = pn
            y = z - eta * (PCIRC + dot2(p, p))
            abs(y - y_prev) < 1e-9 && break
            y_prev = y
            inner += 1
        end
        rate_actual, ee_actual = evaluate(p)
        if abs(ee_actual - eta) < 1e-9
            eta = ee_actual
            break
        end
        eta = ee_actual
        outer += 1
    end
    power = dot2(p, p)
    harvest = dot2(HE, p) * dot2(HE, p)

    best_ee, best_p0, best_p1 = -1e300, 0.0, 0.0    # dense brute force of the TRUE objective
    span = sqrt(PMAX)
    NG = 1200
    for a in 0:NG
        p0 = -span + 2.0 * span * a / NG
        for b in 0:NG
            p1 = -span + 2.0 * span * b / NG
            q = (p0, p1)
            dot2(q, q) > PMAX && continue
            he = dot2(HE, q)
            he * he < ereq && continue
            _, ee = evaluate(q)
            ee > best_ee && ((best_ee, best_p0, best_p1) = (ee, p0, p1))
        end
    end
    ok = abs(ee_actual - best_ee) < 1e-3 * (1.0 + abs(best_ee)) && power <= PMAX + 1e-7 && harvest >= ereq - 1e-7
    println("secure_ee_sdma  K=1 J=1 Nt=2")
    @printf("  E_req = sai_inverse(%.3f) = %.6f\n", EH, ereq)
    @printf("  p = (%.6f, %.6f)  ||p||^2 = %.6f <= %.3f\n", p[1], p[2], power, PMAX)
    @printf("  secrecy rate = %.6f bit/s/Hz   harvested = %.6f >= %.6f\n", rate_actual, harvest, ereq)
    @printf("  SEE = %.6f bit/J/Hz   [brute force %.6f at (%.4f,%.4f)]\n", ee_actual, best_ee, best_p0, best_p1)
    @printf("  outer=%d inner=%d\n", outer + 1, inner + 1)
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
