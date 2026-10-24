#!/usr/bin/env julia
# limit_analysis_dp.jl — Drucker-Prager yield cone of a rigid-plastic limit
# analysis, as a second-order cone program (a0082273/limit-analysis-with-
# node-based-element-and-MOSEK).
#
# yf(sigma) = sqrt(J2) + 3 alpha m - k <= 0 is the cone ||s|| <= sqrt(2)(k - alpha tr(sigma))
# with s the deviatoric 6-vector; alpha = sin(phi)/sqrt(9 + 3 sin^2 phi),
# k = sqrt(3) c cos(phi)/sqrt(3 + sin^2 phi). A proportional stress t sigma0
# is admissible up to t* = k/(sqrt(J2(sigma0)) + alpha tr(sigma0)).
# Case: uniaxial sigma0, c = 1, phi = 30 deg: t* = 1.128237.
# Port of examples/c/limit_analysis_dp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const PI = 3.14159265358979323846
csin(x) = ccall(:sin, Cdouble, (Cdouble,), x)      # the C library's sin/cos
ccos(x) = ccall(:cos, Cdouble, (Cdouble,), x)

function main()
    c, phi = 1.0, 30.0 * PI / 180.0                 # material
    alpha = csin(phi) / sqrt(9.0 + 3.0 * csin(phi) * csin(phi))
    kk = sqrt(3.0) * c * ccos(phi) / sqrt(3.0 + csin(phi) * csin(phi))
    s0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]             # proportional reference stress: uniaxial

    T, SIG, SDEV, RHS, nv = 0, 1, 7, 13, 14
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MAX)
        var_bounds(m, T, BK_FR; lo=-Inf, up=Inf)
        for i in 0:5
            var_bounds(m, SIG + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:5
            var_bounds(m, SDEV + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, RHS, BK_FR; lo=-Inf, up=Inf)
        add_cons(m, 6 + 6 + 1)
        row = 0
        for i in 0:5                                # sigma_i - s0_i t = 0
            a_row(m, row, [SIG + i, T], [1.0, -s0[i+1]])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        for i in 0:5                                # sdev_i - sigma_i + (sigma0+sigma1+sigma2)/3 = 0
            if i < 3
                sub = [SDEV + i, SIG + i, SIG + (i + 1) % 3, SIG + (i + 2) % 3]
                val = [1.0, -2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            else                                    # shear: sdev_i = sigma_i
                sub, val = [SDEV + i, SIG + i], [1.0, -1.0]
            end
            a_row(m, row, sub, val)
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        # RHS - sqrt2 k + sqrt2 alpha (sigma0+sigma1+sigma2) = 0
        a_row(m, row, [RHS, SIG + 0, SIG + 1, SIG + 2], [1.0, sqrt(2.0) * alpha, sqrt(2.0) * alpha, sqrt(2.0) * alpha])
        con_bounds(m, row, BK_FX; lo=sqrt(2.0) * kk, up=sqrt(2.0) * kk); row += 1
        cone(m, CT_QUAD, [RHS, SDEV + 0, SDEV + 1, SDEV + 2, SDEV + 3, SDEV + 4, SDEV + 5])   # RHS >= ||s||
        cj(m, T, 1.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    @printf(stderr, "rc=%d prosta=%d solsta=%d\n", r.rc, r.prosta, r.solsta)
    v = ok ? r.x : zeros(nv)
    tstar = v[T+1]

    # hand value and independent check of the cone at the returned stress
    want = kk / (1.0 / sqrt(3.0) + alpha * 1.0)
    s0v, s1v, s2v = v[SIG+1], v[SIG+2], v[SIG+3]
    mm = (s0v + s1v + s2v) / 3.0
    d = [s0v - mm, s1v - mm, s2v - mm, v[SIG+4], v[SIG+5], v[SIG+6]]
    sq = 0.0
    for i in 1:6
        sq += d[i] * d[i]
    end
    yf = sqrt(sq / 2.0) + 3.0 * alpha * mm - kk     # must be ~ 0 at yield
    good = ok && abs(tstar - want) < 1e-7 && abs(tstar - 1.128237) < 1e-5 && abs(yf) < 1e-7
    @printf("limit_analysis_dp  alpha=%.8f k=%.8f  t*=%.8f (expected %.8f)\n", alpha, kk, tstar, want)
    @printf("  sigma_x=%.6f  sqrt(J2)=%.6f  m=%.6f  yf=%.2e  %s\n", s0v, sqrt(sq / 2.0), mm, yf, good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
