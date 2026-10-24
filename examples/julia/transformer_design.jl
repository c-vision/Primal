#!/usr/bin/env julia
# transformer_design.jl — high-frequency transformer, minimum-loss design, as
# a geometric program (MOSEK Tutorials, transformer-design). Under y = log x
# each posynomial constraint becomes one exponential cone per monomial plus a
# linear row (a single monomial is linear); the posynomial objective adds -t.
# The notebook's 15-variable, 28-constraint HF ferrite model is ported; the
# solution is checked by GP primal feasibility (every posynomial <= 1), not
# against the notebook's optimum (declared deviation).
# Port of examples/c/transformer_design.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const VC, VT, VBW, VHW, VNP, VNS, VNLS, VB, VJ, VPC, VPCU, VVRR, VVRX, VIM, VHB, NVX = 0:15
const NGP = NVX
const MCON = 28
const M_PI = 3.14159265358979323846
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp/pow
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)
cpow(x, y) = ccall(:pow, Cdouble, (Cdouble, Cdouble), x, y)

# one monomial: coefficient and (variable, exponent) pairs
mono(c, pairs...) = (c, [(Int(pairs[2i-1]), Float64(pairs[2i])) for i in 1:length(pairs)÷2])

function main()
    c0, t0, bw0, hw0 = 0.005, 0.015, 0.010, 0.025   # constants
    S, Ep, Es, f, pf = 1200.0, 300.0, 75.0, 100e3, 0.80
    rf = sqrt(1.0 - pf * pf)
    VRm, k_phi, dT = 0.03, 0.02, 65.0
    Ip = S / Ep
    m_layers, d_foil, kCu = 2.0, 0.238e-3, 0.6
    Kc, alpha, beta, Bsat, rho_c = 1.9e-3, 1.24, 2.0, 0.4, 4800.0
    rho_w = 2.26e-8
    mu0 = 4e-7 * M_PI
    mu = mu0 * 2000.0
    kf, ka, Fr, Fl = 0.95, 8.0 * cpow(100.0, 1.75), 1.32, 0.9
    PI = M_PI
    c33 = cpow(2.0, 1.5) * PI * f * kf / Ep         # derived coefficients
    c34 = Es / Ep
    c35 = 2.0 * Ip / kCu
    c36 = (1.0 / Fl) * (1.0 / d_foil) * Ip
    denomT = 1.42 * sqrt(2.0) * ka * cpow(dT, 1.25)
    Ccore = kf * Kc * rho_c * cpow(f, alpha)
    Ccu = kCu * Fr * rho_w
    CIm = sqrt(2.0) / mu
    Cnl = (1.0 / (k_phi * k_phi)) * (1.0 / (Ip * Ip))
    Cr = Fr * rho_w * pf
    inv_p2 = (m_layers * m_layers) / 4.0
    Cx = Ip * rf * f * mu0 * inv_p2
    coeff_vr = (1.0 / Ep) * (1.0 / VRm)

    P = [                                           # the 28 constraint posynomials
        [mono(1.0 / c0, VC, 1)], [mono(c0, VC, -1)],             # fix_var c, t, bw, hw
        [mono(1.0 / t0, VT, 1)], [mono(t0, VT, -1)],
        [mono(1.0 / bw0, VBW, 1)], [mono(bw0, VBW, -1)],
        [mono(1.0 / hw0, VHW, 1)], [mono(hw0, VHW, -1)],
        [mono(c33, VNP, 1, VB, 1, VC, 1, VT, 1)], [mono(1.0 / c33, VNP, -1, VB, -1, VC, -1, VT, -1)],   # (33)-(36)
        [mono(c34, VNP, 1, VNS, -1)], [mono(1.0 / c34, VNP, -1, VNS, 1)],
        [mono(c35, VNP, 1, VJ, -1, VBW, -1, VHW, -1)], [mono(1.0 / c35, VNP, -1, VJ, 1, VBW, 1, VHW, 1)],
        [mono(c36, VJ, -1, VNP, 1, VNS, -1, VNLS, 1, VHW, -1)], [mono(1.0 / c36, VJ, 1, VNP, -1, VNS, 1, VNLS, -1, VHW, 1)],
        [mono(1.0 / denomT, VPC, 1, VHB, 0.25, VC, -0.5, VT, -0.5, VBW, -0.5, VHW, -0.5),      # (37) temperature rise
         mono(1.0 / denomT, VPCU, 1, VHB, 0.25, VC, -0.5, VT, -0.5, VBW, -0.5, VHW, -0.5)],
        [mono(1.0, VHW, 1, VHB, -1), mono(2.0, VC, 1, VHB, -1)],                                # (38) hw + 2c <= hbar
        [mono(4.0 * Ccore, VB, beta, VC, 1, VT, 1, VHW, 1, VPC, -1),                           # (39) core loss
         mono(8.0 * Ccore, VB, beta, VC, 2, VT, 1, VPC, -1),
         mono(4.0 * Ccore, VB, beta, VC, 1, VT, 1, VBW, 1, VPC, -1)],
        [mono(4.0 * Ccu, VC, 1, VBW, 1, VHW, 1, VJ, 2, VPCU, -1),                              # (40) copper loss
         mono(2.0 * Ccu, VT, 1, VBW, 1, VHW, 1, VJ, 2, VPCU, -1),
         mono(PI * Ccu, VBW, 2, VHW, 1, VJ, 2, VPCU, -1)],
        [mono(CIm, VB, 1, VNP, -1, VHW, 1, VIM, -1),                                            # (42) magnetizing current
         mono(CIm, VB, 1, VNP, -1, VBW, 1, VIM, -1),
         mono(2.0 * CIm, VB, 1, VNP, -1, VC, 1, VIM, -1)],
        [mono(1.0 / Bsat, VB, 1)],                                                              # (43) saturation
        [mono(Cnl, VIM, 2), mono(Cnl / (Ep * Ep), VPC, 2)],                                     # (44) no-load current
        [mono(8.0 * Cr, VNP, 1, VC, 1, VJ, 1, VVRR, -1),                                        # (45) resistive regulation
         mono(4.0 * Cr, VNP, 1, VT, 1, VJ, 1, VVRR, -1),
         mono(2.0 * PI * Cr, VNP, 1, VBW, 1, VJ, 1, VVRR, -1)],
        [mono((8.0 * PI / 3.0) * Cx, VNP, 2, VC, 1, VBW, 1, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2),   # (46) reactive
         mono((4.0 * PI / 3.0) * Cx, VNP, 2, VT, 1, VBW, 1, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2),
         mono((2.0 * PI * PI / 3.0) * Cx, VNP, 2, VBW, 2, VHW, -1, VVRX, -1, VNLS, 2, VNS, -2)],
        [mono(1.0, VNLS, -1)], [mono(m_layers, VNLS, 1, VNS, -1)],                              # (47)
        [mono(coeff_vr, VVRR, 1), mono(coeff_vr, VVRX, 1)]]                                     # (48) total regulation
    if length(P) != MCON
        @printf("transformer_design  slot count %d != %d FAIL\n", length(P), MCON)
        return 1
    end
    OBJ = [mono(1.0, VPC, 1), mono(1.0, VPCU, 1)]   # objective posynomial

    # vars: y[NGP] | t | e_i (per multi-mono monomial) | p_i | ONE
    nmulti = sum(length(p) for p in P if length(p) > 1) + 2   # + objective
    YBASE, TV, EBASE = 0, NGP, NGP + 1
    PBASE = EBASE + nmulti
    ONEV = PBASE + nmulti
    m = Model(0, 0)
    r = try
        add_vars(m, ONEV + 1)
        for i in 0:NGP-1
            var_bounds(m, YBASE + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, TV, BK_FR; lo=-Inf, up=Inf)
        for i in 0:nmulti-1
            var_bounds(m, EBASE + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:nmulti-1
            var_bounds(m, PBASE + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, ONEV, BK_FX; lo=1.0, up=1.0)
        add_cons(m, MCON + 1 + 2nmulti)
        ecur, pcur, nrow = 0, 0, 0
        for posy in P
            if length(posy) == 1                    # linear: a.y + ln c <= 0
                coef, terms = posy[1]
                a_row(m, nrow, [YBASE + v for (v, _) in terms], [a for (_, a) in terms])
                con_bounds(m, nrow, BK_UP; lo=-Inf, up=-clog(coef)); nrow += 1
            else
                p0 = pcur
                for (coef, terms) in posy           # e - a.y = ln c ; PEXP(p, ONE, e)
                    a_row(m, nrow, [EBASE + ecur; [YBASE + v for (v, _) in terms]], [1.0; [-a for (_, a) in terms]])
                    con_bounds(m, nrow, BK_FX; lo=clog(coef), up=clog(coef)); nrow += 1
                    cone(m, CT_PEXP, [PBASE + pcur, ONEV, EBASE + ecur])
                    ecur += 1; pcur += 1
                end
                a_row(m, nrow, [PBASE + p0 + q for q in 0:length(posy)-1], ones(length(posy)))
                con_bounds(m, nrow, BK_UP; lo=-Inf, up=1.0); nrow += 1
            end
        end
        p0 = pcur                                   # objective: min t, p_k >= exp(a.y + ln c - t), sum p <= 1
        for (coef, terms) in OBJ
            a_row(m, nrow, [EBASE + ecur; [YBASE + v for (v, _) in terms]; TV], [1.0; [-a for (_, a) in terms]; 1.0])
            con_bounds(m, nrow, BK_FX; lo=clog(coef), up=clog(coef)); nrow += 1
            cone(m, CT_PEXP, [PBASE + pcur, ONEV, EBASE + ecur])
            ecur += 1; pcur += 1
        end
        a_row(m, nrow, [PBASE + p0, PBASE + p0 + 1], [1.0, 1.0])
        con_bounds(m, nrow, BK_UP; lo=-Inf, up=1.0); nrow += 1
        cj(m, TV, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    x, topt = zeros(NGP), 0.0
    if ok
        x = [cexp(r.x[YBASE+i+1]) for i in 0:NGP-1]
        topt = r.x[TV+1]
    end
    posyval(posy) = (s = 0.0; for (coef, terms) in posy
        v = coef
        for (var, a) in terms
            v *= cpow(x[var+1], a)
        end
        s += v
    end; s)
    worst_i, worst = -1, -1e300                     # independent check: every posynomial <= 1
    for i in 1:MCON
        s = posyval(P[i])
        s - 1.0 > worst && ((worst, worst_i) = (s - 1.0, i - 1))
    end
    objf = posyval(OBJ)
    okall = ok && worst < 1e-6 && abs(objf - cexp(topt)) < 1e-6 * (1.0 + objf)
    @printf("transformer_design  n=%d m=%d  loss = %.6f\n", NGP, MCON, objf)
    @printf("  Np=%.4f Ns=%.4f Nls=%.4f B=%.6f J=%.4e hbar=%.4f\n",
            x[VNP+1], x[VNS+1], x[VNLS+1], x[VB+1], x[VJ+1], x[VHB+1])
    @printf("  worst posynomial <= 1: %.3e (slot %d)   exp(t)=%.6f\n", worst, worst_i, cexp(topt))
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
