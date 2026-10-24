#!/usr/bin/env julia
# steering_robustness.jl — steering robustness of the two-qubit Werner state
# (slmaquedano/semidefinite_programming_workbench). The robustness is the
# minimal t with tau~_{a|x} = sum_l D[a,x,l] tau~_l, tau~_l PSD,
# pi~_{a|x} = tau~_{a|x} - sigma_{a|x} PSD, t = sum_l tr(tau~_l) - 1.
#
# Each 2x2 Hermitian PSD block H = [a c+id; c-id b] is written exactly as the
# rotated cone (a, b, sqrt2 c, sqrt2 d) in Qr^4. Declared deviations, as in the
# C sample: the redundant no-signaling rows are omitted and the objective is
# explicit. Reference: SR = 0 below w ~ 0.56, then 0.0017/0.1412/0.2046/0.2679.
# Port of examples/c/steering_robustness.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# complex numbers as (re, im) tuples, multiplied in the C sample's order
cmul(a, b) = (a[1] * b[1] - a[2] * b[2], a[1] * b[2] + a[2] * b[1])
cadd(a, b) = (a[1] + b[1], a[2] + b[2])
cconj(a) = (a[1], -a[2])

# measurement vector (x = 1..3, a = 1..2): the three mutually unbiased bases
function meas(x, a)
    s = 1.0 / sqrt(2.0)
    x == 1 && return [(s, 0.0), (a == 1 ? s : -s, 0.0)]
    x == 2 && return [(s, 0.0), (0.0, a == 1 ? s : -s)]
    return [(a == 1 ? 1.0 : 0.0, 0.0), (a == 2 ? 1.0 : 0.0, 0.0)]
end

# Werner state p|Phi+><Phi+| + (1-p) I/4, p = w
function werner(w)
    r = fill((0.0, 0.0), 4, 4)
    r[1, 1] = ((1.0 - w) / 4, 0.0); r[4, 4] = ((1.0 - w) / 4, 0.0)
    r[2, 2] = ((1.0 + w) / 4, 0.0); r[3, 3] = ((1.0 + w) / 4, 0.0)
    r[2, 3] = (-w / 2, 0.0); r[3, 2] = (-w / 2, 0.0)
    return r
end

# sigma_{a|x} = tr_A[(M M^dag (x) I) rho]
function assemblage(w, x, a)
    rho, m = werner(w), meas(x + 1, a + 1)
    P = [cmul(m[i], cconj(m[j])) for i in 1:2, j in 1:2]
    sig = fill((0.0, 0.0), 2, 2)
    for iB in 0:1, jB in 0:1
        s = (0.0, 0.0)
        for iA in 0:1, mA in 0:1
            s = cadd(s, cmul(P[iA+1, mA+1], rho[2mA+iB+1, 2iA+jB+1]))
        end
        sig[iB+1, jB+1] = s
    end
    return sig
end

# True when deterministic strategy l answers a to setting x.
Dstrategy(a, x, l) = ((l >> x) & 1) == a

const OA = 2
const MA = 3
const L = 8

# Steering robustness at Werner parameter w, as an SOCP over Qr^4.
function sr(w)
    sig = zeros(OA, MA, 4)                          # sigma_{a,x} as (a, b, sqrt2 Re c, sqrt2 Im c)
    for x in 0:MA-1, a in 0:OA-1
        s = assemblage(w, x, a)
        sig[a+1, x+1, :] = [s[1, 1][1], s[2, 2][1], sqrt(2.0) * s[1, 2][1], sqrt(2.0) * s[1, 2][2]]
    end
    TAU, PI = 0, 4L
    nv = PI + 4 * OA * MA
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        add_cons(m, 4 * OA * MA)                    # pi link rows only
        for l in 0:L-1
            var_bounds(m, TAU + 4l, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:4*OA*MA-1
            var_bounds(m, PI + i, BK_FR; lo=-Inf, up=Inf)
        end
        row = 0
        for x in 0:MA-1, a in 0:OA-1
            piv = 4 * (a * MA + x)
            for cp in 0:3
                sub, val = [PI + piv + cp], [1.0]
                for l in 0:L-1
                    Dstrategy(a, x, l) && (push!(sub, TAU + 4l + cp); push!(val, -1.0))
                end
                a_row(m, row, sub, val)
                con_bounds(m, row, BK_FX; lo=-sig[a+1, x+1, cp+1], up=-sig[a+1, x+1, cp+1]); row += 1
            end
        end
        for x in 0:MA-1, a in 0:OA-1
            p = PI + 4 * (a * MA + x)
            cone(m, CT_RQUAD, [p, p + 1, p + 2, p + 3])
        end
        for l in 0:L-1
            cone(m, CT_RQUAD, [TAU + 4l, TAU + 4l + 1, TAU + 4l + 2, TAU + 4l + 3])
        end
        for l in 0:L-1
            cj(m, TAU + 4l, 1.0); cj(m, TAU + 4l + 1, 1.0)
        end
        cfix(m, -1.0)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, (r.rc == 0 ? r.objective : -1.0)
end

function main()
    println("steering_robustness (Werner state)")
    ws = [0.0, 0.4, 0.5, 0.58, 0.8, 0.9, 1.0]
    want = [0.0000, 0.0000, 0.0000, 0.0017, 0.1412, 0.2046, 0.2679]
    bad = 0
    for i in 1:7
        ok, v = sr(ws[i])
        good = ok && v > -1e-6 && abs(v - want[i]) < 1e-3
        good || (bad += 1)
        @printf("  w=%.2f  SR=%.4f  (ref %.4f)  %s\n", ws[i], v, want[i], good ? "OK" : "FAIL")
    end
    @printf("%s\n", bad == 0 ? "OK" : "FAIL")
    return bad == 0 ? 0 : 1
end

exit(main())
