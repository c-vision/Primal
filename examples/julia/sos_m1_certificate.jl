#!/usr/bin/env julia
# sos_m1_certificate.jl — exact rational SOS certificate of the axisymmetric
# sub-case of the "KKT-infused flag algebra" P2 project (jcpaik/p2-kkt-flag-sos).
# On the axisymmetric branch the energy is the bicubic C(s,t) = sum M0[i,j] s^i t^j
# and the committed certificate proves int int C dtau dtau >= 0 via
#
#   C(s,t) = Phi(s)'G Phi(t) + sum_beta lambda_beta B_beta(s,t),  G = P W P',
#
# with W PSD, G v* = 0, lambda >= 0 and corner-vanishing Handelman products.
# The identity is re-checked coefficient by coefficient (as the repository's
# cylinder_cert.py m1 --verify). Port of examples/c/sos_m1_certificate.c.

using Printf

# mode-0 bicubic coefficients
const M0 = [2.0/3 -4 12 -10; -4 84 -270 210; 12 -270 840 -630; -10 210 -630 462]
# face_P: columns span v*^perp, v* = (1, 1/3, 1/3, 1/3)
const FACE_P = [1.0 0 0; -3 1 0; 0 -1 1; 0 0 -1]
# committed rational witness (certificates/m1_axisymmetric.json)
const W = [0.666666666667 -4.634368181351 12.634368181351;
           -4.634368181351 68.817301074893 -175.449591239948;
           12.634368181351 -175.449591239948 450.081881405003]
# Handelman terms: (i,j),(k,l)
const HL = [(0, 5, 1, 2), (1, 2, 1, 6), (1, 2, 2, 6), (1, 2, 4, 0),
            (1, 2, 5, 0), (1, 2, 5, 1), (1, 2, 5, 2), (1, 2, 5, 3)]
const LAM = [5.268736362702, 7.720171650511, 8.532848755941, 0.067019857594,
             13.395507417002, 20.316405933601, 17.878374617311, 8.532848755941]
cpow(x, y) = ccall(:pow, Cdouble, (Cdouble, Cdouble), x, y)   # the C library's pow

# Binomial coefficient n choose k as a double.
function binom(n, k)
    (k < 0 || k > n) && return 0.0
    r = 1.0
    for i in 0:k-1
        r = r * (n - i) / (i + 1)
    end
    return r
end

# coefficient of s^x in s^a (1-s)^j
function s_coef(a, j, x)
    e = x - a
    (e < 0 || e > j) && return 0.0
    s = binom(j, e)
    return e % 2 == 1 ? -s : s
end

# coefficient of s^m t^n in the symmetrized Handelman product p
function handelman(p, m, n)
    f = s_coef(p[1], p[2], m) * s_coef(p[3], p[4], n)
    g = s_coef(p[3], p[4], m) * s_coef(p[1], p[2], n)
    return 0.5 * (f + g)
end

# value of the symmetrized Handelman basis at (s,t)
function handelman_value(p, s, t)
    f = cpow(s, p[1]) * cpow(1 - s, p[2]) * cpow(t, p[3]) * cpow(1 - t, p[4])
    g = cpow(s, p[3]) * cpow(1 - s, p[4]) * cpow(t, p[1]) * cpow(1 - t, p[2])
    return 0.5 * (f + g)
end

# eigenvalues of a symmetric 3x3 matrix (Jacobi)
function eig3(A)
    a = copy(A)
    for _ in 1:100
        off = a[1, 2] * a[1, 2] + a[1, 3] * a[1, 3] + a[2, 3] * a[2, 3]
        off < 1e-30 && break
        for p in 1:2, q in p+1:3
            abs(a[p, q]) < 1e-300 && continue
            theta = (a[q, q] - a[p, p]) / (2.0 * a[p, q])
            t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
            c = 1.0 / sqrt(t * t + 1.0)
            s = t * c
            for k in 1:3
                ap, aq = a[p, k], a[q, k]
                a[p, k] = c * ap - s * aq; a[q, k] = s * ap + c * aq
            end
            for k in 1:3
                ap, aq = a[k, p], a[k, q]
                a[k, p] = c * ap - s * aq; a[k, q] = s * ap + c * aq
            end
        end
    end
    return [a[i, i] for i in 1:3]
end

function main()
    vstar = [1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]
    G = zeros(4, 4)                                 # G = P W P'
    for p in 1:4, q in 1:4
        s = 0.0
        for r in 1:3, t in 1:3
            s += FACE_P[p, r] * W[r, t] * FACE_P[q, t]
        end
        G[p, q] = s
    end
    nres, maxres = 0, 0.0
    for m in 0:8, n in m:8
        gram = (m <= 3 && n <= 3) ? G[m+1, n+1] : 0.0
        hand = 0.0
        for b in 1:8
            hand += LAM[b] * handelman(HL[b], m, n)
        end
        c = (m <= 3 && n <= 3) ? M0[m+1, n+1] : 0.0
        res = abs(gram + hand - c)
        res > maxres && (maxres = res)
        nres += 1
    end
    ev = eig3(W)
    lam_min = minimum(LAM)
    evmin = minimum(ev)
    margin = evmin < lam_min ? evmin : lam_min
    gvmax = 0.0
    for i in 1:4
        s = 0.0
        for j in 1:4
            s += G[i, j] * vstar[j]
        end
        abs(s) > gvmax && (gvmax = abs(s))
    end
    corner_ok = all(abs(handelman_value(HL[b], cs, cd)) <= 1e-12 for b in 1:8 for cs in (0.0, 1.0) for cd in (0.0, 1.0))
    lam_ok = all(l -> l >= 0.0, LAM)
    id_ok, w_ok, gv_ok = maxres < 1e-6, evmin > -1e-9, gvmax < 1e-6
    ok = id_ok && w_ok && gv_ok && lam_ok && corner_ok
    @printf("sos_m1_certificate (deg 8, %d monomials)\n", nres)
    @printf("  (i)   C = Phi^T G Phi + sum lam*B        max|res|=%.3e  %s\n", maxres, id_ok ? "OK" : "FAIL")
    @printf("  (ii)  W PSD                              lambda_min=%.6f  %s\n", evmin, w_ok ? "OK" : "FAIL")
    @printf("  (iii) G v* = 0                           max|=%.3e  %s\n", gvmax, gv_ok ? "OK" : "FAIL")
    @printf("  (iv)  lambda >= 0                        min=%.6f  %s\n", lam_min, lam_ok ? "OK" : "FAIL")
    @printf("  (v)   Handelman corner-vanishing         %s\n", corner_ok ? "OK" : "FAIL")
    @printf("  certified margin min(lmin(W), min lam) = %.6f  ->  int int C dtau dtau >= 0  %s\n",
            margin, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
