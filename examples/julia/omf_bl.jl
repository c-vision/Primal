#!/usr/bin/env julia
# omf_bl.jl — Black-Litterman portfolio optimization of "Optimization Methods
# in Finance" (Cornuejols & Tutuncu, 2007), Section 8.2. Equilibrium returns
# and covariance of Section 8.1.1; two views (tau = 0.1): mu_M = 0.02
# (omega 1e-5) and mu_S - mu_B = 0.05 (omega 1e-3). The posterior mean
#
#   mu_bar = [(tau Sigma)^-1 + P'Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P'Omega^-1 q]
#
# is checked against the book (0.1177, 0.0751, 0.0234), then the Markowitz QP
# is solved for five targets and checked against Table 8.5.
# Port of examples/c/omf_bl.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const PI_0 = [0.1073, 0.0737, 0.0627]
const SIG = [0.02778 0.00387 0.00021; 0.00387 0.01112 -0.00020; 0.00021 -0.00020 0.00115]
const MU_BOOK = [0.1177, 0.0751, 0.0234]
const R = [0.040, 0.045, 0.050, 0.055, 0.060]
const VAR = [0.0012, 0.0015, 0.0020, 0.0025, 0.0032]
const XS = [0.08, 0.11, 0.15, 0.18, 0.22]
const XB = [0.17, 0.21, 0.24, 0.28, 0.31]

# Invert a 3x3 matrix by cofactors.
function invert3(a)
    d = a[1, 1] * (a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]) -
        a[1, 2] * (a[2, 1] * a[3, 3] - a[2, 3] * a[3, 1]) +
        a[1, 3] * (a[2, 1] * a[3, 2] - a[2, 2] * a[3, 1])
    inv = zeros(3, 3)
    inv[1, 1] = (a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]) / d
    inv[1, 2] = (a[1, 3] * a[3, 2] - a[1, 2] * a[3, 3]) / d
    inv[1, 3] = (a[1, 2] * a[2, 3] - a[1, 3] * a[2, 2]) / d
    inv[2, 1] = (a[2, 3] * a[3, 1] - a[2, 1] * a[3, 3]) / d
    inv[2, 2] = (a[1, 1] * a[3, 3] - a[1, 3] * a[3, 1]) / d
    inv[2, 3] = (a[1, 3] * a[2, 1] - a[1, 1] * a[2, 3]) / d
    inv[3, 1] = (a[2, 1] * a[3, 2] - a[2, 2] * a[3, 1]) / d
    inv[3, 2] = (a[1, 2] * a[3, 1] - a[1, 1] * a[3, 2]) / d
    inv[3, 3] = (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) / d
    return inv
end

# Solve the Markowitz QP for target R[k] and check the book portfolio.
function solve_bl(k, mu)
    m = Model(2, 3)
    r = try
        for j in 0:2
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        qobj(m, [(0, 0, 2 * SIG[1, 1]), (0, 1, 2 * SIG[1, 2]), (0, 2, 2 * SIG[1, 3]),
                 (1, 1, 2 * SIG[2, 2]), (1, 2, 2 * SIG[2, 3]), (2, 2, 2 * SIG[3, 3])])
        cj(m, 0, 0.0); cj(m, 1, 0.0); cj(m, 2, 0.0)
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [0, 1, 2], mu)
        con_bounds(m, 1, BK_LO; lo=R[k], up=Inf)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        var, x = r.objective, r.x
        ok = abs(var - VAR[k]) < 1e-4 && abs(x[1] - XS[k]) < 0.02 && abs(x[2] - XB[k]) < 0.02
        @printf("omf_bl R=%.3f  var=%.4f (book %.4f)  x=(%.2f,%.2f,%.2f) (book %.2f,%.2f,%.2f) %s\n",
                R[k], var, VAR[k], x[1], x[2], x[3], XS[k], XB[k], 1.0 - XS[k] - XB[k], ok ? "OK" : "FAIL")
    else
        @printf("omf_bl R=%.3f rc=%d FAIL\n", R[k], r.rc)
    end
    return ok
end

function main()
    tau = 0.1
    om = [0.00001, 0.001]
    q = [0.02, 0.05]
    P = [0 0 1; 1 -1 0]
    sig = [tau * SIG[i, j] for i in 1:3, j in 1:3]
    sinv = invert3(sig)
    # A = (tau Sigma)^-1 + P'Omega^-1 P ; b = (tau Sigma)^-1 pi + P'Omega^-1 q
    A, b = zeros(3, 3), zeros(3)
    for i in 1:3
        for j in 1:3
            s = 0.0
            for k in 1:2
                s += P[k, i] * (1.0 / om[k]) * P[k, j]
            end
            A[i, j] = sinv[i, j] + s
        end
        s = 0.0
        for k in 1:2
            s += P[k, i] * (1.0 / om[k]) * q[k]
        end
        t2 = 0.0
        for j in 1:3
            t2 += sinv[i, j] * PI_0[j]
        end
        b[i] = t2 + s
    end
    Ai = invert3(A)
    mu = zeros(3)
    for i in 1:3, j in 1:3
        mu[i] += Ai[i, j] * b[j]
    end
    @printf("omf_bl mu_bar=(%.4f,%.4f,%.4f) (book %.4f,%.4f,%.4f) %s\n",
            mu[1], mu[2], mu[3], MU_BOOK[1], MU_BOOK[2], MU_BOOK[3],
            all(i -> abs(mu[i] - MU_BOOK[i]) < 1e-3, 1:3) ? "OK" : "FAIL")
    ok = true
    for k in 1:5
        ok &= solve_bl(k, mu)
    end
    return ok ? 0 : 1
end

exit(main())
